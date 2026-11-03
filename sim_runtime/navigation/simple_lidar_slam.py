"""Small dependency-free 2.5D LiDAR SLAM backend for the Isaac simulation.

The real G1 uses Unitree/arise_slam with Mid-360 and IMU messages.  Isaac's
synthetic scanner does not provide Livox per-point timing, so this module uses
kinematic odometry for prediction and correlative scan-to-map matching for
correction.  It deliberately exposes the same mapping products used by the
navigation stack: a corrected pose, registered scans and a cumulative map.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


class CorrelativeLidarSlam:
    """Odometry prediction, local mapping and prior-map relocalization.

    The high-rate odometry frame is allowed to drift.  Local scan matching
    keeps consecutive scans smooth, while a separate immutable 3-D prior map
    periodically estimates ``map -> odom``.  This mirrors the architecture of
    FAST_LIO_LOCALIZATION without requiring Livox per-point timestamps in the
    synthetic Isaac scanner.
    """

    def __init__(
        self,
        resolution: float = 0.10,
        prior_map_path: str | Path | None = None,
        relocalize_every_scans: int = 5,
        odom_scale: tuple[float, float, float] = (1.002, 0.998, 1.001),
    ) -> None:
        self.resolution = float(resolution)
        self.prior_resolution = max(0.16, 2.0 * self.resolution)
        self.relocalize_every_scans = max(1, int(relocalize_every_scans))
        self.odom_scale = np.asarray(odom_scale, dtype=np.float64)
        self.pose = np.zeros(3, dtype=np.float64)
        self.odom_pose = np.zeros(3, dtype=np.float64)
        self.raw_odom_pose = np.zeros(3, dtype=np.float64)
        self.map_to_odom = np.zeros(3, dtype=np.float64)
        self.pose_z = 0.0
        self.last_odom: np.ndarray | None = None
        self.initialized = False
        self.scan_count = 0
        self.match_score = 0.0
        self.last_min_range = 0.0
        self.last_correction = np.zeros(3, dtype=np.float64)
        self.prior_map_path: str | None = None
        self.prior_match_score = 0.0
        self.prior_match_gain = 0.0
        self.last_prior_correction = np.zeros(3, dtype=np.float64)
        self.prior_localized = False
        self.relocalization_attempts = 0
        self.relocalization_successes = 0
        self.relocalization_rejections = 0
        self._prior_voxels: set[tuple[int, int, int]] = set()
        self._prior_points = np.empty((0, 3), dtype=np.float32)
        self._force_relocalization = True
        self._initial_pose_seed: np.ndarray | None = None
        self._voxels: dict[tuple[int, int, int], np.ndarray] = {}
        self._occupied_xy: set[tuple[int, int]] = set()
        self._free_xy: set[tuple[int, int]] = set()
        # Local mapping is a short-term stabilizer.  Once an immutable prior
        # is available it must not overpower the global map-to-odom estimate.
        self.local_mapping_gain = 0.0 if prior_map_path else 0.65
        self.local_minimum_improvement = 0.010
        # Consensus already rejects isolated outliers, so an accepted global
        # correction can carry enough weight to catch up with real drift.
        self.prior_tracking_gain = 0.60
        self.prior_consistency_frames = 3
        self.prior_translation_deadband = 0.14
        self.prior_yaw_deadband = math.radians(3.0)
        self._prior_correction_candidates: list[np.ndarray] = []
        if prior_map_path:
            self.load_prior_map(prior_map_path)

    @property
    def map_point_count(self) -> int:
        return len(self._voxels)

    @property
    def prior_map_point_count(self) -> int:
        return len(self._prior_voxels)

    @property
    def total_map_point_count(self) -> int:
        return len(self._prior_points) + len(self._voxels)

    @staticmethod
    def _compose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Compose two SE(2) poses: ``first * second``."""
        c, s = math.cos(float(first[2])), math.sin(float(first[2]))
        return np.asarray(
            (
                first[0] + c * second[0] - s * second[1],
                first[1] + s * second[0] + c * second[1],
                wrap_angle(first[2] + second[2]),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _inverse(pose: np.ndarray) -> np.ndarray:
        c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
        return np.asarray(
            (-c * pose[0] - s * pose[1], s * pose[0] - c * pose[1], -pose[2]),
            dtype=np.float64,
        )

    def _sync_map_to_odom(self) -> None:
        self.map_to_odom = self._compose(self.pose, self._inverse(self.odom_pose))

    @staticmethod
    def _read_ascii_cloud(path: Path) -> np.ndarray:
        """Read the xyz columns of an ASCII PLY/PCD prior map."""
        suffix = path.suffix.lower()
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            if suffix == ".ply":
                first = stream.readline().strip()
                if first != "ply":
                    raise ValueError(f"not a PLY file: {path}")
                is_ascii = False
                while True:
                    line = stream.readline()
                    if not line:
                        raise ValueError(f"incomplete PLY header: {path}")
                    stripped = line.strip()
                    if stripped == "format ascii 1.0":
                        is_ascii = True
                    if stripped == "end_header":
                        break
                if not is_ascii:
                    raise ValueError("only ASCII PLY prior maps are supported")
            elif suffix == ".pcd":
                fields: list[str] = []
                is_ascii = False
                while True:
                    line = stream.readline()
                    if not line:
                        raise ValueError(f"incomplete PCD header: {path}")
                    stripped = line.strip()
                    if stripped.upper().startswith("FIELDS "):
                        fields = stripped.split()[1:]
                    if stripped.upper() == "DATA ASCII":
                        is_ascii = True
                        break
                    if stripped.upper().startswith("DATA "):
                        break
                if not is_ascii:
                    raise ValueError("only ASCII PCD prior maps are supported")
                if fields[:3] != ["x", "y", "z"]:
                    raise ValueError("PCD prior map must start with x y z fields")
            else:
                raise ValueError(f"unsupported prior-map format: {path.suffix}")
            points = np.loadtxt(stream, dtype=np.float32, usecols=(0, 1, 2))
        return np.asarray(points, dtype=np.float32).reshape(-1, 3)

    def set_prior_map(self, points: np.ndarray, source: str = "memory") -> int:
        """Install an immutable 3-D prior map and return its voxel count."""
        xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        # Ground is excellent for height but provides no horizontal constraint.
        # Retain walls, racks and machinery for an observable x/y/yaw match.
        useful = xyz[(xyz[:, 2] > 0.12) & (xyz[:, 2] < 3.50)]
        if len(useful) < 80:
            raise ValueError("prior map has too few non-ground 3-D points")
        keys = np.rint(useful / self.prior_resolution).astype(np.int32)
        unique_keys, indices = np.unique(keys, axis=0, return_index=True)
        self._prior_voxels = {tuple(row.tolist()) for row in unique_keys}
        self._prior_points = useful[indices].astype(np.float32, copy=True)
        self.prior_map_path = str(source)
        self.prior_localized = False
        self._force_relocalization = True
        self.local_mapping_gain = 0.0
        self._prior_correction_candidates.clear()
        return len(self._prior_voxels)

    def load_prior_map(self, path: str | Path) -> int:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"prior map not found: {source}")
        return self.set_prior_map(self._read_ascii_cloud(source), str(source))

    def prior_map_points(self) -> np.ndarray:
        return self._prior_points

    def set_initial_pose(self, x: float, y: float, yaw: float) -> None:
        """Apply a rough map-frame seed, then request a wide prior-map match."""
        self.pose[:] = (float(x), float(y), wrap_angle(yaw))
        self._initial_pose_seed = self.pose.copy()
        if self.last_odom is not None:
            self._sync_map_to_odom()
        self.prior_localized = False
        self._force_relocalization = True

    def update_odometry(self, x: float, y: float, z: float, yaw: float) -> None:
        odom = np.asarray((x, y, wrap_angle(yaw)), dtype=np.float64)
        self.raw_odom_pose[:] = odom
        self.pose_z = float(z)
        if self.last_odom is None:
            # Align the SLAM map with the authored factory frame on startup.
            # Recorded patrol waypoints can therefore be reused in relocation.
            self.odom_pose[:] = odom
            if self._initial_pose_seed is not None:
                self.pose[:] = self._initial_pose_seed
                self.map_to_odom = self._compose(
                    self.pose, self._inverse(self.odom_pose)
                )
            else:
                self.pose[:] = self._compose(self.map_to_odom, self.odom_pose)
            self.last_odom = odom
            self.initialized = True
            return
        delta = odom - self.last_odom
        delta[2] = wrap_angle(delta[2])
        # This is the high-rate odometry estimate.  A deterministic scale
        # error makes drift observable in simulation; prior-map matching must
        # estimate map->odom instead of secretly reading simulator truth.
        self.odom_pose += delta * self.odom_scale
        self.odom_pose[2] = wrap_angle(self.odom_pose[2])
        self.pose[:] = self._compose(self.map_to_odom, self.odom_pose)
        self.last_odom = odom

    def _transform(self, points: np.ndarray, pose: np.ndarray | None = None) -> np.ndarray:
        pose = self.pose if pose is None else pose
        c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
        result = np.empty_like(points, dtype=np.float32)
        result[:, 0] = c * points[:, 0] - s * points[:, 1] + float(pose[0])
        result[:, 1] = s * points[:, 0] + c * points[:, 1] + float(pose[1])
        result[:, 2] = points[:, 2] + float(self.pose_z)
        return result

    def _score_pose(self, local_xy: np.ndarray, candidate: np.ndarray) -> float:
        c, s = math.cos(float(candidate[2])), math.sin(float(candidate[2]))
        x = c * local_xy[:, 0] - s * local_xy[:, 1] + float(candidate[0])
        y = s * local_xy[:, 0] + c * local_xy[:, 1] + float(candidate[1])
        ix = np.rint(x / self.resolution).astype(np.int32)
        iy = np.rint(y / self.resolution).astype(np.int32)
        exact = near = 0
        for gx, gy in zip(ix.tolist(), iy.tolist()):
            if (gx, gy) in self._occupied_xy:
                exact += 1
            elif any(
                (gx + dx, gy + dy) in self._occupied_xy
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                near += 1
        return (exact + 0.35 * near) / max(1, len(ix))

    def _score_prior_pose(self, local_xyz: np.ndarray, candidate: np.ndarray) -> float:
        if not self._prior_voxels or not len(local_xyz):
            return 0.0
        transformed = self._transform(local_xyz, candidate)
        keys = np.rint(transformed / self.prior_resolution).astype(np.int32)
        exact = near = 0
        # A six-neighbour tolerance absorbs voxel-boundary and ray sampling
        # differences without making parallel factory aisles interchangeable.
        neighbours = (
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        )
        for gx, gy, gz in keys.tolist():
            key = (gx, gy, gz)
            if key in self._prior_voxels:
                exact += 1
            elif any(
                (gx + dx, gy + dy, gz + dz) in self._prior_voxels
                for dx, dy, dz in neighbours
            ):
                near += 1
        return (exact + 0.35 * near) / max(1, len(keys))

    def _match(self, points: np.ndarray) -> None:
        if len(self._occupied_xy) < 80 or len(points) < 30:
            self.match_score = 0.0
            self.last_correction[:] = 0.0
            return
        local_xy = points[::2, :2]
        predicted = self.pose.copy()
        best = predicted.copy()
        best_score = self._score_pose(local_xy, best)
        yaw_offsets = np.deg2rad(np.arange(-3.0, 3.01, 1.0))
        offsets = np.arange(-0.12, 0.121, 0.04)
        for dyaw in yaw_offsets:
            for dx in offsets:
                for dy in offsets:
                    candidate = predicted + np.asarray((dx, dy, dyaw))
                    candidate[2] = wrap_angle(candidate[2])
                    score = self._score_pose(local_xy, candidate)
                    # Prefer a small correction when scores are effectively tied.
                    score -= 0.015 * (abs(dx) + abs(dy)) / 0.12
                    score -= 0.008 * abs(float(dyaw)) / math.radians(3.0)
                    if score > best_score:
                        best_score, best = score, candidate
        improvement = float(best_score - self._score_pose(local_xy, predicted))
        correction = best - predicted
        correction[2] = wrap_angle(correction[2])
        if improvement < self.local_minimum_improvement:
            correction[:] = 0.0
        # Damping avoids visible pose jumps from sparse Mid-360 returns.
        correction[:2] = np.clip(correction[:2], -0.08, 0.08)
        correction[2] = float(np.clip(correction[2], -math.radians(2), math.radians(2)))
        applied = self.local_mapping_gain * correction
        self.pose += applied
        self.pose[2] = wrap_angle(self.pose[2])
        self.last_correction = applied
        self.match_score = max(0.0, float(best_score))
        # Preserve the correction when the next high-rate odometry sample
        # arrives.  It is map->odom that changes, never the raw odometry.
        self._sync_map_to_odom()

    def _match_prior(self, points: np.ndarray) -> bool:
        """Low-frequency scan-to-prior-map correction with rejection gates."""
        self.relocalization_attempts += 1
        if len(self._prior_voxels) < 80 or len(points) < 30:
            self.prior_match_score = 0.0
            self.last_prior_correction[:] = 0.0
            self._prior_correction_candidates.clear()
            self.relocalization_rejections += 1
            return False

        # Match structural points only.  Subsampling bounds CPU cost and gives
        # each height band comparable influence in a dense factory map.
        local = points[(points[:, 2] + self.pose_z > 0.12) & (points[:, 2] + self.pose_z < 3.50)]
        if len(local) < 25:
            local = points
        local = local[::2]
        predicted = self.pose.copy()
        predicted_score = self._score_prior_pose(local, predicted)
        best = predicted.copy()
        best_score = predicted_score

        if self._force_relocalization or not self.prior_localized:
            xy_offsets = np.arange(-0.80, 0.801, 0.20)
            yaw_offsets = np.deg2rad(np.arange(-10.0, 10.01, 2.5))
        else:
            xy_offsets = np.arange(-0.30, 0.301, 0.10)
            yaw_offsets = np.deg2rad(np.arange(-4.0, 4.01, 2.0))

        for dyaw in yaw_offsets:
            for dx in xy_offsets:
                for dy in xy_offsets:
                    candidate = predicted + np.asarray((dx, dy, dyaw))
                    candidate[2] = wrap_angle(candidate[2])
                    score = self._score_prior_pose(local, candidate)
                    # Prefer the odometry prediction when candidates are tied.
                    score -= 0.004 * math.hypot(float(dx), float(dy))
                    score -= 0.002 * abs(float(dyaw)) / math.radians(10.0)
                    if score > best_score:
                        best_score, best = score, candidate

        correction = best - predicted
        correction[2] = wrap_angle(correction[2])
        improvement = float(best_score - predicted_score)
        minimum_score = 0.22 if self.prior_localized else 0.28
        minimum_improvement = 0.008 if self.prior_localized else 0.015
        prediction_is_already_good = predicted_score >= 0.55
        credible = best_score >= minimum_score and (
            improvement >= minimum_improvement or prediction_is_already_good
        )
        if not credible:
            self.prior_match_score = max(0.0, float(best_score))
            self.prior_match_gain = improvement
            self.last_prior_correction[:] = 0.0
            # Consensus must come from consecutive trustworthy scans.  A
            # sparse/ambiguous frame breaks the sequence instead of allowing
            # stale candidates to be combined after an occlusion.
            self._prior_correction_candidates.clear()
            self.relocalization_rejections += 1
            self._force_relocalization = False
            return False

        # FAST_LIO_LOCALIZATION fuses a low-rate global correction with smooth
        # high-rate odometry.  Apply the same idea here instead of teleporting
        # the planner pose.  The first accepted alignment receives more weight.
        # On a high-scoring prediction, a tiny apparent gain is normally a
        # quantization plateau rather than evidence of motion.  Accept the
        # localization health-wise, but do not inject that voxel-sized offset
        # into map->odom.  A real accumulated drift produces a clear score
        # gain and passes the normal correction branch below.
        if improvement < minimum_improvement and prediction_is_already_good:
            correction[:] = 0.0

        # Corrections below the effective CAD/voxel observability are more
        # likely discretization noise than accumulated drift.  Let odometry
        # carry these sub-voxel motions and wait for a measurable residual.
        if (
            float(np.linalg.norm(correction[:2])) < self.prior_translation_deadband
            and abs(float(correction[2])) < self.prior_yaw_deadband
        ):
            correction[:] = 0.0

        # A single sparse scan is ambiguous in repeated factory aisles.  Keep
        # several independent scan-to-prior solutions and only apply their
        # median when they agree spatially.  This prevents alternating voxel
        # optima from being integrated into a false map->odom drift.
        if np.linalg.norm(correction[:2]) > 1e-6 or abs(float(correction[2])) > 1e-6:
            self._prior_correction_candidates.append(correction.copy())
            self._prior_correction_candidates = self._prior_correction_candidates[
                -self.prior_consistency_frames:
            ]
            if len(self._prior_correction_candidates) < self.prior_consistency_frames:
                self.prior_match_score = max(0.0, float(best_score))
                self.prior_match_gain = improvement
                self.last_prior_correction[:] = 0.0
                self.relocalization_rejections += 1
                self._force_relocalization = False
                return False
            candidates = np.stack(self._prior_correction_candidates)
            consensus = np.median(candidates, axis=0)
            yaw_residual = np.asarray(
                [wrap_angle(value - consensus[2]) for value in candidates[:, 2]]
            )
            consistent = (
                float(np.max(np.linalg.norm(candidates[:, :2] - consensus[:2], axis=1)))
                <= 0.14
                and float(np.max(np.abs(yaw_residual))) <= math.radians(2.5)
            )
            if not consistent:
                self.prior_match_score = max(0.0, float(best_score))
                self.prior_match_gain = improvement
                self.last_prior_correction[:] = 0.0
                self.relocalization_rejections += 1
                self._force_relocalization = False
                return False
            correction = consensus
            self._prior_correction_candidates.clear()
        else:
            self._prior_correction_candidates.clear()
        gain = 0.70 if not self.prior_localized else self.prior_tracking_gain
        correction[:2] = np.clip(correction[:2], -0.45, 0.45)
        correction[2] = float(
            np.clip(correction[2], -math.radians(5.0), math.radians(5.0))
        )
        applied = gain * correction
        self.pose += applied
        self.pose[2] = wrap_angle(self.pose[2])
        self._sync_map_to_odom()
        self.prior_match_score = max(0.0, float(best_score))
        self.prior_match_gain = improvement
        self.last_prior_correction = applied
        self.prior_localized = True
        self._force_relocalization = False
        self.relocalization_successes += 1
        return True

    def _raytrace_free(self, endpoints: np.ndarray) -> None:
        sx = int(round(float(self.pose[0]) / self.resolution))
        sy = int(round(float(self.pose[1]) / self.resolution))
        for point in endpoints[::3]:
            ex = int(round(float(point[0]) / self.resolution))
            ey = int(round(float(point[1]) / self.resolution))
            steps = max(abs(ex - sx), abs(ey - sy))
            if steps < 2:
                continue
            for index in range(1, steps - 1):
                ratio = index / steps
                cell = (int(round(sx + ratio * (ex - sx))), int(round(sy + ratio * (ey - sy))))
                if cell not in self._occupied_xy:
                    self._free_xy.add(cell)

    def _integrate(self, points_map: np.ndarray) -> None:
        for point in points_map:
            key = tuple(np.rint(point / self.resolution).astype(np.int32).tolist())
            if key not in self._voxels:
                self._voxels[key] = point.astype(np.float32, copy=True)
            self._occupied_xy.add((key[0], key[1]))
            self._free_xy.discard((key[0], key[1]))
        self._raytrace_free(points_map)

    def update_scan(self, points_sensor: np.ndarray) -> np.ndarray:
        points = np.asarray(points_sensor, dtype=np.float32).reshape(-1, 3)
        if not self.initialized or not len(points):
            return np.empty((0, 3), dtype=np.float32)
        ranges = np.linalg.norm(points[:, :2], axis=1)
        finite_ranges = ranges[np.isfinite(ranges) & (ranges > 0.45)]
        self.last_min_range = float(finite_ranges.min()) if len(finite_ranges) else 0.0
        points = points[np.isfinite(points).all(axis=1) & (ranges > 0.45) & (ranges < 8.0)]
        if not len(points):
            return np.empty((0, 3), dtype=np.float32)
        self._match(points)
        if self._prior_voxels and (
            self._force_relocalization
            or self.scan_count % self.relocalize_every_scans == 0
        ):
            self._match_prior(points)
        registered = self._transform(points)
        self._integrate(registered)
        self.scan_count += 1
        return registered

    def map_points(self) -> np.ndarray:
        if not self._voxels:
            return np.empty((0, 3), dtype=np.float32)
        return np.stack(tuple(self._voxels.values())).astype(np.float32, copy=False)

    def combined_map_points(self) -> np.ndarray:
        live = self.map_points()
        if not len(self._prior_points):
            return live
        if not len(live):
            return self._prior_points
        return np.concatenate((self._prior_points, live), axis=0)

    def occupancy(self) -> tuple[np.ndarray, float, float] | None:
        prior_xy = {
            (
                int(round(float(point[0]) / self.resolution)),
                int(round(float(point[1]) / self.resolution)),
            )
            for point in self._prior_points
        }
        occupied = self._occupied_xy | prior_xy
        cells = occupied | self._free_xy
        if not cells:
            return None
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        min_x, max_x = min(xs) - 3, max(xs) + 3
        min_y, max_y = min(ys) - 3, max(ys) + 3
        grid = np.full((max_y - min_y + 1, max_x - min_x + 1), -1, dtype=np.int8)
        for x, y in self._free_xy:
            grid[y - min_y, x - min_x] = 0
        for x, y in occupied:
            grid[y - min_y, x - min_x] = 100
        return grid, min_x * self.resolution, min_y * self.resolution

    def save(self, path: str | Path) -> int:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Preserve both the immutable prior and newly observed/updated areas in
        # the final patrol artifact.  Runtime topics publish them separately.
        points = self.combined_map_points()
        with output.open("w", encoding="utf-8") as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(points)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x, y, z in points:
                stream.write(f"{float(x):.4f} {float(y):.4f} {float(z):.4f}\n")
        return len(points)
