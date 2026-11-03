"""Isaac Sim side of the G1 ROS2 navigation bridge and Mid-360 sensor."""
from __future__ import annotations
import json
import math
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
ISAAC_RX = ('127.0.0.1', 15001)
ROS_RX = ('127.0.0.1', 15002)

class IsaacNavigationBridge:
    ODOM_PERIOD_S = 0.02
    GOAL_PERIOD_S = 0.5
    COMMAND_TIMEOUT_S = 0.35

    def __init__(self, env, stage, run_command_dds, project_root: str):
        self.env = env
        self.stage = stage
        self.run_command_dds = run_command_dds
        self.root = Path(project_root)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(ISAAC_RX)
        self.sock.setblocking(False)
        self.process = None
        self.goal = None
        self.goal_feedback = None
        self.goal_feedback_at = 0.0
        self.navigation_active = True
        self.navigation_enable_at = None
        self.last_odom = 0.0
        self.last_goal = 0.0
        self.last_command = 0.0
        self.last_command_sent = 0.0
        self.latest_cmd = (0.0, 0.0, 0.0)
        self.status = {'ros_bridge': False, 'odom': False, 'lidar': False, 'path': False, 'cmd_vel': False, 'lidar_points': 0, 'path_poses': 0, 'slam': False, 'slam_map_points': 0, 'slam_live_map_points': 0, 'slam_match_score': 0.0, 'lidar_min_range': 0.0, 'prior_map_loaded': False, 'prior_map_points': 0, 'prior_localized': False, 'prior_match_score': 0.0, 'prior_match_gain': 0.0, 'prior_correction': [0.0, 0.0, 0.0], 'map_to_odom': [0.0, 0.0, 0.0], 'relocalization_attempts': 0, 'relocalization_successes': 0, 'relocalization_rejections': 0, 'sim_truth_position_error_m': 0.0, 'sim_truth_position_rmse_m': 0.0, 'sim_truth_position_max_error_m': 0.0, 'sim_truth_yaw_error_deg': 0.0, 'sim_odom_drift_m': 0.0, 'sim_odom_drift_max_m': 0.0}
        self.slam_pose = None
        self._lidar = None
        self._lidar_hydra_texture = None
        self._old_viewport_tick_rate = None
        self._lidar_sensor_path = None
        self._rtx_graph_ready = False
        self._render_product_path = None
        self._scan_period_s = 0.2
        self._last_scan = 0.0
        self._last_scan_point_count = 0
        self._last_debug = 0.0
        self._mid360_body_index = None
        self._warp_mesh = None
        self._ray_directions_local = None
        self._debug_draw = None
        self._progress_pose = None
        self._progress_check_at = 0.0
        self._stalled_checks = 0
        self._recovery_until = 0.0
        self._recovery_count = 0
        self.status['dynamic_obstacle_avoidance'] = True
        self._setup_mid360_warp()
        self._start_ros_stack()

    def _setup_mid360_warp(self) -> None:
        """Build one GPU ray-cast mesh from the composed factory USD."""
        import numpy as np
        import torch
        from pxr import Usd, UsdGeom
        from isaaclab.utils.warp import convert_to_warp_mesh
        robot = self.env.scene['robot']
        (body_ids, _) = robot.find_bodies('mid360_link')
        if not body_ids:
            raise RuntimeError('G1 mid360_link body was not found')
        self._mid360_body_index = int(body_ids[0])
        all_points: list[np.ndarray] = []
        all_triangles: list[np.ndarray] = []
        vertex_offset = 0
        mesh_count = 0
        triangle_count = 0
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if '/Robot/' in path or not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points_value = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
            counts_value = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
            indices_value = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
            if not points_value or not counts_value or (not indices_value):
                continue
            points = np.asarray(points_value, dtype=np.float64)
            transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).T
            points = points @ transform[:3, :3].T + transform[:3, 3]
            face_counts = np.asarray(counts_value, dtype=np.int64)
            face_indices = np.asarray(indices_value, dtype=np.int64)
            triangles = []
            cursor = 0
            for count in face_counts:
                face = face_indices[cursor:cursor + int(count)]
                cursor += int(count)
                if count < 3:
                    continue
                for index in range(1, int(count) - 1):
                    triangles.append((face[0], face[index], face[index + 1]))
            if not triangles:
                continue
            triangles_np = np.asarray(triangles, dtype=np.int32) + vertex_offset
            all_points.append(points.astype(np.float32))
            all_triangles.append(triangles_np)
            vertex_offset += len(points)
            triangle_count += len(triangles_np)
            mesh_count += 1
        if not all_points:
            raise RuntimeError('factory scene contains no ray-castable USD meshes')
        vertices = np.concatenate(all_points, axis=0)
        triangles = np.concatenate(all_triangles, axis=0)
        geometry_prior_path = os.getenv('G1_EXPORT_GEOMETRY_PRIOR')
        if geometry_prior_path:
            self._export_geometry_prior(vertices, Path(geometry_prior_path), triangles=triangles if os.getenv('G1_SINGLE_POINT_DEMO') == '1' else None)
        self._warp_mesh = convert_to_warp_mesh(vertices, triangles, device=str(self.env.device))
        directions = []
        for pitch_deg in (-20.0, -13.0, -6.0, 1.0, 8.0, 15.0):
            pitch = math.radians(pitch_deg)
            for azimuth_deg in range(0, 360, 4):
                azimuth = math.radians(azimuth_deg)
                directions.append((math.cos(pitch) * math.cos(azimuth), math.cos(pitch) * math.sin(azimuth), math.sin(pitch)))
        self._ray_directions_local = torch.tensor(directions, dtype=torch.float32, device=self.env.device)
        if os.getenv('ISAAC_LIDAR_DEBUG_DRAW', '0') == '1':
            try:
                import omni.kit.app
                omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.util.debug_draw', True)
                from isaacsim.util.debug_draw import _debug_draw
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as exc:
                print(f'[navigation] Isaac point debug draw unavailable: {exc}', flush=True)
        self._rtx_graph_ready = True
        print(f'[navigation] Mid-360 geometry LiDAR ready on mid360_link: {mesh_count} meshes, {len(vertices)} vertices, {triangle_count} triangles, {len(directions)} rays at 5 Hz', flush=True)

    @staticmethod
    def _export_geometry_prior(vertices, path: Path, resolution: float=0.16, triangles=None) -> None:
        """Export a metric, drift-free factory prior from authored USD geometry.

        This is an offline simulation map-building utility, not a localization
        input shortcut: subsequent runs load only the saved PLY and match live
        LiDAR scans in the same way a real robot uses a surveyed/CAD prior.
        """
        import numpy as np
        xyz = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        if triangles is not None:
            spacing = resolution / 2.0
            seed = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.12) & (xyz[:, 2] < 3.5)]
            (_, keep) = np.unique(np.rint(seed / spacing).astype(np.int32), axis=0, return_index=True)
            samples = [seed[keep]]
            indices = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
            for first in range(0, len(indices), 100000):
                faces = xyz[indices[first:first + 100000]]
                edges = faces[:, [1, 2, 0]] - faces
                longest = np.linalg.norm(edges, axis=2).max(axis=1)
                selected = (longest > spacing) & np.isfinite(faces).all(axis=(1, 2)) & (faces[:, :, 2].max(axis=1) > 0.12) & (faces[:, :, 2].min(axis=1) < 3.5)
                for (face, length) in zip(faces[selected], longest[selected]):
                    n = int(np.ceil(length / spacing))
                    for lower in range(0, n + 1, 64):
                        (ii, jj) = np.meshgrid(np.arange(lower, min(n + 1, lower + 64)), np.arange(n + 1), indexing='ij')
                        valid = ii + jj <= n
                        pts = face[0] + ii[valid, None] / n * (face[1] - face[0]) + jj[valid, None] / n * (face[2] - face[0])
                        pts = pts[(pts[:, 2] > 0.12) & (pts[:, 2] < 3.5)]
                        if len(pts):
                            samples.append(pts.astype(np.float32))
                if len(samples) > 2000:
                    pts = np.concatenate(samples)
                    (_, keep) = np.unique(np.rint(pts / spacing).astype(np.int32), axis=0, return_index=True)
                    samples = [pts[keep]]
            xyz = np.concatenate(samples)
        xyz = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.12) & (xyz[:, 2] < 3.5)]
        if triangles is None:
            keys = np.unique(np.rint(xyz / resolution).astype(np.int32), axis=0)
            points = keys.astype(np.float32) * float(resolution)
        else:
            (_, keep) = np.unique(np.rint(xyz / (resolution / 2)).astype(np.int32), axis=0, return_index=True)
            points = xyz[keep]
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream:
            stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
            np.savetxt(stream, points, fmt='%.4f %.4f %.4f')
        print(f"[navigation] drift-free USD geometry prior exported: {path} ({len(points)} points; {('triangle surfaces' if triangles is not None else 'vertices')}; resolution {resolution:.2f} m)", flush=True)

    def _publish_warp_scan(self, now: float) -> None:
        """Cast Mid-360 rays and publish an unregistered body-frame scan."""
        import torch
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        from isaaclab.utils.warp import raycast_mesh
        robot = self.env.scene['robot']
        origin = robot.data.body_link_pos_w[0, self._mid360_body_index]
        orientation = robot.data.body_link_quat_w[0, self._mid360_body_index]
        directions = quat_apply(orientation.unsqueeze(0).repeat(self._ray_directions_local.shape[0], 1), self._ray_directions_local)
        starts = origin.unsqueeze(0).repeat(directions.shape[0], 1)
        (hits, _, _, _) = raycast_mesh(starts.unsqueeze(0), directions.unsqueeze(0), self._warp_mesh, max_dist=8.0)
        hits = hits[0]
        valid = torch.isfinite(hits).all(dim=1)
        hit_points_world = hits[valid]
        root_position = robot.data.root_pos_w[0]
        root_orientation = robot.data.root_quat_w[0]
        relative = hit_points_world - root_position.unsqueeze(0)
        local_points = quat_apply_inverse(root_orientation.unsqueeze(0).repeat(relative.shape[0], 1), relative)
        points = [[round(float(x), 4), round(float(y), 4), round(float(z), 4), 1.0] for (x, y, z) in local_points.detach().cpu().tolist()]
        self._send({'type': 'scan_raw', 'frame_id': 'sensor', 'points': points})
        self._last_scan_point_count = len(points)
        if self._debug_draw is not None:
            self._debug_draw.clear_points()
            display_points = [tuple(point) for point in hit_points_world.detach().cpu().tolist()]
            self._debug_draw.draw_points(display_points, [(0.05, 1.0, 0.25, 1.0)] * len(display_points), [3.0] * len(display_points))

    def _find_robot_prim(self, name: str) -> str:
        matches = [str(p.GetPath()) for p in self.stage.Traverse() if p.GetName() == name]
        if not matches:
            raise RuntimeError(f'G1 {name} prim was not found')
        return next((p for p in matches if '/env_0/Robot/' in p), matches[0])

    def _start_ros_stack(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(ROS_RX)
        except OSError as exc:
            raise RuntimeError('ROS navigation UDP port 15002 is already in use; stop the stale ros2_navigation_bridge.py/localPlanner/pathFollower processes first') from exc
        finally:
            probe.close()
        setup = self.root / 'navigation' / 'install' / 'setup.bash'
        parent_pid = os.getpid()
        command = f"source /opt/ros/humble/setup.bash && source '{setup}' && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && export UNITREE_SIM_PROJECT_ROOT='{self.root}' && export ROS_HOME='{self.root / 'navigation' / 'ros_home'}' && export ROS_LOG_DIR='{self.root / 'navigation' / 'log' / 'runtime'}' && ros2 launch local_planner sim_navigation.launch.py & nav_pid=$!; while kill -0 {parent_pid} 2>/dev/null && kill -0 $nav_pid 2>/dev/null; do sleep 1; done; if kill -0 $nav_pid 2>/dev/null; then kill -TERM -- -$$ 2>/dev/null; fi; wait $nav_pid"
        child_env = os.environ.copy()
        for name in ('AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'PYTHONHOME', 'PYTHONPATH', 'LD_PRELOAD', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            child_env.pop(name, None)
        child_env['LD_LIBRARY_PATH'] = ''
        child_env['PATH'] = '/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        child_env['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
        self.process = subprocess.Popen(['/bin/bash', '--noprofile', '--norc', '-c', command], cwd=str(self.root), env=child_env, start_new_session=True)
        print(f'[navigation] ROS2 local planner started (pid={self.process.pid})', flush=True)

    def set_goal(self, x: float, y: float, yaw: float):
        self.goal = {'type': 'goal', 'x': float(x), 'y': float(y), 'yaw': float(yaw)}
        if os.getenv('G1_SINGLE_POINT_DEMO') == '1':
            import uuid
            self.goal['goal_id'] = uuid.uuid4().hex
            self.goal_feedback = None
        self.navigation_active = True
        self.last_goal = 0.0
        return self.goal.get('goal_id')

    def arrived(self):
        data = self.goal_feedback or {}
        feedback = data.get('feedback', {})
        return bool(self.goal and data.get('goal_id') == self.goal.get('goal_id') and (feedback.get('reached') is True) and (feedback.get('reason') == 'arrival_latched') and (time.monotonic() - self.goal_feedback_at < 1.0))

    def stop_navigation(self) -> None:
        self.navigation_active = False
        self.latest_cmd = (0.0, 0.0, 0.0)
        self._write_velocity(0.0, 0.0, 0.0)

    def _send(self, payload: dict) -> None:
        self.sock.sendto(json.dumps(payload, separators=(',', ':')).encode(), ROS_RX)

    def _publish_odom(self) -> None:
        robot = self.env.scene['robot']
        pos = robot.data.root_pos_w[0].detach().cpu().tolist()
        quat = robot.data.root_quat_w[0].detach().cpu().tolist()
        linear = robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        angular = robot.data.root_ang_vel_w[0].detach().cpu().tolist()
        self._send({'type': 'odom', 'position': pos, 'orientation_wxyz': quat, 'linear_velocity': linear, 'angular_velocity': angular})

    def _drain_udp(self) -> None:
        while True:
            try:
                (raw, _) = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                data = json.loads(raw.decode())
                kind = data.get('type')
                if kind == 'final_follower_ready':
                    self.status['final_follower_ready'] = True
                elif kind == 'goal_feedback':
                    if self.goal and data.get('goal_id') == self.goal.get('goal_id'):
                        self.goal_feedback = data
                        self.goal_feedback_at = time.monotonic()
                elif kind == 'cmd_vel':
                    if self.navigation_active:
                        self.latest_cmd = (float(data['linear_x']), float(data['linear_y']), float(data['angular_z']))
                        self.last_command = time.monotonic()
                elif kind == 'lidar_status':
                    self.status['lidar'] = True
                    self.status['lidar_points'] = int(data.get('points', 0))
                elif kind == 'path_status':
                    self.status['path'] = True
                    self.status['path_poses'] = int(data.get('poses', 0))
                elif kind == 'ros_status':
                    self.status.update({'ros_bridge': True, 'odom': bool(data.get('odom')), 'lidar': bool(data.get('lidar')), 'path': bool(data.get('path')), 'cmd_vel': bool(data.get('cmd_vel')), 'slam': bool(data.get('slam')), 'slam_map_points': int(data.get('slam_map_points', 0)), 'slam_live_map_points': int(data.get('slam_live_map_points', 0)), 'slam_match_score': float(data.get('slam_match_score', 0.0)), 'lidar_min_range': float(data.get('lidar_min_range', 0.0)), 'prior_map_loaded': bool(data.get('prior_map_loaded', False)), 'prior_map_points': int(data.get('prior_map_points', 0)), 'prior_localized': bool(data.get('prior_localized', False)), 'prior_match_score': float(data.get('prior_match_score', 0.0)), 'prior_match_gain': float(data.get('prior_match_gain', 0.0)), 'prior_correction': list(data.get('prior_correction', [0.0, 0.0, 0.0])), 'map_to_odom': list(data.get('map_to_odom', [0.0, 0.0, 0.0])), 'relocalization_attempts': int(data.get('relocalization_attempts', 0)), 'relocalization_successes': int(data.get('relocalization_successes', 0)), 'relocalization_rejections': int(data.get('relocalization_rejections', 0)), 'sim_truth_position_error_m': float(data.get('sim_truth_position_error_m', 0.0)), 'sim_truth_position_rmse_m': float(data.get('sim_truth_position_rmse_m', 0.0)), 'sim_truth_position_max_error_m': float(data.get('sim_truth_position_max_error_m', 0.0)), 'sim_truth_yaw_error_deg': float(data.get('sim_truth_yaw_error_deg', 0.0)), 'sim_odom_drift_m': float(data.get('sim_odom_drift_m', 0.0)), 'sim_odom_drift_max_m': float(data.get('sim_odom_drift_max_m', 0.0))})
                    pose = data.get('slam_pose')
                    if isinstance(pose, list) and len(pose) >= 3:
                        self.slam_pose = tuple((float(value) for value in pose[:3]))
            except Exception as exc:
                print(f'[navigation] ignored malformed UDP packet: {exc}', flush=True)

    def _write_velocity(self, x: float, y: float, yaw: float) -> None:
        self.run_command_dds.write_run_command(str([float(x), float(y), float(yaw), 0.8]))

    def _motion_recovery_command(self, now: float):
        """Detect and unlock the learned walking policy's occasional dead zone."""
        robot = self.env.scene['robot']
        position = robot.data.root_pos_w[0, :2].detach().cpu().tolist()
        if self._progress_pose is None:
            self._progress_pose = position
            self._progress_check_at = now
            return None
        if now < self._recovery_until:
            return (0.6, 0.0, 0.0)
        if now - self._progress_check_at < 2.0:
            return None
        displacement = math.hypot(position[0] - self._progress_pose[0], position[1] - self._progress_pose[1])
        requested_speed = math.hypot(self.latest_cmd[0], self.latest_cmd[1])
        if self.navigation_active and requested_speed >= 0.16 and (displacement < 0.018):
            self._stalled_checks += 1
        else:
            self._stalled_checks = 0
        self._progress_pose = position
        self._progress_check_at = now
        if self._stalled_checks >= 3:
            if float(self.status.get('lidar_min_range', 0.0)) >= 0.75:
                self._recovery_count += 1
                self._recovery_until = now + 1.5
                self._stalled_checks = 0
                print(f'[navigation] walking-policy stall detected; safe gait nudge #{self._recovery_count}', flush=True)
                return (0.6, 0.0, 0.0)
        return None

    def save_slam_map(self, path: str) -> None:
        self._send({'type': 'save_map', 'path': str(path)})

    def step(self, write_commands=True) -> None:
        now = time.monotonic()
        if self.navigation_enable_at is None:
            self.navigation_enable_at = now + 5.0
        if now - self._last_scan >= self._scan_period_s:
            self._publish_warp_scan(now)
            self._last_scan = now
        self._drain_udp()
        if now - self.last_odom >= self.ODOM_PERIOD_S:
            self._publish_odom()
            self.last_odom = now
        if self.goal is not None and self.navigation_active and (now >= self.navigation_enable_at) and (now - self.last_goal >= self.GOAL_PERIOD_S):
            self._send(self.goal)
            self.last_goal = now
        if write_commands and now - self.last_command_sent >= self.ODOM_PERIOD_S:
            recovery_command = self._motion_recovery_command(now)
            if self.navigation_active and now >= self.navigation_enable_at and (now - self.last_command <= self.COMMAND_TIMEOUT_S):
                self._write_velocity(*(recovery_command or self.latest_cmd))
            else:
                self._write_velocity(0.0, 0.0, 0.0)
            self.last_command_sent = now
        if now - self._last_debug >= 1.0:
            robot = self.env.scene['robot']
            pose = robot.data.root_pos_w[0].detach().cpu().tolist()
            quat = robot.data.root_quat_w[0].detach().cpu().tolist()
            (qw, qx, qy, qz) = quat
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            print(f"[navigation] pose=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f},yaw={math.degrees(yaw):.1f}) cmd=({self.latest_cmd[0]:.2f},{self.latest_cmd[1]:.2f},{self.latest_cmd[2]:.2f}) lidar_points={self.status['lidar_points']} ray_hits={self._last_scan_point_count} " + (f'health={self.status}' if os.getenv('G1_SINGLE_POINT_DEMO') != '1' else f"final_goal={(self.goal_feedback or {}).get('feedback', {}).get('reason', 'no_goal')} distance={(self.goal_feedback or {}).get('feedback', {}).get('distance')}"), flush=True)
            self._last_debug = now

    def close(self) -> None:
        try:
            self.stop_navigation()
        except Exception:
            pass
        if self._lidar_hydra_texture is not None:
            try:
                self._lidar_hydra_texture.set_updates_enabled(False)
            except Exception:
                pass
        if self._old_viewport_tick_rate is not None:
            try:
                import carb
                carb.settings.get_settings().set_int('/persistent/app/viewport/defaults/tickRate', self._old_viewport_tick_rate)
            except Exception:
                pass
        self.sock.close()
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                if self.process.poll() is None:
                    self.process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass
