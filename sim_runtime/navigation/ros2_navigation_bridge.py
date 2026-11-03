#!/usr/bin/env python3
"""ROS 2 side of the Isaac Sim navigation bridge.

Isaac Sim 5 runs Python 3.11 while Ubuntu Humble uses Python 3.10.  Keeping
rclpy in this small system-Python process avoids loading two incompatible
Python and DDS runtimes into Kit.  Pose, registered Mid-360 scan, goal and
velocity traffic stay on localhost.
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import (
    PointStamped,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
    TwistStamped,
)
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import TransformBroadcaster

from simple_lidar_slam import CorrelativeLidarSlam


ISAAC_RX = ("127.0.0.1", 15001)
ROS_RX = ("127.0.0.1", 15002)


class IsaacNavigationRosBridge(Node):
    def __init__(self) -> None:
        super().__init__("isaac_navigation_bridge")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(ROS_RX)
        self.sock.setblocking(False)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.odom_pub = self.create_publisher(Odometry, "/state_estimation", 10)
        self.mapping_odom_pub = self.create_publisher(Odometry, "/unitree/slam_mapping/odom", 10)
        self.wheel_odom_pub = self.create_publisher(Odometry, "/wheel_odom", 10)
        self.map_to_odom_pub = self.create_publisher(Odometry, "/map_to_odom", 10)
        self.goal_pub = self.create_publisher(PointStamped, "/way_point", 10)
        self.pose_goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.final_goal_stamp_ns = None
        self.goal_feedback = None
        self.create_subscription(String, "/fastlio_far/goal_status", self._goal_status_callback, 10)
        self.scan_pub = self.create_publisher(PointCloud2, "/registered_scan", 10)
        self.raw_scan_pub = self.create_publisher(
            PointCloud2, "/utlidar/cloud_livox_mid360", qos_profile_sensor_data
        )
        self.mapping_points_pub = self.create_publisher(
            PointCloud2, "/unitree/slam_mapping/points", 10
        )
        self.global_map_pub = self.create_publisher(PointCloud2, "/global_map", 1)
        self.prior_map_pub = self.create_publisher(PointCloud2, "/prior_map", 1)
        self.grid_map_pub = self.create_publisher(OccupancyGrid, "/gridmap", 1)
        self.slam_info_pub = self.create_publisher(String, "/slam_info", 10)
        self.path_map_pub = self.create_publisher(NavPath, "/path_map", 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self._cmd_callback, 10)
        self.create_subscription(NavPath, "/path", self._path_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._initial_pose_callback, 10
        )
        self.create_subscription(PointCloud2, "/registered_scan", self._scan_callback, 10)
        self.create_subscription(
            PointCloud2,
            "/mid360/points_raw",
            self._raw_scan_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(0.005, self._drain_udp)
        self.create_timer(1.0, self._report_health)

        self.latest_goal: dict | None = None
        self.last_odom_at = 0.0
        self.last_cmd_at = 0.0
        self.last_scan_at = 0.0
        self.last_path_at = 0.0
        self.scan_count = 0
        self.last_scan_points = 0
        self.path_count = 0
        self.robot_position: tuple[float, float, float] | None = None
        self.robot_yaw = 0.0
        self.raw_scan_count = 0
        self.raw_scan_points = 0
        prior_path, prior_interval = self._prior_localization_config()
        odom_scale = self._simulation_odom_scale()
        try:
            self.slam = CorrelativeLidarSlam(
                resolution=0.10,
                prior_map_path=prior_path,
                relocalize_every_scans=prior_interval,
                odom_scale=odom_scale,
            )
        except Exception as exc:
            self.get_logger().error(
                f"cannot load configured prior map {prior_path}: {exc}; "
                "continuing in mapping-only mode"
            )
            self.slam = CorrelativeLidarSlam(resolution=0.10)
        self.last_odom_data: dict | None = None
        self.last_map_publish_scan = 0
        self.last_prior_publish_scan = -1000
        self.sim_truth_position_error_m = 0.0
        self.sim_truth_yaw_error_deg = 0.0
        self.sim_odom_drift_m = 0.0
        self.sim_truth_error_samples = 0
        self.sim_truth_position_error_sq_sum = 0.0
        self.sim_truth_position_error_max_m = 0.0
        self.sim_odom_drift_max_m = 0.0
        self._cloud_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.get_logger().info(
            "bridge ready: /utlidar/cloud_livox_mid360 -> odometry + local scan matching "
            "+ 3-D prior-map relocalization "
            "-> /registered_scan + /global_map; UDP odom/goal -> ROS2; "
            "/cmd_vel -> Unitree DDS"
        )
        if self.slam.prior_map_point_count:
            self.get_logger().info(
                f"prior map loaded: {self.slam.prior_map_path} "
                f"({self.slam.prior_map_point_count} structural voxels), "
                f"relocalization every {self.slam.relocalize_every_scans} scans"
            )

    @staticmethod
    def _prior_localization_config() -> tuple[str | None, int]:
        """Resolve an explicit env override or the inspection prior-map file."""
        root = Path(os.environ.get("UNITREE_SIM_PROJECT_ROOT", Path(__file__).parents[1]))
        configured = os.environ.get("G1_PRIOR_MAP")
        interval = 5
        if configured is not None and configured.strip().lower() in {"", "0", "none", "off"}:
            return None, interval
        if configured is None:
            config_path = root / "inspection" / "prior_map.json"
            if config_path.is_file():
                data = json.loads(config_path.read_text(encoding="utf-8"))
                if not bool(data.get("enabled", True)):
                    return None, interval
                configured = data.get("map_path")
                interval = max(1, int(data.get("relocalize_every_scans", interval)))
        if not configured:
            return None, interval
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = root / path
        return str(path.resolve()), interval

    @staticmethod
    def _simulation_odom_scale() -> tuple[float, float, float]:
        """Optional deterministic drift injection used only by regression runs."""
        configured = os.getenv("G1_SIM_ODOM_SCALE", "1.002,0.998,1.001")
        try:
            values = tuple(float(value.strip()) for value in configured.split(","))
        except ValueError as exc:
            raise ValueError("G1_SIM_ODOM_SCALE must contain x,y,yaw scales") from exc
        if len(values) != 3 or any(not math.isfinite(value) for value in values):
            raise ValueError("G1_SIM_ODOM_SCALE must contain three finite numbers")
        return values

    def _send(self, payload: dict) -> None:
        self.sock.sendto(json.dumps(payload, separators=(",", ":")).encode(), ISAAC_RX)

    def _cmd_callback(self, msg: TwistStamped) -> None:
        self.last_cmd_at = time.monotonic()
        self._send({
            "type": "cmd_vel",
            "linear_x": float(msg.twist.linear.x),
            "linear_y": float(msg.twist.linear.y),
            "angular_z": float(msg.twist.angular.z),
        })

    def _initial_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
            1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
        )
        self.slam.set_initial_pose(
            float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), yaw
        )
        self.get_logger().info(
            "rough initial pose received; wide 3-D prior-map relocalization requested"
        )

    def _path_callback(self, msg: NavPath) -> None:
        self.last_path_at = time.monotonic()
        self.path_count += 1
        self._send({"type": "path_status", "poses": len(msg.poses)})
        # localPlanner intentionally publishes a vehicle-relative path for its
        # follower.  Keep that topic untouched and provide RViz with a map
        # projection captured at the same robot pose.
        if self.robot_position is None:
            return
        path_map = NavPath()
        path_map.header = msg.header
        path_map.header.frame_id = "map"
        cos_yaw = math.cos(self.robot_yaw)
        sin_yaw = math.sin(self.robot_yaw)
        origin_x, origin_y, _ = self.robot_position
        for source in msg.poses:
            target = type(source)()
            target.header = source.header
            target.header.frame_id = "map"
            local_x = float(source.pose.position.x)
            local_y = float(source.pose.position.y)
            target.pose.position.x = origin_x + cos_yaw * local_x - sin_yaw * local_y
            target.pose.position.y = origin_y + sin_yaw * local_x + cos_yaw * local_y
            target.pose.position.z = float(source.pose.position.z)
            target.pose.orientation = source.pose.orientation
            path_map.poses.append(target)
        self.path_map_pub.publish(path_map)

    def _raw_scan_callback(self, msg: PointCloud2) -> None:
        """Remove G1 self returns and republish the WORLD-frame RTX cloud."""
        if self.robot_position is None:
            return
        self.raw_scan_count += 1
        try:
            cloud = point_cloud2.read_points(
                msg, field_names=["x", "y", "z"], skip_nans=True
            )
            if cloud.size == 0:
                return
            xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(
                np.float32, copy=False
            )
        except Exception as exc:
            self.get_logger().warning(f"cannot decode RTX point cloud: {exc}")
            return

        robot = np.asarray(self.robot_position, dtype=np.float32)
        delta = xyz - robot
        horizontal_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2
        range_sq = horizontal_sq + delta[:, 2] ** 2
        # The RTX sensor sees the visual G1 meshes.  A 0.60 m horizontal blind
        # disk removes torso/arms/legs while retaining anything outside the
        # physical 0.5 x 0.5 m planner footprint.  Eight metres is more than
        # the planner's 3.5 m adjacent range and keeps RViz responsive.
        keep = (
            np.isfinite(xyz).all(axis=1)
            & (horizontal_sq >= 0.60 ** 2)
            & (range_sq <= 8.0 ** 2)
        )
        xyz = xyz[keep]
        self.raw_scan_points = int(xyz.shape[0])

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = "map"
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        if self.raw_scan_points:
            values = np.column_stack(
                (xyz, np.ones((self.raw_scan_points,), dtype=np.float32))
            ).astype(np.float32, copy=False)
        else:
            values = np.empty((0, 4), dtype=np.float32)
        self.scan_pub.publish(point_cloud2.create_cloud(header, fields, values))

    def _scan_callback(self, msg: PointCloud2) -> None:
        self.last_scan_at = time.monotonic()
        self.scan_count += 1
        point_count = int(msg.width * msg.height)
        self.last_scan_points = point_count
        self._send({
            "type": "lidar_status",
            "points": point_count,
            "frame_id": msg.header.frame_id,
        })

    @staticmethod
    def _yaw_from_wxyz(quaternion) -> float:
        qw, qx, qy, qz = map(float, quaternion)
        return math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def _odom_message(self, data: dict, slam: bool) -> Odometry:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map" if slam else "odom"
        msg.child_frame_id = "sensor"
        p = data["position"]
        q = data["orientation_wxyz"]
        linear = data.get("linear_velocity", [0.0, 0.0, 0.0])
        angular = data.get("angular_velocity", [0.0, 0.0, 0.0])
        if slam:
            sx, sy, syaw = self.slam.pose.tolist()
            msg.pose.pose.position.x = float(sx)
            msg.pose.pose.position.y = float(sy)
            msg.pose.pose.position.z = float(self.slam.pose_z)
            msg.pose.pose.orientation.w = math.cos(0.5 * syaw)
            msg.pose.pose.orientation.z = math.sin(0.5 * syaw)
            msg.pose.covariance[0] = max(0.005, 0.08 * (1.0 - self.slam.match_score))
            msg.pose.covariance[7] = msg.pose.covariance[0]
            msg.pose.covariance[35] = max(0.003, 0.04 * (1.0 - self.slam.match_score))
        else:
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, p)
            msg.pose.pose.orientation.w = float(q[0])
            msg.pose.pose.orientation.x = float(q[1])
            msg.pose.pose.orientation.y = float(q[2])
            msg.pose.pose.orientation.z = float(q[3])
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = map(float, linear)
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z = map(float, angular)
        return msg

    def _map_to_odom_message(self) -> Odometry:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "odom"
        x, y, yaw = self.slam.map_to_odom.tolist()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.w = math.cos(0.5 * yaw)
        msg.pose.pose.orientation.z = math.sin(0.5 * yaw)
        return msg

    def _publish_localization_tf(self) -> None:
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for parent, child, pose, z in (
            ("map", "odom", self.slam.map_to_odom, 0.0),
            ("odom", "sensor", self.slam.odom_pose, self.slam.pose_z),
        ):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = parent
            transform.child_frame_id = child
            transform.transform.translation.x = float(pose[0])
            transform.transform.translation.y = float(pose[1])
            transform.transform.translation.z = float(z)
            transform.transform.rotation.w = math.cos(0.5 * float(pose[2]))
            transform.transform.rotation.z = math.sin(0.5 * float(pose[2]))
            transforms.append(transform)
        self.tf_broadcaster.sendTransform(transforms)

    def _publish_odom(self, data: dict) -> None:
        self.last_odom_data = data
        p = list(map(float, data["position"]))
        yaw = self._yaw_from_wxyz(data["orientation_wxyz"])
        self.slam.update_odometry(p[0], p[1], p[2], yaw)
        self.sim_truth_position_error_m = math.hypot(
            float(self.slam.pose[0]) - p[0], float(self.slam.pose[1]) - p[1]
        )
        self.sim_truth_yaw_error_deg = math.degrees(
            abs((float(self.slam.pose[2]) - yaw + math.pi) % (2.0 * math.pi) - math.pi)
        )
        self.sim_odom_drift_m = math.hypot(
            float(self.slam.odom_pose[0]) - p[0],
            float(self.slam.odom_pose[1]) - p[1],
        )
        # These aggregates are simulation-only acceptance metrics.  They are
        # deliberately calculated after SLAM and are never consumed by the
        # estimator, planner or controller.
        self.sim_truth_error_samples += 1
        self.sim_truth_position_error_sq_sum += self.sim_truth_position_error_m ** 2
        self.sim_truth_position_error_max_m = max(
            self.sim_truth_position_error_max_m, self.sim_truth_position_error_m
        )
        self.sim_odom_drift_max_m = max(
            self.sim_odom_drift_max_m, self.sim_odom_drift_m
        )
        self.wheel_odom_pub.publish(self._odom_message(data, slam=False))
        slam_msg = self._odom_message(data, slam=True)
        self.odom_pub.publish(slam_msg)
        self.mapping_odom_pub.publish(slam_msg)
        self.map_to_odom_pub.publish(self._map_to_odom_message())
        self._publish_localization_tf()
        self.robot_position = (
            float(self.slam.pose[0]), float(self.slam.pose[1]), float(self.slam.pose_z)
        )
        self.robot_yaw = float(self.slam.pose[2])
        self.last_odom_at = time.monotonic()

    def _publish_goal(self, data: dict) -> None:
        if data.get("goal_id"):
            if self.latest_goal and self.latest_goal.get("goal_id") == data["goal_id"]:
                return  # retry of the UDP request must not reset arrival confirmation
            self.latest_goal = data
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            self.final_goal_stamp_ns = pose.header.stamp.sec * 1000000000 + pose.header.stamp.nanosec
            pose.pose.position.x, pose.pose.position.y = float(data["x"]), float(data["y"])
            pose.pose.orientation.w = math.cos(float(data["yaw"]) / 2)
            pose.pose.orientation.z = math.sin(float(data["yaw"]) / 2)
            self.pose_goal_pub.publish(pose)
        self.latest_goal = data
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.point.x = float(data["x"])
        msg.point.y = float(data["y"])
        # This project's G1 local-planner extension carries final world yaw in
        # PointStamped.z.  The upstream planner ignored z entirely.
        msg.point.z = float(data["yaw"])
        self.goal_pub.publish(msg)

    def _goal_status_callback(self, msg):
        try:
            feedback = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        # READY can prove the actual follower is publishing even before a goal.
        self._send({"type": "final_follower_ready"})
        if (not self.latest_goal or not self.latest_goal.get("goal_id")
                or feedback.get("goal_stamp_ns") != self.final_goal_stamp_ns):
            return
        self._send({"type": "goal_feedback", "goal_id": self.latest_goal["goal_id"],
                    "feedback": feedback, "measured_pose": list(self.slam.pose)})

    def _cloud(self, points: np.ndarray, frame_id: str) -> PointCloud2:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        values = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float32)))
        return point_cloud2.create_cloud(header, self._cloud_fields, values)

    def _publish_slam_products(self, data: dict) -> None:
        raw_values = np.asarray(data.get("points", []), dtype=np.float32).reshape(-1, 4)
        raw_xyz = raw_values[:, :3] if len(raw_values) else np.empty((0, 3), np.float32)
        self.raw_scan_count += 1
        self.raw_scan_points = int(len(raw_xyz))
        self.raw_scan_pub.publish(self._cloud(raw_xyz, "sensor"))
        registered = self.slam.update_scan(raw_xyz)
        registered_msg = self._cloud(registered, "map")
        self.scan_pub.publish(registered_msg)
        self.mapping_points_pub.publish(registered_msg)

        if self.last_odom_data is not None:
            slam_msg = self._odom_message(self.last_odom_data, slam=True)
            self.odom_pub.publish(slam_msg)
            self.mapping_odom_pub.publish(slam_msg)
            self.map_to_odom_pub.publish(self._map_to_odom_message())
            self._publish_localization_tf()
            self.robot_position = (
                float(self.slam.pose[0]), float(self.slam.pose[1]), float(self.slam.pose_z)
            )
            self.robot_yaw = float(self.slam.pose[2])

        if self.slam.scan_count - self.last_map_publish_scan >= 5:
            self.global_map_pub.publish(self._cloud(self.slam.map_points(), "map"))
            occupancy = self.slam.occupancy()
            if occupancy is not None:
                grid, origin_x, origin_y = occupancy
                msg = OccupancyGrid()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "map"
                msg.info.resolution = float(self.slam.resolution)
                msg.info.width = int(grid.shape[1])
                msg.info.height = int(grid.shape[0])
                msg.info.origin.position.x = float(origin_x)
                msg.info.origin.position.y = float(origin_y)
                msg.info.origin.orientation.w = 1.0
                msg.data = grid.reshape(-1).astype(np.int8).tolist()
                self.grid_map_pub.publish(msg)
            self.last_map_publish_scan = self.slam.scan_count

        if (
            self.slam.prior_map_point_count
            and self.slam.scan_count - self.last_prior_publish_scan >= 25
        ):
            self.prior_map_pub.publish(
                self._cloud(self.slam.prior_map_points(), "map")
            )
            self.last_prior_publish_scan = self.slam.scan_count

        info = {
            "mode": (
                "prior_map_localization_and_mapping"
                if self.slam.prior_map_point_count else "mapping"
            ),
            "scan_count": self.slam.scan_count,
            "map_points": self.slam.total_map_point_count,
            "live_map_points": self.slam.map_point_count,
            "match_score": round(self.slam.match_score, 4),
            "correction": [round(float(value), 5) for value in self.slam.last_correction],
            "map_to_odom": [round(float(value), 5) for value in self.slam.map_to_odom],
            "prior_map": self.slam.prior_map_path,
            "prior_map_voxels": self.slam.prior_map_point_count,
            "prior_localized": self.slam.prior_localized,
            "prior_match_score": round(self.slam.prior_match_score, 4),
            "prior_match_gain": round(self.slam.prior_match_gain, 4),
            "prior_correction": [
                round(float(value), 5) for value in self.slam.last_prior_correction
            ],
            "relocalization_attempts": self.slam.relocalization_attempts,
            "relocalization_successes": self.slam.relocalization_successes,
            "relocalization_rejections": self.slam.relocalization_rejections,
            "pose": [round(float(value), 5) for value in self.slam.pose],
            "odometry_scale_for_simulation": self.slam.odom_scale.tolist(),
            "simulation_validation": {
                "position_error_m": round(self.sim_truth_position_error_m, 5),
                "position_rmse_m": round(
                    math.sqrt(
                        self.sim_truth_position_error_sq_sum
                        / max(1, self.sim_truth_error_samples)
                    ),
                    5,
                ),
                "position_max_error_m": round(
                    self.sim_truth_position_error_max_m, 5
                ),
                "yaw_error_deg": round(self.sim_truth_yaw_error_deg, 4),
                "uncorrected_odom_drift_m": round(self.sim_odom_drift_m, 5),
                "uncorrected_odom_max_drift_m": round(
                    self.sim_odom_drift_max_m, 5
                ),
            },
        }
        self.slam_info_pub.publish(String(data=json.dumps(info, separators=(",", ":"))))

    def _save_map(self, path: str) -> None:
        try:
            count = self.slam.save(Path(path))
            self.get_logger().info(f"SLAM map saved: {path} ({count} points)")
        except Exception as exc:
            self.get_logger().error(f"cannot save SLAM map {path}: {exc}")

    def _drain_udp(self) -> None:
        while True:
            try:
                raw, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                data = json.loads(raw.decode())
                kind = data.get("type")
                if kind == "odom":
                    self._publish_odom(data)
                elif kind == "goal":
                    self._publish_goal(data)
                elif kind == "scan_raw":
                    self._publish_slam_products(data)
                elif kind == "save_map":
                    self._save_map(str(data["path"]))
            except Exception as exc:
                self.get_logger().warning(f"bad Isaac UDP packet: {exc}")

    def _report_health(self) -> None:
        now = time.monotonic()
        payload = {
            "type": "ros_status",
            "odom": now - self.last_odom_at < 0.5,
            "lidar": now - self.last_scan_at < 2.0 and self.last_scan_points > 0,
            "path": now - self.last_path_at < 2.0,
            "cmd_vel": now - self.last_cmd_at < 0.5,
            "slam": self.slam.initialized and self.slam.scan_count > 0,
            "slam_map_points": self.slam.total_map_point_count,
            "slam_live_map_points": self.slam.map_point_count,
            "slam_match_score": self.slam.match_score,
            "prior_map_loaded": self.slam.prior_map_point_count > 0,
            "prior_map_points": self.slam.prior_map_point_count,
            "prior_localized": self.slam.prior_localized,
            "prior_match_score": self.slam.prior_match_score,
            "prior_match_gain": self.slam.prior_match_gain,
            "prior_correction": self.slam.last_prior_correction.tolist(),
            "map_to_odom": self.slam.map_to_odom.tolist(),
            "relocalization_attempts": self.slam.relocalization_attempts,
            "relocalization_successes": self.slam.relocalization_successes,
            "relocalization_rejections": self.slam.relocalization_rejections,
            # Simulation validation only; never consumed by localization or planning.
            "sim_truth_position_error_m": self.sim_truth_position_error_m,
            "sim_truth_position_rmse_m": math.sqrt(
                self.sim_truth_position_error_sq_sum
                / max(1, self.sim_truth_error_samples)
            ),
            "sim_truth_position_max_error_m": self.sim_truth_position_error_max_m,
            "sim_truth_yaw_error_deg": self.sim_truth_yaw_error_deg,
            "sim_odom_drift_m": self.sim_odom_drift_m,
            "sim_odom_drift_max_m": self.sim_odom_drift_max_m,
            "lidar_min_range": self.slam.last_min_range,
            "slam_pose": self.slam.pose.tolist(),
            "scan_messages": self.scan_count,
            "raw_scan_messages": self.raw_scan_count,
            "raw_scan_points": self.raw_scan_points,
            "path_messages": self.path_count,
        }
        self._send(payload)

    def destroy_node(self) -> bool:
        self.sock.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = IsaacNavigationRosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
