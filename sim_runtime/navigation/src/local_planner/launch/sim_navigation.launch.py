"""Launch G1 local planning, simulated LiDAR SLAM and optional RViz."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("local_planner")
    config = os.path.join(share, "config", "omniDir.yaml")
    robot_config = os.path.join(share, "config", "unitree", "unitree_g1.yaml")
    project_root = os.environ["UNITREE_SIM_PROJECT_ROOT"]
    bridge = os.path.join(project_root, "navigation", "ros2_navigation_bridge.py")

    common = [config, robot_config]
    planner = Node(
        package="local_planner",
        executable="localPlanner",
        name="localPlanner",
        output="screen",
        parameters=[*common, {
            "pathFolder": os.path.join(share, "paths"),
            "useTerrainAnalysis": False,
            # /registered_scan is the filtered RTX Mid-360 cloud.  Replan on
            # every scan and also prevent in-place rotation into close objects.
            "checkObstacle": True,
            "checkRotObstacle": True,
            "autonomyMode": True,
            "twoWayDrive": False,
            # This learned G1 policy can enter a walking dead zone around
            # 0.25 m/s after a stop/turn transition.  A 0.40 m/s cruise stays
            # inside its stable gait regime; obstacle checks remain enabled.
            "maxSpeed": 0.45,
            "autonomySpeed": 0.40,
            # Legacy position-only stopping behind the robot at 0.8 m would
            # suppress every recovery path before the 0.10 m pose arrival gate.
            "goalBehindRange": 0.05 if os.getenv("G1_SINGLE_POINT_DEMO") == "1" else 0.8,
        }],
    )
    follower = Node(
        package="local_planner",
        executable="inspectionPathFollower",
        name="pathFollower",
        output="screen",
        parameters=[*common, {
            "pubSkipNum": 1,
            "realRobot": False,
            "useFinalGoalPose": os.getenv("G1_SINGLE_POINT_DEMO") == "1",
            "autonomyMode": True,
            "twoWayDrive": False,
            "noRotAtGoal": False,
            "useInclRateToSlow": False,
            "useInclToStop": False,
            "maxSpeed": 0.45,
            "autonomySpeed": 0.40,
            "maxAccel": 0.40,
            # In-place yaw below about 0.7 rad/s is inside this locomotion
            # policy's practical dead zone.  The upstream robot limit is
            # 80 deg/s; use it in simulation so final pose alignment works.
            "maxYawRate": 80.0,
            "yawRateGain": 2.0,
            "stopYawRateGain": 2.0,
            # The current 12-DoF G1 locomotion policy has little translational
            # response below roughly 0.18 m/s.  The upstream G1 setting starts
            # slowing at 0.875 m and asymptotically stalls outside the mission
            # tolerance.  Keep the usable 0.25 m/s command until the final
            # approach, then hand over to final-yaw alignment at 0.20 m.
            "slowDwnDisThre": 0.25,
            "stopDisThre": 0.20,
        }],
    )
    udp_bridge = ExecuteProcess(cmd=["/usr/bin/python3", bridge], output="screen")
    processes = [udp_bridge, planner, follower]
    return LaunchDescription(processes)
