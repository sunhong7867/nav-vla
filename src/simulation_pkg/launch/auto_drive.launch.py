#!/usr/bin/env python3
"""One-shot mission launch: lane following + traffic light stop + avoidance.

Brings up everything in a single command:
  - mission_sim (Gazebo, obstacles, traffic light, camera, lidar, lane-follower)
    with traffic-light handling left in motion_planner.
  - navigator_node: listens to the obstacle signal and changes lane to go around
    obstacle vehicles. The obstacle signal is intentionally NOT wired to
    motion_planner in this launch, because motion_planner hard-stops on lidar
    obstacles before lane-change avoidance can happen.

Usage:
    ros2 launch simulation_pkg auto_drive.launch.py
    ros2 launch simulation_pkg auto_drive.launch.py use_avoidance:=false motion_lidar_topic:=lidar_obstacle_info
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_dir = get_package_share_directory("simulation_pkg")
    use_avoidance = LaunchConfiguration("use_avoidance")
    obstacle_topic = LaunchConfiguration("obstacle_topic")
    use_obstacles = LaunchConfiguration("use_obstacles")
    use_traffic_light = LaunchConfiguration("use_traffic_light")
    motion_lidar_topic = LaunchConfiguration("motion_lidar_topic")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_avoidance", default_value="true",
            description="Also enable navigator lane-change avoidance.",
        ),
        DeclareLaunchArgument(
            "obstacle_topic", default_value="/lidar_obstacle_info",
            description="Obstacle Bool topic used by navigator lane-change avoidance.",
        ),
        DeclareLaunchArgument(
            "motion_lidar_topic",
            default_value="/none",
            description="motion_planner lidar-stop topic. Keep /none for lane-change "
                        "avoidance; set lidar_obstacle_info to stop at obstacles.",
        ),
        DeclareLaunchArgument(
            "use_obstacles",
            default_value="true",
            description="Spawn mission obstacle vehicles and the lidar obstacle detector.",
        ),
        DeclareLaunchArgument(
            "use_traffic_light",
            default_value="true",
            description="Spawn the mission traffic light and run traffic-light detection.",
        ),
        # Sim + lane-follower. For avoidance mode, keep traffic light handling in
        # motion_planner but disconnect its lidar-stop input. The navigator uses
        # the same lidar detector output to command lane changes instead.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sim_dir, "launch", "mission_sim.launch.py")
            ),
            launch_arguments={
                "use_obstacles": use_obstacles,
                "use_traffic_light": use_traffic_light,
                "use_mission_events": "false",
                "motion_lidar_topic": motion_lidar_topic,
            }.items(),
        ),
        # Navigator avoidance starts after the perception/driver pipeline is up.
        TimerAction(
            period=12.0,
            actions=[
                Node(
                    condition=IfCondition(use_avoidance),
                    package="nav_vla",
                    executable="navigator_node",
                    parameters=[{
                        "use_obstacle_avoidance": True,
                        "stop_on_start": False,
                        "obstacle_topic": obstacle_topic,
                        "avoidance_commit_delay": 0.25,
                        "avoidance_cooldown": 4.0,
                        "avoidance_return_delay": 5.0,
                    }],
                    output="screen",
                ),
            ],
        ),
    ])
