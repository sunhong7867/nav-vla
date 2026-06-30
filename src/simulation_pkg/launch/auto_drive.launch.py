#!/usr/bin/env python3
"""One-shot launch: lane following + obstacle slow-down + lane-change avoidance.

Brings up everything in a single command:
  - mission_sim (Gazebo, obstacles, traffic light, camera, lidar, lane-follower)
    with the lidar obstacle signal wired to motion_planner so it SLOWS to
    `obstacle_speed` (default 75) on a detected obstacle and restores after
    `obstacle_clear_hold` seconds clear.
  - navigator_node: on the same lidar signal, changes lane to go around.

Usage:
    ros2 launch simulation_pkg auto_drive.launch.py
    ros2 launch simulation_pkg auto_drive.launch.py use_avoidance:=false
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

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_avoidance", default_value="true",
            description="Also enable navigator lane-change avoidance.",
        ),
        DeclareLaunchArgument(
            "obstacle_topic", default_value="/lidar_obstacle_info",
            description="Obstacle Bool topic both the slow-down and the avoidance use.",
        ),
        # Sim + lane-follower. Defaults keep the lidar obstacle signal wired to
        # motion_planner (motion_lidar_topic=lidar_obstacle_info) so it slows
        # (not stops) on obstacles; obstacles + lidar detector run via use_obstacles.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sim_dir, "launch", "mission_sim.launch.py")
            ),
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
                    }],
                    output="screen",
                ),
            ],
        ),
    ])
