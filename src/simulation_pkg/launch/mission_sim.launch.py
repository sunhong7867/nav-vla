#!/usr/bin/env python3

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _create_runtime_world(package_dir, workspace_root):
    source_world = os.path.join(package_dir, "worlds", "track.world")
    texture_candidates = [
        os.path.join(
            workspace_root,
            "src",
            "simulation_pkg",
            "models",
            "race_track",
            "materials",
            "textures",
            "track.png",
        ),
        os.path.join(
            package_dir,
            "models",
            "race_track",
            "materials",
            "textures",
            "track.png",
        ),
    ]

    track_texture_path = None
    for candidate in texture_candidates:
        if os.path.exists(candidate):
            track_texture_path = candidate
            break

    if track_texture_path is None:
        return source_world

    with open(source_world, "r", encoding="utf-8") as file:
        world_text = file.read()

    world_text = world_text.replace(
        "model://race_track/materials/textures/track.png",
        f"file://{track_texture_path}",
    )

    runtime_world = os.path.join(tempfile.gettempdir(), "simulation_pkg_runtime_track.world")
    with open(runtime_world, "w", encoding="utf-8") as file:
        file.write(world_text)

    return runtime_world


def _create_runtime_gui_config(package_dir):
    desired_camera_pose = (
        "-1.463067 0.399495 33.714752 "
        "-3.141590 1.547276 -0.005837"
    )
    source_gui_config = os.path.expanduser("~/.gz/sim/8/gui.config")
    fallback_gui_config = os.path.join(package_dir, "gui", "track_gui.config")

    if os.path.exists(source_gui_config):
        with open(source_gui_config, "r", encoding="utf-8") as file:
            gui_text = file.read()
    else:
        with open(fallback_gui_config, "r", encoding="utf-8") as file:
            gui_text = file.read()

    gui_text = gui_text.replace(
        "<camera_pose>-6 0 6 0 0.5 0</camera_pose>",
        f"<camera_pose>{desired_camera_pose}</camera_pose>",
    )
    gui_text = gui_text.replace(
        "<camera_pose>-3.049776 -0.241982 38.020702 0.000000 1.511954 -0.003112</camera_pose>",
        f"<camera_pose>{desired_camera_pose}</camera_pose>",
    )
    gui_text = gui_text.replace("<start_paused>true</start_paused>", "<start_paused>false</start_paused>")

    runtime_gui_config = os.path.join(tempfile.gettempdir(), "simulation_pkg_runtime_gui.config")
    with open(runtime_gui_config, "w", encoding="utf-8") as file:
        file.write(gui_text)

    return runtime_gui_config


def generate_launch_description():
    workspace_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "..",
        )
    )
    package_dir = get_package_share_directory("simulation_pkg")
    ros_gz_sim_dir = get_package_share_directory("ros_gz_sim")
    world_file = _create_runtime_world(package_dir, workspace_root)
    gui_config = _create_runtime_gui_config(package_dir)

    install_model_path = os.path.join(package_dir, "models")
    source_model_path = os.path.join(workspace_root, "src", "simulation_pkg", "models")
    resource_paths = [install_model_path, package_dir]
    if os.path.isdir(source_model_path):
        resource_paths.insert(0, source_model_path)

    existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH")
    if existing_resource_path:
        resource_paths.append(existing_resource_path)

    resource_path_value = ":".join(resource_paths)
    yolo_model_paths = [
        os.path.join(workspace_root, "traffic_light_sim.pt"),
        os.path.join(workspace_root, "best_cap.pt"),
        os.path.join(workspace_root, "crosswalk.pt"),
        os.path.join(workspace_root, "parking_front.pt"),
        os.path.join(workspace_root, "parking_rear.pt"),
    ]
    # Only weights that actually exist on disk (missing ones break detection).
    # best_cap.pt is the lane model needed for driving; keep it primary.
    lane_model_path = os.path.join(workspace_root, "best_cap.pt")
    existing_models = [p for p in yolo_model_paths if os.path.exists(p)]
    if lane_model_path in existing_models:
        existing_models.remove(lane_model_path)
    existing_models.insert(0, lane_model_path)
    use_debug_visualizers = LaunchConfiguration("use_debug_visualizers")
    use_rqt = LaunchConfiguration("use_rqt")
    use_obstacles = LaunchConfiguration("use_obstacles")
    use_traffic_light = LaunchConfiguration("use_traffic_light")
    use_mission_events = LaunchConfiguration("use_mission_events")
    use_driver = LaunchConfiguration("use_driver")
    driver_cmd_topic = LaunchConfiguration("driver_cmd_topic")
    motion_lidar_topic = LaunchConfiguration("motion_lidar_topic")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_debug_visualizers",
            default_value="true",
            description="Publish lane/path debug image topics for RViz.",
        ),
        DeclareLaunchArgument(
            "use_rqt",
            default_value="false",
            description="Launch rqt for debugging.",
        ),
        DeclareLaunchArgument(
            "use_obstacles",
            default_value="true",
            description="Spawn mission obstacle vehicles.",
        ),
        DeclareLaunchArgument(
            "use_traffic_light",
            default_value="true",
            description="Spawn the mission traffic light stand.",
        ),
        DeclareLaunchArgument(
            "use_mission_events",
            default_value="true",
            description="Enable mission events: lane change at M2/T2 and stop 5s at T3.",
        ),
        DeclareLaunchArgument(
            "use_driver",
            default_value="true",
            description="Run the old YOLO/lane/path/motion driver that owns /cmd_vel. "
                        "Set false so an external controller "
                        "drives while obstacles/traffic light/camera stay spawned.",
        ),
        DeclareLaunchArgument(
            "driver_cmd_topic",
            default_value="/cmd_vel",
            description="Topic the lane-following driver publishes to. Set to /cmd_nav "
                        "if another node should post-process driving commands.",
        ),
        DeclareLaunchArgument(
            "motion_lidar_topic",
            default_value="/none",
            description="motion_planner's lidar-obstacle STOP topic. Set to /none for "
                        "avoidance mode so it does NOT stop on obstacles.",
        ),
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=resource_path_value,
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ros_gz_sim_dir, "launch", "gz_sim.launch.py")
            ),
            launch_arguments={
                "gz_args": f"-r -s {world_file}",
                "gz_version": "8",
                "on_exit_shutdown": "true",
            }.items(),
        ),
        TimerAction(
            period=0.5,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "bash",
                        "-lc",
                        "source /opt/ros/jazzy/setup.bash && "
                        "for i in $(seq 1 40); do "
                        "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz service -l | grep -q '/world/default/gui/info' && break; "
                        "sleep 1; "
                        "done; "
                        f"ruby /opt/ros/jazzy/opt/gz_tools_vendor/bin/gz sim -g --gui-config {gui_config} "
                        "--render-engine-gui ogre --force-version 8"
                    ],
                    output="screen",
                ),
            ],
        ),
        TimerAction(
            period=4.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "bash",
                        "-lc",
                        "source /opt/ros/jazzy/setup.bash && "
                        "for i in $(seq 1 8); do "
                        "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz topic "
                        "-t /gui/camera/pose "
                        "-m gz.msgs.Pose "
                        "-p 'position: {x: -1.4630670547485352 y: 0.39949455857276917 z: 33.714752197265625} "
                        "orientation: {x: 0.71537137031555176 y: -0.0020889292936772108 z: -0.69873839616775513 w: -0.0020404069218784571}'; "
                        "sleep 0.5; "
                        "done"
                    ],
                    output="screen",
                ),
            ],
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/model/ego_vehicle/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
                "/model/ego_vehicle/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            ],
            remappings=[
                ("/model/ego_vehicle/odometry", "/odom"),
                ("/model/ego_vehicle/cmd_vel", "/cmd_vel"),
            ],
            output="screen",
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/camera@sensor_msgs/msg/Image@gz.msgs.Image",
            ],
            remappings=[
                ("/camera", "/camera/image_raw"),
            ],
            output="screen",
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            ],
            output="screen",
        ),
        Node(
            condition=IfCondition(use_rqt),
            package="rqt_gui",
            executable="rqt_gui",
            output="screen",
        ),
        TimerAction(
            period=6.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "python3",
                        "-m",
                        "simulation_pkg.lib.load_ego_car_node",
                    ],
                    output="screen",
                ),
            ],
        ),
        TimerAction(
            period=8.0,
            actions=[
                ExecuteProcess(
                    condition=IfCondition(use_obstacles),
                    cmd=[
                        "python3",
                        "-m",
                        "simulation_pkg.lib.load_obstable_car_node",
                    ],
                    output="screen",
                ),
                ExecuteProcess(
                    condition=IfCondition(use_traffic_light),
                    cmd=[
                        "python3",
                        "-m",
                        "simulation_pkg.lib.load_traffic_light_node",
                    ],
                    output="screen",
                ),
                Node(
                    package="camera_perception_pkg",
                    executable="yolov8_node",
                    parameters=[{
                        "model": lane_model_path,
                        "models": existing_models,
                        "device": "cuda:0",
                        "ignore_class_names": ["crosswalk", "end_line", "endline", "parking_space", "parkingspace"],
                        "inference_period": 0.0,
                        "imgsz": 640,
                    }],
                    output="screen",
                ),
                Node(
                    package="camera_perception_pkg",
                    executable="lane_info_extractor_node",
                    parameters=[{
                        "lane_mode": "keep_lane",
                        "target_lane": "lane2",
                        "use_crosswalk_detection": False,
                        "enable_auto_lane_change": False,
                    }],
                    output="screen",
                ),
                # Lidar obstacle detection (ROS-native; gz CLI is unreliable).
                # Reads /scan directly -> /lidar_obstacle_info (Bool).
                Node(
                    condition=IfCondition(use_obstacles),
                    package="lidar_perception_pkg",
                    executable="lidar_obstacle_detector_node",
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_traffic_light),
                    package="camera_perception_pkg",
                    executable="traffic_light_detector_node",
                    parameters=[{
                        "sub_image_topic": "/camera/image_raw",
                        "sub_detection_topic": "detections",
                    }],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_debug_visualizers),
                    package="debug_pkg",
                    executable="path_visualizer_node",
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_debug_visualizers),
                    package="debug_pkg",
                    executable="yolov8_visualizer_node",
                    parameters=[{
                        "debug_image_topic": "yolov8_seg_debug_image",
                    }],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_driver),
                    package="decision_making_pkg",
                    executable="path_planner_node",
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_driver),
                    package="decision_making_pkg",
                    executable="motion_planner_node",
                    parameters=[{"sub_lidar_obstacle_topic": motion_lidar_topic}],
                    output="screen",
                ),
                ExecuteProcess(
                    condition=IfCondition(use_driver),
                    cmd=[
                        "python3",
                        "-m",
                        "simulation_pkg.simulation_sender_node",
                        "--ros-args",
                        "-p",
                        ["pub_topic:=", driver_cmd_topic],
                    ],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_mission_events),
                    package="simulation_pkg",
                    executable="mission_event_node",
                    output="screen",
                ),
            ],
        ),
    ])
