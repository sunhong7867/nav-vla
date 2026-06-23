#!/usr/bin/env python3

import os
import re
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition, UnlessCondition
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

    gui_text, camera_pose_replacements = re.subn(
        r"<camera_pose>.*?</camera_pose>",
        f"<camera_pose>{desired_camera_pose}</camera_pose>",
        gui_text,
        count=1,
        flags=re.DOTALL,
    )
    if camera_pose_replacements == 0:
        with open(fallback_gui_config, "r", encoding="utf-8") as file:
            gui_text = file.read()
        gui_text = re.sub(
            r"<camera_pose>.*?</camera_pose>",
            f"<camera_pose>{desired_camera_pose}</camera_pose>",
            gui_text,
            count=1,
            flags=re.DOTALL,
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
    rviz_config_path = os.path.join(package_dir, "rviz", "driving_debug.rviz")
    yolo_model_path = os.path.join(workspace_root, "best_cap.pt")
    use_perception_pipeline = LaunchConfiguration("use_perception_pipeline")
    use_lane_mode_gui = LaunchConfiguration("use_lane_mode_gui")
    use_debug_visualizers = LaunchConfiguration("use_debug_visualizers")
    use_rviz = LaunchConfiguration("use_rviz")
    use_yolo_image_view = LaunchConfiguration("use_yolo_image_view")
    use_top_down_view = LaunchConfiguration("use_top_down_view")
    use_track_overview_view = LaunchConfiguration("use_track_overview_view")
    use_lane_tuning_gui = LaunchConfiguration("use_lane_tuning_gui")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_perception_pipeline",
            default_value="true",
            description="Launch the YOLO/lane/path/motion pipeline. False uses the stable lane2 map driver.",
        ),
        DeclareLaunchArgument(
            "use_lane_mode_gui",
            default_value="false",
            description="Launch a small GUI to switch between fixed-lane and lane-change driving modes.",
        ),
        DeclareLaunchArgument(
            "use_debug_visualizers",
            default_value="true",
            description="Publish lane/path debug image topics for RViz.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Launch RViz with driving debug image displays.",
        ),
        DeclareLaunchArgument(
            "use_yolo_image_view",
            default_value="false",
            description="Launch a large OpenCV window for the YOLO segmentation debug image.",
        ),
        DeclareLaunchArgument(
            "use_top_down_view",
            default_value="false",
            description="Launch a vehicle-mounted top-down camera image window.",
        ),
        DeclareLaunchArgument(
            "use_track_overview_view",
            default_value="false",
            description="Launch a fixed overhead camera image window showing the whole track.",
        ),
        DeclareLaunchArgument(
            "use_lane_tuning_gui",
            default_value="false",
            description="Launch a GUI to tune bird-eye ROI and target point values.",
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
            condition=IfCondition(use_perception_pipeline),
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
            condition=IfCondition(use_top_down_view),
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/ego_top_camera@sensor_msgs/msg/Image@gz.msgs.Image",
            ],
            remappings=[
                ("/ego_top_camera", "/ego_top_camera/image_raw"),
            ],
            output="screen",
        ),
        Node(
            condition=IfCondition(use_track_overview_view),
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/top_camera@sensor_msgs/msg/Image@gz.msgs.Image",
            ],
            remappings=[
                ("/top_camera", "/top_camera/image_raw"),
            ],
            output="screen",
        ),
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package="simulation_pkg",
                    executable="load_ego_car_node",
                    output="screen",
                ),
            ],
        ),
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    condition=UnlessCondition(use_perception_pipeline),
                    package="simulation_pkg",
                    executable="simple_track_driver_node",
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_perception_pipeline),
                    package="camera_perception_pkg",
                    executable="yolov8_node",
                    parameters=[{
                        "model": yolo_model_path,
                        "device": "cuda:0",
                        "allowed_class_names": ["lane1", "lane2"],
                        "ignore_class_names": ["crosswalk"],
                        "inference_period": 0.0,
                        "imgsz": 640,
                    }],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_perception_pipeline),
                    package="camera_perception_pkg",
                    executable="lane_info_extractor_node",
                    parameters=[{
                        "lane_mode": "keep_lane",
                        "target_lane": "lane2",
                    }],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_lane_mode_gui),
                    package="simulation_pkg",
                    executable="lane_mode_gui_node",
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
                    condition=IfCondition(use_rviz),
                    package="rviz2",
                    executable="rviz2",
                    arguments=["-d", rviz_config_path],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_yolo_image_view),
                    package="simulation_pkg",
                    executable="yolo_debug_image_viewer_node",
                    parameters=[{
                        "image_topic": "/yolov8_seg_debug_image",
                    }],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_top_down_view),
                    package="simulation_pkg",
                    executable="yolo_debug_image_viewer_node",
                    parameters=[{
                        "image_topic": "/ego_top_camera/image_raw",
                        "window_name": "Vehicle Top-Down View",
                    }],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_track_overview_view),
                    package="simulation_pkg",
                    executable="yolo_debug_image_viewer_node",
                    parameters=[{
                        "image_topic": "/top_camera/image_raw",
                        "window_name": "Full Track View",
                    }],
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_lane_tuning_gui),
                    package="simulation_pkg",
                    executable="lane_tuning_gui_node",
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_perception_pipeline),
                    package="decision_making_pkg",
                    executable="path_planner_node",
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_perception_pipeline),
                    package="decision_making_pkg",
                    executable="motion_planner_node",
                    output="screen",
                ),
                Node(
                    condition=IfCondition(use_perception_pipeline),
                    package="simulation_pkg",
                    executable="sim_simulation_sender_node",
                    output="screen",
                ),
            ],
        ),
    ])
