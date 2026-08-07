"""Thor 실차 VLA 브링업 — 카메라 + 어댑터 + 시리얼.

정책 서버/브리지는 시뮬과 동일하게 별도 기동한다 (LeRobot venv):
    vla_policy_server.py --checkpoint <ckpt> --endpoint ipc:///tmp/nav_vla.sock
    ros2 run nav_vla_pkg vla_bridge_node --ros-args \
        -p image_topic:=/image_raw/compressed -p speed_slew:=0.08 ...

이 런치의 세 노드가 나머지 실차 경로다:
    camera_publisher -> (bridge/server) -> /cmd_vel
        -> cmd_vel_motion_adapter (안전 게이트) -> MotionCommand
        -> serial_sender -> Arduino

트랙 주행 시 require_pose:=true 로 올려 Hesai pose 유실 시 정지하게 한다.
wheels-up/벤치는 기본 false.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("data_source", default_value="camera"),
        DeclareLaunchArgument("video_path", default_value=""),
        DeclareLaunchArgument("exposure_us", default_value="30"),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("baud", default_value="115200"),
        DeclareLaunchArgument("require_pose", default_value="false"),
        DeclareLaunchArgument("cap_speed_int", default_value="100"),
    ]
    camera = Node(
        package="thor_vehicle_pkg",
        executable="camera_publisher_node",
        name="camera_publisher_node",
        output="screen",
        parameters=[{
            "data_source": LaunchConfiguration("data_source"),
            "video_path": LaunchConfiguration("video_path"),
            "exposure_us": LaunchConfiguration("exposure_us"),
        }],
    )
    adapter = Node(
        package="thor_vehicle_pkg",
        executable="cmd_vel_motion_adapter_node",
        name="cmd_vel_motion_adapter_node",
        output="screen",
        parameters=[{
            "require_pose": LaunchConfiguration("require_pose"),
            "cap_speed_int": LaunchConfiguration("cap_speed_int"),
        }],
    )
    serial_sender = Node(
        package="thor_vehicle_pkg",
        executable="serial_sender_node",
        name="serial_sender_node",
        output="screen",
        parameters=[{
            "port": LaunchConfiguration("serial_port"),
            "baud": LaunchConfiguration("baud"),
        }],
    )
    return LaunchDescription([*args, camera, adapter, serial_sender])
