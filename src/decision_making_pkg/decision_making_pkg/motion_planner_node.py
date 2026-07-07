import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from std_msgs.msg import Bool
from std_msgs.msg import String

from interfaces_pkg.msg import DetectionArray
from interfaces_pkg.msg import MotionCommand
from interfaces_pkg.msg import PathPlanningResult
from .lib import decision_making_func_lib as DMFL


SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
SUB_MOTION_CONTROL_TOPIC_NAME = "motion_control_command"
SUB_DIRECT_MOTION_TOPIC_NAME = "direct_motion_command"
PUB_TOPIC_NAME = "topic_control_signal"
TIMER = 0.1
TARGET_SPEED_RAW = 150
MIN_CORNER_SPEED_RAW = 100
LANE_CHANGE_SPEED_RAW = 70
LANE_CHANGE_STEERING_GAIN = 2.0
LANE_CHANGE_HOLD_SEC = 8.0
CORNER_SLOWDOWN_STEERING = 2
MAX_STEERING_COMMAND = 7
TARGET_POINT_INDEX_FROM_END = 10


def convert_steeringangle2command(max_target_angle, target_angle):
    command = round(7 / (max_target_angle ** 3) * (target_angle ** 3))
    return max(-7, min(7, command))


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__("motion_planner_node")

        self.sub_detection_topic = self.declare_parameter(
            "sub_detection_topic", SUB_DETECTION_TOPIC_NAME
        ).value
        self.sub_path_topic = self.declare_parameter("sub_path_topic", SUB_PATH_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter("sub_lane_topic", self.sub_path_topic).value
        self.sub_traffic_light_topic = self.declare_parameter(
            "sub_traffic_light_topic", SUB_TRAFFIC_LIGHT_TOPIC_NAME
        ).value
        self.sub_lidar_obstacle_topic = self.declare_parameter(
            "sub_lidar_obstacle_topic", SUB_LIDAR_OBSTACLE_TOPIC_NAME
        ).value
        self.sub_motion_control_topic = self.declare_parameter(
            "sub_motion_control_topic", SUB_MOTION_CONTROL_TOPIC_NAME
        ).value
        self.sub_direct_motion_topic = self.declare_parameter(
            "sub_direct_motion_topic", SUB_DIRECT_MOTION_TOPIC_NAME
        ).value
        self.direct_motion_timeout = float(
            self.declare_parameter("direct_motion_timeout", 0.3).value
        )
        # Direct-motion override lets an external node (navigator) seize /cmd via
        # direct_motion_command. Off by default: it was never actually run and its
        # zero-speed stop commands freeze normal perception-driven lane driving.
        self.use_direct_motion_override = bool(
            self.declare_parameter("use_direct_motion_override", False).value
        )
        self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value
        self.timer_period = self.declare_parameter("timer", TIMER).value
        self.target_speed_raw = int(
            self.declare_parameter("target_speed_raw", TARGET_SPEED_RAW).value
        )
        self.target_speed_raw = max(0, min(255, self.target_speed_raw))
        self.min_corner_speed_raw = int(
            self.declare_parameter("min_corner_speed_raw", MIN_CORNER_SPEED_RAW).value
        )
        self.min_corner_speed_raw = max(0, min(255, self.min_corner_speed_raw))
        self.lane_change_speed_raw = int(
            self.declare_parameter("lane_change_speed_raw", LANE_CHANGE_SPEED_RAW).value
        )
        self.lane_change_speed_raw = max(0, min(255, self.lane_change_speed_raw))
        self.corner_slowdown_steering = int(
            self.declare_parameter("corner_slowdown_steering", CORNER_SLOWDOWN_STEERING).value
        )
        self.corner_slowdown_steering = max(0, min(MAX_STEERING_COMMAND, self.corner_slowdown_steering))
        # Steering is a weak cubic of path slope; on curves the mid-slope command is
        # too small to pull fully into the new lane, so scale it up during a lane
        # change only (normal driving is unaffected).
        self.lane_change_steering_gain = float(
            self.declare_parameter("lane_change_steering_gain", LANE_CHANGE_STEERING_GAIN).value
        )
        self.lane_change_steering_gain = max(1.0, self.lane_change_steering_gain)
        # Once a lane change starts, force the low-speed + steering-gain window for
        # this long, regardless of when the perception is_lane_changing flag clears
        # (it can clear before the car has physically settled into the new lane).
        self.lane_change_hold_sec = float(
            self.declare_parameter("lane_change_hold_sec", LANE_CHANGE_HOLD_SEC).value
        )
        self.lane_change_hold_until = 0.0

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.motion_control_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self.detection_data = None
        self.path_data = None
        self.path_is_lane_changing = False
        self.traffic_light_data = None
        self.lidar_data = None
        self.motion_stopped = False
        self.direct_motion_command = None
        self.direct_motion_time = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.timer_count = 0

        self.create_subscription(
            DetectionArray, self.sub_detection_topic, self.detection_callback, self.qos_profile
        )
        self.create_subscription(
            PathPlanningResult, self.sub_path_topic, self.path_callback, self.qos_profile
        )
        self.create_subscription(
            String, self.sub_traffic_light_topic, self.traffic_light_callback, self.qos_profile
        )
        self.create_subscription(
            Bool, self.sub_lidar_obstacle_topic, self.lidar_callback, self.qos_profile
        )
        self.create_subscription(
            String,
            self.sub_motion_control_topic,
            self.motion_control_callback,
            self.motion_control_qos_profile,
        )
        self.create_subscription(
            MotionCommand,
            self.sub_direct_motion_topic,
            self.direct_motion_callback,
            self.qos_profile,
        )

        self.publisher = self.create_publisher(MotionCommand, self.pub_topic, self.qos_profile)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def detection_callback(self, msg):
        self.detection_data = msg

    def path_callback(self, msg):
        self.path_data = list(zip(msg.x_points, msg.y_points))
        is_changing = bool(msg.is_lane_changing)
        if is_changing and not self.path_is_lane_changing:
            # rising edge: hold the lane-change regime for a fixed window
            now = self.get_clock().now().nanoseconds * 1e-9
            self.lane_change_hold_until = now + self.lane_change_hold_sec
        self.path_is_lane_changing = is_changing

    def traffic_light_callback(self, msg):
        self.traffic_light_data = msg

    def lidar_callback(self, msg):
        self.lidar_data = msg

    def motion_control_callback(self, msg):
        command = str(msg.data or "").strip().lower()
        if command in {"stop", "pause"}:
            self.motion_stopped = True
            self.get_logger().info("motion control: stop")
        elif command in {"start", "resume"}:
            self.motion_stopped = False
            self.get_logger().info("motion control: start")
        else:
            self.get_logger().warn(f"unknown motion control command: {msg.data}")

    def direct_motion_callback(self, msg):
        self.direct_motion_command = msg
        self.direct_motion_time = self.get_clock().now().nanoseconds * 1e-9

    def timer_callback(self):
        self.timer_count += 1
        direct_command = (
            self._fresh_direct_motion_command() if self.use_direct_motion_override else None
        )
        if direct_command is not None:
            self.steering_command = int(direct_command.steering)
            self.left_speed_command = int(direct_command.left_speed)
            self.right_speed_command = int(direct_command.right_speed)
            msg = MotionCommand()
            msg.steering = self.steering_command
            msg.left_speed = self.left_speed_command
            msg.right_speed = self.right_speed_command
            self.publisher.publish(msg)
            return

        if not self.path_data:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            if self.timer_count % 30 == 1:
                self.get_logger().warn("waiting for path data; publishing zero speed")
        else:
            if len(self.path_data) >= TARGET_POINT_INDEX_FROM_END:
                target_point = self.path_data[-TARGET_POINT_INDEX_FROM_END]
                start_point = self.path_data[-1]
            else:
                target_point = self.path_data[0]
                start_point = self.path_data[-1]

            now = self.get_clock().now().nanoseconds * 1e-9
            in_lane_change = self.path_is_lane_changing or now < self.lane_change_hold_until

            target_slope = DMFL.calculate_slope_between_points(target_point, start_point)
            self.steering_command = convert_steeringangle2command(90, target_slope)
            if in_lane_change and self.lane_change_steering_gain != 1.0:
                boosted = round(self.steering_command * self.lane_change_steering_gain)
                self.steering_command = max(
                    -MAX_STEERING_COMMAND, min(MAX_STEERING_COMMAND, boosted)
                )
            target_speed = self._target_speed_for_steering(self.steering_command)
            if in_lane_change:
                target_speed = min(target_speed, self.lane_change_speed_raw)
            self.left_speed_command = target_speed
            self.right_speed_command = target_speed

        if self.traffic_light_data is not None and self.traffic_light_data.data == "Red":
            if self.detection_data is not None:
                for detection in self.detection_data.detections:
                    if detection.class_name == "traffic_light":
                        y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2)
                        if y_max < 255:
                            self.steering_command = 0
                            self.left_speed_command = 0
                            self.right_speed_command = 0
                            break

        if self.lidar_data is not None and self.lidar_data.data is True:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0

        if self.motion_stopped:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0
            if self.timer_count % 30 == 1:
                self.get_logger().info("motion control is stopped; publishing zero speed")

        msg = MotionCommand()
        msg.steering = self.steering_command
        msg.left_speed = self.left_speed_command
        msg.right_speed = self.right_speed_command
        self.publisher.publish(msg)

    def _fresh_direct_motion_command(self):
        if self.direct_motion_command is None or self.direct_motion_time is None:
            return None
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.direct_motion_time > self.direct_motion_timeout:
            return None
        return self.direct_motion_command

    def _target_speed_for_steering(self, steering_command):
        steering_abs = abs(int(steering_command))
        if steering_abs <= self.corner_slowdown_steering:
            return self.target_speed_raw

        slowdown_range = max(1, MAX_STEERING_COMMAND - self.corner_slowdown_steering)
        slowdown_ratio = (steering_abs - self.corner_slowdown_steering) / slowdown_range
        speed = self.target_speed_raw - (
            self.target_speed_raw - self.min_corner_speed_raw
        ) * slowdown_ratio
        return int(round(max(self.min_corner_speed_raw, min(self.target_speed_raw, speed))))


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
