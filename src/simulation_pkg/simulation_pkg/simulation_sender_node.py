import math

import rclpy
from geometry_msgs.msg import Twist
from interfaces_pkg.msg import MotionCommand
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

from simulation_pkg.config import SimulationSenderSettings as set


SUB_TOPIC_NAME = set.MOTION_PLANNER_TOPIC
PUB_TOPIC_NAME = set.GAZEBO_CONTROL_TOPIC

STEER = set.STEERING
DIRECT = set.DIRECTION
MAX_SPEED = set.MAX_SPEED
MAX_STEER_ANGLE = 0.6
WHEEL_BASE = 2.86


class SendSignal:
    def map_to_steer_angle(self, input_value):
        input_min = -7.0
        input_max = 7.0
        normalized_value = (input_value - input_min) / (input_max - input_min) * 2.0 - 1.0
        return normalized_value * MAX_STEER_ANGLE

    def map_to_speed(self, input_speed):
        input_min = -255.0
        input_max = 255.0
        normalized_input_speed = (input_speed - input_min) / (input_max - input_min) * 2.0 - 1.0
        return normalized_input_speed * MAX_SPEED

    def process(self, motor):
        left_speed = self.map_to_speed(DIRECT * motor.left_speed)
        right_speed = self.map_to_speed(DIRECT * motor.right_speed)
        speed = (left_speed + right_speed) * 0.5
        steer_angle = self.map_to_steer_angle(STEER * motor.steering)
        # The Gazebo model currently consumes Twist through a drive system.
        # Convert the front-wheel steering command to bicycle-model yaw rate so
        # the simulated car cannot rotate in place and matches Ackermann motion.
        yaw_rate = speed * math.tan(steer_angle) / WHEEL_BASE
        return yaw_rate, left_speed, right_speed


class MotorControlNode(Node):
    def __init__(self, sub_topic=SUB_TOPIC_NAME, pub_topic=PUB_TOPIC_NAME):
        super().__init__("simulation_sender_node")

        self.declare_parameter("sub_topic", sub_topic)
        self.declare_parameter("pub_topic", pub_topic)

        self.sub_topic = self.get_parameter("sub_topic").get_parameter_value().string_value
        self.pub_topic = self.get_parameter("pub_topic").get_parameter_value().string_value

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.simul = SendSignal()
        self.subscription = self.create_subscription(
            MotionCommand,
            self.sub_topic,
            self.data_callback,
            qos_profile,
        )

        self.publisher = self.create_publisher(Twist, self.pub_topic, qos_profile)
        self.timer = self.create_timer(0.05, self.send_cmd_vel)
        self.velocity = Twist()
        self.command_count = 0

    def send_cmd_vel(self):
        if rclpy.ok():
            self.publisher.publish(self.velocity)

    def data_callback(self, motor):
        yaw_rate, left, right = self.simul.process(motor)

        self.velocity.angular.z = float(yaw_rate)
        self.velocity.linear.x = float((left + right) * 0.5)
        self.command_count += 1
        if self.command_count % 20 == 1:
            self.get_logger().info(
                f"cmd_vel update: steering={motor.steering}, left={motor.left_speed}, "
                f"right={motor.right_speed}, linear={self.velocity.linear.x:.2f}, "
                f"angular={self.velocity.angular.z:.2f}"
            )

    def stop_cmd(self):
        self.velocity.linear.x = 0.0
        if rclpy.ok():
            self.publisher.publish(self.velocity)
        self.get_logger().error("Robot stopped")


def main(args=None):
    rclpy.init(args=args)
    node = MotorControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.stop_cmd()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
