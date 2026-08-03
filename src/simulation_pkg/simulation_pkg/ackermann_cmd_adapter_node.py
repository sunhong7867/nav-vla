"""Mirror a body yaw-rate command onto both visible front steering joints.

Gazebo DiffDrive continues to consume ``/cmd_vel`` unchanged so learned-policy
dynamics stay identical to the training environment.  This node only computes
one display steering angle and publishes that same target to both front wheels.
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64


class AckermannCommandAdapter(Node):
    def __init__(self):
        super().__init__("ackermann_cmd_adapter")
        self.input_topic = self.declare_parameter("input_topic", "/cmd_vel").value
        self.steer_topic = self.declare_parameter(
            "steer_topic", "/front_steer_cmd"
        ).value
        self.wheel_base = float(self.declare_parameter("wheel_base", 2.86).value)
        self.max_steer = float(self.declare_parameter("max_steer", 0.6458).value)
        self.min_speed = float(self.declare_parameter("min_speed", 1.0e-3).value)

        self.steer_publisher = self.create_publisher(Float64, self.steer_topic, 10)
        self.subscription = self.create_subscription(
            Twist, self.input_topic, self._command_callback, 10
        )

    def _command_callback(self, command):
        steering = Float64()

        if abs(command.linear.x) >= self.min_speed:
            steer = math.atan(
                self.wheel_base * command.angular.z / command.linear.x
            )
            steering.data = max(-self.max_steer, min(self.max_steer, steer))
        else:
            # A steering angle cannot be inferred from yaw rate at zero speed.
            steering.data = 0.0

        self.steer_publisher.publish(steering)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannCommandAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
