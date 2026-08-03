"""Keyboard teleop for the Ackermann car (nav-vla).

teleop_twist_keyboard's j/l are in-place rotation (linear=0), which a car cannot
do, and it doesn't hold steering. This holds speed + steering as STATE and keeps
publishing them, so the car drives smoothly for manual data collection.

Keys (focus this terminal):
    w / s : speed up / down (forward, negative = reverse)
    a / d : steer LEFT / RIGHT one level (-7..7, held; a then d returns to center)
    x     : center steering (level 0)
    space : stop (speed 0)
    q     : quit

Steering uses discrete levels -7..7 (like the project's steering command),
mapped to a real steering ANGLE via the bicycle model so it feels consistent at
any speed: angular.z = speed * tan(level/7 * MAX_STEER) / WHEEL_BASE.

Publishes /cmd_vel.  Run with the bare sim:  mission_sim.launch.py use_driver:=false
"""

import math
import sys
import termios
import threading
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

SPEED_STEP = 0.3
SPEED_MAX = 3.0
STEER_LEVELS = 7          # -7..7
MAX_STEER = 0.6           # rad, matches AckermannSteering steering_limit
WHEEL_BASE = 2.86         # m, matches the prius model


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class CarTeleop(Node):
    def __init__(self):
        super().__init__("car_teleop_node")
        cmd_topic = self.declare_parameter("cmd_vel_topic", "/cmd_vel").value
        self.speed = 0.0
        self.steer_level = 0          # integer -7..7
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_timer(0.05, self._tick)  # 20 Hz continuous publish (holds state)

    def _tick(self):
        steer_angle = (self.steer_level / STEER_LEVELS) * MAX_STEER
        msg = Twist()
        msg.linear.x = self.speed
        # bicycle model: yaw rate for this steering angle at the current speed
        msg.angular.z = self.speed * math.tan(steer_angle) / WHEEL_BASE
        self.pub.publish(msg)

    def on_key(self, k):
        if k == "w":
            self.speed = clamp(self.speed + SPEED_STEP, -SPEED_MAX, SPEED_MAX)
        elif k == "s":
            self.speed = clamp(self.speed - SPEED_STEP, -SPEED_MAX, SPEED_MAX)
        elif k == "a":
            self.steer_level = clamp(self.steer_level + 1, -STEER_LEVELS, STEER_LEVELS)
        elif k == "d":
            self.steer_level = clamp(self.steer_level - 1, -STEER_LEVELS, STEER_LEVELS)
        elif k == "x":
            self.steer_level = 0
        elif k == " ":
            self.speed = 0.0
        else:
            return
        self.get_logger().info(
            f"speed={self.speed:+.2f} m/s  steer={self.steer_level:+d}/7")


def main():
    rclpy.init()
    node = CarTeleop()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(__doc__)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            k = sys.stdin.read(1)
            if k == "q" or k == "\x03":  # q or Ctrl-C
                break
            node.on_key(k)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.pub.publish(Twist())  # stop on exit
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
