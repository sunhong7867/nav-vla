"""Ground-truth obstacle monitor for nav-vla (bootstrap perception).

Publishes /obstacle_ahead (Bool) = True when any non-ego model is within
stop_distance and inside the ego's forward cone. Uses Gazebo ground-truth poses
(gz CLI) -- a reliable bootstrap to validate the governor architecture.

To be REPLACED by a learned camera VLA that publishes the same /obstacle_ahead
signal (the research goal: stop from vision, not ground truth).

Usage:
    ros2 run nav_vla obstacle_monitor_node
    ros2 run nav_vla obstacle_monitor_node --ros-args -p stop_distance:=6.0
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from nav_vla.gz_pose import (
    WorldPoseStream,
    list_models,
    query_world_pose,
    resolve_gz_bin,
)

DEFAULT_EXCLUDE = ["ego_vehicle", "ground", "top_camera", "sun", "light"]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class ObstacleMonitor(Node):
    def __init__(self):
        super().__init__("obstacle_monitor_node")
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.stop_distance = float(self.declare_parameter("stop_distance", 6.0).value)
        self.cone_deg = float(self.declare_parameter("cone_deg", 55.0).value)
        self.yaw_offset = float(
            self.declare_parameter("yaw_offset", -math.pi / 2).value)
        self.refresh_sec = float(self.declare_parameter("refresh_sec", 3.0).value)
        excl = self.declare_parameter("exclude", DEFAULT_EXCLUDE).value
        self.exclude = set(excl) | {self.model_name}
        self.gz = resolve_gz_bin(self.declare_parameter("gz_bin", "").value)

        self.obstacles = {}          # name -> (x, y), cached (mostly static)
        self._last_refresh = 0.0
        self.stream = WorldPoseStream(self.gz, self.model_name).start()
        seed = query_world_pose(self.gz, self.model_name)
        if seed:
            self.stream.latest = seed

        self.pub = self.create_publisher(Bool, "/obstacle_ahead", 10)
        self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f"obstacle_monitor: stop<{self.stop_distance}m, cone +-{self.cone_deg}deg")

    def _refresh_obstacles(self):
        names = [n for n in list_models(self.gz)
                 if n not in self.exclude
                 and not any(e in n for e in self.exclude)]
        obs = {}
        for n in names:
            p = query_world_pose(self.gz, n)
            if p:
                obs[n] = (p[0], p[1])
        self.obstacles = obs
        self.get_logger().info(f"obstacles: {list(obs)}")

    def _tick(self):
        now = time.monotonic()
        if now - self._last_refresh > self.refresh_sec:
            self._last_refresh = now
            self._refresh_obstacles()
        ego = self.stream.latest
        if ego is None:
            return
        ex, ey, eyaw = ego
        head = eyaw + self.yaw_offset
        cone = math.radians(self.cone_deg)
        blocked = False
        for (ox, oy) in self.obstacles.values():
            dx, dy = ox - ex, oy - ey
            dist = math.hypot(dx, dy)
            if dist <= self.stop_distance and abs(wrap(math.atan2(dy, dx) - head)) <= cone:
                blocked = True
                break
        self.pub.publish(Bool(data=blocked))


def main():
    rclpy.init()
    node = ObstacleMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
