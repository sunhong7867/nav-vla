"""Lidar obstacle detector (self-contained).

Subscribes the raw /scan (LaserScan) and publishes /lidar_obstacle_info (Bool) =
True when something is within [range_min, range_max] inside the forward angular
sector [start_angle, end_angle] (degrees), debounced over a few consecutive hits.

Rewritten to NOT depend on lidar_perception_func_lib (its source is missing; only
a Python 3.10 .pyc remained, unimportable on 3.12) and to read /scan directly
(no lidar_processor needed).

Front sector + range are live ROS params (tune with `ros2 param set` without
rebuild). The body has a ~90 deg yaw offset, so the real forward sector may not
be 0-30 deg -- find it via /scan argmin when an obstacle is dead ahead.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32

SUB_TOPIC_NAME = "scan"
PUB_TOPIC_NAME = "lidar_obstacle_info"
DIST_TOPIC_NAME = "lidar_obstacle_distance"


def sector_indices(n, start_deg, end_deg):
    """Indices covering [start_deg, end_deg] for n samples over 360 deg,
    handling wraparound (start > end crosses 0)."""
    si = int(round(start_deg * n / 360.0)) % n
    ei = int(round(end_deg * n / 360.0)) % n
    if si <= ei:
        return range(si, ei + 1)
    return list(range(si, n)) + list(range(0, ei + 1))


class ObjectDetection(Node):
    def __init__(self):
        super().__init__("lidar_obstacle_detector_node")
        self.start_angle = self.declare_parameter("start_angle", 0).value
        self.end_angle = self.declare_parameter("end_angle", 15).value
        self.range_min = self.declare_parameter("range_min", 0.5).value
        # detect EARLY: the camera-based lane change takes ~5 s and freezes once
        # the obstacle blocks the lane markings, so the change must start while
        # the obstacle is still far. 5 m gives room to weave before the view is
        # blocked. (Curve-wall false hits are filtered by commit_delay in the
        # avoidance node, not by a short range.) Live-tunable.
        self.range_max = self.declare_parameter("range_max", 5.0).value
        self.consec = int(self.declare_parameter("consec_count", 3).value)
        # consecutive MISSES needed to clear the latch (mild hysteresis only).
        # The real anti-weave guard is commit_delay in the avoidance node; the
        # latch must NOT be sticky or it holds a momentary wall-glance and
        # triggers a spurious lane change.
        self.clear_consec = int(self.declare_parameter("clear_count", 2).value)
        self.debug_argmin = bool(self.declare_parameter("debug_argmin", True).value)
        self._frame = 0
        self._misses = 0

        # /scan from ros_gz_bridge is RELIABLE; match it so the callback connects.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.sub = self.create_subscription(LaserScan, SUB_TOPIC_NAME, self._cb, qos)
        self.pub = self.create_publisher(Bool, PUB_TOPIC_NAME, 10)
        # front-sector nearest distance (m) — richer label for camera training
        self.dist_pub = self.create_publisher(Float32, DIST_TOPIC_NAME, 10)
        self._hits = 0
        self._latched = False

    def _cb(self, msg):
        s = int(self.get_parameter("start_angle").value)
        e = int(self.get_parameter("end_angle").value)
        rmin = float(self.get_parameter("range_min").value)
        rmax = float(self.get_parameter("range_max").value)
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            return

        hit = False
        nearest = math.inf
        for i in sector_indices(n, s, e):
            r = ranges[i]
            if r is None or math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue
            if r < nearest:
                nearest = r
            if rmin <= r <= rmax:
                hit = True
                break

        # debounce with hysteresis: need `consec` consecutive hits to latch True,
        # and `clear_consec` consecutive misses to drop back to False. This stops
        # a single empty/noisy beam from flickering the signal (which would make
        # the avoidance node weave back and forth).
        clear_consec = max(1, int(self.get_parameter("clear_count").value))
        if hit:
            self._hits = min(self._hits + 1, self.consec)
            self._misses = 0
        else:
            self._misses = min(self._misses + 1, clear_consec)
            self._hits = 0
        if self._latched:
            detected = self._misses < clear_consec   # stay latched until enough misses
        else:
            detected = self._hits >= self.consec     # need enough hits to latch

        if detected != self._latched:
            self._latched = detected
            near = "%.2fm" % nearest if math.isfinite(nearest) else "inf"
            self.get_logger().info(
                f"obstacle {'DETECTED' if detected else 'clear'} "
                f"(front {s}-{e}deg, nearest {near})")
        self.pub.publish(Bool(data=detected))
        # publish nearest front distance (range_max sentinel if nothing in sector)
        self.dist_pub.publish(Float32(data=float(nearest if math.isfinite(nearest) else rmax)))

        # forward-finding aid: log the globally nearest beam (any direction)
        self._frame += 1
        if self.debug_argmin and self._frame % 10 == 0:
            best_i, best_r = -1, math.inf
            for i, r in enumerate(ranges):
                if r is None or math.isinf(r) or math.isnan(r) or r <= 0.0:
                    continue
                if r < best_r:
                    best_r, best_i = r, i
            if best_i >= 0:
                deg = best_i * 360.0 / n
                front = "%.2fm" % nearest if math.isfinite(nearest) else "inf"
                self.get_logger().info(
                    f"[argmin] nearest beam overall: idx={best_i} (~{deg:.0f}deg) {best_r:.2f}m"
                    f"  | front {s}-{e}deg nearest={front} detected={detected}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection()
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
