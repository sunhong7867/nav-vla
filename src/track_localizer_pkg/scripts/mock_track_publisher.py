#!/usr/bin/env python3
"""Synthetic /track/* publisher — stands in for the laptop + LiDAR.

Publishes the exact topic/QoS contract of track_pose_node (circle trajectory,
10 Hz) so the Thor-side receiver can be brought up and validated BEFORE the
LiDAR chain exists. Run it on the laptop to test the WiFi link alone, or on
Thor itself for a loopback test of the tooling.

    source track_link_env.sh <server-ip>
    python3 mock_track_publisher.py [--rate 10] [--duration 0]
"""

import argparse
import json
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, String


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MockTrackPublisher(Node):
    def __init__(self, rate_hz, duration_s):
        super().__init__("mock_track_publisher")
        # Same QoS as track_pose_node: pose is BEST_EFFORT depth 1 — a late
        # pose is worse than a missing one. The receiver must match this.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.pose_pub = self.create_publisher(
            Odometry, "/track/vehicle_pose", sensor_qos)
        self.map_pub = self.create_publisher(
            Odometry, "/track/vehicle_pose_map", sensor_qos)
        self.status_pub = self.create_publisher(
            String, "/track/vehicle_status", 10)
        self.estop_pub = self.create_publisher(
            Bool, "/track/geofence_estop", 10)
        self.rate_hz = rate_hz
        self.duration_s = duration_s
        self.n = 0
        self.speed = 1.5           # m/s, circle radius 3 m
        self.radius = 3.0
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"mock /track/* at {rate_hz} Hz"
            + (f" for {duration_s} s" if duration_s > 0 else ""))

    def _tick(self):
        t = self.n / self.rate_hz
        self.n += 1
        if 0 < self.duration_s < t:
            raise SystemExit(0)
        ang = (self.speed / self.radius) * t
        # sensor-frame-ish circle around (7.5, 1.0); heading = tangent
        fx = 7.5 + self.radius * math.cos(ang)
        fy = 1.0 + self.radius * math.sin(ang)
        yaw = ang + math.pi / 2.0
        stamp = self.get_clock().now().to_msg()

        for pub, ox in ((self.pose_pub, 0.0), (self.map_pub, 0.5)):
            m = Odometry()
            m.header.stamp = stamp
            m.header.frame_id = "track" if ox == 0.0 else "track_map"
            m.child_frame_id = "base_link"
            m.pose.pose.position.x = fx + ox
            m.pose.pose.position.y = fy + ox
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            m.pose.pose.orientation.z = qz
            m.pose.pose.orientation.w = qw
            m.pose.covariance[0] = m.pose.covariance[7] = 0.0036
            m.pose.covariance[35] = 1e-4          # heading valid
            m.twist.twist.linear.x = self.speed * math.cos(yaw)
            m.twist.twist.linear.y = self.speed * math.sin(yaw)
            pub.publish(m)

        self.status_pub.publish(String(data=json.dumps({
            "status": "MOCK", "heading_valid": True,
            "speed_mps": self.speed, "forward_m": round(fx, 3),
            "lateral_m": round(fy, 3)})))
        self.estop_pub.publish(Bool(data=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=0.0,
                   help="seconds; 0 = run until Ctrl-C")
    args = p.parse_args()
    rclpy.init()
    node = MockTrackPublisher(args.rate, args.duration)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
