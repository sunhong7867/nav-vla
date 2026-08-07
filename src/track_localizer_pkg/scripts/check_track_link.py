#!/usr/bin/env python3
"""Thor-side receiver validator for the laptop->Thor /track/* link.

Subscribes with track_pose_node's exact QoS contract and measures, over a
fixed window: message rate, worst inter-arrival gap, and stamp age
(receive time - header.stamp; includes clock offset until chrony is up —
that is the point: this number IS the D2 sync gate once the link is real).

Gates (master plan D2 / §4.4):
    pose rate       >= 10 Hz (>= 0.9x expected)
    worst gap       <  0.5 s  (the pose-loss stop rule threshold)
    estop received  yes

    source track_link_env.sh <laptop-ip>
    python3 check_track_link.py [--duration 15] [--expect-rate 10]

Exit 0 = PASS, 1 = FAIL (usable from scripts).
"""

import argparse
import json
import statistics as st
import time

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


class LinkCheck(Node):
    def __init__(self):
        super().__init__("check_track_link")
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.t = {"pose": [], "map": [], "status": [], "estop": []}
        self.age = []
        self.heading_valid = 0
        self.status_n = 0
        self.create_subscription(
            Odometry, "/track/vehicle_pose",
            lambda m: self._pose(m, "pose"), sensor_qos)
        self.create_subscription(
            Odometry, "/track/vehicle_pose_map",
            lambda m: self._pose(m, "map"), sensor_qos)
        self.create_subscription(
            String, "/track/vehicle_status", self._status, 10)
        self.create_subscription(
            Bool, "/track/geofence_estop",
            lambda m: self.t["estop"].append(time.monotonic()), 10)

    def _pose(self, msg, key):
        now = time.monotonic()
        self.t[key].append(now)
        if key == "pose":
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            wall = time.time()
            self.age.append(wall - stamp)

    def _status(self, msg):
        self.t["status"].append(time.monotonic())
        self.status_n += 1
        try:
            if json.loads(msg.data).get("heading_valid"):
                self.heading_valid += 1
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--expect-rate", type=float, default=10.0)
    p.add_argument("--max-gap", type=float, default=0.5)
    args = p.parse_args()

    rclpy.init()
    node = LinkCheck()
    end = time.monotonic() + args.duration
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    fails = []
    print(f"\n=== track link report ({args.duration:.0f} s) ===")
    for key, label in (("pose", "/track/vehicle_pose"),
                       ("map", "/track/vehicle_pose_map"),
                       ("status", "/track/vehicle_status"),
                       ("estop", "/track/geofence_estop")):
        ts = node.t[key]
        rate = len(ts) / args.duration
        gap = max((b - a for a, b in zip(ts, ts[1:])), default=float("inf"))
        line = f"  {label:28s} {len(ts):4d} msgs  {rate:5.1f} Hz"
        if len(ts) >= 2:
            line += f"  worst gap {gap:.3f} s"
        print(line)
        if key == "pose":
            if rate < 0.9 * args.expect_rate:
                fails.append(f"pose rate {rate:.1f} < {0.9*args.expect_rate:.1f} Hz")
            if gap > args.max_gap:
                fails.append(f"pose gap {gap:.3f} s > {args.max_gap} s "
                             "(pose-loss stop rule would trip)")
        if key == "estop" and not ts:
            fails.append("no estop messages — safety topic not arriving")

    if node.age:
        med = st.median(node.age)
        p95 = sorted(node.age)[int(len(node.age) * 0.95) - 1]
        print(f"  stamp age (clock offset incl.)  median {med*1000:+.1f} ms"
              f"  p95 {p95*1000:+.1f} ms")
        print("    ^ chrony 이전엔 클럭 오프셋이 섞인 값. 동기 후 이 값이"
              " D2 게이트(<30 ms + 전송지연)다.")
    if node.status_n:
        print(f"  heading_valid: {node.heading_valid}/{node.status_n}")

    if fails:
        print("FAIL:")
        for f in fails:
            print(f"  - {f}")
        print("  힌트: 양쪽 ROS_DOMAIN_ID/ROS_DISCOVERY_SERVER 일치 여부,"
              " 서버 프로세스(fastdds discovery) 생존, 방화벽 11811/udp 확인")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
