#!/usr/bin/env python3
"""Measure how much the oracle's trajectory actually changes with the instruction.

This is the number the whole plan exists to move. The current stack scores
**0.063 m** on it: over 5,523 pose-matched pairs whose goal zone differed, the
trajectories separated by six centimetres, because the lane follower steers the
same way regardless of goal. A policy trained on that learns to ignore language.

Method: reset to one pose, issue instruction A, record; reset to the *same* pose,
issue instruction B, record. Two runs from an identical start differing only in
what was said.

Reported separately, because they are different claims:

* ``D_shape``  — discrete Fréchet distance between the two paths as *point sets*,
  time discarded. Speed-invariant, so a slow run and a fast run down the same
  line score zero.
* ``D_sched``  — difference in arc length travelled. Shape-invariant.

Splitting them matters: a single ``max |Δposition|`` is passed by speed alone.
Driving the same curve at 0.7 vs 1.8 m/s puts ~6 m between the two after six
seconds while the *navigation* is identical. That metric was rejected for exactly
this reason.

The floor is not zero. GATE G1 measured 9.5 cm of endpoint spread between two
runs that were given the *same* instruction, so anything below that is noise. A
same-instruction pair is therefore always measured alongside, and the ratio
matters more than the absolute.

Usage (sim + route_oracle running)::

    python3 src/nav_vla_pkg/scripts/measure_counterfactual.py
    python3 src/nav_vla_pkg/scripts/measure_counterfactual.py --duration 14
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from nav_vla_pkg.gz_pose import query_world_pose, resolve_gz_bin  # noqa: E402
from nav_vla_pkg.gz_reset import SimResetter, _load_zone  # noqa: E402

G1_FLOOR_M = 0.095      # measured reset noise floor, docs/ver/20260728_1453


def frechet(P, Q):
    """Discrete Fréchet distance. O(nm), fine for a few hundred points."""
    n, m = len(P), len(Q)
    ca = [[-1.0] * m for _ in range(n)]

    def d(i, j):
        return math.hypot(P[i][0] - Q[j][0], P[i][1] - Q[j][1])

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                ca[i][j] = d(0, 0)
            elif i == 0:
                ca[i][j] = max(ca[0][j - 1], d(0, j))
            elif j == 0:
                ca[i][j] = max(ca[i - 1][0], d(i, 0))
            else:
                ca[i][j] = max(min(ca[i - 1][j], ca[i - 1][j - 1], ca[i][j - 1]),
                               d(i, j))
    return ca[n - 1][m - 1]


def arclen(P):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(P, P[1:]))


def truncate_to(P, limit):
    """Prefix of P covering at most `limit` metres of arc length."""
    out, run = [P[0]], 0.0
    for a, b in zip(P, P[1:]):
        run += math.hypot(b[0] - a[0], b[1] - a[1])
        if run > limit:
            break
        out.append(b)
    return out


class Runner(Node):
    def __init__(self, args):
        super().__init__("counterfactual_probe")
        self.args = args
        self.gz = resolve_gz_bin("")
        self.goal_pub = self.create_publisher(String, "/oracle_goal", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=5)
        self.create_subscription(TFMessage, args.tf_topic, self._tf_cb, qos)
        self.tf = None
        self.idx = None
        self.reset = SimResetter(gz_bin=self.gz, logger=self.get_logger())

    def _tf_cb(self, msg):
        if self.idx is None:
            truth = query_world_pose(self.gz, "ego_vehicle")
            if truth and msg.transforms:
                d = [(math.hypot(t.transform.translation.x - truth[0],
                                 t.transform.translation.y - truth[1]), i)
                     for i, t in enumerate(msg.transforms)]
                d.sort()
                if d[0][0] < 2.0:
                    self.idx = d[0][1]
            return
        if self.idx < len(msg.transforms):
            t = msg.transforms[self.idx].transform
            self.tf = (t.translation.x, t.translation.y)

    def run_once(self, start, instruction):
        self.goal_pub.publish(String(data=json.dumps({"goal": "stop"})))
        time.sleep(0.3)
        self.reset.reset(start[0], start[1], start[2], cmd_pub=self.cmd_pub)
        time.sleep(1.0)
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.05)
        self.goal_pub.publish(String(data=json.dumps(instruction)))
        track, t0 = [], time.monotonic()
        while time.monotonic() - t0 < self.args.duration:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.tf:
                track.append(self.tf)
            time.sleep(0.05)
        self.goal_pub.publish(String(data=json.dumps({"goal": "stop"})))
        time.sleep(0.4)
        # thin to ~1 point per 0.2 m so Frechet stays cheap
        thin = [track[0]] if track else []
        for p in track[1:]:
            if math.hypot(p[0] - thin[-1][0], p[1] - thin[-1][1]) > 0.2:
                thin.append(p)
        return thin

    def run(self):
        a = self.args
        for _ in range(80):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.idx is not None:
                break
        if self.idx is None:
            print("could not locate the ego vehicle — is the sim running?",
                  file=sys.stderr)
            return 1
        start = _load_zone(a.start_zone)

        if a.mode == "parking":
            cases = [
                ("SAME bay (noise floor)",
                 {"goal": "bay2", "target_speed": 1.0},
                 {"goal": "bay2", "target_speed": 1.0}),
                ("bay2 vs bay3 (adjacent)",
                 {"goal": "bay2", "target_speed": 1.0},
                 {"goal": "bay3", "target_speed": 1.0}),
                ("bay1 vs bay4 (extremes)",
                 {"goal": "bay1", "target_speed": 1.0},
                 {"goal": "bay4", "target_speed": 1.0}),
                ("bay2 vs bay4",
                 {"goal": "bay2", "target_speed": 1.0},
                 {"goal": "bay4", "target_speed": 1.0}),
            ]
            start = (a.aisle_x, a.aisle_y, math.pi / 2 + math.pi / 2)
            print(f"start aisle ({a.aisle_x}, {a.aisle_y}), "
                  f"{a.duration}s per run, G1 floor {G1_FLOOR_M * 100:.1f} cm\n")
            print(f"{'case':32s} {'D_shape':>9s} {'D_sched':>9s} {'vs floor':>9s}")
            results = []
            for name, ia, ib in cases:
                A = self.run_once(start, ia)
                B = self.run_once(start, ib)
                if len(A) < 5 or len(B) < 5:
                    print(f"{name:32s}  too few points ({len(A)}, {len(B)})")
                    continue
                la, lb = arclen(A), arclen(B)
                common = min(la, lb)
                ds = frechet(truncate_to(A, common), truncate_to(B, common))
                dl = abs(la - lb)
                # How far the two runs stay together before separating: the
                # ordinal claim lives here. If they diverge from the first metre,
                # something other than the instruction is driving the difference.
                shared = 0.0
                run = 0.0
                for (pa, qa), (pb, qb) in zip(A, B):
                    if math.hypot(pa - pb, qa - qb) > 0.30:
                        break
                    shared = run
                    run += 0.2
                results.append({"case": name, "D_shape_m": ds, "D_sched_m": dl,
                                "shared_m": shared, "instr_a": ia, "instr_b": ib})
                print(f"{name:32s} {ds:8.3f}m {dl:8.3f}m {ds / G1_FLOOR_M:8.1f}x"
                      f"   shared {shared:5.1f} m")
            if a.out:
                with open(a.out, "w", encoding="utf-8") as f:
                    json.dump({"mode": "parking", "results": results}, f, indent=2)
                print(f"\nwrote {a.out}")
            return 0

        cases = [
            ("SAME instruction (noise floor)",
             {"goal": a.goal_a, "lane": "lane2", "target_speed": 1.5},
             {"goal": a.goal_a, "lane": "lane2", "target_speed": 1.5}),
            ("LANE differs",
             {"goal": a.goal_a, "lane": "lane1", "target_speed": 1.5},
             {"goal": a.goal_a, "lane": "lane2", "target_speed": 1.5}),
            ("GOAL differs",
             {"goal": a.goal_a, "lane": "lane2", "target_speed": 1.5},
             {"goal": a.goal_b, "lane": "lane2", "target_speed": 1.5}),
            ("SPEED differs",
             {"goal": a.goal_b, "lane": "lane2", "target_speed": 0.7},
             {"goal": a.goal_b, "lane": "lane2", "target_speed": 1.8}),
        ]
        print(f"start {a.start_zone} {tuple(round(v, 2) for v in start)}, "
              f"{a.duration}s per run, G1 floor {G1_FLOOR_M * 100:.1f} cm\n")
        print(f"{'case':32s} {'D_shape':>9s} {'D_sched':>9s} {'vs floor':>9s}")
        results = []
        for name, ia, ib in cases:
            A = self.run_once(start, ia)
            B = self.run_once(start, ib)
            if len(A) < 5 or len(B) < 5:
                print(f"{name:32s}  too few points ({len(A)}, {len(B)})")
                continue
            # Schedule is compared over the fixed wall-clock window; shape must
            # be compared over a common arc length, otherwise a slower run is a
            # strict prefix of a faster one and Frechet returns the length of the
            # leftover instead of a shape difference. That artefact reported
            # 12.15 m for a pure speed change down an identical line.
            la, lb = arclen(A), arclen(B)
            dl = abs(la - lb)
            common = min(la, lb)
            ds = frechet(truncate_to(A, common), truncate_to(B, common))
            results.append({"case": name, "D_shape_m": ds, "D_sched_m": dl,
                            "arclen_a_m": la, "arclen_b_m": lb,
                            "n_a": len(A), "n_b": len(B),
                            "instr_a": ia, "instr_b": ib})
            print(f"{name:32s} {ds:8.3f}m {dl:8.3f}m {ds / G1_FLOOR_M:8.1f}x")

        print()
        base = next((r for r in results if r["case"].startswith("SAME")), None)
        if base:
            print(f"measured same-instruction floor: {base['D_shape_m'] * 100:.1f} cm")
            for r in results[1:]:
                ratio = r["D_shape_m"] / max(base["D_shape_m"], 1e-6)
                print(f"  {r['case']:30s} {ratio:6.1f}x the floor")
        print("\nreference: the current stack scores 0.063 m on goal-differs pairs")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump({"start_zone": a.start_zone, "duration": a.duration,
                           "g1_floor_m": G1_FLOOR_M, "results": results}, f, indent=2)
            print(f"wrote {a.out}")
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-zone", default="Start")
    p.add_argument("--goal-a", default="M2")
    p.add_argument("--goal-b", default="T2")
    p.add_argument("--duration", type=float, default=16.0)
    p.add_argument("--tf-topic", default="/world/default/dynamic_pose/info")
    p.add_argument("--mode", default="ring", choices=["ring", "parking"])
    p.add_argument("--aisle-x", type=float, default=-2.5)
    p.add_argument("--aisle-y", type=float, default=-17.6)
    p.add_argument("--out", default="counterfactual_report.json")
    args = p.parse_args()

    rclpy.init()
    node = Runner(args)
    try:
        return node.run()
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist())
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
