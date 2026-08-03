#!/usr/bin/env python3
"""Commanded x reached over all four bays — the demo question, measured.

`probe_policy_counterfactual.py` established that two sentences produce two
trajectories. This asks the stronger, user-facing question:

    say "the N-th bay" -> does the car STOP, INSIDE bay N?

For each of the four ordinals (optionally repeated), the car is reset to the
corpus start, told one sentence, and driven until it either comes to rest or a
timeout expires. Three verdicts per trial:

    reached   nearest bay centre at the final position, and how far off
    stopped   displacement over the last 3 s under 0.15 m
    in_bay    final position inside the bay rectangle (not just nearest)

The output table is the confusion matrix the plan's A2 ablation needs, and
`stopped` is the direct test of the hold-frame padding fix — the unpadded v2
model drove 6 m past the back wall on every trial.

Requires: sim + vla_bridge_node + vla_policy_server running.

    python3 probe_policy_bays.py --repeats 1
    python3 probe_policy_bays.py --repeats 3 --timeout 75
"""

import argparse
import json
import math
import os
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
from nav_vla_pkg.gz_reset import SimResetter  # noqa: E402

DEFAULT_START = (-12.30, -22.16, 1.112)

# Geometry from extract_track_paths.py / docs/ver/20260728_2115.
BAYS = {"bay1": -9.36, "bay2": -5.72, "bay3": -2.09, "bay4": 1.54}
BAY_X_OPEN, BAY_X_BACK = 0.81, 6.47
BAY_HALF_WIDTH = 3.63 / 2.0

SENTENCES = {
    "bay1": "Park in the first bay on the right, slowly.",
    "bay2": "Park in the second bay on the right, slowly.",
    "bay3": "Park in the third bay on the right, slowly.",
    "bay4": "Park in the fourth bay on the right, slowly.",
}

# The A3 ablation, closed loop: templates AND slot words drawn only from the
# heldout split of instructions.json, committed before training. None of these
# strings — not the sentence frames, not "number three", not "at a crawl" —
# appears anywhere in the training corpus. A regex parser keyed to the training
# phrasing fails this by construction; a policy that reads language passes.
SENTENCES_HELDOUT = {
    "bay1": "Find the number one bay on your right and park there, at a crawl.",
    "bay2": "I want you in the number two parking space, taking it easy.",
    "bay3": "Bring the car to a stop inside the number three bay, at a crawl.",
    "bay4": "Put it in the last slot on the right, taking it easy.",
}


def nearest_bay(x, y):
    k = min(BAYS, key=lambda b: math.hypot(x - 3.64, y - BAYS[b]))
    return k, round(math.hypot(x - 3.64, y - BAYS[k]), 2)


def in_bay(x, y, bay):
    return (BAY_X_OPEN <= x <= BAY_X_BACK
            and abs(y - BAYS[bay]) <= BAY_HALF_WIDTH)


class BayProbe(Node):
    def __init__(self, args):
        super().__init__("policy_bay_probe")
        self.args = args
        self.gz = resolve_gz_bin("")
        latched = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.instr_pub = self.create_publisher(String, "/vla/instruction", latched)
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
                d = sorted((math.hypot(t.transform.translation.x - truth[0],
                                       t.transform.translation.y - truth[1]), i)
                           for i, t in enumerate(msg.transforms))
                if d[0][0] < 2.0:
                    self.idx = d[0][1]
            return
        if self.idx < len(msg.transforms):
            t = msg.transforms[self.idx].transform
            self.tf = (t.translation.x, t.translation.y)

    def _spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def trial(self, bay):
        a = self.args
        self.instr_pub.publish(String(data=""))
        self._spin(0.5)
        ok, msg, _ = self.reset.reset(a.x, a.y, a.yaw, cmd_pub=self.cmd_pub)
        if not ok:
            return {"commanded": bay, "error": f"reset: {msg}"}
        self._spin(a.settle_s)

        bank = SENTENCES_HELDOUT if self.args.heldout else SENTENCES
        self.instr_pub.publish(String(data=bank[bay]))
        track, t0 = [], time.monotonic()
        stopped_at = None
        while time.monotonic() - t0 < a.timeout:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.tf:
                track.append((time.monotonic() - t0, *self.tf))
            # Rest = under 0.15 m of movement across the last 3 s, once clear of
            # the start. Checked online so a stopped car ends the trial early
            # instead of idling out the clock.
            if len(track) > 80 and track[-1][0] > 10.0:
                ref = next((p for p in reversed(track)
                            if track[-1][0] - p[0] >= 3.0), None)
                if ref and math.hypot(track[-1][1] - ref[1],
                                      track[-1][2] - ref[2]) < 0.15:
                    stopped_at = track[-1][0]
                    break
            time.sleep(0.04)
        self.instr_pub.publish(String(data=""))
        self._spin(0.5)

        if not track:
            return {"commanded": bay, "error": "no pose data"}
        _, fx, fy = track[-1]
        nb, dist = nearest_bay(fx, fy)
        return {"commanded": bay, "end": [round(fx, 2), round(fy, 2)],
                "reached": nb, "off_centre_m": dist,
                "in_bay": in_bay(fx, fy, bay),
                "stopped": stopped_at is not None,
                "stop_t_s": round(stopped_at, 1) if stopped_at else None,
                "correct": nb == bay and stopped_at is not None
                           and in_bay(fx, fy, bay)}

    def run(self):
        a = self.args
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.idx is not None:
                break
        if self.idx is None:
            print("no ego — is the sim running?", file=sys.stderr)
            return 1

        rows = []
        for rep in range(a.repeats):
            for bay in BAYS:
                r = self.trial(bay)
                r["rep"] = rep
                rows.append(r)
                if "error" in r:
                    print(f"{bay}: ERROR {r['error']}")
                else:
                    print(f"cmd {bay}  -> ended {r['end']}  nearest {r['reached']}"
                          f" ({r['off_centre_m']} m)  "
                          f"stopped={'yes @' + str(r['stop_t_s']) + 's' if r['stopped'] else 'NO'}"
                          f"  in_bay={r['in_bay']}  "
                          f"{'CORRECT' if r['correct'] else 'wrong'}")

        good = [r for r in rows if "error" not in r]
        n_ok = sum(1 for r in good if r["correct"])
        n_stop = sum(1 for r in good if r["stopped"])
        print(f"\nconfusion (commanded -> reached):")
        for bay in BAYS:
            got = [r["reached"] for r in good if r["commanded"] == bay]
            print(f"  {bay}: {got}")
        print(f"\nstopped {n_stop}/{len(good)}   fully correct {n_ok}/{len(good)}")
        print("(unpadded v2 reference: stopped 0/4, drove ~6 m past the wall)")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            print(f"wrote {a.out}")
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x", type=float, default=DEFAULT_START[0])
    p.add_argument("--y", type=float, default=DEFAULT_START[1])
    p.add_argument("--yaw", type=float, default=DEFAULT_START[2])
    p.add_argument("--timeout", type=float, default=75.0)
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--heldout", action="store_true",
                   help="use never-trained paraphrases (A3 ablation)")
    p.add_argument("--tf-topic", default="/world/default/dynamic_pose/info")
    p.add_argument("--out", default="bay_confusion.json")
    args = p.parse_args()

    rclpy.init()
    node = BayProbe(args)
    try:
        return node.run()
    except KeyboardInterrupt:
        node.instr_pub.publish(String(data=""))
        node.cmd_pub.publish(Twist())
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
