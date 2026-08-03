#!/usr/bin/env python3
"""Measure the rotation between the model's reported yaw and its heading.

The action label is an ego-frame SE(2) delta, so it is only as good as the frame
it is expressed in. Getting this constant wrong rotates every label in the corpus
by the same amount, and nothing downstream can detect it — the data stays
perfectly self-consistent, just in the wrong frame.

Method: drive **straight**. A turn produces sideslip, and sideslip adds an
unknown angle to whatever frame offset exists; with no turn what remains is the
frame offset alone. The car is driven straight several times, and for each step
the direction of travel is compared with the reported yaw.

Two sources are measured separately, because they disagree:
``gz model -p`` (the model origin) and the bridged ``dynamic_pose`` TF. If they
still disagree after this, do not average them — find out why first, because one
of them is not describing the body you think it is.

Usage (sim running)::

    python3 src/nav_vla_pkg/scripts/calibrate_yaw_offset.py
    python3 src/nav_vla_pkg/scripts/calibrate_yaw_offset.py --runs 4 --speed 1.5
"""

import argparse
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
from tf2_msgs.msg import TFMessage  # noqa: E402

from nav_vla_pkg.gz_pose import query_world_pose, resolve_gz_bin  # noqa: E402
from nav_vla_pkg.gz_reset import SimResetter, _load_zone  # noqa: E402


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def offsets_from_track(samples, min_step=0.05):
    """(travel direction - reported yaw) for each step long enough to define one."""
    out = []
    for a, b in zip(samples, samples[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) < min_step:
            continue
        out.append(wrap(math.atan2(dy, dx) - a[2]))
    return out


class Calibrator(Node):
    def __init__(self, args):
        super().__init__("yaw_offset_calibrator")
        self.args = args
        self.gz_bin = resolve_gz_bin("")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=5)
        self.create_subscription(TFMessage, args.tf_topic, self._tf_cb, qos)
        self.tf_latest = None
        self.resetter = SimResetter(gz_bin=self.gz_bin, model=args.model,
                                    logger=self.get_logger())

    def _tf_cb(self, msg):
        if self.args.tf_index < len(msg.transforms):
            tf = msg.transforms[self.args.tf_index].transform
            q = tf.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.tf_latest = (tf.translation.x, tf.translation.y, yaw)

    def drive_straight(self, x, y, yaw):
        self.resetter.reset(x, y, yaw, cmd_pub=self.cmd_pub)
        time.sleep(0.5)
        msg = Twist()
        msg.linear.x = float(self.args.speed)
        msg.angular.z = 0.0          # straight: no sideslip to contaminate the angle

        cli_track, tf_track = [], []
        deadline = time.monotonic() + self.args.duration
        while time.monotonic() < deadline:
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            p = query_world_pose(self.gz_bin, self.args.model)
            if p:
                cli_track.append(p)
            if self.tf_latest:
                tf_track.append(self.tf_latest)
            time.sleep(0.05)
        self.cmd_pub.publish(Twist())
        time.sleep(0.3)
        return cli_track, tf_track

    def run(self):
        a = self.args
        bx, by, byaw = _load_zone(a.zone)
        print(f"straight-line calibration: {a.runs} runs of {a.duration}s "
              f"at {a.speed} m/s from {a.zone}\n")

        cli_all, tf_all = [], []
        for i in range(a.runs):
            # Vary the starting heading: a frame offset is constant in the body
            # frame, so it must come out the same from every direction. Anything
            # that changes with heading is not a frame offset.
            yaw = byaw + i * (math.pi / 2)
            cli_track, tf_track = self.drive_straight(bx, by, yaw)
            c = offsets_from_track(cli_track)
            t = offsets_from_track(tf_track)
            cli_all += c
            tf_all += t
            fmt = lambda v: f"{math.degrees(statistics.median(v)):+7.2f}" if v else "    n/a"
            print(f"  run {i + 1}: start_yaw={math.degrees(yaw):+7.1f}deg  "
                  f"cli={fmt(c)}deg (n={len(c)})  tf={fmt(t)}deg (n={len(t)})")

        print("\n" + "=" * 60)
        verdicts = {}
        for name, vals in (("gz model -p (model origin)", cli_all),
                           ("dynamic_pose TF", tf_all)):
            if len(vals) < 20:
                print(f"{name:28s}: too few samples ({len(vals)})")
                continue
            med = statistics.median(vals)
            sd = statistics.pstdev(vals)
            verdicts[name] = (med, sd, len(vals))
            print(f"{name:28s}: {math.degrees(med):+7.2f} deg  "
                  f"sd {math.degrees(sd):5.2f} deg  n={len(vals)}")
        if len(verdicts) == 2:
            (m1, s1, _), (m2, s2, _) = verdicts.values()
            d = abs(math.degrees(wrap(m1 - m2)))
            print(f"\nsources disagree by {d:.2f} deg")
            if d > 2.0:
                print("  -> Do NOT average them. A disagreement this large means one "
                      "source is not describing the body it is assumed to, or its "
                      "position and orientation are not sampled at the same instant. "
                      "Resolve it before collecting a corpus.")
            else:
                print("  -> consistent; either may be used")
        print("=" * 60)
        best = min(verdicts.items(), key=lambda kv: kv[1][1]) if verdicts else None
        if best:
            print(f"\nlowest-variance source: {best[0]}")
            print(f"  --yaw-offset-deg {math.degrees(best[1][0]):.2f}")
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--speed", type=float, default=1.2)
    p.add_argument("--zone", default="Start")
    p.add_argument("--model", default="ego_vehicle")
    p.add_argument("--tf-topic", default="/world/default/dynamic_pose/info")
    p.add_argument("--tf-index", type=int, default=0)
    args = p.parse_args()

    rclpy.init()
    node = Calibrator(args)
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
