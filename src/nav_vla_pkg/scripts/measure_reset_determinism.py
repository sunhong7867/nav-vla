#!/usr/bin/env python3
"""GATE G1 — how repeatable is a reset?

Every counterfactual number downstream is a difference between two runs that
started "at the same place". This measures what "the same place" is actually
worth: reset N times to one pose, drive the *same* open-loop command for the same
duration, and look at how far apart the endpoints land.

That spread is the noise floor. A language-induced divergence smaller than it is
unmeasurable, so G1 has to pass before anything else is worth running.

    PASS   spread <= 0.10 m   proceed as planned
    WARN   0.10 - 0.50 m      proceed, but raise every D_diff threshold to 3x the
                              measured floor and say so in the results
    STOP   > 0.50 m           fix this before anything else — every downstream
                              number is noise

A curved command is used on purpose: driving straight hides initial-condition
sensitivity, a turn amplifies it.

Usage (sim must already be running)::

    python3 src/nav_vla_pkg/scripts/measure_reset_determinism.py
    python3 src/nav_vla_pkg/scripts/measure_reset_determinism.py \
        --trials 20 --duration 6.0 --linear 1.2 --angular 0.15 --zone Start
"""

import argparse
import itertools
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

from nav_vla_pkg.gz_pose import query_world_pose, resolve_gz_bin  # noqa: E402
from nav_vla_pkg.gz_reset import SimResetter, _load_zone  # noqa: E402

PASS_M = 0.10
WARN_M = 0.50


class G1Runner(Node):
    def __init__(self, args):
        super().__init__("g1_reset_determinism")
        self.args = args
        self.gz_bin = resolve_gz_bin("")
        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.resetter = SimResetter(
            gz_bin=self.gz_bin, model=args.model, world=args.world,
            settle_ms=args.settle_ms, logger=self.get_logger(),
        )

    def drive_open_loop(self, linear, angular, duration, rate_hz=20.0):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        period = 1.0 / rate_hz
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)
        self.cmd_pub.publish(Twist())
        # Let the last zero command actually take effect before sampling.
        time.sleep(0.2)

    def run(self):
        a = self.args
        if a.zone:
            x, y, yaw = _load_zone(a.zone)
        else:
            x, y, yaw = a.x, a.y, a.yaw

        print("=" * 68)
        print("GATE G1 — reset repeatability (this is NOT a 'does reset work' test)")
        print("=" * 68)
        print("What happens, x%d:" % a.trials)
        print(f"  reset to ({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f}deg)"
              f"  ->  drive linear={a.linear} angular={a.angular} OPEN LOOP"
              f" for {a.duration}s  ->  record endpoint")
        print()
        print("  The car will steer off the track and ignore lane markings. That is")
        print("  intended: no perception is running, this is a constant Twist. The")
        print("  curve is deliberate — a straight line hides initial-condition")
        print("  sensitivity, a turn amplifies it. The ground is a flat plane, so")
        print("  leaving the road costs nothing.")
        print()
        print("  Perfectly deterministic physics would put all endpoints on one")
        print("  point. However far they spread is the NOISE FLOOR: any later")
        print("  language-induced divergence smaller than it is unmeasurable.")
        print(f"\n  Estimated wall clock: ~{a.trials * (a.duration + 1.5) / 60:.1f} min\n")

        endpoints, reset_errors, failures = [], [], 0
        for i in range(a.trials):
            ok, msg, actual = self.resetter.reset(x, y, yaw, cmd_pub=self.cmd_pub)
            if not ok:
                print(f"  trial {i + 1:2d}: RESET FAILED — {msg}")
                failures += 1
                continue
            dp = math.hypot(actual[0] - x, actual[1] - y)
            reset_errors.append(dp)

            self.drive_open_loop(a.linear, a.angular, a.duration)
            end = query_world_pose(self.gz_bin, a.model)
            if end is None:
                print(f"  trial {i + 1:2d}: endpoint read FAILED")
                failures += 1
                continue
            endpoints.append(end)
            print(f"  trial {i + 1:2d}: reset_err={dp * 100:5.1f} cm  "
                  f"end=({end[0]:8.3f}, {end[1]:8.3f}, {math.degrees(end[2]):7.2f}deg)")

        if len(endpoints) < 3:
            print(f"\nonly {len(endpoints)} usable trials — is the sim running "
                  f"and is the model named '{a.model}'?")
            return 1

        spreads = [math.hypot(p[0] - q[0], p[1] - q[1])
                   for p, q in itertools.combinations(endpoints, 2)]
        max_spread = max(spreads)
        yaws = [e[2] for e in endpoints]
        yaw_spread = max(
            abs(math.atan2(math.sin(p - q), math.cos(p - q)))
            for p, q in itertools.combinations(yaws, 2)
        )

        verdict = ("PASS" if max_spread <= PASS_M
                   else "WARN" if max_spread <= WARN_M else "STOP")
        print("\n" + "=" * 64)
        print(f"trials usable        : {len(endpoints)} / {a.trials} ({failures} failed)")
        print(f"reset error          : mean {statistics.fmean(reset_errors) * 100:.1f} cm, "
              f"max {max(reset_errors) * 100:.1f} cm")
        print(f"endpoint spread max  : {max_spread * 100:.1f} cm   <-- G1 metric")
        print(f"endpoint spread mean : {statistics.fmean(spreads) * 100:.1f} cm")
        print(f"endpoint yaw spread  : {math.degrees(yaw_spread):.2f} deg")
        print(f"\nG1: {verdict}   (PASS <= {PASS_M * 100:.0f} cm, "
              f"WARN <= {WARN_M * 100:.0f} cm)")
        if verdict == "WARN":
            print(f"  -> raise every D_diff threshold to >= {max_spread * 3 * 100:.0f} cm "
                  f"and report the measured floor")
        elif verdict == "STOP":
            print("  -> fix before anything else. Suspects, in order: residual body "
                  "velocity surviving the paused teleport (raise --settle-ms), a "
                  "controller still publishing (check `ros2 topic info /cmd_vel`), "
                  "or free-run physics nondeterminism (then do the stepped-world "
                  "migration).")
        print("=" * 64)

        if a.out:
            record = {
                "trials": a.trials, "usable": len(endpoints), "failures": failures,
                "target": [x, y, yaw], "linear": a.linear, "angular": a.angular,
                "duration": a.duration, "settle_ms": a.settle_ms,
                "reset_error_mean_m": statistics.fmean(reset_errors),
                "reset_error_max_m": max(reset_errors),
                "endpoint_spread_max_m": max_spread,
                "endpoint_spread_mean_m": statistics.fmean(spreads),
                "endpoint_yaw_spread_rad": yaw_spread,
                "verdict": verdict,
                "endpoints": endpoints,
            }
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            print(f"\nwrote {a.out}")
        return 0 if verdict != "STOP" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--linear", type=float, default=1.2, help="m/s")
    p.add_argument("--angular", type=float, default=0.15,
                   help="rad/s — keep non-zero, a straight line hides scatter")
    p.add_argument("--zone", default="Start")
    p.add_argument("--x", type=float, default=3.70)
    p.add_argument("--y", type=float, default=24.5943)
    p.add_argument("--yaw", type=float, default=-1.5707)
    p.add_argument("--model", default="ego_vehicle")
    p.add_argument("--world", default="default")
    p.add_argument("--cmd-topic", default="/cmd_vel")
    p.add_argument("--settle-ms", type=int, default=150)
    p.add_argument("--out", default="g1_reset_determinism.json")
    args = p.parse_args()

    rclpy.init()
    node = G1Runner(args)
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
