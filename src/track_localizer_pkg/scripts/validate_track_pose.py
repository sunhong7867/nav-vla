#!/usr/bin/env python3
"""Measure track_pose_node accuracy against tape-measured ground truth.

Two gates decide whether the Hesai pose is good enough to drive on:

    position   < 5 cm  mean error
    heading    < 3 deg RMS

Heading matters more than it looks. At a 3 m lookahead, 5 deg of yaw error is
26 cm of lateral error at the target — larger than any trajectory tolerance
downstream. Measure it before trusting the pose.

Usage
-----
Park the car on a surveyed point and hold still::

    python3 validate_track_pose.py static --truth 5.00 1.20 --label P1

Drive a straight surveyed line, slowly and without steering::

    python3 validate_track_pose.py heading --from 3.0 0.0 --to 10.0 0.0

Both append one JSON record per run to --out (default track_pose_eval.jsonl).
Then::

    python3 validate_track_pose.py report

Surveyed coordinates are (forward_m, lateral_m) in the same track frame the
4-point homography was picked in — the origin is the LiDAR, +forward across the
track, +lateral to the sensor's left.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

DEFAULT_OUT = "track_pose_eval.jsonl"
POSITION_GATE_M = 0.05
HEADING_GATE_DEG = 3.0


def _collect(topic_samples, duration, predicate=None):
    """Spin a subscriber on /track/vehicle_pose, returning matching samples."""
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10,
    )

    samples = []

    class Collector(Node):
        def __init__(self):
            super().__init__("track_pose_validator")
            self.create_subscription(Odometry, "/track/vehicle_pose", self._cb, qos)
            self.start = self.get_clock().now().nanoseconds * 1e-9

        def _cb(self, msg):
            q = msg.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            sample = {
                "forward": msg.pose.pose.position.x,
                "lateral": msg.pose.pose.position.y,
                "yaw": yaw,
                # track_pose_node encodes "heading is held, not measured" as a
                # huge yaw variance. Anything above 1.0 is a stale heading.
                "heading_valid": msg.pose.covariance[35] < 1.0,
                "speed": math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y),
                "pos_std": math.sqrt(max(msg.pose.covariance[0], 0.0)),
            }
            if predicate is None or predicate(sample):
                samples.append(sample)
            if len(samples) >= topic_samples:
                raise SystemExit(0)

        def elapsed(self):
            return self.get_clock().now().nanoseconds * 1e-9 - self.start

    rclpy.init()
    node = Collector()
    try:
        while rclpy.ok() and len(samples) < topic_samples and node.elapsed() < duration:
            rclpy.spin_once(node, timeout_sec=0.2)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return samples


def _append(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cmd_static(args):
    truth_f, truth_l = args.truth
    print(f"[static] truth=({truth_f:.3f}, {truth_l:.3f}) m — hold the car still.")
    samples = _collect(args.samples, args.timeout)
    if len(samples) < 5:
        print(f"only {len(samples)} samples — is track_pose_node publishing?", file=sys.stderr)
        return 1

    errors = [math.hypot(s["forward"] - truth_f, s["lateral"] - truth_l) for s in samples]
    fwd = [s["forward"] for s in samples]
    lat = [s["lateral"] for s in samples]
    record = {
        "kind": "static",
        "label": args.label,
        "truth": [truth_f, truth_l],
        "n": len(samples),
        "mean_error_m": statistics.fmean(errors),
        "max_error_m": max(errors),
        "bias_forward_m": statistics.fmean(fwd) - truth_f,
        "bias_lateral_m": statistics.fmean(lat) - truth_l,
        "jitter_forward_m": statistics.pstdev(fwd) if len(fwd) > 1 else 0.0,
        "jitter_lateral_m": statistics.pstdev(lat) if len(lat) > 1 else 0.0,
    }
    _append(args.out, record)
    ok = record["mean_error_m"] < POSITION_GATE_M
    print(
        f"[static] n={record['n']} mean={record['mean_error_m'] * 100:.1f} cm "
        f"max={record['max_error_m'] * 100:.1f} cm "
        f"bias=({record['bias_forward_m'] * 100:+.1f}, {record['bias_lateral_m'] * 100:+.1f}) cm "
        f"jitter=({record['jitter_forward_m'] * 100:.1f}, {record['jitter_lateral_m'] * 100:.1f}) cm "
        f"-> {'PASS' if ok else 'FAIL'} (gate {POSITION_GATE_M * 100:.0f} cm)"
    )
    if not ok:
        print(
            "  A large constant bias is usually the homography or the centroid "
            "sitting on the sensor-facing face, not noise. Check bias vs jitter.",
        )
    return 0 if ok else 1


def cmd_heading(args):
    ax, ay = getattr(args, "from")
    bx, by = args.to
    truth = math.atan2(by - ay, bx - ax)
    length = math.hypot(bx - ax, by - ay)
    print(
        f"[heading] line ({ax:.2f},{ay:.2f}) -> ({bx:.2f},{by:.2f}), "
        f"bearing {math.degrees(truth):.2f} deg, length {length:.2f} m. Drive it straight."
    )

    def on_line(sample):
        if not sample["heading_valid"] or sample["speed"] < args.min_speed:
            return False
        # Reject samples off the surveyed line so a curved approach or the
        # turnaround at either end does not pollute the estimate.
        dx, dy = bx - ax, by - ay
        t = ((sample["forward"] - ax) * dx + (sample["lateral"] - ay) * dy) / (length ** 2)
        if not (args.margin <= t <= 1.0 - args.margin):
            return False
        lateral_off = abs((sample["forward"] - ax) * dy - (sample["lateral"] - ay) * dx) / length
        return lateral_off <= args.corridor

    samples = _collect(args.samples, args.timeout, predicate=on_line)
    if len(samples) < 10:
        print(
            f"only {len(samples)} usable samples — drive faster than "
            f"{args.min_speed} m/s and stay inside the {args.corridor} m corridor.",
            file=sys.stderr,
        )
        return 1

    def wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    errs = [math.degrees(wrap(s["yaw"] - truth)) for s in samples]
    rms = math.sqrt(statistics.fmean([e * e for e in errs]))
    record = {
        "kind": "heading",
        "label": args.label,
        "from": [ax, ay],
        "to": [bx, by],
        "truth_deg": math.degrees(truth),
        "n": len(samples),
        "rms_deg": rms,
        "bias_deg": statistics.fmean(errs),
        "max_abs_deg": max(abs(e) for e in errs),
        "mean_speed_mps": statistics.fmean([s["speed"] for s in samples]),
    }
    _append(args.out, record)
    ok = rms < HEADING_GATE_DEG
    print(
        f"[heading] n={record['n']} rms={rms:.2f} deg bias={record['bias_deg']:+.2f} deg "
        f"max={record['max_abs_deg']:.2f} deg at {record['mean_speed_mps']:.2f} m/s "
        f"-> {'PASS' if ok else 'FAIL'} (gate {HEADING_GATE_DEG:.0f} deg)"
    )
    if not ok:
        print(
            "  A consistent bias means the travel-direction assumption or the "
            "frame convention is off. Scatter means measurement_std/process_accel_std "
            "need retuning, or the car is too far for enough returns."
        )
    return 0 if ok else 1


def cmd_report(args):
    path = Path(args.out)
    if not path.exists():
        print(f"no results at {path}", file=sys.stderr)
        return 1
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    statics = [r for r in records if r["kind"] == "static"]
    headings = [r for r in records if r["kind"] == "heading"]

    print(f"=== {path} — {len(statics)} static, {len(headings)} heading runs ===")
    pass_all = True
    if statics:
        worst = max(statics, key=lambda r: r["mean_error_m"])
        overall = statistics.fmean([r["mean_error_m"] for r in statics])
        ok = worst["mean_error_m"] < POSITION_GATE_M
        pass_all &= ok
        print(
            f"position: mean {overall * 100:.1f} cm, worst {worst['mean_error_m'] * 100:.1f} cm "
            f"at {worst.get('label') or worst['truth']} -> {'PASS' if ok else 'FAIL'}"
        )
        for r in sorted(statics, key=lambda r: -r["mean_error_m"]):
            print(
                f"  {str(r.get('label') or r['truth']):<14} "
                f"mean {r['mean_error_m'] * 100:5.1f} cm  "
                f"bias ({r['bias_forward_m'] * 100:+5.1f}, {r['bias_lateral_m'] * 100:+5.1f}) cm"
            )
    if headings:
        worst = max(headings, key=lambda r: r["rms_deg"])
        ok = worst["rms_deg"] < HEADING_GATE_DEG
        pass_all &= ok
        print(
            f"heading:  worst RMS {worst['rms_deg']:.2f} deg "
            f"at {worst.get('label') or worst['from']} -> {'PASS' if ok else 'FAIL'}"
        )
        for r in sorted(headings, key=lambda r: -r["rms_deg"]):
            print(
                f"  {str(r.get('label') or r['from']):<14} "
                f"rms {r['rms_deg']:5.2f} deg  bias {r['bias_deg']:+5.2f} deg  n={r['n']}"
            )
    if not statics or not headings:
        print("incomplete: run both static and heading before calling the gate.")
        return 1
    print(f"\nGATE: {'PASS' if pass_all else 'FAIL'}")
    return 0 if pass_all else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("static", help="accuracy at a surveyed standing point")
    s.add_argument("--truth", nargs=2, type=float, required=True, metavar=("FORWARD", "LATERAL"))
    s.add_argument("--label", default="")
    s.add_argument("--samples", type=int, default=100)
    s.add_argument("--timeout", type=float, default=30.0)
    s.set_defaults(func=cmd_static)

    h = sub.add_parser("heading", help="heading RMS along a surveyed straight line")
    h.add_argument("--from", nargs=2, type=float, required=True, metavar=("FORWARD", "LATERAL"))
    h.add_argument("--to", nargs=2, type=float, required=True, metavar=("FORWARD", "LATERAL"))
    h.add_argument("--label", default="")
    h.add_argument("--samples", type=int, default=200)
    h.add_argument("--timeout", type=float, default=120.0)
    h.add_argument("--min-speed", type=float, default=0.3, dest="min_speed")
    h.add_argument("--corridor", type=float, default=0.5, help="max lateral offset from the line (m)")
    h.add_argument("--margin", type=float, default=0.1, help="fraction trimmed from each end")
    h.set_defaults(func=cmd_heading)

    r = sub.add_parser("report", help="aggregate runs and call the gate")
    r.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
