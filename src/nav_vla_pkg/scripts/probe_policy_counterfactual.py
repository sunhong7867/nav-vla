#!/usr/bin/env python3
"""Closed-loop counterfactual probe for the TRAINED policy.

The same experiment `measure_counterfactual.py` runs against the oracle, pointed
at the learned stack instead: reset to one pose, speak sentence A, record; reset
to the same pose, speak sentence B, record. The claim under test is the entire
point of the project —

    does changing ONLY the sentence change where the car drives?

The oracle scored D_shape 6.4 m on ordinal pairs with a 14 m shared prefix.
The old stack scored 0.063 m. Whatever the policy scores sits between those two
numbers, and that single figure is the honest summary of whether training
worked.

Differences from the oracle probe, all deliberate:

* Commands go to ``/vla/instruction`` as plain English — the same topic and the
  same latched QoS the demo uses. No oracle, no goal JSON, no zone lookup.
* The policy's chunk queue is flushed on instruction change by the bridge, so
  switching sentences between runs needs no restart.
* ``--settle-s`` waits after reset before speaking, because the policy consumes
  a live camera feed: the first chunk must be inferred from the settled view,
  not from frames of the car mid-teleport.

Requires: sim + vla_bridge_node + vla_policy_server (with a checkpoint) running.

    python3 probe_policy_counterfactual.py                      # bay2 vs bay4
    python3 probe_policy_counterfactual.py --duration 40
    python3 probe_policy_counterfactual.py \\
        --say-a "Park in the second bay on the right, slowly." \\
        --say-b "Park in the fourth bay on the right, slowly."
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
from nav_vla_pkg.gz_reset import SimResetter, pause_sim  # noqa: E402

# Same start the corpus used: on the ring->aisle connector, model yaw for
# heading-along-connector (heading + 90 deg, YAW_OFFSET convention).
DEFAULT_START = (-12.30, -22.16, 1.112)

# Oracle reference numbers, for the printout only (docs/ver/20260729_2030).
ORACLE_D_SHAPE = 6.407
OLD_STACK_D_SHAPE = 0.063


def frechet(P, Q):
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
    out, run = [P[0]], 0.0
    for a, b in zip(P, P[1:]):
        run += math.hypot(b[0] - a[0], b[1] - a[1])
        if run > limit:
            break
        out.append(b)
    return out


def resample_arclen(P, step=0.2):
    if len(P) < 2:
        return list(P)
    out, target, run = [P[0]], step, 0.0
    for a, b in zip(P, P[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 0:
            continue
        while run + seg >= target:
            f = (target - run) / seg
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            target += step
        run += seg
    return out


def text_divergence(lines_a, lines_b):
    """1 - Jaccard over word sets of the two reasoning streams.

    Crude on purpose: it only needs to separate "same narration" from
    "different narration" relative to the SAME-sentence floor, mirroring
    how D_shape is only read against its own floor.
    """
    ta = {w for line in lines_a for w in line.lower().split()}
    tb = {w for line in lines_b for w in line.lower().split()}
    if not ta and not tb:
        return 0.0
    return 1.0 - len(ta & tb) / max(1, len(ta | tb))


def shared_prefix_m(A, B, tol=0.30, step=0.2):
    RA, RB = resample_arclen(A, step), resample_arclen(B, step)
    run = 0.0
    for a, b in zip(RA, RB):
        if math.hypot(a[0] - b[0], a[1] - b[1]) > tol:
            return run
        run += step
    return run


class Probe(Node):
    def __init__(self, args):
        super().__init__("policy_cf_probe")
        self.args = args
        self.gz = resolve_gz_bin("")
        # Latched, matching the bridge's TRANSIENT_LOCAL subscription — the
        # instruction must survive even if the bridge re-identifies mid-run.
        latched = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.instr_pub = self.create_publisher(String, "/vla/instruction", latched)
        self.goal_pub = self.create_publisher(String, "/vla_goal", latched)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=5)
        self.create_subscription(TFMessage, args.tf_topic, self._tf_cb, qos)
        self.create_subscription(String, "/vla/status", self._status_cb, 10)
        self.create_subscription(String, "/vla/plan", self._plan_cb, 10)
        # Reasoning text stream (reasoning_vla server + bridge republish).
        # Silent topic on a plain checkpoint — the list just stays empty.
        self.create_subscription(String, "/vla/reasoning",
                                 self._reasoning_cb, 10)
        self.reasoning_lines = []
        self.tf = None
        self.idx = None
        self.status = {}
        self.status_events = []
        self.plan_count = 0
        self.reset = SimResetter(gz_bin=self.gz, logger=self.get_logger())

    def _reasoning_cb(self, msg):
        try:
            self.reasoning_lines.append(json.loads(msg.data).get("text", ""))
        except Exception:
            self.reasoning_lines.append(msg.data)

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

    def _status_cb(self, msg):
        try:
            self.status = json.loads(msg.data)
            self.status_events.append(self.status)
        except (json.JSONDecodeError, TypeError):
            pass

    def _plan_cb(self, _msg):
        self.plan_count += 1

    def _spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def run_once(self, sentence, goal=""):
        a = self.args
        # Silence first: an empty instruction makes the bridge flush its queue
        # and stop asking for chunks, so the reset happens with the car passive.
        self.instr_pub.publish(String(data=""))
        self.goal_pub.publish(String(data="none"))
        self._spin(0.5)
        ok, msg, _ = self.reset.reset(a.x, a.y, a.yaw, cmd_pub=self.cmd_pub)
        if not ok:
            print(f"  reset failed: {msg}", file=sys.stderr)
            return None
        self._spin(a.settle_s)

        # Goal-conditioned bridges (goal_conditioning:=true) read /vla_goal;
        # plain bridges have no subscriber and the publish is inert.
        self.goal_pub.publish(String(data=goal or "none"))
        self.reasoning_lines = []
        self.status_events = []

        # Arm the policy while simulation time is stopped.  The bridge control
        # timer uses simulation time, whereas its inference worker uses wall
        # time.  Pausing here therefore lets the first action chunk arrive
        # without consuming control ticks or manufacturing startup underruns.
        ok, msg = pause_sim(self.gz, True)
        if not ok:
            print(f"  prefill pause failed: {msg}", file=sys.stderr)
            return None
        before = self.plan_count
        prefill_t0 = time.monotonic()
        self.instr_pub.publish(String(data=sentence))
        deadline = prefill_t0 + a.prefill_timeout_s
        while self.plan_count <= before and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.01)
        prefill_s = time.monotonic() - prefill_t0
        if self.plan_count <= before:
            pause_sim(self.gz, False)
            self.instr_pub.publish(String(data=""))
            print(f"  first action chunk did not arrive within "
                  f"{a.prefill_timeout_s:.1f}s", file=sys.stderr)
            return None
        ok, msg = pause_sim(self.gz, False)
        if not ok:
            self.instr_pub.publish(String(data=""))
            print(f"  prefill unpause failed: {msg}", file=sys.stderr)
            return None

        track, t0 = [], time.monotonic()
        while time.monotonic() - t0 < a.duration:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.tf:
                track.append(self.tf)
            time.sleep(0.04)
        run_status = {
            "watchdog_hits": max(
                (int(e.get("watchdog_hits", 0)) for e in self.status_events),
                default=0),
            "underrun_pct": max(
                (float(e.get("underrun_pct", 0.0)) for e in self.status_events),
                default=0.0),
            "latency_ms": next(
                (float(e["latency_ms"]) for e in reversed(self.status_events)
                 if "latency_ms" in e), None),
            "prefill_s": prefill_s,
            "contract": next(
                (e["contract"] for e in reversed(self.status_events)
                 if "contract" in e), None),
        }
        self.instr_pub.publish(String(data=""))
        self._spin(0.5)

        thin = [track[0]] if track else []
        for p in track[1:]:
            if math.hypot(p[0] - thin[-1][0], p[1] - thin[-1][1]) > 0.2:
                thin.append(p)
        return thin, list(self.reasoning_lines), run_status

    def run(self):
        a = self.args
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.idx is not None:
                break
        if self.idx is None:
            print("no ego in dynamic_pose — is the sim running?", file=sys.stderr)
            return 1

        pairs = [("SAME sentence (noise floor)",
                  (a.say_a, a.goal_a), (a.say_a, a.goal_a)),
                 (a.case_label,
                  (a.say_a, a.goal_a), (a.say_b, a.goal_b))]
        print(f"start ({a.x:.2f}, {a.y:.2f}), {a.duration:.0f}s per run, "
              f"{a.repeats} repeat(s)\n")
        results = []
        for name, (sa, ga), (sb, gb) in pairs:
            for rep in range(a.repeats):
                A, reasA, statA = self.run_once(sa, ga) or (None, [], {})
                B, reasB, statB = self.run_once(sb, gb) or (None, [], {})
                if not A or not B or len(A) < 5 or len(B) < 5:
                    la = len(A or []); lb = len(B or [])
                    print(f"{name}: too few points ({la}, {lb}) — did the "
                          "car move? check /vla/status watchdog_hits")
                    continue
                la, lb = arclen(A), arclen(B)
                common = min(la, lb)
                ds = frechet(truncate_to(A, common), truncate_to(B, common))
                sh = shared_prefix_m(A, B)
                # Bay centres from extract_track_paths.py; nearest one names
                # where each run actually ended, which is the demo question —
                # "did 'second bay' park in bay 2" — that D_shape alone
                # cannot answer.
                bays = {"bay1": (3.64, -9.36), "bay2": (3.64, -5.72),
                        "bay3": (3.64, -2.09), "bay4": (3.64, 1.54)}
                def nearest_bay(pt):
                    k = min(bays, key=lambda b: math.hypot(
                        pt[0] - bays[b][0], pt[1] - bays[b][1]))
                    return k, round(math.hypot(pt[0] - bays[k][0],
                                               pt[1] - bays[k][1]), 2)
                ea, eb = nearest_bay(A[-1]), nearest_bay(B[-1])
                print(f"{'':28s} A ended {A[-1][0]:+.1f},{A[-1][1]:+.1f} "
                      f"-> {ea[0]} ({ea[1]} m off)   "
                      f"B ended {B[-1][0]:+.1f},{B[-1][1]:+.1f} "
                      f"-> {eb[0]} ({eb[1]} m off)")
                case_tag = "same" if name.startswith("SAME") else "different"
                pair_id = f"{a.run_id}:{case_tag}:r{rep}"
                row = {"run_id": a.run_id, "pair_id": pair_id,
                       "rollout_id_a": f"{pair_id}:a",
                       "rollout_id_b": f"{pair_id}:b",
                       "start_label": a.start_label,
                       "reset_pose": [a.x, a.y, a.yaw],
                       "case": name, "rep": rep, "D_shape_m": ds,
                       "shared_m": sh, "arclen_a": la, "arclen_b": lb,
                       "end_a": list(A[-1]), "end_b": list(B[-1]),
                       "nearest_bay_a": ea, "nearest_bay_b": eb,
                       "track_a": A, "track_b": B,
                       "say_a": sa, "say_b": sb,
                       "status_a": statA, "status_b": statB}
                if reasA or reasB:
                    row["reasoning_a"], row["reasoning_b"] = reasA, reasB
                    row["text_div"] = text_divergence(reasA, reasB)
                    print(f"{'':28s} reasoning: {len(reasA)}/{len(reasB)} "
                          f"lines, divergence {row['text_div']:.2f}")
                results.append(row)
                print(f"{name:28s} D_shape {ds:6.3f} m   shared {sh:5.1f} m   "
                      f"(drove {la:.1f} / {lb:.1f} m)")
                print(f"{'':28s} watchdog {statA.get('watchdog_hits', '?')}/"
                      f"{statB.get('watchdog_hits', '?')}   underrun "
                      f"{statA.get('underrun_pct', '?')}/"
                      f"{statB.get('underrun_pct', '?')}%   prefill "
                      f"{statA.get('prefill_s', 0):.2f}/"
                      f"{statB.get('prefill_s', 0):.2f}s")

        floor = next((r["D_shape_m"] for r in results
                      if r["case"].startswith("SAME")), None)
        diff = [r for r in results if r["case"].startswith("DIFFERENT")]
        # Text-side mirror of the trajectory methodology: divergence between
        # the two runs' reasoning streams, judged against the same-sentence
        # floor. Grounded reasoning should diverge with behaviour; a parrot
        # reads the same on both sides of the counterfactual.
        tfloor = next((r["text_div"] for r in results
                       if r["case"].startswith("SAME") and "text_div" in r),
                      None)
        tdiff = [r["text_div"] for r in diff if "text_div" in r]
        if tfloor is not None and tdiff:
            print(f"\nreasoning text: floor {tfloor:.2f}   "
                  f"different {max(tdiff):.2f} "
                  f"({max(tdiff) / max(tfloor, 1e-6):.1f}x the floor)")
        print(f"\nreference:  oracle {ORACLE_D_SHAPE:.3f} m   "
              f"old stack {OLD_STACK_D_SHAPE:.3f} m")
        if floor is not None and diff:
            d = max(r["D_shape_m"] for r in diff)
            print(f"policy:     floor {floor:.3f} m   different-bay {d:.3f} m "
                  f"({d / max(floor, 1e-6):.1f}x the floor)")
            print("\nRR (policy / oracle) = "
                  f"{d / ORACLE_D_SHAPE:.2f}   (plan section 7.3 asks >= 0.7)")
        if self.status:
            print(f"\nbridge status at end: {self.status}")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"wrote {a.out}")
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--say-a",
                   default="Park in the second bay on the right, slowly.")
    p.add_argument("--say-b",
                   default="Park in the fourth bay on the right, slowly.")
    p.add_argument("--goal-a", default="",
                   help="zone name (or 'x,y') published on /vla_goal for run A; "
                        "only goal-conditioned bridges listen")
    p.add_argument("--goal-b", default="")
    p.add_argument("--case-label", default="DIFFERENT instruction")
    p.add_argument("--run-id", default="",
                   help="globally unique identifier copied into every row")
    p.add_argument("--start-label", default="",
                   help="human-readable reset condition, e.g. inner/outer")
    p.add_argument("--x", type=float, default=DEFAULT_START[0])
    p.add_argument("--y", type=float, default=DEFAULT_START[1])
    p.add_argument("--yaw", type=float, default=DEFAULT_START[2])
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--prefill-timeout-s", type=float, default=8.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--tf-topic", default="/world/default/dynamic_pose/info")
    p.add_argument("--out", default="policy_cf_report.json")
    args = p.parse_args()

    rclpy.init()
    node = Probe(args)
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
