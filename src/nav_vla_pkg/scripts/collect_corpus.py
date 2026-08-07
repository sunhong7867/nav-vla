#!/usr/bin/env python3
"""Corpus collection driver: enumerate a grid, reset, command, record, label.

This is the piece that turns the oracle into a dataset. Everything before it
proved that a *pair* of episodes can be made to diverge on instruction alone
(docs/ver/20260728_1906). This drives that pairing thousands of times and writes
down which episodes belong to which comparison, at collection time, because that
bookkeeping cannot be reconstructed afterwards.

Structure
---------
The unit of collection is a **counterfactual group**: one start pose, several
instructions. Every episode in a group is reset to the *same* pose, so any
trajectory difference inside a group is attributable to the sentence and nothing
else. Groups get a ``cf_group_id``; episodes inside get a ``cf_variant_id``.

    group g0007  start (-11.3, -19.4, 178.4 deg)
      v0  "Park in the second bay on the right, slowly."      bay2 / 0.7
      v1  "Take the fourth bay, quickly."                      bay4 / 1.8
      v2  "Pull into the first parking space, steadily."       bay1 / 1.2
      ...

Three things this driver does that a naive loop would not:

**It varies where the episode starts.** Every group samples a different point
along the ring→aisle connector, with lateral and heading jitter. If every episode
began from an identical pose, "turn right after 4.2 seconds" would be a perfect
policy on the training set and useless anywhere else. The jitter also supplies
the recovery behaviour a pure-pursuit demonstrator otherwise never shows: from an
offset start the oracle must steer back, and those frames are the only place the
corpus contains a correction.

**It shuffles variant order inside a group.** Anything that drifts over a long
session — physics state, disk pressure, thermal — would otherwise correlate with
bay index and enter the corpus as a confound that looks like signal.

**It records why an episode failed.** The recorder's ``termination`` field is
six-valued for this reason. An episode that timed out is not an episode that
succeeded, and a session where 5% time out is a different dataset from one where
0% do. Outcomes land in ``manifest.jsonl`` next to the plan, so a run that is
killed halfway is still interpretable.

Group completeness matters more than episode count: a group missing one variant
contributes no pair for that comparison. The summary reports complete groups
separately from total episodes.

Usage (sim + route_oracle + episode_recorder running)::

    python3 src/nav_vla_pkg/scripts/collect_corpus.py --dry-run
    python3 src/nav_vla_pkg/scripts/collect_corpus.py --groups 40 --split train
    python3 src/nav_vla_pkg/scripts/collect_corpus.py --groups 8 --split heldout \\
        --session eval_20260728
"""

import argparse
import collections
import json
import math
import os
import random
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
from std_msgs.msg import Int32, String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from nav_vla_pkg.gz_pose import query_world_pose, resolve_gz_bin  # noqa: E402
from nav_vla_pkg.gz_reset import SimResetter  # noqa: E402
from nav_vla_pkg.speed_control import raw_speed_to_mps  # noqa: E402

CFG = os.path.join(PKG_PARENT, "config")
PATHS_FILE = os.path.join(CFG, "track_paths.json")
BANK_FILE = os.path.join(CFG, "instructions.json")

# route_oracle_node.py: heading = model_yaw + YAW_OFFSET, measured -90.00 deg.
YAW_OFFSET = -math.pi / 2

# Ring waypoints that are far enough apart to be distinct goals. T3 (index 124)
# and crosswalk_stop (125) are the same point on the loop, as are Start (325) and
# T1/M1 (328); using both of a pair would add rows without adding signal.
RING_GOALS = ["T2", "M3", "T3", "T4", "Start", "M2"]

# Bounds on how far a ring goal may be from the start, in metres of forward arc.
# The loop is one-way, so a goal 2 m *behind* the car is 140 m ahead of it: an
# unbounded choice produces either a 2-second episode or a four-minute lap,
# neither of which is what the row claims to be.
RING_GAP_MIN_M = 22.0
RING_GAP_MAX_M = 55.0


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


# --------------------------------------------------------------------- bank

class InstructionBank:
    """Templates + slot lexicon, split into train / heldout surface forms.

    The split is over phrasing only. Every bay and every zone appears in both
    splits, so a held-out failure means "did not parse this wording" rather than
    "never saw this goal" — otherwise the A3 ablation measures two things at once.
    """

    def __init__(self, path, split, rng):
        with open(path, "r", encoding="utf-8") as f:
            self.d = json.load(f)
        if split not in ("train", "heldout"):
            raise ValueError(f"unknown split {split!r}")
        self.split = split
        self.rng = rng
        # The bank's speed_map speaks raw motor units (0..250), the vehicle's
        # native scale; the oracle and timeouts below need m/s.
        self.speed_map_raw = {k: v for k, v in self.d["speed_map"].items()
                              if not k.startswith("_")}
        self.speed_map = {k: raw_speed_to_mps(v)
                          for k, v in self.speed_map_raw.items()}

    def word(self, kind, value):
        entry = self.d["lexicon"][kind][value]
        opts = entry.get(self.split) or entry["train"]
        return self.rng.choice(opts)

    def render(self, intent, slots):
        tpl_set = self.d["templates"][intent]
        tpl = self.rng.choice(tpl_set[self.split])
        fill = {k: self.word(k, v) for k, v in slots.items()}
        text = tpl.format(**fill)
        return text, tpl, fill

    def n_templates(self, intent):
        return len(self.d["templates"][intent][self.split])


# --------------------------------------------------------------------- plan

def connector_starts(paths, n, rng, lat_jitter, yaw_jitter_deg,
                     index_lo=0.0, index_hi=0.6):
    """Sample start poses along the ring→aisle connector.

    Sampling stays inside the connector rather than upstream on the ring: the
    oracle's parking path *begins* at the connector, so a pose 15 m back on the
    loop has no nearest point on it and pure pursuit would drive to the connector
    entrance instead of following it. Reaching further back needs the ring-goal →
    parking-goal handoff, which does not exist yet.

    `index_hi` stops short of the connector's end so the whole shared aisle stays
    inside every episode. The shared prefix is what the ordinal axis rests on;
    starting past it would shorten the very thing being measured.
    """
    pts = [tuple(p) for p in paths["ring_to_aisle"]["lane2"]["path"]]
    out = []
    for _ in range(n):
        u = rng.uniform(index_lo, index_hi)
        i = min(int(u * (len(pts) - 2)), len(pts) - 2)
        x, y = pts[i]
        nx, ny = pts[i + 1]
        heading = math.atan2(ny - y, nx - x)
        # Offset perpendicular to travel, so the oracle has to steer back and the
        # corpus contains recovery rather than only nominal following.
        lat = rng.uniform(-lat_jitter, lat_jitter)
        x += -math.sin(heading) * lat
        y += math.cos(heading) * lat
        heading = wrap_pi(heading + math.radians(
            rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)))
        out.append({"x": x, "y": y, "heading": heading,
                    "model_yaw": wrap_pi(heading - YAW_OFFSET),
                    "connector_index": i, "lateral_m": lat})
    return out


def lane_starts(paths, n, rng, lat_jitter, yaw_jitter_deg, windows=None):
    """Sample start poses anywhere on the ring lanes (for ring-goal episodes).

    windows: optional list of (lo, hi) ring-index ranges; when given, start
    indices are drawn only from those ranges (used to target sparse spots
    like the parking OUT-road junction upstream)."""
    out = []
    for _ in range(n):
        lane = rng.choice(["lane1", "lane2"])
        pts = [tuple(p) for p in paths[lane]]
        if windows:
            lo, hi = windows[rng.randrange(len(windows))]
            i = rng.randrange(lo, hi + 1) % len(pts)
        else:
            i = rng.randrange(len(pts))
        x, y = pts[i]
        nx, ny = pts[(i + 1) % len(pts)]
        heading = math.atan2(ny - y, nx - x)
        lat = rng.uniform(-lat_jitter, lat_jitter)
        x += -math.sin(heading) * lat
        y += math.cos(heading) * lat
        heading = wrap_pi(heading + math.radians(
            rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)))
        out.append({"x": x, "y": y, "heading": heading,
                    "model_yaw": wrap_pi(heading - YAW_OFFSET),
                    "lane_index": i, "lane": lane, "lateral_m": lat})
    return out


def polyline_len(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def forward_gap_m(goal_idx, start_idx, n, spacing):
    """Arc distance from start to goal in the one legal direction of travel."""
    return ((goal_idx - start_idx) % n) * spacing


def pick_ring_goal(zones, start_idx, n, spacing, rng):
    """A waypoint far enough ahead to be a drive, near enough to be one episode.

    Required on *both* lanes: the group's whole point is a lane1/lane2 pair from
    one start, and a goal that is a comfortable 30 m ahead on lane1 but 3 m ahead
    on lane2 would give the two variants nothing to compare.
    """
    ok = []
    for z in RING_GOALS:
        e = zones.get(z)
        if not e or not e.get("on_ring"):
            continue
        gaps = [forward_gap_m(e[lane]["index"], start_idx, n, spacing)
                for lane in ("lane1", "lane2")]
        if all(RING_GAP_MIN_M <= g <= RING_GAP_MAX_M for g in gaps):
            ok.append(z)
    return rng.choice(ok) if ok else None


def build_plan(args, paths, bank, rng):
    """Enumerate groups. A group is one start pose and a set of variants."""
    groups = []

    # --- parking: the non-separable ordinal axis -------------------------
    #
    # Each group varies exactly ONE axis and holds everything else fixed. A group
    # that changed both bay and speed would produce pairs differing in two ways,
    # and no amount of downstream analysis can split a single divergence figure
    # back into its causes. `cf_axis` records which axis a group is for, so the
    # verifier can report the axes separately instead of averaging them.
    bays = ["bay1", "bay2", "bay3", "bay4"]
    speeds = ["slow", "normal", "fast"]

    # Route length per bay, so the timeout scales with the drive being asked for
    # rather than with a constant that is slack for bay1 and a guillotine for
    # bay4 at 0.7 m/s. A guillotined run enters the corpus as a timeout, which
    # reads as a controller failure it never was.
    conn_len = polyline_len(paths["ring_to_aisle"]["lane2"]["path"])
    park_len = {k: conn_len + polyline_len(v["path"])
                for k, v in paths["parking"].items()}

    def park_variant(vi, bay, level):
        slots = {"ordinal": bay, "speed": level}
        text, tpl, fill = bank.render("park_bay", slots)
        return {
            "cf_variant_id": f"v{vi}",
            "intent_id": "park_bay",
            "instruction": text,
            "template": tpl,
            "intent_slots": {"goal": bay, "speed_level": level,
                             "target_speed": bank.speed_map[level],
                             "target_speed_raw": bank.speed_map_raw[level],
                             "words": fill},
            "oracle_goal": {"goal": bay, "from": "ring",
                            "target_speed": bank.speed_map[level]},
            "timeout_s": round(park_len[bay] / bank.speed_map[level]
                               * args.timeout_slack + 20.0, 1),
        }

    ordinal_starts = connector_starts(
        paths, args.groups, rng, args.lat_jitter, args.yaw_jitter)
    for gi, sp in enumerate(ordinal_starts):
        # Speed is fixed inside the group and rotated across groups, so it stays
        # a balanced label without contaminating the ordinal comparison.
        level = speeds[gi % len(speeds)]
        order = bays[:]
        rng.shuffle(order)
        groups.append({
            "cf_group_id": f"{args.group_prefix}p{gi:04d}",
            "kind": "parking", "cf_axis": "ordinal",
            "start": sp,
            "variants": [park_variant(vi, bay, level)
                         for vi, bay in enumerate(order)],
        })

    # Floor groups: the SAME instruction repeated from one start. Every ratio in
    # the analysis divides by a noise floor, and the 9.5 cm currently used for
    # that came from GATE G1 — free-running physics with no goal, measured on a
    # different day. This measures the floor under the conditions the corpus was
    # actually collected in, so the denominator stops being an assumption.
    for gi, sp in enumerate(connector_starts(
            paths, args.floor_groups, rng, args.lat_jitter, args.yaw_jitter)):
        bay, level = bays[gi % len(bays)], speeds[gi % len(speeds)]
        # The two variants draw templates independently, so they are usually
        # different wordings of the same request. For the oracle that changes
        # nothing — it never sees the sentence, only `oracle_goal` — so the
        # measured spread is pure physics. The same pair becomes a paraphrase-
        # invariance test the moment a trained policy is driving instead.
        groups.append({
            "cf_group_id": f"{args.group_prefix}f{gi:04d}",
            "kind": "parking", "cf_axis": "floor",
            "start": sp,
            "variants": [park_variant(vi, bay, level) for vi in range(2)],
        })

    # Speed groups: same bay, three speeds, one start. Without these the corpus
    # contains no pair that differs in speed alone, so the schedule axis could
    # only ever be argued for from a separate probe run rather than from the
    # training data itself.
    for gi, sp in enumerate(connector_starts(
            paths, args.speed_groups, rng, args.lat_jitter, args.yaw_jitter)):
        bay = bays[gi % len(bays)]
        order = speeds[:]
        rng.shuffle(order)
        groups.append({
            "cf_group_id": f"{args.group_prefix}s{gi:04d}",
            "kind": "parking", "cf_axis": "speed",
            "start": sp,
            "variants": [park_variant(vi, bay, lv) for vi, lv in enumerate(order)],
        })

    # --- ring: lane axis, plus goal as a label ---------------------------
    n_ring = len(paths["lane1"])
    spacing = float(paths["meta"]["spacing_m"])
    zones = paths["zones"]
    skipped_ring = 0
    for gi, sp in enumerate(lane_starts(
            paths, args.ring_groups, rng, args.lat_jitter, args.yaw_jitter)):
        level = speeds[gi % len(speeds)]
        goal = pick_ring_goal(zones, sp["lane_index"], n_ring, spacing, rng)
        if goal is None:
            # No waypoint sits in the usable window ahead of this start. Drop the
            # group rather than emit one that will time out or finish in 3 s.
            skipped_ring += 1
            continue
        variants = []
        for vi, lane in enumerate(["lane1", "lane2"]):
            slots = {"zone": goal, "lane": lane, "speed": level}
            text, tpl, fill = bank.render("ring_goal", slots)
            variants.append({
                "cf_variant_id": f"v{vi}",
                "intent_id": "ring_goal",
                "instruction": text,
                "template": tpl,
                "intent_slots": {"goal": goal, "lane": lane,
                                 "speed_level": level,
                                 "target_speed": bank.speed_map[level],
                                 "target_speed_raw": bank.speed_map_raw[level],
                                 "words": fill},
                "oracle_goal": {"goal": goal, "lane": lane,
                                "target_speed": bank.speed_map[level]},
                "timeout_s": round(
                    forward_gap_m(zones[goal][lane]["index"], sp["lane_index"],
                                  n_ring, spacing)
                    / bank.speed_map[level] * args.timeout_slack + 20.0, 1),
            })
        groups.append({"cf_group_id": f"{args.group_prefix}r{gi:04d}", "kind": "ring",
                       "cf_axis": "lane", "start": sp, "goal": goal,
                       "variants": variants})

    # --- cruise: lane axis with NO endpoint in the sentence ----------------
    # The demo's first utterance is "start driving", not "drive to T2". These
    # episodes end by duration (the collector cancels mid-cruise), so the
    # demonstration never contains a stop — to_lerobot must not hold-pad them.
    cruise_windows = None
    if args.start_windows:
        cruise_windows = [tuple(int(v) for v in w.split("-"))
                          for w in args.start_windows.split(",")]
    if args.start_poses_file:
        # DAgger-style corrections: explicit start poses sampled from the
        # POLICY'S OWN error states (position + heading where it drifted off
        # the lane line). No jitter — the pose already is the perturbation.
        # Each entry: {"x", "y", "heading", "lane"}. The teacher recovers from
        # exactly the states the current policy visits, which is the data
        # plain jittered starts cannot provide.
        entries = json.load(open(args.start_poses_file))
        cruise_sps = []
        for e in entries:
            pts = [tuple(p) for p in paths[e["lane"]]]
            i = min(range(len(pts)),
                    key=lambda k: (pts[k][0]-e["x"])**2 + (pts[k][1]-e["y"])**2)
            cruise_sps.append({"x": e["x"], "y": e["y"],
                               "heading": e["heading"],
                               "model_yaw": wrap_pi(e["heading"] - YAW_OFFSET),
                               "lane_index": i, "lane": e["lane"],
                               "lateral_m": e.get("dev", 0.0),
                               "dagger": True})
    else:
        cruise_sps = lane_starts(
            paths, args.cruise_groups, rng, args.lat_jitter, args.yaw_jitter,
            windows=cruise_windows)
    for gi, sp in enumerate(cruise_sps):
        level = speeds[gi % len(speeds)]
        # Vary the window so episode length is not a hidden constant the
        # policy could latch onto instead of the instruction.
        dur = [18.0, 24.0, 30.0][(gi // len(speeds)) % 3]
        variants = []
        if sp.get("dagger"):
            cruise_lanes = [sp["lane"]]  # instruction must match the lane
            # the policy was following when it drifted into this state
        elif args.cruise_lane == "both":
            cruise_lanes = ["lane1", "lane2"]
        else:
            cruise_lanes = [args.cruise_lane]
        for vi, lane in enumerate(cruise_lanes):
            slots = {"lane": lane, "speed": level}
            text, tpl, fill = bank.render("cruise", slots)
            variants.append({
                "cf_variant_id": f"v{vi}",
                "intent_id": "cruise",
                "instruction": text,
                "template": tpl,
                "intent_slots": {"lane": lane,
                                 "speed_level": level,
                                 "target_speed": bank.speed_map[level],
                                 "target_speed_raw": bank.speed_map_raw[level],
                                 "cruise_s": dur,
                                 "words": fill},
                "oracle_goal": {"goal": "cruise", "lane": lane,
                                "target_speed": bank.speed_map[level]},
                "cruise_s": dur,
                "timeout_s": dur + 15.0,
            })
        groups.append({"cf_group_id": f"{args.group_prefix}c{gi:04d}",
                       "kind": "cruise", "cf_axis": "lane", "start": sp,
                       "variants": variants})

    if skipped_ring:
        print(f"note: dropped {skipped_ring} ring group(s) — no waypoint "
              f"{RING_GAP_MIN_M:.0f}-{RING_GAP_MAX_M:.0f} m ahead of the start")
    rng.shuffle(groups)
    return groups


# ------------------------------------------------------------------- driver

class Collector(Node):

    def __init__(self, args, plan, ring_pts=None):
        super().__init__("corpus_collector")
        self.args = args
        self.plan = plan
        # Coarse ring polyline for the off-ring guard. The teacher's lane
        # follower can latch onto the parking OUT-road at the junction during
        # goal-less cruises and drive off the ring; duration-terminated
        # episodes would still be marked success and poison the corpus
        # (found 2026-08-04: 16 such episodes across v3y/supp/y5j).
        self.ring_pts = ring_pts or []
        self.gz = resolve_gz_bin("")
        self.goal_pub = self.create_publisher(String, "/oracle_goal", 10)
        self.rec_pub = self.create_publisher(String, "/recorder/command", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # --driver yolo: the demonstrator is the tuned camera lane-follow
        # stack plus the user's navigator_node, commanded exactly like the
        # chat GUI commands it: /nav_goal for zone runs (navigator owns the
        # tuned stop logic), /lane_mode_command for goal-less cruising, and
        # /speed_command (raw Int32 limit into motion_planner) for the tier.
        # motion_planner subscribes its command topics RELIABLE +
        # TRANSIENT_LOCAL (depth 1); a default VOLATILE publisher is silently
        # incompatible and the commands never arrive. Same profile the
        # navigator uses for its own command publishers.
        cmd_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.lane_pub = self.create_publisher(String, "/lane_mode_command", cmd_qos)
        self.speed_pub = self.create_publisher(Int32, "/speed_command", cmd_qos)
        self.motion_pub = self.create_publisher(
            String, "/motion_control_command", cmd_qos)
        self.zone_goal_pub = self.create_publisher(String, "/nav_goal", 10)

        self.create_subscription(String, "/nav_status", self._nav_cb, 20)
        self.create_subscription(String, "/recorder/status", self._rec_cb, 20)
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=5)
        self.create_subscription(TFMessage, args.tf_topic, self._tf_cb, qos)

        self.reset = SimResetter(gz_bin=self.gz, logger=self.get_logger())
        self.tf = None
        self.idx = None
        self.nav = None            # latest /nav_status string
        self.rec_state = None      # latest recorder state dict
        self.manifest = None

    # ---------------------------------------------------------------- ros

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

    def _nav_cb(self, msg):
        self.nav = msg.data

    def _rec_cb(self, msg):
        try:
            self.rec_state = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.rec_state = None

    def _spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _halt_driver(self):
        if self.args.driver == "yolo":
            # Cancel any navigator goal, close the planner's motion gate, and
            # clamp speed — cruise episodes have no goal, so the latter two
            # are what actually stop them.
            self.zone_goal_pub.publish(String(data="stop"))
            self.motion_pub.publish(String(data="stop"))
            self.speed_pub.publish(Int32(data=0))
        else:
            self.goal_pub.publish(String(data=json.dumps({"goal": "stop"})))

    # ------------------------------------------------------------ episode

    def _await_recorder(self, want, timeout=8.0, needs=None):
        """Block until the recorder confirms it changed state.

        Fire-and-forget would let the driver publish a goal before recording
        started, silently truncating the first second of every episode — the
        part that contains the response to the instruction.

        `needs` names a field the message must carry. The recorder also emits a
        bare ``idle`` heartbeat, which is indistinguishable from the ``idle`` that
        follows a stop unless the episode summary is required with it; without
        that check a heartbeat landing in the wait window would report every
        episode as zero frames.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            st = self.rec_state
            if not st:
                continue
            if st.get("state") == "error":
                return st
            if st.get("state") == want and (needs is None or needs in st):
                return st
        return None

    def run_episode(self, group, variant):
        sp = group["start"]
        row = {
            "cf_group_id": group["cf_group_id"],
            "cf_variant_id": variant["cf_variant_id"],
            "kind": group["kind"],
            "instruction": variant["instruction"],
            "intent_id": variant["intent_id"],
            "intent_slots": variant["intent_slots"],
            "start_requested": [sp["x"], sp["y"], sp["model_yaw"]],
        }

        self._halt_driver()
        self._spin(0.3)

        # Reset, with retries. A group whose variants started from different
        # poses yields no valid pair, so a bad reset is worth spending time on.
        ok, msg, actual = False, "", None
        for attempt in range(self.args.reset_retries + 1):
            ok, msg, actual = self.reset.reset(
                sp["x"], sp["y"], sp["model_yaw"], cmd_pub=self.cmd_pub)
            if ok:
                break
            self.get_logger().warn(f"reset attempt {attempt + 1} failed: {msg}")
            self._spin(1.0)
        row["reset_ok"] = ok
        row["reset_detail"] = msg
        row["start_actual"] = list(actual) if actual else None
        if not ok:
            row["termination"] = "reset_failed"
            row["episode"] = None
            return row

        self._spin(1.0)

        self.rec_state = None
        self.rec_pub.publish(String(data=json.dumps({
            "cmd": "start",
            "instruction": variant["instruction"],
            "intent_id": variant["intent_id"],
            "intent_slots": variant["intent_slots"],
            "cf_group_id": group["cf_group_id"],
            "cf_variant_id": variant["cf_variant_id"],
            "cf_axis": group["cf_axis"],
            "seed": self.args.seed,
            "world": "default",
        })))
        st = self._await_recorder("recording")
        if st is None or st.get("state") != "recording":
            row["termination"] = "recorder_failed"
            row["recorder_detail"] = (st or {}).get("detail", "no status")
            row["episode"] = None
            self.rec_pub.publish(String(data=json.dumps(
                {"cmd": "abort", "termination": "operator_abort"})))
            return row
        row["episode"] = st.get("episode")

        # Go. The `settled` event is what the resampler uses as grid origin
        # (resample_episodes.py:218). Marking it here rather than at recorder
        # start excludes the gap between the two, during which the car sits
        # still with recording already running — frames that would teach the
        # policy to hold position immediately after being given an instruction.
        self.nav = None
        self.rec_pub.publish(String(data=json.dumps(
            {"cmd": "event", "kind": "settled",
             "data": {"oracle_goal": variant["oracle_goal"]}})))
        if self.args.driver == "yolo":
            og = variant["oracle_goal"]
            self.speed_pub.publish(Int32(
                data=int(variant["intent_slots"]["target_speed_raw"])))
            if variant.get("cruise_s") is not None:
                self.lane_pub.publish(String(data=og.get("lane", "lane2")))
                # No navigator goal in cruise: release the planner's motion
                # gate ourselves (the navigator's "start" would otherwise
                # never fire and the car stays parked).
                self.motion_pub.publish(String(data="start"))
            else:
                self.zone_goal_pub.publish(String(data=json.dumps(
                    {"zone": og["goal"], "lane": og.get("lane", "lane2")})))
        else:
            self.goal_pub.publish(String(data=json.dumps(variant["oracle_goal"])))

        termination, detail = "timeout", ""
        cruise_s = variant.get("cruise_s")
        t0 = time.monotonic()
        last_xy, last_move = self.tf, t0
        off_ring_since = None
        while time.monotonic() - t0 < variant["timeout_s"]:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.ring_pts and self.tf and variant["intent_id"] != "park_bay":
                x, y = self.tf[0], self.tf[1]
                d_ring = min(math.hypot(px - x, py - y)
                             for px, py in self.ring_pts)
                if d_ring > 3.0:
                    if off_ring_since is None:
                        off_ring_since = time.monotonic()
                    elif time.monotonic() - off_ring_since > 2.0:
                        termination, detail = "off_ring", (
                            f"{d_ring:.1f} m from ring for >2 s")
                        break
                else:
                    off_ring_since = None
            if cruise_s is not None and time.monotonic() - t0 >= cruise_s:
                # Cruise has no arrival; the window elapsing IS success. The
                # cancel below then cuts the driver off mid-drive on purpose.
                termination, detail = "success", f"cruise window {cruise_s:.0f}s"
                break
            n = self.nav
            if n:
                # Both drivers speak these prefixes on /nav_status: the
                # route_oracle directly, the navigator with a longer tail.
                if n.startswith(("arrived:", "direct arrived:")):
                    termination, detail = "success", n
                    break
                if n.startswith("abort:"):
                    # The oracle only aborts on "N m off path".
                    termination, detail = "off_track", n
                    break
                if n.startswith("error:"):
                    termination, detail = "operator_abort", n
                    break
            if self.tf and last_xy:
                if math.hypot(self.tf[0] - last_xy[0],
                              self.tf[1] - last_xy[1]) > 0.05:
                    last_xy, last_move = self.tf, time.monotonic()
                elif time.monotonic() - last_move > self.args.stuck_s:
                    termination, detail = "stuck", (
                        f"no motion for {self.args.stuck_s:.0f}s")
                    break
            elif self.tf:
                last_xy = self.tf

        if cruise_s is not None:
            # Recorder first, driver second: the cancel halts the car
            # instantly, and frames recorded past it would put an unexplained
            # stop at the tail of a "keep driving" demonstration.
            self.rec_state = None
            self.rec_pub.publish(String(data=json.dumps(
                {"cmd": "stop", "termination": termination})))
            self._halt_driver()
            self._spin(0.5)
        else:
            self._halt_driver()
            self._spin(0.5)
            self.rec_state = None
            self.rec_pub.publish(String(data=json.dumps(
                {"cmd": "stop", "termination": termination})))
        st = self._await_recorder("idle", timeout=40.0, needs="n_frames") or {}
        row["termination"] = termination
        row["detail"] = detail
        row["duration_s"] = round(time.monotonic() - t0, 2)
        row["n_frames"] = st.get("n_frames")
        row["dropped"] = st.get("dropped")
        row["valid"] = bool(st.get("valid")) and termination == "success"
        return row

    # ---------------------------------------------------------------- run

    def run(self):
        a = self.args
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.idx is not None:
                break
        if self.idx is None:
            print("could not locate the ego vehicle — is the sim running?",
                  file=sys.stderr)
            return 1
        # The recorder announces its session on every status message; without it
        # the manifest would land somewhere other than the episodes it describes.
        st = self._await_recorder_any(timeout=10.0)
        if st is None:
            print("no /recorder/status — is episode_recorder_node running?",
                  file=sys.stderr)
            return 1
        session = st.get("session", "unknown")
        sess_dir = os.path.join(a.out_dir, session)
        os.makedirs(sess_dir, exist_ok=True)
        with open(os.path.join(sess_dir, "plan.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(a), "groups": self.plan}, f, indent=2,
                      ensure_ascii=False)
        man_path = os.path.join(sess_dir, "manifest.jsonl")
        self.manifest = open(man_path, "a", encoding="utf-8", buffering=1)

        total = sum(len(g["variants"]) for g in self.plan)
        print(f"session {session} — {len(self.plan)} groups, {total} episodes")
        print(f"plan + manifest in {sess_dir}\n")

        done, counts, complete_groups = 0, {}, 0
        t_start = time.monotonic()
        for g in self.plan:
            grp_ok = 0
            for v in g["variants"]:
                row = self.run_episode(g, v)
                done += 1
                counts[row["termination"]] = counts.get(row["termination"], 0) + 1
                if row.get("valid"):
                    grp_ok += 1
                self.manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                el = time.monotonic() - t_start
                eta = el / done * (total - done)
                print(f"[{done:4d}/{total}] {g['cf_group_id']}/{v['cf_variant_id']} "
                      f"{row['termination']:14s} "
                      f"{str(row.get('n_frames', '-')):>5s}f  "
                      f"eta {eta / 60:5.1f} min   {v['instruction'][:52]}")
            # A group missing a variant contributes no pair for that comparison,
            # so completeness is tracked separately from the episode count.
            if grp_ok == len(g["variants"]):
                complete_groups += 1
            self.manifest.write(json.dumps({
                "cf_group_id": g["cf_group_id"], "group_summary": True,
                "valid_variants": grp_ok, "n_variants": len(g["variants"]),
                "complete": grp_ok == len(g["variants"]),
            }, ensure_ascii=False) + "\n")

        print(f"\n{done} episodes in {(time.monotonic() - t_start) / 60:.1f} min")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:16s} {v:5d}  ({100.0 * v / max(done, 1):.1f}%)")
        print(f"  complete groups  {complete_groups}/{len(self.plan)}")
        if complete_groups < len(self.plan):
            print("\n  incomplete groups yield no counterfactual pair — rerun "
                  "those, or the ordinal axis loses that much statistical power.")
        self.manifest.close()
        return 0

    def _await_recorder_any(self, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.rec_state:
                return self.rec_state
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", type=int, default=30,
                   help="parking ordinal groups: 4 bays, one speed (4 eps each)")
    p.add_argument("--speed-groups", type=int, default=9,
                   help="parking speed groups: one bay, 3 speeds (3 eps each)")
    p.add_argument("--ring-groups", type=int, default=12,
                   help="ring lane groups: one goal, 2 lanes (2 eps each)")
    p.add_argument("--driver", choices=["oracle", "yolo"], default="oracle",
                   help="oracle = route_oracle pure pursuit on stored paths; "
                        "yolo = the tuned camera lane-follow stack, goal "
                        "stops supervised from the zone map")
    p.add_argument("--cruise-groups", type=int, default=0,
                   help="lane-pair groups with no endpoint in the sentence; "
                        "the episode ends by duration while still cruising")
    p.add_argument("--floor-groups", type=int, default=6,
                   help="same request twice: measures this session's noise floor")
    p.add_argument("--split", default="train", choices=["train", "heldout"],
                   help="'heldout' phrasings are for evaluation only")
    p.add_argument("--group-prefix", default="",
                   help="prepended to every cf_group_id. Required when a corpus "
                        "is collected across several sessions: group ids are "
                        "positional (p0000, s0000...) and would otherwise repeat, "
                        "silently merging two unrelated start poses into one "
                        "counterfactual group.")
    p.add_argument("--start-poses-file", default=None,
                   help="JSON list of {x, y, heading, lane} start poses for "
                        "cruise groups (DAgger corrections — no jitter added); "
                        "overrides --cruise-groups count and --start-windows")
    p.add_argument("--cruise-lane", default="both",
                   choices=["both", "lane1", "lane2"],
                   help="restrict cruise variants to one lane (corpus "
                        "rebalancing packs; breaks lane counterfactual pairs)")
    p.add_argument("--start-windows", default=None,
                   help="comma list of LO-HI ring-index windows restricting "
                        "cruise start poses, e.g. '120-180,290-350'")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--lat-jitter", type=float, default=0.6,
                   help="metres of perpendicular start offset")
    p.add_argument("--yaw-jitter", type=float, default=6.0,
                   help="degrees of start heading offset")
    p.add_argument("--timeout-slack", type=float, default=2.5,
                   help="episode timeout = route_len / speed * SLACK + 20 s")
    p.add_argument("--stuck-s", type=float, default=8.0)
    p.add_argument("--reset-retries", type=int, default=2)
    p.add_argument("--tf-topic", default="/world/default/dynamic_pose/info")
    p.add_argument("--out-dir", default=os.path.join(PKG_PARENT, "data_v2"))
    p.add_argument("--dry-run", action="store_true",
                   help="print the grid and exit; no sim needed")
    args = p.parse_args()

    rng = random.Random(args.seed)
    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        paths = json.load(f)
    bank = InstructionBank(BANK_FILE, args.split, rng)
    plan = build_plan(args, paths, bank, rng)

    total = sum(len(g["variants"]) for g in plan)
    uniq = {v["instruction"] for g in plan for v in g["variants"]}
    print(f"split={args.split}  seed={args.seed}")
    print(f"{len(plan)} groups, {total} episodes, {len(uniq)} distinct sentences")
    print(f"templates available: park_bay={bank.n_templates('park_bay')}, "
          f"ring_goal={bank.n_templates('ring_goal')}, "
          f"cruise={bank.n_templates('cruise')}")
    axes = collections.Counter(g["cf_axis"] for g in plan)
    print("axes: " + ", ".join(
        f"{k}={v} groups ({sum(len(g['variants']) for g in plan if g['cf_axis'] == k)} eps)"
        for k, v in sorted(axes.items())))
    # Slot balance, printed before anything is collected. An imbalanced corpus
    # lets the policy score above chance from the label prior alone, and there
    # is no fixing it after the fact without discarding episodes.
    goals = collections.Counter(v["intent_slots"].get("goal", "(cruise)")
                                for g in plan for v in g["variants"])
    levels = collections.Counter(v["intent_slots"]["speed_level"]
                                 for g in plan for v in g["variants"])
    bays = [goals[b] for b in ("bay1", "bay2", "bay3", "bay4")]
    print("goals:  " + ", ".join(f"{k}={v}" for k, v in sorted(goals.items())))
    print("speeds: " + ", ".join(f"{k}={v}" for k, v in levels.most_common()))
    print(f"bay balance {max(bays) / max(min(bays), 1):.2f}x, "
          f"speed balance {max(levels.values()) / max(min(levels.values()), 1):.2f}x")
    est = sum(v["timeout_s"] * 0.45 + 12 for g in plan for v in g["variants"])
    print(f"estimate: {est / 3600:.1f} h wall clock, "
          f"{sum(len(g['variants']) for g in plan) * 0.024:.1f} GB on disk")
    if args.dry_run:
        seen = set()
        for g in plan:
            if g["cf_axis"] in seen:
                continue
            seen.add(g["cf_axis"])
            sp = g["start"]
            print(f"\n{g['cf_group_id']}  axis={g['cf_axis']}  start "
                  f"({sp['x']:.2f}, {sp['y']:.2f}, "
                  f"{math.degrees(sp['heading']):.1f} deg heading, "
                  f"lat {sp['lateral_m']:+.2f} m)")
            for v in g["variants"]:
                print(f"   {v['cf_variant_id']}  {v['oracle_goal']['goal']:6s} "
                      f"@{v['oracle_goal']['target_speed']:.1f}  "
                      f"t/o {v['timeout_s']:5.0f}s  {v['instruction']}")
        est = sum(v["timeout_s"] * 0.5 + 6 for g in plan for v in g["variants"])
        print(f"\nrough upper bound on wall clock: {est / 60:.0f} min "
              "(assumes half the timeout per episode plus reset overhead)")
        return 0

    rclpy.init()
    ring_pts = [tuple(p) for p in paths["ring_center"][::2]]
    node = Collector(args, plan, ring_pts=ring_pts)
    try:
        return node.run()
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist())
        node.rec_pub.publish(String(data=json.dumps(
            {"cmd": "abort", "termination": "operator_abort"})))
        if node.manifest:
            node.manifest.close()
        print("\ninterrupted — manifest.jsonl holds everything finished so far")
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
