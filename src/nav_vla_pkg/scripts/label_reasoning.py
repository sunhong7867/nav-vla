#!/usr/bin/env python3
"""Offline reasoning-label generator for packed v3y episodes.

For every frame of a packed episode (``meta.json`` + ``resampled_10hz.jsonl``)
this derives a *fact skeleton* purely from recorded ground truth — pose,
speed, the 30-step action window, and the track/zone config — and composes a
deterministic English sentence describing why the vehicle is doing what it is
doing. No model, no ROS, no image is involved: every claim is checkable
against the sim state, which is the whole point (labels a VLM paraphrases
later must stay anchored to these facts).

Facts per frame:
  - curvature class of the ring at the ego (curve / gentle curve / straight)
  - arc-length distance ALONG THE INSTRUCTED LANE to the goal zone
    (Euclidean is wrong on this compact ring — see chat_gui_node._arc_ahead)
  - nearest zone within 8 m (context even without a goal)
  - measured speed, planned end-of-chunk speed, and the accel/decel/hold
    trend (thresholds shared with vla_narrator)
  - net planned yaw over the chunk -> turning left / right
  - instruction context: goal, lane, commanded speed tier

Adjacent frames share an event, so sentences are generated per *segment*
(a maximal run of frames whose discrete facts agree) and frames link to the
segment. Numbers in the sentence are frozen at segment start; the per-frame
facts keep the exact values.

Output: ``reasoning.jsonl`` sidecar in each episode dir (originals are never
touched). First line is an episode header with the segment table; following
lines are ``{"k", "segment_id", "facts", "skeleton"}`` per frame.

Usage::

    python3 src/nav_vla_pkg/scripts/label_reasoning.py \
        src/nav_vla_pkg/data_v3y/packed_v3y [more packed dirs...] [--limit N]
"""

import argparse
import json
import math
import os
import sys

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

from scripts.vla_narrator import (  # noqa: E402  (ROS-free import)
    NarratorCore, load_ring, CURV_TIGHT, CURV_STRAIGHT,
    SPEED_TREND_EPS, TURN_YAW_RAD, RATE_HZ, PATHS_FILE,
)

CHUNK = 30                  # action window = policy chunk (train_smolvla.sh)
GOAL_NEAR_M = 12.0          # start talking about the goal inside this arc
GOAL_AT_M = 3.0             # "reaching" band
ZONE_NEAR_M = 8.0           # non-goal zone context radius (narrator's value)
GOAL_BUCKET_M = 3.0         # goal-distance bucket that splits segments
RAW_PER_MPS = 51.0          # raw motor units per m/s (255 / 5.0 m/s)
OB_NEAR_M = 14.0            # a blocking car this close is worth announcing
OB_WATCH_M = 11.0           # standstill inside this arc = watching the car
OB_BESIDE_M = 10.0          # other-lane car mentioned inside this arc

# "slow" as a tier word would collide with the paraphrase verifier's
# deceleration check ("slow pace" reads as a trend claim); "leisurely" keeps
# the tier out of the trend vocabulary.
SPEED_WORD = {"slow": "leisurely", "normal": "moderate", "fast": "brisk"}
LANE_WORD = {"lane1": "inner", "lane2": "outer"}
# Human-facing zone names; crosswalk_stop/T3 are co-located (~2 cm).
ZONE_NAME = {"crosswalk_stop": "the T3 crosswalk", "Start": "the start line",
             "T1/M1": "T1"}


def zone_label(name):
    return ZONE_NAME.get(name, name)


class LaneGeo:
    """Cumulative arc length along one lane polyline of track_paths.json."""

    def __init__(self, pts):
        self.pts = [tuple(p) for p in pts]
        self.n = len(self.pts)
        cum, s = [0.0], 0.0
        for i in range(1, self.n):
            s += math.hypot(self.pts[i][0] - self.pts[i - 1][0],
                            self.pts[i][1] - self.pts[i - 1][1])
            cum.append(s)
        self.loop = s + math.hypot(self.pts[0][0] - self.pts[-1][0],
                                   self.pts[0][1] - self.pts[-1][1])
        self.cum = cum

    def nearest_idx(self, x, y):
        return min(range(self.n), key=lambda k: (self.pts[k][0] - x) ** 2
                   + (self.pts[k][1] - y) ** 2)

    def arc_ahead(self, ex, ey, hx, hy, target_idx):
        """Signed along-lane distance from ego to the lane point at
        ``target_idx`` (positive = ahead in the travel direction). Mirrors
        chat_gui_node._arc_ahead. NOTE: the target must be an index into
        THIS lane's polyline — the zones' ``s_m`` in track_paths.json is
        the ring-center parametrization (index * 0.35 m) and does not match
        a lane's own arc length (lane1 loop is 128.3 m vs ring 142.0 m)."""
        ei = self.nearest_idx(ex, ey)
        tx = self.pts[(ei + 1) % self.n][0] - self.pts[ei][0]
        ty = self.pts[(ei + 1) % self.n][1] - self.pts[ei][1]
        sign = 1.0 if (tx * hx + ty * hy) >= 0.0 else -1.0
        fwd = ((self.cum[target_idx % self.n] - self.cum[ei]) * sign) \
            % self.loop
        return fwd if fwd <= self.loop / 2.0 else fwd - self.loop


def load_track(path=PATHS_FILE):
    with open(path) as f:
        tp = json.load(f)
    lanes = {ln: LaneGeo(tp[ln]) for ln in ("lane1", "lane2")}
    # zone -> per-lane polyline index, on-ring zones only
    zone_idx = {}
    for name, z in tp["zones"].items():
        if not z.get("on_ring"):
            continue
        zone_idx[name] = {ln: z[ln]["index"] for ln in ("lane1", "lane2")
                          if ln in z}
    return lanes, zone_idx


# ------------------------------------------------------------------ facts

def frame_facts(rows, k, core, lanes, zone_idx, ep):
    r = rows[k]
    x, y = r["x"], r["y"]
    heading = r["heading"]
    hx, hy = math.cos(heading), math.sin(heading)
    lane = ep["lane"] or "lane1"
    geo = lanes[lane]

    # state[0] is a pose-delta derivative and spikes frame to frame
    # (observed 0.5 -> 4.7 -> 1.9 on adjacent rows); a 5-frame median is
    # the honest speed a narration should quote.
    v_win = sorted((rows[j]["state"] or [0.0])[0]
                   for j in range(max(0, k - 2), min(len(rows), k + 3)))
    v_now = v_win[len(v_win) // 2]
    # The resampler nulls `action` on tail rows past the last control sample.
    win = [a["action"] for a in rows[k:k + CHUNK] if a.get("action")]
    tail = win[-5:]
    v_end = (sum(abs(a[0]) for a in tail) / len(tail) * RATE_HZ
             if tail else v_now)
    dv = v_end - v_now
    trend = ("accel" if dv > SPEED_TREND_EPS
             else "decel" if dv < -SPEED_TREND_EPS else "hold")

    yaw_sum = sum(a[2] for a in win)
    turn = ("left" if yaw_sum > TURN_YAW_RAD
            else "right" if yaw_sum < -TURN_YAW_RAD else None)

    kcurv = core.curvature_at(x, y)
    curv = ("curve" if kcurv > CURV_TIGHT
            else "straight" if kcurv < CURV_STRAIGHT else "gentle")

    goal_d = None
    if ep["goal"] and ep["goal"] in zone_idx and lane in zone_idx[ep["goal"]]:
        goal_d = geo.arc_ahead(x, y, hx, hy, zone_idx[ep["goal"]][lane])

    near = core.nearest_zone(x, y)

    # Actual lane from geometry, not from the instruction: during a v9
    # avoidance swerve the ego is deliberately NOT on its instructed lane,
    # and that difference is the label. 0.5 m of hysteresis so lateral
    # jitter (starts are jittered ±0.6 m) cannot flip it.
    other = "lane2" if lane == "lane1" else "lane1"
    d_inst = min(math.hypot(px - x, py - y) for px, py in lanes[lane].pts)
    d_oth = min(math.hypot(px - x, py - y) for px, py in lanes[other].pts)
    lane_actual = other if d_oth + 0.5 < d_inst else lane

    ob_fact = None
    ob = ep.get("obstacle")
    if ob and ob.get("present"):
        ob_arc = lanes[ob["lane"]].arc_ahead(x, y, hx, hy, ob["index"])
        # gap_m is what an onlooker (and the camera) sees: center distance
        # minus the two half-bodies (prius 2.4 + hatchback 2.2)
        ob_fact = {"rel": "ours" if ob.get("in_our_lane") else "other",
                   "arc_m": round(ob_arc, 2),
                   "gap_m": round(max(abs(ob_arc) - 4.6, 0.3), 2)
                   * (1 if ob_arc >= 0 else -1)}

    facts = {
        "v_mps": round(v_now, 2), "v_plan_end_mps": round(v_end, 2),
        # raw motor units (0-255, m/s * 51) — the project's own speed
        # vocabulary (/speed_command tiers 70/110/150); sentences quote
        # these instead of m/s per user preference.
        # rounded to the nearest 10: exact 3-digit integers ("speed 113")
        # came out digit-truncated from the 500M head (r2: "speed 11/10"),
        # and the user's own convention is coarse steps (0/70/140/...)
        "v_raw": int(round(v_now * RAW_PER_MPS / 10.0) * 10),
        "v_plan_end_raw": int(round(v_end * RAW_PER_MPS / 10.0) * 10),
        "trend": trend, "turn": turn, "curv": curv,
        "goal": ep["goal"], "goal_arc_m": None if goal_d is None
        else round(goal_d, 2),
        "near_zone": near, "lane": lane,
        "lane_actual": lane_actual,
        "obstacle": ob_fact,
        "speed_level": ep["speed_level"],
        "target_v_mps": round(ep["target_v"], 2) if ep["target_v"] else None,
        "target_raw": ep["target_raw"],
    }
    return facts


def obstacle_phase(f):
    """None/ahead/watching/passing/returning/beside from one frame's facts.

    The demonstration is watch-then-avoid: the ego brakes behind the car,
    holds still "watching" it, and only then passes. "watching" is derived
    purely from the facts (close + standing still), not from events, so
    the labeler needs nothing beyond the resampled rows.
    """
    ob = f.get("obstacle")
    if not ob:
        return None
    arc = ob["arc_m"]
    if ob["rel"] == "ours":
        if f["lane_actual"] != f["lane"]:
            return "passing" if arc > -2.0 else "returning"
        if 0.0 < arc <= OB_WATCH_M and f["v_mps"] < 0.15:
            return "watching"
        if 0.0 < arc <= OB_NEAR_M:
            return "ahead"
        return None
    return "beside" if abs(arc) <= OB_BESIDE_M else None


def discrete_key(f):
    """Fields whose change starts a new segment."""
    gd = f["goal_arc_m"]
    if gd is None or gd < 0 or gd > GOAL_NEAR_M:
        gbucket = None
    else:
        gbucket = int(gd // GOAL_BUCKET_M)
    phase = obstacle_phase(f)
    near = f["near_zone"] if gbucket is None and phase is None else None
    return (f["trend"], f["turn"], f["curv"], gbucket, near, phase)


# --------------------------------------------------------------- sentence

def compose(f, ep):
    """Deterministic English skeleton from one frame's facts."""
    lane_w = LANE_WORD.get(f["lane"], f["lane"])
    actual_w = LANE_WORD.get(f.get("lane_actual", f["lane"]))
    other_w = "outer" if lane_w == "inner" else "inner"
    parts = []

    phase = obstacle_phase(f)
    gd = f["goal_arc_m"]
    # Epistemically honest phrasing: the driver cannot KNOW the car is
    # parked — it slows, watches, and only after the watch calls it parked.
    if phase == "ahead":
        # the speed clause after this states slowing/holding truthfully
        parts.append(f"A car {f['obstacle']['gap_m']:.1f} m ahead in our"
                     f" {lane_w} lane")
    elif phase == "watching":
        parts.append(f"Stopped {f['obstacle']['gap_m']:.1f} m behind the"
                     f" car ahead, watching whether it moves")
    elif phase == "passing":
        parts.append(f"The car ahead has not moved — treating it as parked"
                     f" and passing in the {actual_w} lane")
    elif phase == "returning":
        parts.append(f"Returning to the {lane_w} lane after passing the"
                     f" parked car")
    elif phase == "beside":
        parts.append(f"A car sits in the {other_w} lane;"
                     f" keeping to the {lane_w} lane")
    elif gd is not None and 0 <= gd <= GOAL_NEAR_M:
        gname = zone_label(f["goal"])
        if gd <= GOAL_AT_M:
            parts.append(f"Reaching {gname}, the goal, {gd:.1f} m ahead"
                         f" in the {lane_w} lane")
        else:
            parts.append(f"Approaching the goal, {gname}, {gd:.1f} m ahead"
                         f" along the {lane_w} lane")
    elif f["near_zone"] and f["goal"] and f["near_zone"] != f["goal"]:
        parts.append(f"Passing {zone_label(f['near_zone'])} in the"
                     f" {lane_w} lane en route to {zone_label(f['goal'])}")
    elif f["near_zone"]:
        parts.append(f"Cruising past {zone_label(f['near_zone'])}"
                     f" in the {lane_w} lane")
    else:
        seg = {"curve": "a tight curve", "gentle": "a gentle curve",
               "straight": "a straight section"}[f["curv"]]
        verb = "Following" if f["goal"] else "Cruising along"
        parts.append(f"{verb} {seg} in the {lane_w} lane")

    # Speed clause — the only number quoted is the COMMANDED TIER
    # (70/110/150 raw). Per-frame speed integers collapsed the 500M head
    # onto the modal token (r2 "110"-spam, r3 "speed 10"-spam, measured on
    # heldout), so the current speed is expressed qualitatively and the
    # trend word carries the per-frame grounding.
    tier = SPEED_WORD.get(f["speed_level"], f["speed_level"])
    tgt = f["target_raw"]
    tier_q = f"the {tier} tier of {tgt}" if tgt else f"the {tier} pace"
    if phase == "watching":
        # the main clause already says we are stopped; a tier clause here
        # ("holding near 110") would contradict the standstill
        pass
    elif f["trend"] == "decel":
        why = ("to stop at the goal" if gd is not None and gd <= GOAL_NEAR_M
               else "for the curve" if f["curv"] == "curve"
               else f"back toward {tier_q}")
        parts.append(f"slowing down {why}")
    elif f["trend"] == "accel":
        parts.append(f"speeding up toward {tier_q}")
    else:
        parts.append(f"holding near {tier_q}")

    if f["turn"]:
        parts.append(f"turning {f['turn']} through the corner"
                     if f["curv"] != "straight" else f"turning {f['turn']}")

    if len(parts) == 1:
        return parts[0] + "."
    return parts[0] + "; " + ", ".join(parts[1:]) + "."


# ------------------------------------------------------------------ main

def label_episode(ep_dir, core, lanes, zone_s):
    with open(os.path.join(ep_dir, "meta.json")) as f:
        meta = json.load(f)
    slots = meta.get("intent_slots") or {}
    ep = {"goal": slots.get("goal"), "lane": slots.get("lane"),
          "speed_level": slots.get("speed_level"),
          "target_v": slots.get("target_speed"),
          "target_raw": slots.get("target_speed_raw"),
          "obstacle": slots.get("obstacle")}
    rows = [json.loads(l) for l in
            open(os.path.join(ep_dir, "resampled_10hz.jsonl"))]
    if not rows:
        return None

    facts = [frame_facts(rows, k, core, lanes, zone_s, ep)
             for k in range(len(rows))]

    # Hysteresis: the trend/turn read of a single frame is noise, not an
    # event. Majority-vote each discrete field over a 5-frame window so a
    # one-frame flip cannot start a segment.
    def majority(vals):
        return max(set(vals), key=vals.count)
    for field in ("trend", "turn"):
        sm = [majority([facts[j][field]
                        for j in range(max(0, k - 2),
                                       min(len(facts), k + 3))])
              for k in range(len(facts))]
        for k, v in enumerate(sm):
            facts[k][field] = v

    segments, seg_of = [], []
    for k, f in enumerate(facts):
        key = discrete_key(f)
        if not segments or key != segments[-1]["key"]:
            segments.append({"key": key, "k0": k,
                             "skeleton": compose(f, ep)})
        segments[-1]["k1"] = k
        seg_of.append(len(segments) - 1)

    # Merge segments shorter than 0.5 s into their predecessor: sub-second
    # "events" are jitter, and per-frame facts keep the exact values anyway.
    MIN_LEN = 5
    merged, remap = [], []
    for i, s in enumerate(segments):
        if merged and (s["k1"] - s["k0"] + 1) < MIN_LEN:
            merged[-1]["k1"] = s["k1"]
        else:
            merged.append(s)
        remap.append(len(merged) - 1)
    segments = merged
    seg_of = [remap[i] for i in seg_of]

    out = os.path.join(ep_dir, "reasoning.jsonl")
    with open(out, "w") as f:
        f.write(json.dumps({
            "_meta": {"episode": os.path.basename(ep_dir),
                      "instruction": meta.get("instruction"),
                      "intent_id": meta.get("intent_id"),
                      "n_frames": len(rows), "n_segments": len(segments),
                      "segments": [{"id": i, "k0": s["k0"], "k1": s["k1"],
                                    "skeleton": s["skeleton"]}
                                   for i, s in enumerate(segments)]},
        }, ensure_ascii=False) + "\n")
        for k in range(len(rows)):
            f.write(json.dumps({
                "k": k, "segment_id": seg_of[k], "facts": facts[k],
                "skeleton": segments[seg_of[k]]["skeleton"],
            }, ensure_ascii=False) + "\n")
    return len(segments)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packed_dirs", nargs="+")
    ap.add_argument("--limit", type=int, default=0,
                    help="label only the first N episodes (smoke)")
    args = ap.parse_args()

    core = NarratorCore(load_ring(), {})     # curvature only; zones below
    lanes, zone_s = load_track()
    # narrator zones (Euclidean context radius) from zone_map via its loader
    from scripts.vla_narrator import load_zones
    core.zones = load_zones()

    total_ep = total_seg = 0
    for root in args.packed_dirs:
        eps = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
        if args.limit:
            eps = eps[:args.limit]
        for name in eps:
            n = label_episode(os.path.join(root, name), core, lanes, zone_s)
            if n is None:
                print(f"  SKIP {name} (empty)")
                continue
            total_ep += 1
            total_seg += n
        print(f"{root}: {len(eps)} episodes")
    print(f"labeled {total_ep} episodes, {total_seg} segments "
          f"({total_seg / max(total_ep, 1):.1f}/ep)")


if __name__ == "__main__":
    main()
