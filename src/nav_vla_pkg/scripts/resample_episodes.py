#!/usr/bin/env python3
"""Join the four recorder-v2 streams onto an exact 10 Hz grid, and reject loudly.

The recorder deliberately does no rate limiting — that is what produced the old
corpus's +15.4% systematic timing error. The exact grid is built here instead,
where the whole episode is available and a bad join can be *detected* rather than
silently averaged away.

This matters more than it looks. LeRobot fabricates ``timestamp = frame_index/fps``
and then validates against its own fabrication, so nothing downstream will ever
tell you the timing was wrong. The acceptance gate below is the only place that
can.

Per grid point k at ``t = t0 + k/fps``:

    frame    nearest in time          (rejected if the gap is too large)
    pose     linear interpolation     (yaw unwrapped first, then interpolated)
    control  zero-order hold          (last value at or before t)
    action   [dx, dy, dyaw] ego-frame SE(2) delta from pose[k] to pose[k+1],
             computed from the float64 interpolated GT pose — never from cmd_vel,
             which is a 9-value quantized symbol table
    state    [speed_mps, yaw_rate_radps, steer_angle_rad]

``t0`` is the first frame at or after the ``settled`` event, so the spawn
transient is not in the data.

Usage::

    python3 src/nav_vla_pkg/scripts/resample_episodes.py \
        --session src/nav_vla_pkg/data_v2/session_20260728_150000
    python3 src/nav_vla_pkg/scripts/resample_episodes.py --session <dir> --verbose
"""

import argparse
import json
import math
import os
import statistics
import sys

# ---- acceptance gate (plan section 5.3) -------------------------------------
# A raw source gap only matters insofar as it forces a grid point to take a
# stale frame — and that is measured directly, per point, by MAX_NEAREST_ERR_S
# with a 5% budget. A single 0.23 s hiccup in a 15 s episode touches ~1% of grid
# points and is already inside that budget, so rejecting the whole episode for it
# would discard good data over a criterion that is a proxy for the real one.
# This threshold is kept only to catch a genuinely broken stream.
MAX_SOURCE_GAP_S = 0.50
MAX_NEAREST_ERR_S = 0.05     # per-grid-point tolerance
MAX_FRAC_OVER_TOL = 0.05     # at most 5% of grid points may exceed it
# Gap tolerance for pose interpolation. Set from the geometry, not from taste:
# linear interpolation across a gap chords the true arc, and the worst-case
# sagitta is (v*dt)^2 / (8*R). At the planned extreme (1.8 m/s through a 2 m
# radius turn) a 0.25 s gap gives 8 mm — two orders below the 9.5 cm reset noise
# floor measured in G1, so it cannot dominate any label. Tightening this further
# only rejects episodes over an error the vehicle dynamics cannot produce.
MAX_POSE_GAP_S = 1.00        # hard cap: beyond this the stream is broken, not gappy
# What actually matters is the ERROR linear interpolation introduces, not the gap
# that produced it. Chording an arc of speed v and yaw rate w over dt leaves a
# sagitta of |w|*v*dt^2/8. Gating on that is self-tuning: a long gap while nearly
# straight is harmless, a short one mid-corner is not. 0.02 m sits far below the
# 9.5 cm reset noise floor measured in G1, so an accepted episode's labels can
# never be interpolation-dominated.
MAX_INTERP_ERR_M = 0.02

# Rotation between the model's reported yaw and its direction of travel.
#
# There IS one, and it is not optional: projecting motion onto the raw yaw put
# the entire forward displacement into the lateral channel — dx collapsed to ~0
# while dy grew to the full step length, i.e. a car apparently driving sideways
# at 1.3 m/s. The existing stack already carries a correction of this kind
# (policy_node.py's `direct_yaw_offset`, default -pi/2).
#
# But the value is NOT -pi/2 here, and it is not guessed. Measured against the
# `gz model -p` pose it is a very stable -79.5 deg (sd 0.15 deg over 271
# samples); measured against the bridged TF pose it is -67.5 deg (sd 2.09 deg).
# The two disagree by 12 deg, which is far too large to hand-wave, and until it
# is resolved a wrong constant here would silently rotate every action label in
# the corpus. So there is no default: --yaw-offset-deg must be passed, and
# calibrate_yaw_offset.py measures it from straight-line driving (no turn means
# no sideslip, so what remains is the frame offset alone).
YAW_OFFSET_RAD = None
MAX_MEDIAN_ERR_S = 0.017     # half a 30 Hz source period


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unwrap(seq):
    out, prev, offset = [], None, 0.0
    for a in seq:
        if prev is not None:
            d = a - prev
            if d > math.pi:
                offset -= 2 * math.pi
            elif d < -math.pi:
                offset += 2 * math.pi
        out.append(a + offset)
        prev = a
    return out


def interp_pose(poses_t, poses_x, poses_y, poses_yaw_u, t):
    """Linear interpolation. Returns (x, y, yaw_unwrapped, gap) or None."""
    if not poses_t or t < poses_t[0] or t > poses_t[-1]:
        return None
    lo, hi = 0, len(poses_t) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if poses_t[mid] <= t:
            lo = mid
        else:
            hi = mid
    t0, t1 = poses_t[lo], poses_t[hi]
    gap = t1 - t0
    if gap <= 0:
        return poses_x[lo], poses_y[lo], poses_yaw_u[lo], 0.0
    a = (t - t0) / gap
    return (poses_x[lo] + a * (poses_x[hi] - poses_x[lo]),
            poses_y[lo] + a * (poses_y[hi] - poses_y[lo]),
            poses_yaw_u[lo] + a * (poses_yaw_u[hi] - poses_yaw_u[lo]),
            gap)


def zoh(rows, t):
    """Zero-order hold: last row with row['t'] <= t."""
    best = None
    for r in rows:
        if r["t"] <= t:
            best = r
        else:
            break
    return best


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def resample_episode(ep_dir, fps=10, verbose=False, yaw_offset=0.0,
                     pose_source="any"):
    meta_path = os.path.join(ep_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {"episode": os.path.basename(ep_dir), "ok": False,
                "reason": "no meta.json"}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    name = os.path.basename(ep_dir)
    result = {"episode": name, "ok": False,
              "termination": meta.get("termination"),
              "instruction": meta.get("instruction"),
              "cf_group_id": meta.get("cf_group_id"),
              "start_pose_key": meta.get("start_pose_key")}

    if meta.get("valid") is False:
        result["reason"] = meta.get("invalid_reason", "recorder marked invalid")
        return result

    frames = load_jsonl(os.path.join(ep_dir, "frames.jsonl"))
    poses = load_jsonl(os.path.join(ep_dir, "poses.jsonl"))
    if pose_source != "any":
        poses = [p for p in poses if p.get("src") == pose_source]
    control = load_jsonl(os.path.join(ep_dir, "control.jsonl"))
    events = load_jsonl(os.path.join(ep_dir, "events.jsonl"))
    if len(frames) < 5 or len(poses) < 5:
        result["reason"] = f"too few samples (frames={len(frames)}, poses={len(poses)})"
        return result

    frames.sort(key=lambda r: r["t"])
    poses.sort(key=lambda r: r["t"])
    control.sort(key=lambda r: r["t"])

    # --- gate 0: are the timestamps real? --------------------------------
    # A stopped clock (use_sim_time true with nothing publishing /clock) writes
    # every row at t=0.0. Every later check would still "pass" on that data
    # while the join silently means nothing, so this is checked first.
    for stream_name, rows in (("poses", poses), ("control", control)):
        if not rows:
            continue
        ts = [r["t"] for r in rows]
        span = ts[-1] - ts[0]
        uniq = len(set(ts))
        if span <= 0.0 or uniq < 2:
            result["reason"] = (
                f"{stream_name}.jsonl has a stopped clock: {len(ts)} rows, "
                f"{uniq} distinct timestamp(s), span {span:.6f}s. "
                "Nothing is publishing /clock while use_sim_time is true."
            )
            return result
        if uniq < 0.5 * len(ts):
            result["reason"] = (
                f"{stream_name}.jsonl timestamps are {100 * (1 - uniq / len(ts)):.0f}% "
                f"duplicates ({uniq}/{len(ts)} distinct) — clock resolution is "
                "coarser than the sample rate"
            )
            return result

    # --- gate 1: no large hole in the source frame stream ---------------
    gaps = [b["t"] - a["t"] for a, b in zip(frames, frames[1:])]
    max_gap = max(gaps) if gaps else 0.0
    result["source_fps_median"] = 1.0 / statistics.median(gaps) if gaps else 0.0
    result["source_gap_max_s"] = max_gap
    if max_gap > MAX_SOURCE_GAP_S:
        result["reason"] = f"source frame gap {max_gap:.3f}s > {MAX_SOURCE_GAP_S}s"
        return result

    # --- grid origin: first frame at or after `settled` AND inside every stream
    #
    # Anchoring only to the frame stream throws the episode away when the first
    # image beats the first pose row, which it routinely does by a few
    # milliseconds — the image callback and the pose thread start independently.
    # Measured on a healthy session: frames began 10 ms before poses and 10 of 11
    # episodes were rejected as "grid point 0 outside the pose stream". Ten
    # milliseconds is 0.7 cm of travel; there is nothing wrong with the data.
    settled_t = next((e["t"] for e in events if e.get("kind") == "settled"), None)
    floor_t = poses[0]["t"]
    if control:
        # Control uses zero-order hold, so the grid must start at or after the
        # first command; before it there is no value to hold.
        floor_t = max(floor_t, control[0]["t"])
    if settled_t is not None:
        floor_t = max(floor_t, settled_t)
    t0 = next((f["t"] for f in frames if f["t"] >= floor_t), None)
    if t0 is None:
        result["reason"] = ("no frame after the later of settle, first pose and "
                            "first control")
        return result
    t_end = min(frames[-1]["t"], poses[-1]["t"])
    n = int(math.floor((t_end - t0) * fps)) + 1
    if n < 5:
        result["reason"] = f"only {n} grid points after settle"
        return result

    poses_t = [p["t"] for p in poses]
    poses_x = [p["x"] for p in poses]
    poses_y = [p["y"] for p in poses]
    poses_yaw_u = unwrap([p["yaw"] for p in poses])

    fi = 0
    samples, nearest_errs, pose_gaps = [], [], []
    for k in range(n):
        t = t0 + k / fps
        while fi + 1 < len(frames) and abs(frames[fi + 1]["t"] - t) <= abs(frames[fi]["t"] - t):
            fi += 1
        err = abs(frames[fi]["t"] - t)
        nearest_errs.append(err)

        got = interp_pose(poses_t, poses_x, poses_y, poses_yaw_u, t)
        if got is None:
            result["reason"] = f"grid point {k} outside the pose stream"
            return result
        x, y, yaw_u, gap = got
        pose_gaps.append(gap)

        c = zoh(control, t) or {}
        samples.append({
            "k": k, "t": t,
            "frame": frames[fi]["file"], "frame_t": frames[fi]["t"], "frame_err": err,
            "x": x, "y": y, "yaw": wrap_pi(yaw_u), "yaw_u": yaw_u,
            "heading": wrap_pi(yaw_u + yaw_offset),
            "v_cmd": c.get("v_cmd"), "yaw_rate_cmd": c.get("yaw_rate_cmd"),
            "oracle_curvature": c.get("oracle_curvature"),
            "oracle_steer_rad": c.get("oracle_steer_rad"),
            "steer_float": c.get("steer_float"),
            "source": c.get("source"), "override": c.get("override"),
        })

    # --- gates 2-4 -------------------------------------------------------
    # Interpolation error per grid point, from the locally measured motion.
    interp_errs = []
    for k in range(1, len(samples)):
        a, b = samples[k - 1], samples[k]
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        v = math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt
        w = abs(wrap_pi(b["heading"] - a["heading"])) / dt
        interp_errs.append(w * v * (pose_gaps[k] ** 2) / 8.0)
    max_interp_err = max(interp_errs) if interp_errs else 0.0
    result["interp_err_max_m"] = max_interp_err

    over = sum(1 for e in nearest_errs if e > MAX_NEAREST_ERR_S)
    frac_over = over / len(nearest_errs)
    median_err = statistics.median(nearest_errs)
    max_pose_gap = max(pose_gaps)
    pose_gaps_sorted = sorted(pose_gaps)
    result.update({
        "n_grid": n,
        "pose_gap_p95_s": pose_gaps_sorted[int(len(pose_gaps_sorted) * 0.95)],
        "nearest_err_median_s": median_err,
        "nearest_err_max_s": max(nearest_errs),
        "frac_over_tol": frac_over,
        "pose_gap_max_s": max_pose_gap,
    })
    if frac_over > MAX_FRAC_OVER_TOL:
        result["reason"] = (f"{frac_over * 100:.1f}% of grid points off by more than "
                            f"{MAX_NEAREST_ERR_S}s (limit {MAX_FRAC_OVER_TOL * 100:.0f}%)")
        return result
    if max_pose_gap > MAX_POSE_GAP_S:
        result["reason"] = (f"pose stream broken: {max_pose_gap:.3f}s gap "
                            f"> {MAX_POSE_GAP_S}s")
        return result
    if max_interp_err > MAX_INTERP_ERR_M:
        result["reason"] = (f"pose interpolation error {max_interp_err * 100:.1f} cm "
                            f"> {MAX_INTERP_ERR_M * 100:.0f} cm "
                            f"(worst gap {max_pose_gap:.3f}s while turning)")
        return result
    if median_err > MAX_MEDIAN_ERR_S:
        result["reason"] = f"median nearest error {median_err:.4f}s > {MAX_MEDIAN_ERR_S}s"
        return result

    # --- actions and state ----------------------------------------------
    # SE(2) delta in the ego frame at k, from float64 interpolated GT pose.
    for k in range(n - 1):
        a, b = samples[k], samples[k + 1]
        dxw, dyw = b["x"] - a["x"], b["y"] - a["y"]
        h = a["heading"]
        c, s = math.cos(h), math.sin(h)
        dx = dxw * c + dyw * s
        dy = -dxw * s + dyw * c
        dyaw = wrap_pi(b["heading"] - a["heading"])
        a["action"] = [dx, dy, dyaw]
        dist = math.hypot(dx, dy)
        a["state"] = [
            dist * fps,                                   # speed m/s
            dyaw * fps,                                   # yaw rate rad/s
            a["oracle_steer_rad"] if a["oracle_steer_rad"] is not None else 0.0,
        ]
    samples[-1]["action"] = None   # no successor; the loader pads this
    samples[-1]["state"] = samples[-2]["state"] if n > 1 else [0.0, 0.0, 0.0]

    out = os.path.join(ep_dir, f"resampled_{fps}hz.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for smp in samples:
            f.write(json.dumps(smp) + "\n")

    dyaws = [abs(s["action"][2]) for s in samples[:-1]]
    speeds = [s["state"][0] for s in samples[:-1]]
    result.update({
        "ok": True, "out": os.path.basename(out),
        "speed_mean_mps": statistics.fmean(speeds) if speeds else 0.0,
        "speed_max_mps": max(speeds) if speeds else 0.0,
        "dyaw_max_rad": max(dyaws) if dyaws else 0.0,
    })
    if verbose:
        print(f"  {name}: {n} pts, median_err={median_err * 1000:.1f}ms, "
              f"speed={result['speed_mean_mps']:.2f} m/s")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="session directory holding ep_XXXX/")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--pose-source", default="cli", choices=["cli", "tf", "any"],
                   help="which pose stream to build labels from. Default cli "
                        "(model origin) — the tf stream disagrees during turns "
                        "and the discrepancy is unresolved.")
    p.add_argument("--yaw-offset-deg", type=float, required=True,
                   help="measured rotation from reported yaw to direction of "
                        "travel; run calibrate_yaw_offset.py to obtain it")
    args = p.parse_args()

    session = os.path.abspath(os.path.expanduser(args.session))
    eps = sorted(d for d in os.listdir(session) if d.startswith("ep_"))
    if not eps:
        print(f"no ep_* directories in {session}", file=sys.stderr)
        return 1

    print(f"resampling {len(eps)} episodes from {session} onto a {args.fps} Hz grid\n")
    yaw_offset = math.radians(args.yaw_offset_deg)
    print(f"yaw offset: {args.yaw_offset_deg:+.2f} deg, pose source: {args.pose_source}\n")
    results = [resample_episode(os.path.join(session, e), args.fps, args.verbose,
                                yaw_offset, args.pose_source)
               for e in eps]

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]

    print(f"\n{'=' * 66}")
    print(f"accepted : {len(ok)} / {len(results)}")
    if ok:
        med = statistics.median([r["nearest_err_median_s"] for r in ok])
        src = statistics.median([r["source_fps_median"] for r in ok])
        print(f"source rate (median)   : {src:.2f} Hz")
        print(f"nearest-error (median) : {med * 1000:.1f} ms   "
              f"(gate {MAX_MEDIAN_ERR_S * 1000:.0f} ms)")
        print(f"grid points total      : {sum(r['n_grid'] for r in ok)}")
    if bad:
        print(f"\nrejected : {len(bad)}")
        reasons = {}
        for r in bad:
            reasons.setdefault(r.get("reason", "?"), []).append(r["episode"])
        for reason, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            shown = ", ".join(names[:5]) + (" ..." if len(names) > 5 else "")
            print(f"  {len(names):3d}x  {reason}\n         {shown}")
    print("=" * 66)

    report = os.path.join(session, "resample_report.json")
    with open(report, "w", encoding="utf-8") as f:
        json.dump({"session": session, "fps": args.fps,
                   "yaw_offset_deg": args.yaw_offset_deg,
                   "accepted": len(ok), "rejected": len(bad),
                   "gates": {"max_source_gap_s": MAX_SOURCE_GAP_S,
                             "max_nearest_err_s": MAX_NEAREST_ERR_S,
                             "max_frac_over_tol": MAX_FRAC_OVER_TOL,
                             "max_pose_gap_s": MAX_POSE_GAP_S,
                             "max_median_err_s": MAX_MEDIAN_ERR_S},
                   "episodes": results}, f, indent=2, ensure_ascii=False)
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
