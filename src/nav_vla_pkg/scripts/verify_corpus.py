#!/usr/bin/env python3
"""Acceptance gate for a collection session: did it produce what it claims to?

A collection run that finishes without errors has proved only that it finished.
The properties the corpus is *for* — that trajectories inside a group diverge on
instruction alone, that streams are joinable, that the language axis is balanced
— are none of them checked by the driver, and every one of them has failed
silently at least once in this project's history:

* the aisle sharing broke to a single point and nothing reported it
  (docs/ver/20260728_1906 section 4.1)
* every pose timestamp was 0.0 while the frame counts looked healthy
  (docs/ver/20260728_1713)
* two recorders wrote one session and both episodes had plausible meta.json
  (docs/ver, first collection smoke test)

So this reads the session back off disk and checks it, and exits non-zero if a
gate fails. Run it after every collection, before resampling.

Gates
-----
G-I   integrity   every episode has meta.json, streams are non-empty, timestamps
                  strictly increase, no duplicate stamps, frame files exist
G-R   rate        frame rate within tolerance of the source rate, no drops
G-C   counterfactual  inside each complete group, the variant pairs diverge well
                  above the G1 reset floor and share a real prefix first
G-L   language    distinct sentences, template coverage, slot balance

Usage::

    python3 src/nav_vla_pkg/scripts/verify_corpus.py <session_dir>
    python3 src/nav_vla_pkg/scripts/verify_corpus.py <session_dir> --json report.json
"""

import argparse
import collections
import json
import math
import os
import statistics
import sys

G1_FLOOR_M = 0.095      # measured reset noise floor, docs/ver/20260728_1453
MIN_RATIO = 5.0         # a pair must beat the floor by this much to count
SHARED_TOL_M = 0.30     # two runs count as "together" within this distance
TARGET_HZ = 30.0
HZ_TOL = 0.15           # +-15% on the frame rate
# The check exists to catch a STOPPED clock (~100% duplicates). The 0.20
# bar was set when /clock was raw (943 Hz-ish, dup ~1%); since the 100 Hz
# clock_throttle_node (2026-08-28), a 58 Hz tf stream lands on a 100-tick
# grid and legitimately shares stamps ~40-55% of the time (measured on the
# v9 pilot, 2026-09-04 — every episode "failed" at 22-58%). 0.70 still
# catches the stopped-clock case with a wide margin.
DUP_TOL = 0.70


def load_jsonl(path, limit=None):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(out) >= limit:
                break
    return out


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


def thin(track, pitch=0.2):
    if not track:
        return []
    out = [track[0]]
    for p in track[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > pitch:
            out.append(p)
    return out


def resample_arclen(P, step=0.2):
    """Points at exact multiples of `step` metres along the path.

    Pairing two tracks by list index compares "the 40th recorded point of A" with
    "the 40th of B", which is only the same place if both ran at identical speed.
    Two runs down one line, one 0.25 m ahead of the other, then read as 0.25 m
    apart. Resampling by arc length asks the only question that matters: after
    driving the same distance, are they in the same place?
    """
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


def shared_prefix_m(A, B, step=0.2):
    """Distance both runs cover before separating by more than SHARED_TOL_M.

    The ordinal claim lives here, not in the divergence figure. Two paths can
    differ by 9 m because they left from different places; what makes the
    difference attributable to the sentence is that they were on top of each
    other first, seeing the same thing, and then split.
    """
    RA, RB = resample_arclen(A, step), resample_arclen(B, step)
    run = 0.0
    for a, b in zip(RA, RB):
        if math.hypot(a[0] - b[0], a[1] - b[1]) > SHARED_TOL_M:
            return run
        run += step
    return run


def pair_axis(sa, sb):
    """Which slot actually differs between two episodes.

    Derived from the recorded slots rather than trusted from the group's
    declared `cf_axis`, so that a plan bug which varies two things at once is
    caught by the data instead of being described away by the label.
    """
    diff = [k for k in ("goal", "lane", "speed_level")
            if sa.get(k) != sb.get(k)]
    # v9: the obstacle axis varies the WORLD, not the sentence slots — the
    # whole point is that goal/lane/speed stay identical while the parked
    # car moves. Compare the obstacle spec (minus coordinates) instead.
    oa = sa.get("obstacle") or {}
    ob = sb.get("obstacle") or {}
    if (oa.get("present"), oa.get("lane")) != (ob.get("present"),
                                               ob.get("lane")):
        diff.append("obstacle")
    if not diff:
        return "none"
    if len(diff) > 1:
        return "mixed"
    return {"goal": "ordinal", "lane": "lane", "speed_level": "speed",
            "obstacle": "obstacle"}[diff[0]]


def slot_label(s):
    return f"{s.get('goal', '?')}/{s.get('lane', '-')}/{s.get('speed_level', '?')}"


# --------------------------------------------------------------- integrity

def check_episode(ep_dir):
    """Read one episode and return its facts plus any integrity failures."""
    fail, note = [], []
    meta_p = os.path.join(ep_dir, "meta.json")
    if not os.path.exists(meta_p):
        return None, [f"{os.path.basename(ep_dir)}: no meta.json"], []
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    name = os.path.basename(ep_dir)

    frames = load_jsonl(os.path.join(ep_dir, "frames.jsonl"))
    poses = load_jsonl(os.path.join(ep_dir, "poses.jsonl"))
    control = load_jsonl(os.path.join(ep_dir, "control.jsonl"))

    # What has to hold is what the offline join needs, not textbook tidiness.
    # The resampler sorts by `t` and matches nearest within 0.05 s, so:
    #   * arrival order is irrelevant  — three writer threads deliberately
    #     interleave, and image stamps come from the message header, not the
    #     callback, so seq order and stamp order legitimately disagree
    #   * a handful of ties is irrelevant — the sim clock has 1 ms resolution
    #     and two messages can land inside one tick
    # What is fatal is a clock that is not running, which shows up as every
    # stamp identical. An earlier corpus was lost to exactly that, so the test
    # is on the *fraction* of ties, not their presence.
    for label, rows in (("frames", frames), ("poses", poses),
                        ("control", control)):
        if not rows:
            fail.append(f"{name}: {label}.jsonl is empty")
            continue
        streams = ({"poses[tf]": [r["t"] for r in rows if r.get("src") == "tf"],
                    "poses[cli]": [r["t"] for r in rows if r.get("src") == "cli"]}
                   if label == "poses" else
                   {label: [r.get("t", 0.0) for r in rows]})
        for sname, ts in streams.items():
            if not ts:
                fail.append(f"{name}: {sname} is empty")
                continue
            if max(ts) <= 0.0:
                fail.append(f"{name}: every {sname} timestamp is 0 — /clock stopped")
                continue
            dup_frac = 1.0 - len(set(ts)) / len(ts)
            if dup_frac > DUP_TOL:
                fail.append(f"{name}: {sname} is {dup_frac * 100:.0f}% duplicate "
                            f"timestamps (>{DUP_TOL * 100:.0f}%) — clock may be "
                            "coarser than the source rate")
            span = max(ts) - min(ts)
            if span <= 0:
                fail.append(f"{name}: {sname} spans 0 s over {len(ts)} rows")

    # Frame seq must be a complete run: a hole means a frame was written whose
    # image is gone, or two frames share an index and one overwrote the other.
    seqs = [r.get("seq") for r in frames if r.get("seq") is not None]
    if seqs and sorted(seqs) != list(range(len(seqs))):
        fail.append(f"{name}: frame seq is not 0..{len(seqs) - 1} "
                    f"({len(set(seqs))} distinct of {len(seqs)})")

    # The tf bridge is the stream the SE(2) action labels are built from. It can
    # die for a whole episode while the CLI fallback keeps `n_poses` looking
    # healthy — that is exactly what happened on the first full smoke collection,
    # where five of ten episodes had zero tf rows and all ten reported valid.
    n_tf_rows = sum(1 for r in poses if r.get("src") == "tf")
    dur = (meta.get("t_end", 0.0) or 0.0) - (meta.get("t_start", 0.0) or 0.0)
    if dur > 0 and n_tf_rows / dur < 5.0:
        fail.append(f"{name}: tf pose stream {n_tf_rows / dur:.1f} Hz "
                    f"({n_tf_rows} rows in {dur:.1f}s) — labels come from this")

    # Direction matters. The whole downstream pipeline is row-driven: the
    # resampler picks frames by the `file` field, so a JPEG nobody references is
    # dead weight, not corruption. A row pointing at a file that is not there is
    # the fatal case.
    #
    # One orphan appears occasionally because the writer thread saves the image
    # and then appends the row; a stop landing between those two leaves the file
    # without its row. Measured: 1 orphan across 53 episodes.
    fdir = os.path.join(ep_dir, "frames")
    on_disk = set(os.listdir(fdir)) if os.path.isdir(fdir) else set()
    referenced = {os.path.basename(r.get("file", "")) for r in frames}
    missing = referenced - on_disk
    orphans = on_disk - referenced
    if missing:
        fail.append(f"{name}: {len(missing)} frame row(s) reference files that "
                    f"do not exist (e.g. {sorted(missing)[0]})")
    if orphans:
        note.append(f"{name}: {len(orphans)} orphan frame file(s) with no row "
                    "(harmless — the pipeline reads rows)")
    if meta.get("dropped_frames"):
        fail.append(f"{name}: {meta['dropped_frames']} dropped frames")
    if meta.get("valid") is False:
        # The recorder flagging its own episode is the pipeline WORKING, not
        # a session-integrity failure: the episode is already excluded from
        # `good` (and from packing) by its valid=False. Failing the whole
        # session for it blocked the v9 pilot (12/97 tf-gap episodes,
        # 2026-09-04) even though 85 clean episodes were sitting there.
        note.append(f"{name}: dropped — recorder marked invalid "
                    f"({meta.get('invalid_reason', 'no reason given')})")

    hz = len(frames) / dur if dur > 0 else 0.0

    # tf is the pose stream the action labels are built from.
    tf_track = [(r["x"], r["y"]) for r in poses if r.get("src") == "tf"]
    return {
        "name": name, "meta": meta, "dur_s": dur, "hz": hz,
        "n_frames": len(frames), "n_poses": len(poses), "n_control": len(control),
        "n_tf": len(tf_track), "track": thin(tf_track),
    }, fail, note


# ------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir", nargs="+",
                   help="one or more session directories. A corpus split across "
                        "sessions is one corpus: pass them all so the pair count "
                        "and the balance check see the whole thing.")
    p.add_argument("--json", default="")
    p.add_argument("--min-ratio", type=float, default=MIN_RATIO)
    args = p.parse_args()

    sessions = [os.path.abspath(s) for s in args.session_dir]
    eps = []
    for sd in sessions:
        found = sorted(d for d in os.listdir(sd) if d.startswith("ep_"))
        if not found:
            print(f"no episodes in {sd}", file=sys.stderr)
            return 1
        eps.extend(os.path.join(sd, d) for d in found)
        print(f"session {os.path.basename(sd)} — {len(found)} episode dirs")
    print()

    # ---------------------------------------------------------- G-I / G-R
    loaded, fails, notes = [], [], []
    for d in eps:
        info, f, nt = check_episode(d)
        fails.extend(f)
        notes.extend(nt)
        if info:
            loaded.append(info)

    # Group ids are positional within a session (p0000, s0000...). Two sessions
    # collected without distinct --group-prefix would merge unrelated start poses
    # into one "counterfactual group", and the start-pose check below is the only
    # thing that would notice.
    seen = collections.defaultdict(set)
    for i in loaded:
        seen[i["meta"].get("cf_group_id")].add(i["meta"].get("start_pose_key"))
    clashes = [g for g, k in seen.items() if g and len(k) > 1]
    if clashes and len(sessions) > 1:
        fails.append(f"{len(clashes)} cf_group_id(s) span different start poses "
                     "— sessions were collected without distinct --group-prefix")

    # The recorder's image topic is a parameter, and the wide VLA lens lives on a
    # different topic from the narrow one the YOLO pipeline uses. Forgetting the
    # override produces a complete, healthy-looking corpus shot through the wrong
    # camera — nothing else in the pipeline would notice.
    topics = collections.Counter(i["meta"].get("image_topic") for i in loaded)
    if len(topics) > 1:
        fails.append("session mixes image topics: " + ", ".join(
            f"{k}={v}" for k, v in topics.most_common()))
    elif topics:
        print(f"camera: {next(iter(topics))}\n")

    terms = collections.Counter(i["meta"].get("termination") for i in loaded)
    good = [i for i in loaded if i["meta"].get("valid")
            and i["meta"].get("termination") == "success"]

    print("G-I integrity")
    if fails:
        for m in fails[:20]:
            print(f"  FAIL  {m}")
        if len(fails) > 20:
            print(f"  ... {len(fails) - 20} more")
    else:
        print(f"  PASS  {len(loaded)} episodes, streams non-empty, "
              "timestamps strictly increasing")
    for m in notes[:5]:
        print(f"  note  {m}")
    if len(notes) > 5:
        print(f"  note  ... {len(notes) - 5} more")

    print("\nG-R rate")
    hzs = [i["hz"] for i in loaded if i["hz"] > 0]
    rate_ok = True
    if hzs:
        med = statistics.median(hzs)
        lo, hi = min(hzs), max(hzs)
        rate_ok = abs(med - TARGET_HZ) / TARGET_HZ <= HZ_TOL
        print(f"  frames  median {med:.2f} Hz  (min {lo:.2f}, max {hi:.2f}) "
              f"target {TARGET_HZ:.0f} +-{HZ_TOL * 100:.0f}%  "
              f"{'PASS' if rate_ok else 'FAIL'}")
        print(f"  poses   {sum(i['n_poses'] for i in loaded)} rows "
              f"({sum(i['n_tf'] for i in loaded)} from the tf bridge)")
        print(f"  control {sum(i['n_control'] for i in loaded)} rows")
    dropped = sum(i["meta"].get("dropped_frames", 0) or 0 for i in loaded)
    print(f"  dropped {dropped}  {'PASS' if dropped == 0 else 'FAIL'}")

    print("\n  terminations: " + ", ".join(
        f"{k}={v}" for k, v in terms.most_common()))

    # ------------------------------------------------------------- G-C
    print("\nG-C counterfactual")
    groups = collections.defaultdict(list)
    for i in good:
        gid = i["meta"].get("cf_group_id")
        if gid:
            groups[gid].append(i)

    pairs, cf_rows = [], []
    for gid, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        # Every start in a group is the same pose by construction; if it is not,
        # the pair measures the start difference and not the sentence.
        keys = {i["meta"].get("start_pose_key") for i in members}
        if len(keys) > 1:
            print(f"  FAIL  {gid}: variants started from {len(keys)} different "
                  "poses — not a counterfactual group")
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                A, B = members[a]["track"], members[b]["track"]
                if len(A) < 5 or len(B) < 5:
                    continue
                la, lb = arclen(A), arclen(B)
                common = min(la, lb)
                # Shape over a common arc length, schedule over the full runs.
                # Comparing shape over the raw tracks makes a slow run a strict
                # prefix of a fast one and Frechet then returns the leftover
                # length, which once reported 12.15 m for two runs down an
                # identical line (docs/ver/20260728_1821 section 4.1).
                ds = frechet(truncate_to(A, common), truncate_to(B, common))
                sa = members[a]["meta"].get("intent_slots", {})
                sb = members[b]["meta"].get("intent_slots", {})
                # Schedule for goal-terminated episodes is *duration*, not arc
                # length. Both runs of a speed pair drive the same route to the
                # same bay, so their arc lengths agree to within a few cm and
                # the arc-length metric reports ~0 for a change that halved the
                # journey. The 12.68 m figure in docs/ver/20260728_1821 came
                # from a fixed 14 s window, where the faster run simply got
                # further — that framing does not exist here.
                da, db = members[a]["dur_s"], members[b]["dur_s"]
                row = {"group": gid, "axis": pair_axis(sa, sb),
                       "declared_axis": members[a]["meta"].get("cf_axis"),
                       "a": slot_label(sa), "b": slot_label(sb),
                       "D_shape_m": ds, "D_sched_m": abs(la - lb),
                       "D_time_s": abs(da - db),
                       "time_ratio": max(da, db) / max(min(da, db), 1e-6),
                       "shared_m": shared_prefix_m(A, B),
                       "ratio": ds / G1_FLOOR_M}
                cf_rows.append(row)
                pairs.append(row)

    # Prefer a floor measured in this session over the G1 constant: G1 was
    # free-running physics with no goal, on another day. If floor groups were
    # collected, they are the honest denominator.
    floor_rows = [r for r in cf_rows if r["axis"] == "none"]
    floor = G1_FLOOR_M
    if floor_rows:
        floor = max(r["D_shape_m"] for r in floor_rows)
        print(f"  floor: {len(floor_rows)} same-request pair(s), worst "
              f"{floor * 100:.1f} cm (G1 constant was {G1_FLOOR_M * 100:.1f} cm)")
        for r in cf_rows:
            r["ratio"] = r["D_shape_m"] / max(floor, 1e-6)
    else:
        print(f"  floor: no same-request pairs collected — falling back to the "
              f"G1 constant {G1_FLOOR_M * 100:.1f} cm. Use --floor-groups.")

    cf_ok = bool(pairs)
    if pairs:
        # Report per axis. Ordinal and speed pairs are different claims measured
        # on different quantities: an ordinal pair must move D_shape, a speed pair
        # must move D_sched while leaving D_shape at the floor. Averaging them
        # would let a strong ordinal result hide a dead speed axis.
        by_axis = collections.defaultdict(list)
        for r in pairs:
            by_axis[r["axis"]].append(r)
        # A "floor" group declares that nothing varies, which shows up in the
        # slots as "none".
        # A "zone" group (direct-to-zone, v8) varies the goal slot, which the
        # slot classifier reports under its historical name "ordinal".
        expected = {"floor": "none", "zone": "ordinal"}
        mismatch = [r for r in pairs if r["declared_axis"] and r["axis"] !=
                    expected.get(r["declared_axis"], r["declared_axis"])]
        print(f"  {len(pairs)} pairs from {len(groups)} groups")
        print(f"  {'axis':9s} {'n':>4s} {'D_shape med':>12s} {'dur ratio':>10s} "
              f"{'shared med':>11s} {'min ratio':>10s}")
        for ax, rs in sorted(by_axis.items()):
            print(f"  {ax:9s} {len(rs):4d} "
                  f"{statistics.median(r['D_shape_m'] for r in rs):11.3f}m "
                  f"{statistics.median(r['time_ratio'] for r in rs):9.2f}x "
                  f"{statistics.median(r['shared_m'] for r in rs):10.1f}m "
                  f"{min(r['ratio'] for r in rs):9.1f}x")

        # Only the shape-moving axes are gated on D_shape. A speed pair is
        # *supposed* to leave shape at the floor — that orthogonality is the
        # claim, so failing it here would penalise the corpus for being correct.
        shape_axes = {"ordinal", "lane", "goal", "mixed"}
        gated = [r for r in pairs if r["axis"] in shape_axes]
        weak = [r for r in gated if r["ratio"] < args.min_ratio]
        if weak:
            cf_ok = False
            print(f"\n  FAIL  {len(weak)}/{len(gated)} shape-axis pair(s) below "
                  f"{args.min_ratio:.0f}x the {floor * 100:.1f} cm floor:")
            for r in weak[:6]:
                print(f"        {r['group']} {r['a']} vs {r['b']}: "
                      f"{r['D_shape_m']:.3f} m ({r['ratio']:.1f}x)")
        elif gated:
            print(f"\n  PASS  all {len(gated)} shape-axis pairs above "
                  f"{args.min_ratio:.0f}x the floor")

        speed_pairs = by_axis.get("speed", [])
        if speed_pairs:
            worst = max(r["D_shape_m"] for r in speed_pairs)
            tmin = min(r["time_ratio"] for r in speed_pairs)
            # Two claims, opposite directions. The speed axis must move the
            # schedule; it must NOT move the shape, because that orthogonality
            # is the claim. Gating it on D_shape alongside the other axes would
            # fail the corpus for being correct.
            # 0.3 m is the plan's own orthogonality threshold (sim_vla_plan.md
            # section 5.6), not a number chosen to fit the result. It is close:
            # the smoke session measured 0.275 m against a 0.174 m noise floor,
            # so roughly half the remaining margin is physics scatter.
            orth = worst <= 0.30
            moved = tmin >= 1.25
            print(f"  {'PASS' if orth else 'FAIL'}  speed pairs stay on one line: "
                  f"worst D_shape {worst:.3f} m (need <= 0.300, "
                  f"floor {floor:.3f})")
            print(f"  {'PASS' if moved else 'FAIL'}  speed pairs change duration: "
                  f"worst ratio {tmin:.2f}x (need >= 1.25x)")
            cf_ok = cf_ok and orth and moved
        if mismatch:
            cf_ok = False
            print(f"  FAIL  {len(mismatch)} pair(s) vary an axis other than the "
                  "one their group declared")
    else:
        print("  FAIL  no usable pairs — were any groups completed?")

    # ------------------------------------------------------------- G-L
    print("\nG-L language")
    sents = [i["meta"].get("instruction", "") for i in good]
    tmpl = collections.Counter()
    slots = collections.Counter()
    for i in good:
        s = i["meta"].get("intent_slots", {})
        tmpl[i["meta"].get("intent_id")] += 1
        if s.get("goal"):
            slots[str(s["goal"])] += 1
    uniq = len(set(sents))
    print(f"  {len(sents)} valid episodes, {uniq} distinct sentences "
          f"({uniq / max(len(sents), 1) * 100:.0f}% unique)")
    print("  intents: " + ", ".join(f"{k}={v}" for k, v in tmpl.most_common()))
    print("  goals:   " + ", ".join(f"{k}={v}" for k, v in slots.most_common()))
    if slots:
        spread = max(slots.values()) / max(min(slots.values()), 1)
        bal_ok = spread <= 2.0
        print(f"  balance: most/least = {spread:.1f}x  "
              f"{'PASS' if bal_ok else 'WARN (goal frequency is itself a cue)'}")
    else:
        bal_ok = False

    ok = (not fails) and rate_ok and dropped == 0 and cf_ok
    print(f"\n{'PASS' if ok else 'FAIL'} — "
          f"{len(good)}/{len(loaded)} episodes usable")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"sessions": sessions, "n_episodes": len(loaded),
                       "n_valid": len(good), "terminations": dict(terms),
                       "integrity_failures": fails, "pairs": cf_rows,
                       "distinct_sentences": uniq, "pass": ok},
                      f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
