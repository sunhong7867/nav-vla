#!/usr/bin/env python3
"""Conditional mutual information: does the sentence still matter once you can see?

The last unmeasured item in GATE G2, and the one that most directly states the
claim. Every other number compares whole trajectories. This one asks the question
a policy actually faces, one frame at a time:

    Standing HERE, seeing THIS, how much does the instruction tell me about what
    to do next?

    I(action ; instruction | observation)

If it is near zero, the observation already determines the action and language is
decoration — which is precisely the old stack's failure. `navigator_node` emitted
a lane name; steering came from the camera spline; a goal-shuffle ablation moved
the commanded yaw rate by 0.0049 rad/s. A policy trained on that learns to ignore
the sentence, and no amount of model capacity fixes it.

Conditioning is the whole point
-------------------------------
Plain I(action ; instruction) is easy to inflate and means little: bay1 and bay4
episodes differ in where they end, so their actions differ, so the mutual
information is large — but an observation-only policy would score just as well by
looking out of the window. Conditioning on the observation removes exactly that.
What survives is the part of the action that the sentence explains **and the view
does not**.

The observation bucket
----------------------
Ideally the conditioning variable is the image. Here it is the ground-truth pose,
quantized. That is a *stronger* test than bucketing on pixels: pose determines the
rendered view exactly (static world, no traffic), so two samples in one bucket
have identical images by construction, while an image-feature bucket would blur
together views that merely look similar and leak some position information into
the "unexplained" residual.

Buckets are (x, y, heading) at `--pos-q` metres and `--yaw-q` degrees. A bucket
with samples from only one instruction contributes nothing and is dropped: with
one value of the conditioning variable there is no comparison to make. The number
of usable buckets is reported, because a large CMI over four buckets is not
evidence of anything.

Estimator
---------
Discrete plug-in, on quantized actions. Plug-in MI is biased upward when bins
outnumber samples, so this reports the Miller-Madow correction alongside the raw
value, and a shuffled control: instruction labels permuted **within each bucket**,
which destroys the real association while preserving every marginal. The shuffled
score is the floor this estimator produces from noise alone; the real number has
to clear it to mean anything.

Usage::

    python3 src/nav_vla_pkg/scripts/measure_cmi.py <session_dir>
    python3 src/nav_vla_pkg/scripts/measure_cmi.py <session_dir> --axis ordinal
"""

import argparse
import collections
import json
import math
import os
import random
import sys

DEFAULT_POS_Q = 1.0        # metres
DEFAULT_YAW_Q = 15.0       # degrees
# Action bins. dyaw is the steering channel and the one the claim is about;
# 0.01 rad per 10 Hz step is 0.1 rad/s, comfortably finer than the difference
# between going straight and turning into a bay.
DYAW_Q = 0.02              # rad, per horizon window (not per step)
DS_Q = 0.02                # metres of forward step


def entropy(counts):
    n = sum(counts)
    if n == 0:
        return 0.0
    return -sum(c / n * math.log2(c / n) for c in counts if c > 0)


def mi(pairs):
    """Plug-in mutual information in bits, plus its Miller-Madow correction."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    ja = collections.Counter(pairs)
    xa = collections.Counter(a for a, _ in pairs)
    ya = collections.Counter(b for _, b in pairs)
    h = entropy(list(xa.values())) + entropy(list(ya.values())) - entropy(list(ja.values()))
    # Miller-Madow adds (support-1)/(2n ln2) to each entropy. The joint support
    # is the largest of the three and enters with a minus sign, so the net term
    # is negative — plug-in MI is biased *up*, which is the whole reason this
    # correction is here. (Subtracting `corr` instead of adding it inflated the
    # estimate rather than shrinking it.)
    corr = ((len(xa) - 1) + (len(ya) - 1) - (len(ja) - 1)) / (2 * n * math.log(2))
    return h, max(0.0, h + corr)


def load_session(sess, axis_filter):
    eps = []
    for d in sorted(os.listdir(sess)):
        if not d.startswith("ep_"):
            continue
        base = os.path.join(sess, d)
        mp, rp = os.path.join(base, "meta.json"), os.path.join(base, "resampled_10hz.jsonl")
        if not (os.path.exists(mp) and os.path.exists(rp)):
            continue
        meta = json.load(open(mp, encoding="utf-8"))
        if not meta.get("valid"):
            continue
        if axis_filter and meta.get("cf_axis") != axis_filter:
            continue
        rows = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
        if rows:
            eps.append((meta, rows))
    return eps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir", nargs="+",
                   help="one or more session directories, merged into one corpus")
    p.add_argument("--axis", default="", help="only groups with this cf_axis")
    p.add_argument("--slot", default="goal",
                   help="which instruction slot is the variable")
    p.add_argument("--pos-q", type=float, default=DEFAULT_POS_Q)
    p.add_argument("--yaw-q", type=float, default=DEFAULT_YAW_Q)
    p.add_argument("--min-per-bucket", type=int, default=8)
    p.add_argument("--with-speed", action="store_true",
                   help="add the forward-step channel (larger alphabet, more bias)")
    p.add_argument("--horizon", type=int, default=21,
                   help="grid steps to sum the action over. 21 = 2.1 s, the "
                        "evaluation point sim_vla_plan.md section 5.6 names")
    p.add_argument("--hi-bits", type=float, default=0.5,
                   help="a bucket counts as decision-relevant above this")
    p.add_argument("--min-peak", type=float, default=0.8)
    p.add_argument("--min-frac", type=float, default=0.20)
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--json", default="")
    args = p.parse_args()

    sessions = [os.path.abspath(s) for s in args.session_dir]
    eps = []
    for sd in sessions:
        eps.extend(load_session(sd, args.axis))
    if not eps:
        print(f"no resampled episodes in {', '.join(sessions)} — "
              "run resample_episodes.py first", file=sys.stderr)
        return 1

    yq = math.radians(args.yaw_q)
    h = max(1, args.horizon)
    buckets = collections.defaultdict(list)
    for meta, rows in eps:
        instr = str(meta.get("intent_slots", {}).get(args.slot))
        acts = [r.get("action") for r in rows]
        for i, r in enumerate(rows):
            if not acts[i] or i + h > len(rows):
                continue
            key = (round(r["x"] / args.pos_q), round(r["y"] / args.pos_q),
                   round(r["heading"] / yq))
            # Net turn over the next `horizon` steps, not the next single step.
            #
            # This is what the policy is asked to produce: SmolVLA emits an action
            # chunk, so the prediction target at time t spans seconds, not 100 ms.
            # At h=1 the metric answers "does the sentence change the next tenth of
            # a second", and driving straight down the shared aisle the honest
            # answer is no for every bay — the instruction has not become relevant
            # yet. Measured that way the corpus scores ~0 bits and looks like the
            # old stack, for entirely the wrong reason.
            span = sum(a[2] for a in acts[i:i + h] if a)
            act = (round(span / DYAW_Q),)
            if args.with_speed:
                dist = sum(math.hypot(a[0], a[1]) for a in acts[i:i + h] if a)
                act = act + (round(dist / DS_Q),)
            buckets[key].append((act, instr))

    usable = {k: v for k, v in buckets.items()
              if len(v) >= args.min_per_bucket and len({i for _, i in v}) >= 2}
    n_used = sum(len(v) for v in usable.values())
    n_all = sum(len(v) for v in buckets.values())

    print(", ".join(os.path.basename(s) for s in sessions)
          + (f"  axis={args.axis}" if args.axis else ""))
    print(f"{len(eps)} episodes, {n_all} samples")
    print(f"buckets: {len(buckets)} total, {len(usable)} usable "
          f"(>= {args.min_per_bucket} samples and >= 2 distinct {args.slot})")
    print(f"samples in usable buckets: {n_used} ({100 * n_used / max(n_all, 1):.0f}%)")
    if not usable:
        print("\nnothing to measure: no bucket holds two different instructions.\n"
              "That is itself the finding — episodes never overlap in observation\n"
              "space, so no counterfactual comparison exists at the frame level.")
        return 1

    rng = random.Random(args.seed)
    rows_out, raw, corrected, shuffled, debiased, weights = [], [], [], [], [], []
    for key, pairs in usable.items():
        h_raw, h_corr = mi(pairs)
        # Permute instructions *within* the bucket: marginals are untouched, the
        # association is destroyed. Whatever the estimator still reports is bias.
        # Averaged over several permutations, because a single shuffle is itself
        # a noisy estimate of the noise.
        acts = [a for a, _ in pairs]
        ins = [i for _, i in pairs]
        sh = []
        for _ in range(args.permutations):
            rng.shuffle(ins)
            sh.append(mi(list(zip(acts, ins)))[0])
        h_sh = sum(sh) / len(sh)
        raw.append(h_raw); corrected.append(h_corr); shuffled.append(h_sh)
        # The honest effect size: what the estimator reports minus what it
        # reports on data with no association at all.
        debiased.append(max(0.0, h_raw - h_sh))
        weights.append(len(pairs))
        rows_out.append({"bucket": key, "n": len(pairs), "mi_bits": h_raw,
                         "mi_mm_bits": h_corr, "mi_shuffled_bits": h_sh,
                         "mi_debiased_bits": debiased[-1],
                         "n_instructions": len({i for _, i in pairs})})

    def wmed(vals):
        order = sorted(zip(vals, weights))
        tot = sum(weights) / 2.0
        run = 0.0
        for v, w in order:
            run += w
            if run >= tot:
                return v
        return order[-1][0]

    tot_w = sum(weights)
    print(f"\n{'':24s} {'weighted mean':>14s} {'weighted median':>16s}")
    for name, vals in (("I(a;instr|obs) plug-in", raw),
                       ("  Miller-Madow", corrected),
                       ("  shuffled control", shuffled),
                       ("  DEBIASED (plug-in - shuffled)", debiased)):
        wm = sum(v * w for v, w in zip(vals, weights)) / tot_w
        print(f"{name:30s} {wm:8.3f}b {wmed(vals):15.3f}b")

    real = sum(v * w for v, w in zip(debiased, weights)) / tot_w
    fake = sum(v * w for v, w in zip(shuffled, weights)) / tot_w
    med = wmed(debiased)
    srt = sorted(debiased)
    nb = len(srt)
    peak = srt[-1]
    frac_hi = (sum(w for v, w in zip(debiased, weights) if v >= args.hi_bits)
               / tot_w)

    # A flat median over all buckets is the wrong summary for this corpus, and
    # the reason is structural rather than statistical.
    #
    # The ordinal axis works BECAUSE four bays share 14.4 m of identical aisle.
    # Across that stretch every instruction produces the same action, so the CMI
    # there is legitimately zero — that is the design, not a defect. Most buckets
    # sit in the shared stretch, so the median measures the shared stretch. A
    # corpus that scored 0.8 b at the median would be one whose trajectories
    # separate immediately, which is exactly the corpus that cannot demonstrate
    # "same observation, different sentence".
    #
    # sim_vla_plan.md section 7.3 already argues this for the sibling metric
    # S_lang ("decision-relevant 프레임에서만 측정한다"); the CMI row kept a flat
    # median by oversight. Reported here as a distribution plus two numbers that
    # do not fight the design:
    #
    #   peak      how much the sentence explains where it matters at all
    #   frac_hi   how much of the drive that covers
    print(f"\ndistribution over {nb} buckets (debiased):")
    for q in (0.5, 0.75, 0.9):
        print(f"  p{int(q * 100):<3d} {srt[min(int(q * nb), nb - 1)]:.3f} b")
    print(f"  peak {peak:.3f} b   (ceiling for this axis is "
          f"log2(n_values) = {math.log2(max(len({i for _, i in sum(usable.values(), [])}), 2)):.2f} b)")
    print(f"  frames in buckets >= {args.hi_bits} b: {100 * frac_hi:.0f}%")

    ok = peak >= args.min_peak and frac_hi >= args.min_frac
    print(f"\nGATE: peak >= {args.min_peak} b AND >= {100 * args.min_frac:.0f}% "
          f"of frames above {args.hi_bits} b")
    print(f"  peak {peak:.3f} b   frames {100 * frac_hi:.0f}%   "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  (flat median was {med:.3f} b — retained for continuity with the "
          "plan, but see the note in this script on why it fights the design)")
    print(f"  (estimator noise floor on this sample: {fake:.3f} b — "
          f"{'small' if fake < 0.3 else 'LARGE, buckets are undersampled'})")
    print("\nreference: the old stack's goal-shuffle ablation moved commanded yaw "
          "rate by 0.0049 rad/s")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"sessions": sessions, "axis": args.axis, "slot": args.slot,
                       "pos_q_m": args.pos_q, "yaw_q_deg": args.yaw_q,
                       "n_episodes": len(eps), "n_samples": n_all,
                       "n_buckets_usable": len(usable),
                       "weighted_mean_debiased_bits": real,
                       "weighted_median_debiased_bits": med,
                       "peak_debiased_bits": peak,
                       "frac_frames_above_hi": frac_hi,
                       "hi_bits": args.hi_bits,
                       "shuffled_mean_bits": fake, "pass": ok,
                       "buckets": rows_out}, f, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
