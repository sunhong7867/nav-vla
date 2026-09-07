#!/usr/bin/env python3
"""Held-out decode evaluation for a reasoning_vla checkpoint.

Samples frames from packed HELD-OUT episodes (never converted into the
training dataset), runs the full serving-style path — jpeg -> preproc ->
``generate_reasoning`` — and scores each generated sentence against that
frame's ground-truth facts from ``reasoning.jsonl``:

  zone   : the fact's goal/near zone token appears in the text
           (scored only when the frame has one)
  lane   : inner/outer matches the instructed lane
  trend  : no CONTRADICTING trend claim (a hold frame saying "slowing"
           fails; not mentioning speed at all passes)
  speed  : every "N.N m/s" quoted lies within tol of the frame's measured
           or plan-end speed

This is the anti-parrot scorecard: a model that memorized one sentence
scores high on fluency but fails zone/speed on frames it never saw.

Usage (lab server)::

    CUDA_VISIBLE_DEVICES=1 ./venv/bin/python code/eval_reasoning_decode.py \
        --checkpoint runs/navvla_reasoning_r1/checkpoints/last/pretrained_model \
        --packed data/packed_v3y_heldout data/packed_v3y_supp_heldout \
        --n 60 --out logs/reasoning_eval_r1.jsonl
"""

import argparse
import io
import json
import os
import random
import re
import sys

NUM_MS = re.compile(r"(\d+\.?\d*)\s*m/s")
LANE_WORD = {"lane1": "inner", "lane2": "outer"}
TREND_PAT = {"decel": r"slow|brak|decel|reduc",
             "accel": r"accel|speed(ing)? up|pick|increas",
             "hold": r"hold|maintain|steady|constant|keep"}


def zone_tokens(name):
    """Tokens that count as mentioning a zone (T3 crosswalk aliases etc.)."""
    if not name:
        return []
    alias = {"crosswalk_stop": ["crosswalk", "t3"], "Start": ["start"],
             "T1/M1": ["t1", "m1"]}
    return alias.get(name, [name.lower()])


def score(text, facts):
    low = text.lower()
    row = {}
    zone = facts.get("goal") if facts.get("goal_arc_m") is not None and \
        0 <= (facts.get("goal_arc_m") or 99) <= 12 else facts.get("near_zone")
    if zone:
        row["zone"] = any(t in low for t in zone_tokens(zone))
    lane = facts.get("lane")
    if lane:
        want = LANE_WORD[lane]
        other = "outer" if want == "inner" else "inner"
        # mentioning BOTH lanes is a contradiction, not a match
        row["lane"] = want in low and other not in low
    tr = facts.get("trend")
    if tr:
        ok = True
        for other, pat in TREND_PAT.items():
            if other != tr and re.search(pat, low):
                ok = False
        row["trend"] = ok
    nums = [float(x) for x in NUM_MS.findall(text)]
    if nums:
        refs = [facts.get("v_mps"), facts.get("v_plan_end_mps"),
                facts.get("target_v_mps")]
        refs = [r for r in refs if r is not None]
        row["speed"] = all(any(abs(n - r) <= 0.35 for r in refs)
                           for n in nums)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--packed", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--rp", type=float, default=1.0,
                    help="repetition penalty for the decode")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from reasoning_vla import ReasoningSmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device("cuda")
    policy = ReasoningSmolVLAPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(device).eval()
    preproc, _ = make_pre_post_processors(policy.config, args.checkpoint)
    cam_key = [k for k in policy.config.input_features
               if k.startswith("observation.images.")][0]

    # collect (ep_dir, meta, frame rows with reasoning facts)
    pool = []
    for root in args.packed:
        for name in sorted(os.listdir(root)):
            ep = os.path.join(root, name)
            rj = os.path.join(ep, "reasoning.jsonl")
            rs = os.path.join(ep, "resampled_10hz.jsonl")
            if os.path.isfile(rj) and os.path.isfile(rs):
                pool.append(ep)
    rng.shuffle(pool)

    picks = []
    for ep in pool:
        if len(picks) >= args.n:
            break
        rows = [json.loads(l) for l in open(
            os.path.join(ep, "resampled_10hz.jsonl"))]
        rlines = [json.loads(l) for l in open(
            os.path.join(ep, "reasoning.jsonl"))]
        meta = json.load(open(os.path.join(ep, "meta.json")))
        k = rng.randrange(len(rows) - 1)
        picks.append((ep, meta, rows[k], rlines[1 + k]))

    counts, totals = {}, {}
    out_rows = []
    for i, (ep, meta, row, rrow) in enumerate(picks):
        img = np.asarray(Image.open(os.path.join(
            ep, row["frame"])).convert("RGB"))
        t_img = (torch.from_numpy(img).permute(2, 0, 1).float()
                 / 255.0).unsqueeze(0)
        batch = {cam_key: t_img,
                 "observation.state": torch.tensor(
                     row["state"], dtype=torch.float32).unsqueeze(0),
                 "task": [meta.get("instruction", "")]}
        batch = preproc(batch)
        with torch.inference_mode():
            text = policy.generate_reasoning(
                batch, repetition_penalty=args.rp)[0]
        sc = score(text, rrow["facts"])
        for k2, v in sc.items():
            totals[k2] = totals.get(k2, 0) + 1
            counts[k2] = counts.get(k2, 0) + bool(v)
        out_rows.append({"ep": os.path.basename(ep), "k": rrow["k"],
                         "facts": rrow["facts"],
                         "skeleton": rrow["skeleton"],
                         "generated": text, "score": sc})
        flag = "".join(k2[0].upper() if v else k2[0] for k2, v in sc.items())
        print(f"[{i:2d}] {flag:5s} {text[:110]}")

    print("\nfact-match rates (held-out frames):")
    for k2 in sorted(totals):
        print(f"  {k2:6s} {counts[k2]}/{totals[k2]} "
              f"({counts[k2] / totals[k2]:.0%})")
    uniq = len({r['generated'] for r in out_rows})
    print(f"unique sentences: {uniq}/{len(out_rows)}")
    if args.out:
        with open(args.out, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
