#!/usr/bin/env python3
"""Open-loop sensitivity probe: does the policy use the image, and the sentence?

No simulator, no ROS — loads the checkpoint directly and interrogates it with
frames from the recorded corpus. Answers, in one minute, the two questions that
decide whether a closed-loop run is even worth setting up:

    S_img   same sentence, different images   -> action delta
    S_lang  same image, different sentences   -> action delta
    S_noise same image, same sentence, seeds  -> reproducibility floor

The failure modes these catch are the two halves of "not actually a VLA":

* S_img at the noise floor: the model ignores the camera. Plausible here
  because training padded the checkpoint's camera2/3 slots and fed the real
  view through camera1 — if that wiring were wrong, loss would still fall (the
  action is predictable from state + time alone on this track) and nothing
  upstream would have noticed.
* S_lang at the noise floor **on decision frames**: the model ignores language,
  which is the old stack's failure reproduced at great expense.

Stratification matters for S_lang exactly as it does for the corpus CMI: on the
shared aisle every bay demands the same action, so language sensitivity there is
*supposed* to be zero. Frames are therefore sampled from two pools — early
(shared approach) and late (post-divergence) — and reported separately.

Usage::

    ~/venv/navvla/bin/python probe_policy_openloop.py --checkpoint CKPT \\
        --episodes-dir src/nav_vla_pkg/data_v2/corpus_v2_train_b
"""

import argparse
import io
import json
import os
import random
import sys

import numpy as np


def load_rows(ep_dir):
    rp = os.path.join(ep_dir, "resampled_10hz.jsonl")
    if not os.path.exists(rp):
        return []
    rows = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("action")]


def in_decision_zone(r):
    """In the aisle, at or approaching the bay turn points.

    Geometry from extract_track_paths.py: the aisle runs at x = -4.42; bay turn
    arcs begin at turn_start_y = bay_y - 5.23, i.e. y = -14.6 (bay1) .. -3.7
    (bay4). A frame here is one where two different ordinals genuinely demand
    different steering RIGHT NOW — bay1 says turn, bay4 says keep going.

    The first version of this probe stratified by episode *fraction* and never
    sampled this band: 0.15-0.45 is the shared approach (language rightly
    dormant) and 0.75-0.9 is post-divergence (the image already shows the chosen
    bay, so language is redundant with it). Both measured ~0 language
    sensitivity and both were *supposed to*. The corpus CMI made the same
    point: the 0.9-bit peak sits in the turn window, not at the ends.
    """
    return -5.6 <= r["x"] <= -3.2 and -15.2 <= r["y"] <= 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes-dir", required=True,
                    help="session dir with resampled episodes (parking ones)")
    ap.add_argument("--n-frames", type=int, default=6,
                    help="frames per stratum")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="openloop_report.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint} on {device}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    preproc, postproc = make_pre_post_processors(policy.config, args.checkpoint)
    cam_key = next(k for k in policy.config.input_features
                   if k.startswith("observation.images."))
    print(f"camera key {cam_key}")

    def act_mean(img_arr, state, task, n_seeds):
        """Seed-averaged chunk. Flow matching samples its initial noise, and a
        single draw carries enough of it to bury a small conditioning effect —
        measured: S_noise 0.0035 vs a language effect of 0.0002. Averaging n
        draws shrinks the sampling term by sqrt(n) without touching the model."""
        return np.mean([act(img_arr, state, task, s) for s in range(n_seeds)],
                       axis=0)

    def act(img_arr, state, task, seed):
        torch.manual_seed(seed)
        t_img = (torch.from_numpy(img_arr.copy()).permute(2, 0, 1).float()
                 / 255.0).unsqueeze(0)
        batch = {cam_key: t_img,
                 "observation.state": torch.tensor(
                     state, dtype=torch.float32).unsqueeze(0),
                 "task": [task]}
        batch = preproc(batch)
        with torch.inference_mode():
            out = policy.predict_action_chunk(batch)
        out = postproc(out)
        return out.squeeze(0).float().cpu().numpy()

    def delta(a, b):
        """Mean |dyaw| difference over the chunk — the steering channel."""
        n = min(len(a), len(b))
        return float(np.mean(np.abs(a[:n, 2] - b[:n, 2])))

    # ---- collect parking episodes with distinct bays ------------------------
    sess = os.path.abspath(args.episodes_dir)
    eps = []
    for d in sorted(os.listdir(sess)):
        mp = os.path.join(sess, d, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp, encoding="utf-8"))
        if (meta.get("valid") and meta.get("intent_id") == "park_bay"
                and meta.get("termination") == "success"):
            rows = load_rows(os.path.join(sess, d))
            if rows:
                eps.append((meta, os.path.join(sess, d), rows))
    if len(eps) < 4:
        print(f"only {len(eps)} usable parking episodes in {sess}", file=sys.stderr)
        return 1
    rng = random.Random(20260729)
    rng.shuffle(eps)
    print(f"{len(eps)} parking episodes available")

    from PIL import Image as PILImage

    def load_img(p):
        return np.asarray(PILImage.open(p).convert("RGB"))

    sentences = ["Park in the first bay on the right, slowly.",
                 "Park in the second bay on the right, slowly.",
                 "Park in the third bay on the right, slowly.",
                 "Park in the fourth bay on the right, slowly."]

    def frac_pick(rows, fracs):
        out = []
        for f in fracs:
            out.append(rows[min(int(f * len(rows)), len(rows) - 1)])
        return out

    strata = {
        "early (shared aisle)": lambda rows: frac_pick(rows, [0.15, 0.45]),
        "DECISION (turn window)": lambda rows: [r for r in rows
                                                if in_decision_zone(r)][::4],
        "late (post-divergence)": lambda rows: frac_pick(rows, [0.75, 0.9]),
    }
    report = {}
    for sname, pick_fn in strata.items():
        picks = []
        for meta, base, rows in eps:
            for row in pick_fn(rows):
                p = os.path.join(base, row["frame"])
                if os.path.exists(p):
                    picks.append((p, row))
            if len(picks) >= args.n_frames:
                break
        picks = picks[:args.n_frames]
        if not picks:
            report[sname] = {"n_frames": 0}
            continue

        s_noise, s_img, s_lang = [], [], []
        K = args.seeds
        for i, (p, row) in enumerate(picks):
            img = load_img(p)
            st = row["state"]
            base_m = act_mean(img, st, sentences[1], K)
            # noise floor AT THE AVERAGED LEVEL: two disjoint seed sets. This is
            # the honest denominator — comparing an average against a single
            # draw would make every other signal look large.
            alt_m = np.mean([act(img, st, sentences[1], s)
                             for s in range(K, 2 * K)], axis=0)
            s_noise.append(delta(base_m, alt_m))
            # image sensitivity: a different real frame, same sentence
            q, qrow = picks[(i + 1) % len(picks)]
            if q != p:
                s_img.append(delta(base_m, act_mean(load_img(q), st,
                                                    sentences[1], K)))
            grey = np.full_like(img, 128)
            s_img.append(delta(base_m, act_mean(grey, st, sentences[1], K)))
            # language sensitivity: same frame, other bays
            for s2 in (sentences[0], sentences[2], sentences[3]):
                s_lang.append(delta(base_m, act_mean(img, st, s2, K)))

        r = {"n_frames": len(picks),
             "S_noise": float(np.median(s_noise)) if s_noise else 0.0,
             "S_img": float(np.median(s_img)) if s_img else 0.0,
             "S_lang": float(np.median(s_lang)) if s_lang else 0.0}
        r["lang_over_noise"] = r["S_lang"] / max(r["S_noise"], 1e-9)
        r["img_over_noise"] = r["S_img"] / max(r["S_noise"], 1e-9)
        report[sname] = r

    print(f"\n{'stratum':26s} {'S_noise':>9s} {'S_img':>9s} {'S_lang':>9s} "
          f"{'lang/noise':>11s} {'img/noise':>10s}")
    for sname, r in report.items():
        print(f"{sname:26s} {r['S_noise']:9.5f} {r['S_img']:9.5f} "
              f"{r['S_lang']:9.5f} {r['lang_over_noise']:10.1f}x "
              f"{r['img_over_noise']:9.1f}x")

    print("\nhow to read this (plan section 7.3, Level 0):")
    print("  img/noise  >> 1 everywhere      : the model looks at the camera")
    print("  lang/noise >> 1 on LATE frames  : the model reads the sentence "
          "where it matters")
    print("  lang/noise ~ 1 on EARLY frames  : correct — shared aisle, "
          "language should not matter yet")
    print("  (old stack's S_lang/S_img was 0.029)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
