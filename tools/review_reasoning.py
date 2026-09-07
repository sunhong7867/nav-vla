#!/usr/bin/env python3
"""Review report for reasoning labels produced by label_reasoning.py.

Two outputs:
  1. Corpus-wide stats on stdout — sentence duplication rate (the parrot
     metric), position-vs-sentence diversity, token/word length, fact
     coverage, paraphrase reject rate when present.
  2. A self-contained HTML page sampling random segments with their frame
     image, facts, skeleton, and paraphrases, for eyeball verification that
     every claim matches the pixels.

Usage::

    python3 tools/review_reasoning.py src/nav_vla_pkg/data_v3y/packed_v3y \
        [more dirs...] [--n 60] [--out reasoning_review.html]

Token lengths use the SmolVLM2 tokenizer when transformers is importable
(run under /home/sh/venv/navvla for that); otherwise word counts are
reported and marked as such.
"""

import argparse
import base64
import collections
import html
import json
import math
import os
import random


def load_episode(ep_dir):
    path = os.path.join(ep_dir, "reasoning.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    return lines[0]["_meta"], lines[1:]


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packed_dirs", nargs="+")
    ap.add_argument("--n", type=int, default=60,
                    help="segments to sample into the HTML report")
    ap.add_argument("--out", default="reasoning_review.html")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    episodes = []          # (ep_dir, meta, rows)
    for root in args.packed_dirs:
        for name in sorted(os.listdir(root)):
            ep_dir = os.path.join(root, name)
            if not os.path.isdir(ep_dir):
                continue
            got = load_episode(ep_dir)
            if got:
                episodes.append((ep_dir, got[0], got[1]))
    if not episodes:
        raise SystemExit("no reasoning.jsonl found")

    # ---------------------------------------------------------------- stats
    all_sents, seg_records = [], []
    kind_counter = collections.Counter()
    pos_bucket = collections.defaultdict(list)   # ring position -> sentences
    para_total = para_kept = 0
    for ep_dir, meta, rows in episodes:
        by_seg = {}
        for r in rows:
            by_seg.setdefault(r["segment_id"], r)
        for seg in meta["segments"]:
            sent = seg["skeleton"]
            all_sents.append(sent)
            r0 = by_seg.get(seg["id"])
            if r0:
                f = r0["facts"]
                kind_counter[f["curv"]] += 1
                kind_counter["trend:" + f["trend"]] += 1
                if f["turn"]:
                    kind_counter["turn"] += 1
                if f["goal_arc_m"] is not None and 0 <= f["goal_arc_m"] <= 12:
                    kind_counter["goal_approach"] += 1
                # bucket by coarse world position (2 m grid) for the
                # place-vs-sentence check
                # (facts carry no ring index; grid of the segment start
                #  frame is a fair proxy)
            paras = seg.get("paraphrases")
            if paras is not None:
                para_total += seg.get("n_paraphrase_tries", len(paras))
                para_kept += len(paras)
            seg_records.append((ep_dir, meta, seg))
        for r in rows:
            if r["segment_id"] in by_seg and r is by_seg[r["segment_id"]]:
                pass
    # place bucket needs coordinates: re-read start frames' rows
    for ep_dir, meta, rows in episodes:
        firsts = {}
        for r in rows:
            firsts.setdefault(r["segment_id"], r)
        for sid, r in firsts.items():
            # facts have no x/y; use goal/near/curv discrete place proxy:
            key = (r["facts"]["near_zone"], r["facts"]["curv"])
            pos_bucket[key].append(meta["segments"][sid]["skeleton"])

    n_seg = len(all_sents)
    n_unique = len(set(all_sents))
    dup_rate = 1.0 - n_unique / n_seg
    top_dupes = collections.Counter(all_sents).most_common(5)

    tok = get_tokenizer()
    if tok:
        lens = [len(tok(s)["input_ids"]) for s in set(all_sents)]
        unit = "tokens (SmolVLM2)"
    else:
        lens = [len(s.split()) for s in set(all_sents)]
        unit = "words (transformers unavailable)"
    lens.sort()

    def pct(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))]

    print(f"episodes: {len(episodes)}   segments: {n_seg}   "
          f"unique sentences: {n_unique}")
    print(f"duplication rate: {dup_rate:.1%}  (target < 30%)")
    print(f"sentence length [{unit}]: p50={pct(.5)} p90={pct(.9)} "
          f"max={lens[-1]}")
    if para_total:
        print(f"paraphrases kept: {para_kept}/{para_total} "
              f"({1 - para_kept / para_total:.1%} rejected)")
    else:
        print("paraphrases: none yet (skeleton-only corpus)")
    print("\nfact coverage (segment starts):")
    for k, v in kind_counter.most_common():
        print(f"  {k:16s} {v:5d}  ({v / n_seg:.0%})")
    print("\nplace-vs-sentence diversity (same (near_zone, curv) place):")
    worst = sorted(pos_bucket.items(),
                   key=lambda kv: len(set(kv[1])) / len(kv[1]))[:5]
    for key, sents in worst:
        print(f"  {str(key):32s} {len(set(sents)):4d} unique / "
              f"{len(sents):4d} segments")
    print("\ntop repeated sentences:")
    for s, c in top_dupes:
        print(f"  x{c:4d}  {s[:90]}")

    # ----------------------------------------------------------------- html
    sample = rng.sample(seg_records, min(args.n, len(seg_records)))
    cards = []
    for ep_dir, meta, seg in sample:
        # frame at the middle of the segment
        rows = load_episode(ep_dir)[1]
        mid_k = (seg["k0"] + seg["k1"]) // 2
        row = rows[mid_k]
        # resampled row for the image file name
        rs = os.path.join(ep_dir, "resampled_10hz.jsonl")
        with open(rs) as f:
            for i, line in enumerate(f):
                if i == mid_k:
                    frame_rel = json.loads(line)["frame"]
                    break
        img_path = os.path.join(ep_dir, frame_rel)
        b64 = base64.b64encode(open(img_path, "rb").read()).decode()
        paras = seg.get("paraphrases") or []
        cards.append(f"""
<div class="card">
 <img src="data:image/jpeg;base64,{b64}">
 <div class="txt">
  <div class="ep">{html.escape(os.path.basename(ep_dir))}
      [k {seg['k0']}-{seg['k1']}] — {html.escape(meta['instruction'])}</div>
  <div class="skel">{html.escape(seg['skeleton'])}</div>
  <div class="facts">{html.escape(json.dumps(row['facts']))}</div>
  {''.join(f'<div class="para">{html.escape(p)}</div>' for p in paras)}
 </div>
</div>""")

    doc = f"""<!doctype html><meta charset="utf-8">
<title>Reasoning label review</title>
<style>
body{{font-family:system-ui;margin:20px;background:#fafafa}}
.card{{display:flex;gap:14px;background:#fff;border:1px solid #ddd;
      border-radius:8px;padding:10px;margin-bottom:12px}}
.card img{{width:320px;height:240px;object-fit:cover;border-radius:4px}}
.ep{{color:#888;font-size:12px;margin-bottom:6px}}
.skel{{font-size:15px;margin-bottom:6px}}
.facts{{font-family:monospace;font-size:11px;color:#555}}
.para{{font-size:13px;color:#036;margin-top:4px}}
</style>
<h2>Reasoning label review — {len(sample)} random segments</h2>
<p>dup rate {dup_rate:.1%} · {n_seg} segments · {len(episodes)} episodes</p>
{''.join(cards)}"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"\nHTML report: {args.out}")


if __name__ == "__main__":
    main()
