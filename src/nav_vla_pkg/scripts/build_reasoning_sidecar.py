#!/usr/bin/env python3
"""Join per-episode reasoning.jsonl into a dataset-level training sidecar.

Reads a converted LeRobot dataset's ``nav_vla_index.jsonl`` (the
lerobot_episode_index -> packed_episode map written by to_lerobot.py) plus
the packed episodes' ``reasoning.jsonl`` (from label_reasoning.py /
paraphrase_reasoning.py), and writes ``reasoning_labels.jsonl`` into the
dataset root, one row per segment:

    {"episode_index": int, "f0": int, "f1": int, "variants": [skeleton, ...]}

Frame indices are the LeRobot dataset's own frame_index, which for the v3y
conversion equals the packed row index k (the converter only drops the tail
row whose action is null, so ranges are clipped to the episode length, read
from meta/episodes/*.parquet). Terminal-copy episodes (park_bay tails) map
with an offset; v3y has none, but the offset path is handled.

Usage::

    python3 build_reasoning_sidecar.py <lerobot_root> <packed_dir> [...]
"""

import argparse
import glob
import json
import os
import sys


def episode_lengths(root):
    import pandas as pd
    files = sorted(glob.glob(
        os.path.join(root, "meta/episodes/**/*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no meta/episodes parquet under {root}")
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "length"])
                    for f in files])
    return dict(zip(df["episode_index"], df["length"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lerobot_root")
    ap.add_argument("packed_dirs", nargs="+")
    args = ap.parse_args()

    packed_of = {}
    for d in args.packed_dirs:
        for name in os.listdir(d):
            if os.path.isdir(os.path.join(d, name)):
                packed_of[name] = os.path.join(d, name)

    lengths = episode_lengths(args.lerobot_root)
    out_path = os.path.join(args.lerobot_root, "reasoning_labels.jsonl")
    n_rows = n_eps = n_missing = 0
    with open(os.path.join(args.lerobot_root, "nav_vla_index.jsonl")) as f, \
            open(out_path, "w") as out:
        for line in f:
            row = json.loads(line)
            ep = row["lerobot_episode_index"]
            name = row["packed_episode"]
            length = int(lengths.get(ep, 0))
            rj = os.path.join(packed_of.get(name, ""), "reasoning.jsonl")
            if not os.path.exists(rj):
                print(f"  no reasoning.jsonl for {name} (ep {ep})",
                      file=sys.stderr)
                n_missing += 1
                continue
            head = json.loads(open(rj).readline())["_meta"]
            # terminal copies replay the packed tail: frame 0 of the copy is
            # packed k = n_packed - n_frames_of_copy
            offset = 0
            if row.get("terminal_copy"):
                offset = head["n_frames"] - int(row["n_frames"])
            for seg in head["segments"]:
                f0 = max(0, seg["k0"] - offset)
                f1 = min(length - 1, seg["k1"] - offset)
                if f1 < f0:
                    continue
                variants = [seg["skeleton"]] + (seg.get("paraphrases") or [])
                out.write(json.dumps({
                    "episode_index": ep, "f0": f0, "f1": f1,
                    "variants": variants}, ensure_ascii=False) + "\n")
                n_rows += 1
            n_eps += 1
    print(f"wrote {out_path}: {n_rows} segment rows over {n_eps} episodes"
          + (f" ({n_missing} missing)" if n_missing else ""))


if __name__ == "__main__":
    main()
