#!/usr/bin/env python3
"""Pack verified sessions into the compact tree that gets shipped to the GPU box.

Training happens on the lab server; collection happens here. What crosses the
network should be the smallest thing that can still reproduce the dataset, and
that is **not** the raw session: the recorder writes at 30 Hz on purpose so the
timing can be recovered offline, but training consumes the 10 Hz grid. Two of
every three frames are dead weight the moment `resample_episodes.py` has run.

    raw session      ~24 MB/episode      30 Hz, four un-joined streams
    packaged         ~8 MB/episode       10 Hz, joined, only referenced frames

Frames are **hard-linked**, not copied, so packaging is instantaneous and costs
no disk. `tar`/`rsync` dereference them into real files at the far end. If the
destination is on another filesystem the script falls back to copying and says so.

What is deliberately kept
-------------------------
`meta.json` in full. It carries `cf_group_id`, `cf_axis`, `start_pose_key` and
`instruction`, which is the entire counterfactual bookkeeping — none of it is
reconstructible from the frames, and the analysis on the server needs it.

Original frame filenames are preserved rather than renumbered, so a row in the
packaged corpus can still be traced back to the raw episode it came from.

Usage::

    python3 src/nav_vla_pkg/scripts/package_corpus.py OUT_DIR SESSION [SESSION ...]
    python3 src/nav_vla_pkg/scripts/package_corpus.py /tmp/corpus_pack \\
        src/nav_vla_pkg/data_v2/corpus_v2_train src/nav_vla_pkg/data_v2/corpus_v2_train_b
"""

import argparse
import collections
import json
import os
import shutil
import sys

RESAMPLED = "resampled_10hz.jsonl"


def link_or_copy(src, dst, state):
    if state["mode"] == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            # Cross-device, or a filesystem without hard links. Say so once and
            # keep going rather than fail a two-hour pipeline on an optimisation.
            state["mode"] = "copy"
            print("  note: hard links unavailable (different filesystem?) — copying")
    shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--include-invalid", action="store_true",
                   help="pack episodes the recorder marked invalid too")
    args = p.parse_args()

    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    state = {"mode": "link"}

    packed, skipped, n_frames, n_rows = 0, collections.Counter(), 0, 0
    index = []
    seen_groups = collections.Counter()

    for sd in args.sessions:
        sd = os.path.abspath(sd)
        sname = os.path.basename(sd)
        for d in sorted(os.listdir(sd)):
            if not d.startswith("ep_"):
                continue
            src = os.path.join(sd, d)
            mp, rp = os.path.join(src, "meta.json"), os.path.join(src, RESAMPLED)
            if not os.path.exists(mp):
                skipped["no meta.json"] += 1
                continue
            meta = json.load(open(mp, encoding="utf-8"))
            if not os.path.exists(rp):
                skipped["not resampled"] += 1
                continue
            if not meta.get("valid") and not args.include_invalid:
                skipped[f"invalid: {meta.get('invalid_reason', '?')[:40]}"] += 1
                continue
            if meta.get("termination") != "success":
                skipped[f"termination={meta.get('termination')}"] += 1
                continue

            rows = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
            if not rows:
                skipped["empty resampled"] += 1
                continue

            # Session name is part of the packed episode id: group ids are only
            # unique within a session unless --group-prefix was used, and a
            # collision here would silently fuse two counterfactual groups.
            ep_id = f"{sname}__{d}"
            dst = os.path.join(out, ep_id)
            os.makedirs(os.path.join(dst, "frames"), exist_ok=True)

            wanted = []
            for r in rows:
                rel = r["frame"]
                s = os.path.join(src, rel)
                t = os.path.join(dst, rel)
                if not os.path.exists(s):
                    continue
                if not os.path.exists(t):
                    link_or_copy(s, t, state)
                wanted.append(rel)

            if len(wanted) != len(rows):
                skipped[f"{len(rows) - len(wanted)} missing frame files"] += 1
                shutil.rmtree(dst, ignore_errors=True)
                continue

            with open(os.path.join(dst, RESAMPLED), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            meta["packed_from"] = {"session": sname, "episode": d}
            with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            seen_groups[meta.get("cf_group_id")] += 1
            index.append({
                "episode": ep_id, "n_rows": len(rows),
                "instruction": meta.get("instruction"),
                "cf_group_id": meta.get("cf_group_id"),
                "cf_variant_id": meta.get("cf_variant_id"),
                "cf_axis": meta.get("cf_axis"),
                "intent_id": meta.get("intent_id"),
                "intent_slots": meta.get("intent_slots"),
                "start_pose_key": meta.get("start_pose_key"),
            })
            packed += 1
            n_frames += len(wanted)
            n_rows += len(rows)

    with open(os.path.join(out, "index.jsonl"), "w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # A group that lost members to the skip filters no longer yields the pairs it
    # was collected for. Reported here because this is the last place the whole
    # corpus is in one hand before it leaves the machine.
    axis_of = {r["cf_group_id"]: r["cf_axis"] for r in index}
    expected = {"ordinal": 4, "speed": 3, "lane": 2, "floor": 2}
    partial = [(g, n, expected.get(axis_of.get(g), 0))
               for g, n in seen_groups.items()
               if n < expected.get(axis_of.get(g), 0)]

    print(f"\npacked {packed} episodes -> {out}")
    print(f"  {n_rows} grid rows, {n_frames} frames "
          f"({'hard-linked' if state['mode'] == 'link' else 'copied'})")
    print(f"  {len(seen_groups)} counterfactual groups, "
          f"{len(seen_groups) - len(partial)} complete")
    if partial:
        print(f"  {len(partial)} incomplete group(s) — these contribute no pair:")
        for g, n, e in partial[:8]:
            print(f"      {g}: {n}/{e} variants")
    if skipped:
        print("  skipped:")
        for k, v in skipped.most_common():
            print(f"      {v:4d}  {k}")
    print("\nship with:  tar -C {0} -cf - . | zstd -T0 > corpus.tar.zst".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
