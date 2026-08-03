#!/usr/bin/env python3
"""Packaged corpus -> LeRobotDataset, using the library rather than the layout.

Runs wherever `lerobot` is importable — normally the GPU server, since it is not
installed on the collection machine. That constraint is the reason this script
exists in this form: the LeRobot on-disk layout (parquet shards, `meta/info.json`,
`meta/episodes_stats.jsonl`, AV1 videos under `videos/chunk-XXX/<key>/`) is
versioned and has changed shape between v2.0, v2.1 and v3.0. Hand-writing it here
would mean emitting a binary format that nothing on this machine can read back,
and a subtly wrong dataset does not fail at conversion — it fails after a training
run, as a model that learned the wrong thing.

So: `LeRobotDataset.create()` / `add_frame()` / `save_episode()`, and let the
installed version decide what the bytes look like.

Feature layout
--------------
    observation.images.front   video, 480x640x3 — the wide VLA lens
    observation.state          float32[3]  speed m/s, yaw rate rad/s, steer rad
    action                     float32[3]  ego-frame SE(2) delta [dx, dy, dyaw]
    task                       the English instruction, verbatim

`action` is the SE(2) step between consecutive interpolated ground-truth poses,
never `cmd_vel` — the emitted command passes through a 15-symbol quantizer, and
the old corpus had exactly 9 distinct angular values as a result.

The counterfactual bookkeeping (`cf_group_id`, `cf_axis`, `start_pose_key`) has
no home in the LeRobot schema, so it is written alongside as `nav_vla_index.jsonl`
with the LeRobot episode index joined on. Ablations need it and it cannot be
recovered from the dataset itself.

Usage (on a machine with lerobot)::

    python3 to_lerobot.py PACKED_DIR --repo-id sunhong/navvla_sim_v2 --out ~/data/navvla
    python3 to_lerobot.py PACKED_DIR --repo-id ... --out ... --limit 20   # smoke
"""

import argparse
import json
import os
import sys
from pathlib import Path

RESAMPLED = "resampled_10hz.jsonl"
CAM_KEY = "observation.images.front"


def import_lerobot():
    """Return LeRobotDataset across the import paths different versions use."""
    tried = []
    for mod in ("lerobot.datasets.lerobot_dataset",
                "lerobot.common.datasets.lerobot_dataset"):
        try:
            m = __import__(mod, fromlist=["LeRobotDataset"])
            return m.LeRobotDataset, mod
        except ImportError as e:
            tried.append(f"{mod}: {e}")
    print("could not import LeRobotDataset. Tried:\n  " + "\n  ".join(tried),
          file=sys.stderr)
    print("\nInstall lerobot in this environment, or run this script on the GPU "
          "server. The packaged corpus is self-contained; nothing else is needed.",
          file=sys.stderr)
    raise SystemExit(2)


def load_episodes(packed, limit=0):
    eps = []
    for d in sorted(os.listdir(packed)):
        base = os.path.join(packed, d)
        mp, rp = os.path.join(base, "meta.json"), os.path.join(base, RESAMPLED)
        if not (os.path.isdir(base) and os.path.exists(mp) and os.path.exists(rp)):
            continue
        meta = json.load(open(mp, encoding="utf-8"))
        rows = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
        # The last grid point of every episode has action == null: the SE(2)
        # delta needs pose[k+1] and there is no k+1. An episode of N points
        # therefore yields N-1 (observation, action) pairs, and feeding the
        # actionless one through numpy turns it into a NaN scalar that LeRobot
        # rejects as "shape () does not have the expected shape (3,)".
        keep = [r for r in rows
                if isinstance(r.get("action"), (list, tuple))
                and len(r["action"]) == 3]
        dropped = len(rows) - len(keep)
        if keep:
            eps.append((d, base, meta, keep, dropped))
        if limit and len(eps) >= limit:
            break
    return eps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("packed_dir")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--out", required=True, help="dataset root directory")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--robot-type", default="prius_sim_ackermann")
    p.add_argument("--limit", type=int, default=0, help="first N episodes only")
    p.add_argument("--terminal-copies", type=int, default=0,
                   help="extra short episodes replaying each parking episode's "
                        "final N seconds (--terminal-tail-s), plus hold padding. "
                        "The plan (sim_vla_plan.md section 7.2) prescribes "
                        "oversampling the terminal deceleration ~10x because the "
                        "ramp is ~1%% of frames; duplicating frames INSIDE an "
                        "episode would corrupt chunk temporal structure, so the "
                        "oversample is expressed as standalone approach-and-stop "
                        "episodes instead. 2 copies of a 6 s tail on a ~40 s "
                        "episode weights the stop region to roughly 10%%.")
    p.add_argument("--terminal-tail-s", type=float, default=6.0)
    p.add_argument("--hold-frames", type=int, default=15,
                   help="zero-action copies of the final frame appended to every "
                        "episode. The corpus ends the moment the oracle arrives, "
                        "so it contains no 'stay stopped' behaviour at all, and "
                        "LeRobot pads end-of-episode action chunks by repeating "
                        "the LAST action — a small forward creep. Trained on "
                        "that, the closed-loop policy drove straight through "
                        "the bay and 6 m past the back wall. 15 frames = 1.5 s "
                        "of 'at the bay, speed zero -> stay at zero'. This is "
                        "data, not a rule in the control path, so criterion (d) "
                        "is untouched. 0 disables.")
    p.add_argument("--no-videos", action="store_true",
                   help="store frames as images instead of encoded video")
    p.add_argument("--video-backend", default="pyav",
                   help="decoder used when the dataset is read back. LeRobot "
                        "defaults to torchcodec, which dlopen()s FFmpeg 5+ "
                        "shared libraries; on Ubuntu 22.04 (FFmpeg 4.4, "
                        "libavutil.so.56) that fails at import with a bare "
                        "OSError and the dataset looks corrupt when it is fine. "
                        "PyAV ships its own FFmpeg and needs no system package, "
                        "so it is the default here.")
    args = p.parse_args()

    import numpy as np
    from PIL import Image

    LeRobotDataset, modpath = import_lerobot()
    print(f"using {modpath}")

    packed = os.path.abspath(args.packed_dir)
    eps = load_episodes(packed, args.limit)
    if not eps:
        print(f"no packaged episodes in {packed}", file=sys.stderr)
        return 1
    total_rows = sum(len(r) for _, _, _, r, _ in eps)
    total_dropped = sum(d for *_, d in eps)
    print(f"{len(eps)} episodes, {total_rows} frames at {args.fps} Hz")
    # One per episode is expected and structural. More than that means the
    # resampler produced rows it could not label, which is a data problem.
    if total_dropped:
        print(f"  dropped {total_dropped} row(s) with no action label "
              f"({total_dropped / len(eps):.2f} per episode; 1.00 is the "
              "structural end-of-episode row)")
    if total_dropped > 2 * len(eps):
        print("  more unlabelled rows than end-of-episode ones — check the "
              "resampler before training on this", file=sys.stderr)
        return 1

    first_img = None
    for _, base, _, rows, _ in eps:
        cand = os.path.join(base, rows[0]["frame"])
        if os.path.exists(cand):
            first_img = np.asarray(Image.open(cand).convert("RGB"))
            break
    if first_img is None:
        print("no readable frame in the packaged corpus", file=sys.stderr)
        return 1
    h, w, c = first_img.shape
    print(f"image {w}x{h}x{c}")

    features = {
        CAM_KEY: {"dtype": "video" if not args.no_videos else "image",
                  "shape": (h, w, c), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (3,),
                              "names": ["speed_mps", "yaw_rate_radps",
                                        "steer_rad"]},
        "action": {"dtype": "float32", "shape": (3,),
                   "names": ["dx_m", "dy_m", "dyaw_rad"]},
    }

    root = Path(os.path.abspath(args.out))
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, root=root,
        robot_type=args.robot_type, features=features,
        use_videos=not args.no_videos, video_backend=args.video_backend,
    )

    index_rows = []
    lerobot_ep = 0   # actual dataset episode index; terminal copies shift it
    for ep_i, (name, base, meta, rows, _drop) in enumerate(eps):
        task = meta.get("instruction") or ""
        hold = []
        # Pad only episodes that actually END stopped (state[0] is speed in
        # m/s; arrivals ramp to ~0). Cruise episodes are cut by duration while
        # still moving — zero-padding those would teach a spontaneous stop at
        # speed, the exact inverse of the creep bug the padding exists to fix.
        if args.hold_frames > 0 and rows and rows[-1]["state"][0] < 0.3:
            last = rows[-1]
            hold = [{"frame": last["frame"],
                     "state": [0.0, 0.0, 0.0],
                     "action": [0.0, 0.0, 0.0]}] * args.hold_frames
        for r in rows + hold:
            img_p = os.path.join(base, r["frame"])
            img = np.asarray(Image.open(img_p).convert("RGB"))
            frame = {
                CAM_KEY: img,
                "observation.state": np.asarray(r["state"], dtype=np.float32),
                "action": np.asarray(r["action"], dtype=np.float32),
            }
            # Verified against lerobot 0.4.4: `add_frame(self, frame)` takes no
            # `task` argument, so the instruction rides inside the frame dict.
            # Older releases accepted `task=` positionally; the TypeError fallback
            # covers those without making the common path depend on an exception.
            try:
                ds.add_frame({**frame, "task": task})
            except TypeError:
                ds.add_frame(frame, task=task)
        # 0.4.4: save_episode(self, episode_data=None, parallel_encoding=...)
        try:
            ds.save_episode()
        except TypeError:
            ds.save_episode(task=task)
        # The full episode's dataset index is claimed HERE, before any terminal
        # copies are saved after it — the index row must carry the index the
        # dataset actually assigned, not the counter's value after the copies.
        index_rows.append({
            "lerobot_episode_index": lerobot_ep,
            "packed_episode": name,
            "instruction": task,
            "cf_group_id": meta.get("cf_group_id"),
            "cf_variant_id": meta.get("cf_variant_id"),
            "cf_axis": meta.get("cf_axis"),
            "intent_id": meta.get("intent_id"),
            "intent_slots": meta.get("intent_slots"),
            "start_pose_key": meta.get("start_pose_key"),
            "n_frames": len(rows),
        })
        lerobot_ep += 1

        # Terminal oversample: emit the approach-and-stop tail as its own
        # episode(s). Same frames, same actions, correct chunk structure.
        n_tail = int(args.terminal_tail_s * args.fps)
        if (args.terminal_copies > 0 and meta.get("intent_id") == "park_bay"
                and len(rows) > n_tail + 5):
            for _ in range(args.terminal_copies):
                for r in rows[-n_tail:] + hold:
                    img = np.asarray(Image.open(
                        os.path.join(base, r["frame"])).convert("RGB"))
                    frame = {
                        CAM_KEY: img,
                        "observation.state": np.asarray(r["state"],
                                                        dtype=np.float32),
                        "action": np.asarray(r["action"], dtype=np.float32),
                    }
                    try:
                        ds.add_frame({**frame, "task": task})
                    except TypeError:
                        ds.add_frame(frame, task=task)
                try:
                    ds.save_episode()
                except TypeError:
                    ds.save_episode(task=task)
                index_rows.append({
                    "lerobot_episode_index": lerobot_ep,
                    "packed_episode": name, "terminal_copy": True,
                    "instruction": task,
                    "cf_group_id": meta.get("cf_group_id"),
                    "n_frames": n_tail + len(hold),
                })
                lerobot_ep += 1

        if (ep_i + 1) % 10 == 0:
            print(f"  {ep_i + 1}/{len(eps)} episodes")

    # LeRobot has nowhere to put this, and every ablation needs it.
    side = root / "nav_vla_index.jsonl"
    with open(side, "w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nwrote {root}")
    print(f"  {len(eps)} episodes, {total_rows} frames")
    print(f"  counterfactual index -> {side}")
    print("\nverify by loading it back:")
    print(f"  python3 -c \"from {modpath} import LeRobotDataset; "
          f"d=LeRobotDataset('{args.repo_id}', root='{root}', "
          f"video_backend='{args.video_backend}'); "
          "print(len(d), d[0].keys())\"")
    print(f"\nthis dataset must be read with video_backend='{args.video_backend}'; "
          "the default decoder will not open it on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
