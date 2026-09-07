"""LeRobotDataset wrapper that attaches per-frame reasoning text.

Sidecar contract (built by scripts/build_reasoning_sidecar.py): a jsonl file,
one row per (lerobot episode, frame range):

    {"episode_index": int, "f0": int, "f1": int, "variants": ["...", ...]}

``variants[0]`` is the deterministic skeleton; the rest are fact-checked
paraphrases. ``__getitem__`` samples one uniformly — a fresh draw every
epoch, which is deliberate anti-memorization augmentation. Frames not
covered by any row (should not happen) get an empty string, which the
policy treats as "no reasoning supervision for this sample".
"""

import json
import random

import torch


class ReasoningDataset(torch.utils.data.Dataset):
    def __init__(self, lerobot_dataset, sidecar_path, seed=0,
                 key="reasoning"):
        self.ds = lerobot_dataset
        self.key = key
        self.rng = random.Random(seed)
        self.by_ep = {}
        n_rows = 0
        with open(sidecar_path) as f:
            for line in f:
                r = json.loads(line)
                self.by_ep.setdefault(r["episode_index"], []).append(
                    (r["f0"], r["f1"], r["variants"]))
                n_rows += 1
        for segs in self.by_ep.values():
            segs.sort()
        self.n_rows = n_rows

    def __len__(self):
        return len(self.ds)

    def _lookup(self, ep, frame):
        for f0, f1, variants in self.by_ep.get(ep, ()):
            if f0 <= frame <= f1:
                return variants
        return None

    def __getitem__(self, idx):
        item = self.ds[idx]
        ep = int(item["episode_index"])
        frame = int(item["frame_index"])
        variants = self._lookup(ep, frame)
        item[self.key] = self.rng.choice(variants) if variants else ""
        return item


def collate_with_text(batch, text_keys=("reasoning", "task")):
    """default_collate for tensors; plain lists for the text fields."""
    from torch.utils.data._utils.collate import default_collate
    texts = {k: [b.pop(k, "") for b in batch] for k in text_keys
             if any(k in b for b in batch)}
    out = default_collate(batch)
    out.update(texts)
    return out
