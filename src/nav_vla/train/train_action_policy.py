"""Train high-level nav-vla action policy.

Learns:
    English command text + current lane -> action JSON steps

This is not a low-level driving model. It predicts the same high-level plan
schema used by chat_gui_node, so the stable navigator/lane-follow pipeline can
execute the result.

Usage:
    python3 src/nav_vla/train/train_action_policy.py
    python3 src/nav_vla/train/train_action_policy.py --epochs 100
    python3 src/nav_vla/train/train_action_policy.py --data path/to/samples.jsonl
"""

import argparse
import json
import os
import random
import sys
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from nav_vla.action_policy_model import (  # noqa: E402
    MAX_STEPS,
    ActionPolicyNet,
    build_vocabs,
    encode_text,
    outputs_to_plan,
    plan_to_targets,
)


DATA_ROOT = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/data_actions")
OUT_DIR = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/train/checkpoints")


def latest_samples_path(root):
    paths = sorted(glob(os.path.join(root, "session_*", "samples.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no samples.jsonl under {root}")
    return paths[-1]


def read_samples(path):
    samples = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


class ActionDataset(Dataset):
    def __init__(self, samples, vocab, zone_vocab):
        self.samples = samples
        self.vocab = vocab
        self.zone_vocab = zone_vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        current_lane = sample.get("current_lane", "lane2")
        ids = encode_text(sample.get("text", ""), current_lane, self.vocab)
        targets = plan_to_targets(sample.get("teacher", {}), self.zone_vocab)
        return {
            "ids": ids,
            "count": targets["count"],
            "actions": targets["actions"],
            "zones": targets["zones"],
            "lanes": targets["lanes"],
            "sample": sample,
        }


def collate(batch):
    flat_ids = []
    offsets = []
    cursor = 0
    for item in batch:
        offsets.append(cursor)
        flat_ids.extend(item["ids"])
        cursor += len(item["ids"])
    return {
        "token_ids": torch.tensor(flat_ids, dtype=torch.long),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "count": torch.tensor([item["count"] for item in batch], dtype=torch.long),
        "actions": torch.tensor([item["actions"] for item in batch], dtype=torch.long),
        "zones": torch.tensor([item["zones"] for item in batch], dtype=torch.long),
        "lanes": torch.tensor([item["lanes"] for item in batch], dtype=torch.long),
        "samples": [item["sample"] for item in batch],
    }


def compute_loss(outputs, batch, lossf):
    loss = lossf(outputs["count"], batch["count"])
    count_targets = batch["count"]
    for i in range(MAX_STEPS):
        active = count_targets >= i
        if not torch.any(active):
            continue
        loss = loss + lossf(outputs["actions"][i][active], batch["actions"][active, i])
        loss = loss + lossf(outputs["zones"][i][active], batch["zones"][active, i])
        loss = loss + lossf(outputs["lanes"][i][active], batch["lanes"][active, i])
    return loss


def plan_key(plan):
    return json.dumps({"steps": plan.get("steps", [])}, sort_keys=True, ensure_ascii=False)


def target_plan(sample):
    return {"steps": sample.get("teacher", {}).get("steps", [])}


def evaluate(model, loader, device, zone_vocab, lossf):
    model.eval()
    total_loss = 0.0
    total = 0
    exact = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch["token_ids"], batch["offsets"])
            loss = compute_loss(outputs, batch, lossf)
            total_loss += loss.item() * len(batch["samples"])
            total += len(batch["samples"])
            for row, sample in enumerate(batch["samples"]):
                one = {
                    "count": outputs["count"][row:row + 1],
                    "actions": [head[row:row + 1] for head in outputs["actions"]],
                    "zones": [head[row:row + 1] for head in outputs["zones"]],
                    "lanes": [head[row:row + 1] for head in outputs["lanes"]],
                }
                pred = outputs_to_plan(one, zone_vocab)
                if plan_key(pred) == plan_key(target_plan(sample)):
                    exact += 1
    return total_loss / max(1, total), exact / max(1, total)


def move_batch(batch, device):
    moved = dict(batch)
    for key in ("token_ids", "offsets", "count", "actions", "zones", "lanes"):
        moved[key] = moved[key].to(device)
    return moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", help="samples.jsonl path. Empty uses latest data_actions session.")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--final-all-epochs", type=int, default=80)
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "action_policy.pt"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = args.data or latest_samples_path(args.data_root)
    samples = read_samples(data_path)
    random.shuffle(samples)
    vocab, zone_vocab = build_vocabs(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    train_ds = ActionDataset(train_samples, vocab, zone_vocab)
    val_ds = ActionDataset(val_samples, vocab, zone_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActionPolicyNet(
        len(vocab),
        len(zone_vocab),
        emb_dim=args.emb_dim,
        hidden=args.hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()

    print(
        f"device={device} samples={len(samples)} train={len(train_samples)} "
        f"val={len(val_samples)} vocab={len(vocab)} zones={len(zone_vocab)} data={data_path}"
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_acc = -1.0
    best_loss = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad()
            outputs = model(batch["token_ids"], batch["offsets"])
            loss = compute_loss(outputs, batch, lossf)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch["samples"])
        train_loss /= max(1, len(train_samples))
        val_loss, val_acc = evaluate(model, val_loader, device, zone_vocab, lossf)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "zone_vocab": zone_vocab,
                    "emb_dim": args.emb_dim,
                    "hidden": args.hidden,
                    "data": data_path,
                },
                args.out,
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"ep {epoch:03d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | exact {val_acc * 100:.1f}%"
            )
    print(f"best validation exact {best_acc * 100:.1f}% val_loss {best_loss:.4f}")

    if args.final_all_epochs > 0:
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
        all_ds = ActionDataset(samples, vocab, zone_vocab)
        all_loader = DataLoader(all_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
        for epoch in range(1, args.final_all_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in all_loader:
                batch = move_batch(batch, device)
                opt.zero_grad()
                outputs = model(batch["token_ids"], batch["offsets"])
                loss = compute_loss(outputs, batch, lossf)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(batch["samples"])
            total_loss /= max(1, len(samples))
            if epoch == 1 or epoch % 20 == 0 or epoch == args.final_all_epochs:
                all_loss, all_acc = evaluate(model, all_loader, device, zone_vocab, lossf)
                print(
                    f"final {epoch:03d} | train_all {total_loss:.4f} | "
                    f"exact_all {all_acc * 100:.1f}%"
                )
        torch.save(
            {
                "model": model.state_dict(),
                "vocab": vocab,
                "zone_vocab": zone_vocab,
                "emb_dim": args.emb_dim,
                "hidden": args.hidden,
                "data": data_path,
                "best_val_exact": best_acc,
                "best_val_loss": best_loss,
            },
            args.out,
        )
    print(f"done. saved {args.out}")


if __name__ == "__main__":
    main()
