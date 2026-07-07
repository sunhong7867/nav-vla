"""Stage-A VLA behavior cloning for nav-vla.

Learns a vision+goal+relative-pose -> action policy:
pi(camera image, goal zone, lane/direct mode, rel_goal) -> cmd_vel.
This is the "vision + control" half of the hierarchical VLA (qwen handles
language -> zone upstream). Trained by imitating the oracle's recorded actions.

Caveat: the oracle acted on ground-truth pose, so the forward camera only
partially observes the state -> BC is approximate. v0 baseline.

Usage:
    python3 src/nav_vla/train/train_stage_a.py
    python3 src/nav_vla/train/train_stage_a.py --epochs 30 --img 128
    python3 src/nav_vla/train/train_stage_a.py --sessions session_20260702_lane1,2
"""

import argparse
import glob
import json
import math
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DATA_ROOT = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/data")
ZONE_MAP = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml")
OUT_DIR = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/train/checkpoints")
DIRECT_ONLY_ZONES = {
    "IN",
    "OUT(통과직전)",
    "OUT(통과직후)",
    "Slot1",
    "Slot2",
    "Slot3",
    "Slot4",
}
LANE_VOCAB = {"default": 0, "lane1": 1, "lane2": 2, "direct": 3}
STATE_DIM = 5
POS_SCALE = 50.0

# action normalization (so linear/angular contribute comparably to the loss)
LIN_SCALE = 2.5
ANG_SCALE = 1.5


def load_zone_vocab():
    import yaml
    with open(ZONE_MAP, "r", encoding="utf-8") as f:
        zones = (yaml.safe_load(f) or {}).get("zones", {})
    names = sorted(zones)
    return {n: i for i, n in enumerate(names)}


def lane_index(meta, record, nav_mode):
    if nav_mode == "direct" or meta.get("nav_mode") == "direct":
        return LANE_VOCAB["direct"]
    lane = str(record.get("goal_lane") or meta.get("goal_lane") or "default")
    return LANE_VOCAB.get(lane, LANE_VOCAB["default"])


def is_lane_changing_record(record):
    """Frames captured mid lane-change. These teach a lane2->lane1 swerve at the
    spawn, which contaminates the spawn behaviour of a lane-agnostic policy."""
    lane = record.get("lane")
    if not isinstance(lane, str):
        return False
    try:
        info = json.loads(lane)
    except (json.JSONDecodeError, TypeError):
        return False
    return bool(info.get("is_lane_changing"))


def state_features(record):
    rel = record.get("rel_goal") or {}
    pose = record.get("pose") or {}
    dx = float(rel.get("dx", 0.0))
    dy = float(rel.get("dy", 0.0))
    dist = float(record.get("dist_to_goal", (dx * dx + dy * dy) ** 0.5))
    yaw = float(pose.get("yaw", 0.0))
    return [
        dx / POS_SCALE,
        dy / POS_SCALE,
        dist / POS_SCALE,
        math.sin(yaw),
        math.cos(yaw),
    ]


def parse_sessions(text, data_root):
    if not text or str(text).strip().lower() in {"all", "*"}:
        return None
    text = str(text).strip()
    if os.path.isdir(os.path.join(data_root, text)):
        return {text}
    if ";" in text:
        return {s.strip() for s in text.split(";") if s.strip()}
    return {s.strip() for s in str(text).split(",") if s.strip()}


def scan_episodes(data_root, vocab, success_only=True, nav_mode="lane", sessions=None,
                  drop_lane_changes=True):
    """Return episode-level samples so train/val can split by rollout."""
    episodes = []
    dropped_lc = 0
    for ep in sorted(glob.glob(os.path.join(data_root, "session_*", "ep_*"))):
        if sessions is not None and os.path.basename(os.path.dirname(ep)) not in sessions:
            continue
        mp = os.path.join(ep, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        if success_only and not meta.get("success"):
            continue
        if nav_mode != "all" and meta.get("nav_mode", "lane") != nav_mode:
            continue
        zname = meta.get("goal_zone")
        if zname not in vocab:
            continue
        if nav_mode == "lane" and zname in DIRECT_ONLY_ZONES:
            continue
        zidx = vocab[zname]
        sp = os.path.join(ep, "steps.jsonl")
        if not os.path.exists(sp):
            continue
        samples = []
        for line in open(sp):
            r = json.loads(line)
            if drop_lane_changes and is_lane_changing_record(r):
                dropped_lc += 1
                continue
            img = os.path.join(ep, r["image"])
            if os.path.exists(img):
                lidx = lane_index(meta, r, nav_mode)
                samples.append((
                    img,
                    zidx,
                    lidx,
                    state_features(r),
                    r["cmd"]["linear"],
                    r["cmd"]["angular"],
                ))
        if samples:
            episodes.append({"path": ep, "zone": zname, "samples": samples})
    if drop_lane_changes and dropped_lc:
        print(f"dropped {dropped_lc} lane-changing frames (spawn-swerve contamination)")
    return episodes


def flatten_episodes(episodes):
    return [sample for ep in episodes for sample in ep["samples"]]


class EpisodeDataset(Dataset):
    def __init__(self, samples, img_size, train=True):
        self.samples = samples
        aug = [transforms.ColorJitter(0.2, 0.2, 0.2)] if train else []
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, zidx, lidx, state, lin, ang = self.samples[i]
        img = self.tf(Image.open(path).convert("RGB"))
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor([lin / LIN_SCALE, ang / ANG_SCALE], dtype=torch.float32)
        return img, zidx, lidx, state, action


class VisionGoalPolicy(nn.Module):
    def __init__(
        self,
        n_zones,
        n_lanes=len(LANE_VOCAB),
        emb=32,
        state_dim=STATE_DIM,
        pretrained=False,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        bb = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])  # -> (B,512,1,1)
        self.zone_emb = nn.Embedding(n_zones, emb)
        self.lane_emb = nn.Embedding(n_lanes, 8)
        self.head = nn.Sequential(
            nn.Linear(512 + emb + 8 + state_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, img, zidx, lidx, state):
        f = self.backbone(img).flatten(1)
        z = self.zone_emb(zidx)
        lane = self.lane_emb(lidx)
        return self.head(torch.cat([f, z, lane, state], dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", default=DATA_ROOT)
    ap.add_argument(
        "--sessions",
        default="all",
        help="Comma-separated session folder names to train on, or 'all'.",
    )
    ap.add_argument("--nav-mode", choices=["lane", "direct", "all"], default="lane")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--keep-lane-changes",
        action="store_true",
        help="Keep mid lane-change frames (default: drop them to avoid spawn-swerve).",
    )
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = load_zone_vocab()
    sessions = parse_sessions(args.sessions, args.data)
    episodes = scan_episodes(
        args.data, vocab, nav_mode=args.nav_mode, sessions=sessions,
        drop_lane_changes=not args.keep_lane_changes,
    )
    random.Random(args.seed).shuffle(episodes)
    samples = flatten_episodes(episodes)
    print(
        f"device={dev} | nav_mode={args.nav_mode} | zones={len(vocab)} "
        f"| sessions={sorted(sessions) if sessions else 'all'} "
        f"| episodes={len(episodes)} "
        f"| samples={len(samples)}"
    )
    if len(samples) < 50 or len(episodes) < 2:
        print("Not enough samples."); return

    n_val_ep = max(1, int(args.val_ratio * len(episodes)))
    val_eps = episodes[:n_val_ep]
    train_eps = episodes[n_val_ep:]
    train_samples = flatten_episodes(train_eps)
    val_samples = flatten_episodes(val_eps)
    print(
        f"train episodes={len(train_eps)} samples={len(train_samples)} | "
        f"val episodes={len(val_eps)} samples={len(val_samples)}"
    )
    if args.epochs <= 0:
        print("scan only: --epochs <= 0, training skipped.")
        return

    tr = EpisodeDataset(train_samples, args.img, train=True)
    va = EpisodeDataset(val_samples, args.img, train=False)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = VisionGoalPolicy(len(vocab), pretrained=args.pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.SmoothL1Loss()

    os.makedirs(OUT_DIR, exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        tloss = 0.0
        for img, z, lane, state, a in tl:
            img = img.to(dev)
            z = z.to(dev)
            lane = lane.to(dev)
            state = state.to(dev)
            a = a.to(dev)
            opt.zero_grad()
            loss = lossf(model(img, z, lane, state), a)
            loss.backward(); opt.step()
            tloss += loss.item() * img.size(0)
        tloss /= len(train_samples)

        model.eval()
        vloss = 0.0
        mae_lin = mae_ang = 0.0
        with torch.no_grad():
            for img, z, lane, state, a in vl:
                img = img.to(dev)
                z = z.to(dev)
                lane = lane.to(dev)
                state = state.to(dev)
                a = a.to(dev)
                p = model(img, z, lane, state)
                vloss += lossf(p, a).item() * img.size(0)
                err = (p - a).abs().mean(0)
                mae_lin += err[0].item() * img.size(0)
                mae_ang += err[1].item() * img.size(0)
        vloss /= len(val_samples); mae_lin /= len(val_samples); mae_ang /= len(val_samples)
        # de-normalize MAE to physical units
        print(f"ep {ep+1:2d} | train {tloss:.4f} | val {vloss:.4f} | "
              f"MAE lin {mae_lin*LIN_SCALE:.3f} m/s, ang {mae_ang*ANG_SCALE:.3f} rad/s")
        if vloss < best:
            best = vloss
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "lane_vocab": LANE_VOCAB,
                        "state_dim": STATE_DIM, "pos_scale": POS_SCALE,
                        "img": args.img, "lin_scale": LIN_SCALE, "ang_scale": ANG_SCALE},
                       os.path.join(OUT_DIR, "stage_a.pt"))
    print(f"done. best val {best:.4f}. saved {OUT_DIR}/stage_a.pt")


if __name__ == "__main__":
    main()
