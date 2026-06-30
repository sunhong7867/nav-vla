"""Stage-A VLA behavior cloning for nav-vla.

Learns a vision+goal -> action policy:  pi(camera image, goal zone) -> cmd_vel.
This is the "vision + control" half of the hierarchical VLA (qwen handles
language -> zone upstream). Trained by imitating the oracle's recorded actions.

Caveat: the oracle acted on ground-truth pose, so the forward camera only
partially observes the state -> BC is approximate. v0 baseline.

Usage:
    python3 src/nav_vla/train/train_stage_a.py
    python3 src/nav_vla/train/train_stage_a.py --epochs 30 --img 128
"""

import argparse
import glob
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

DATA_ROOT = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/data")
ZONE_MAP = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml")
OUT_DIR = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/train/checkpoints")

# action normalization (so linear/angular contribute comparably to the loss)
LIN_SCALE = 2.5
ANG_SCALE = 1.5


def load_zone_vocab():
    import yaml
    with open(ZONE_MAP, "r", encoding="utf-8") as f:
        zones = (yaml.safe_load(f) or {}).get("zones", {})
    names = sorted(zones)
    return {n: i for i, n in enumerate(names)}


def scan_samples(data_root, vocab, success_only=True):
    """Return list of (image_path, zone_idx, lin, ang)."""
    samples = []
    for ep in sorted(glob.glob(os.path.join(data_root, "session_*", "ep_*"))):
        mp = os.path.join(ep, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        if success_only and not meta.get("success"):
            continue
        zname = meta.get("goal_zone")
        if zname not in vocab:
            continue
        zidx = vocab[zname]
        sp = os.path.join(ep, "steps.jsonl")
        if not os.path.exists(sp):
            continue
        for line in open(sp):
            r = json.loads(line)
            img = os.path.join(ep, r["image"])
            if os.path.exists(img):
                samples.append((img, zidx, r["cmd"]["linear"], r["cmd"]["angular"]))
    return samples


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
        path, zidx, lin, ang = self.samples[i]
        img = self.tf(Image.open(path).convert("RGB"))
        action = torch.tensor([lin / LIN_SCALE, ang / ANG_SCALE], dtype=torch.float32)
        return img, zidx, action


class VisionGoalPolicy(nn.Module):
    def __init__(self, n_zones, emb=32):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])  # -> (B,512,1,1)
        self.zone_emb = nn.Embedding(n_zones, emb)
        self.head = nn.Sequential(
            nn.Linear(512 + emb, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, img, zidx):
        f = self.backbone(img).flatten(1)
        z = self.zone_emb(zidx)
        return self.head(torch.cat([f, z], dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", default=DATA_ROOT)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = load_zone_vocab()
    samples = scan_samples(args.data, vocab)
    print(f"device={dev} | zones={len(vocab)} | samples={len(samples)}")
    if len(samples) < 50:
        print("Not enough samples."); return

    full = EpisodeDataset(samples, args.img, train=True)
    n_val = max(1, int(0.15 * len(full)))
    n_tr = len(full) - n_val
    tr, va = random_split(full, [n_tr, n_val],
                          generator=torch.Generator().manual_seed(0))
    va.dataset = EpisodeDataset(samples, args.img, train=False)  # no aug for val
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4)

    model = VisionGoalPolicy(len(vocab)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.SmoothL1Loss()

    os.makedirs(OUT_DIR, exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        tloss = 0.0
        for img, z, a in tl:
            img, z, a = img.to(dev), z.to(dev), a.to(dev)
            opt.zero_grad()
            loss = lossf(model(img, z), a)
            loss.backward(); opt.step()
            tloss += loss.item() * img.size(0)
        tloss /= n_tr

        model.eval()
        vloss = 0.0
        mae_lin = mae_ang = 0.0
        with torch.no_grad():
            for img, z, a in vl:
                img, z, a = img.to(dev), z.to(dev), a.to(dev)
                p = model(img, z)
                vloss += lossf(p, a).item() * img.size(0)
                err = (p - a).abs().mean(0)
                mae_lin += err[0].item() * img.size(0)
                mae_ang += err[1].item() * img.size(0)
        vloss /= n_val; mae_lin /= n_val; mae_ang /= n_val
        # de-normalize MAE to physical units
        print(f"ep {ep+1:2d} | train {tloss:.4f} | val {vloss:.4f} | "
              f"MAE lin {mae_lin*LIN_SCALE:.3f} m/s, ang {mae_ang*ANG_SCALE:.3f} rad/s")
        if vloss < best:
            best = vloss
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "img": args.img, "lin_scale": LIN_SCALE, "ang_scale": ANG_SCALE},
                       os.path.join(OUT_DIR, "stage_a.pt"))
    print(f"done. best val {best:.4f}. saved {OUT_DIR}/stage_a.pt")


if __name__ == "__main__":
    main()
