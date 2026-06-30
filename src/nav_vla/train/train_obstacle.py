"""Train a camera -> "obstacle ahead" classifier for nav-vla (learned reactive VLA).

Uses frames auto-labeled by the lidar (obstacle_data_collector). The obstacle is
visible in the camera, so this learns well (contrast: localization could not).
Binary classification with class-imbalance handling (clear >> obstacle).

Usage:
    python3 src/nav_vla/train/train_obstacle.py
    python3 src/nav_vla/train/train_obstacle.py --epochs 15 --img 128
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

DATA_ROOT = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/data_obstacle")
OUT_DIR = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/train/checkpoints")


def scan_samples(root):
    samples = []
    for sess in sorted(glob.glob(os.path.join(root, "session_*"))):
        lf = os.path.join(sess, "labels.jsonl")
        if not os.path.exists(lf):
            continue
        for line in open(lf):
            r = json.loads(line)
            p = os.path.join(sess, r["image"])
            if os.path.exists(p):
                samples.append((p, int(r["label"])))
    return samples


class ObstacleDataset(Dataset):
    def __init__(self, samples, img, train=True):
        self.samples = samples
        aug = [transforms.ColorJitter(0.2, 0.2, 0.2),
               transforms.RandomResizedCrop(img, scale=(0.8, 1.0))] if train else \
              [transforms.Resize((img, img))]
        self.tf = transforms.Compose([
            *aug, transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.tf(Image.open(p).convert("RGB")), torch.tensor([float(y)])


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", default=DATA_ROOT)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    samples = scan_samples(args.data)
    pos = sum(y for _, y in samples)
    neg = len(samples) - pos
    print(f"device={dev} | samples={len(samples)} (obstacle={pos}, clear={neg})")
    if pos < 20 or neg < 20:
        print("Need more data of BOTH classes (>=20 each)."); return

    full = ObstacleDataset(samples, args.img, train=True)
    n_val = max(1, int(0.15 * len(full)))
    tr, va = random_split(full, [len(full) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    va.dataset = ObstacleDataset(samples, args.img, train=False)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4)

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # weight the rarer class so recall on obstacles stays high
    pos_weight = torch.tensor([neg / max(1, pos)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(OUT_DIR, exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            lossf(model(x), y).backward()
            opt.step()
        # eval: accuracy + recall/precision on the obstacle class
        model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for x, y in vl:
                x = x.to(dev)
                pred = (torch.sigmoid(model(x)).cpu() > 0.5).float()
                for pr, gt in zip(pred, y):
                    if gt > 0.5:
                        tp += int(pr > 0.5); fn += int(pr <= 0.5)
                    else:
                        fp += int(pr > 0.5); tn += int(pr <= 0.5)
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"ep {ep+1:2d} | acc {acc:.3f} | obstacle recall {rec:.3f} "
              f"prec {prec:.3f} f1 {f1:.3f}")
        if f1 > best:
            best = f1
            torch.save({"model": model.state_dict(), "img": args.img},
                       os.path.join(OUT_DIR, "obstacle.pt"))
    print(f"done. best f1 {best:.3f}. saved {OUT_DIR}/obstacle.pt")


if __name__ == "__main__":
    main()
