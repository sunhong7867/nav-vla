#!/usr/bin/env python3
"""Co-train SmolVLA action + reasoning head (custom loop; lerobot 0.4.4).

`lerobot-train` cannot express a second loss, so this is a minimal loop
around ReasoningSmolVLAPolicy: LeRobotDataset + reasoning sidecar ->
preprocessor pipeline -> policy.forward (flow + CE) -> AdamW.

Run on the lab server (GPU1 = 3090) from ~/sunhong/nav-vla::

    CUDA_VISIBLE_DEVICES=1 ./venv/bin/python code/train_reasoning.py \
        --dataset data/lerobot/v3y \
        --base-checkpoint runs/navvla_smolvla_v3y/checkpoints/last/pretrained_model \
        --out runs/navvla_reasoning_r1 --steps 20000

The base checkpoint provides action competence and the normalizer stats;
the vision encoder is frozen (it already adapted during the base run) and
the trunk/expert train at a low LR while the reasoning head catches up at
a higher one. Every checkpoint dir is served exactly like a stock one —
the action path is untouched by the subclass.
"""

import argparse
import math
import os
import shutil
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--lr-trunk", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=250)
    ap.add_argument("--ce-weight", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--sample-decode", action="store_true",
                    help="greedy-decode 2 samples at every save (slow)")
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists — refusing to overwrite "
                         "(same rule as train_smolvla.sh)")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from reasoning_vla import ReasoningDataset, ReasoningSmolVLAPolicy
    from reasoning_vla.reasoning_dataset import collate_with_text
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from safetensors.torch import load_file

    # PreTrainedConfig dispatches on the "type" field in config.json;
    # calling SmolVLAConfig.from_pretrained directly chokes on it (draccus)
    cfg = PreTrainedConfig.from_pretrained(args.base_checkpoint)
    cfg.freeze_vision_encoder = True   # already adapted in the base run
    cfg.device = "cuda"
    policy = ReasoningSmolVLAPolicy(cfg, ce_weight=args.ce_weight)
    sd = load_file(os.path.join(args.base_checkpoint, "model.safetensors"))
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    # only the new head may be missing; anything else is a real mismatch
    bad = [k for k in missing if not k.startswith("reasoning_head")]
    assert not bad and not unexpected, (bad, unexpected)
    # head init from the checkpoint's lm_head (the ctor used base weights)
    with torch.no_grad():
        policy.reasoning_head.weight.copy_(
            sd["model.vlm_with_expert.vlm.lm_head.weight"])
    policy = policy.to(device).train()

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M")

    fps = 10
    delta_timestamps = {
        "action": [i / fps for i in range(cfg.chunk_size)]}
    # pyav: the lab box has FFmpeg 4.4, torchcodec needs 5+ (same reason
    # train_smolvla.sh passes --dataset.video_backend=pyav)
    ds = LeRobotDataset("navvla/reasoning", root=args.dataset,
                        delta_timestamps=delta_timestamps,
                        video_backend="pyav")
    sidecar = os.path.join(args.dataset, "reasoning_labels.jsonl")
    rds = ReasoningDataset(ds, sidecar, seed=args.seed)
    print(f"dataset: {len(rds)} frames, sidecar {rds.n_rows} segment rows")
    dl = torch.utils.data.DataLoader(
        rds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_with_text,
        pin_memory=True, drop_last=True, persistent_workers=True)

    preproc, _ = make_pre_post_processors(policy.config,
                                          args.base_checkpoint)

    head_params = list(policy.reasoning_head.parameters())
    head_ids = {id(p) for p in head_params}
    trunk_params = [p for p in policy.parameters()
                    if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": trunk_params, "lr": args.lr_trunk},
         {"params": head_params, "lr": args.lr_head}],
        weight_decay=1e-10, betas=(0.9, 0.95))

    def lr_scale(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * p))

    def save(step):
        d = os.path.join(args.out, "checkpoints", f"{step:06d}",
                         "pretrained_model")
        os.makedirs(d, exist_ok=True)
        policy.save_pretrained(d)
        # serving needs the processor/normalizer files next to the model
        for f in os.listdir(args.base_checkpoint):
            if f not in ("model.safetensors", "config.json") and \
                    os.path.isfile(os.path.join(args.base_checkpoint, f)):
                shutil.copy2(os.path.join(args.base_checkpoint, f),
                             os.path.join(d, f))
        link = os.path.join(args.out, "checkpoints", "last")
        if os.path.islink(link):
            os.remove(link)
        os.symlink(f"{step:06d}", link)
        print(f"saved {d}")

    step, t0, t_log = 0, time.time(), time.time()
    it = iter(dl)
    sample_batch = None
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        reasoning = batch.pop("reasoning")
        batch = preproc(batch)
        batch = {k: (v.to(device, non_blocking=True)
                     if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        batch["reasoning"] = reasoning
        if sample_batch is None and args.sample_decode:
            sample_batch = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    sample_batch[k] = v[:2].clone()
                elif isinstance(v, (list, tuple)):
                    sample_batch[k] = v[:2]
                else:
                    sample_batch[k] = v

        for g, base in zip(opt.param_groups,
                           (args.lr_trunk, args.lr_head)):
            g["lr"] = base * lr_scale(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ld = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            args.grad_clip)
        opt.step()
        step += 1

        if step % args.log_freq == 0:
            dt = (time.time() - t_log) / args.log_freq
            t_log = time.time()
            eta_h = dt * (args.steps - step) / 3600
            print(f"step {step:6d}  loss {ld['loss']:.4f}  "
                  f"action {ld.get('action_loss', 0):.4f}  "
                  f"ce {ld.get('reasoning_ce', 0):.3f}  "
                  f"{dt:.3f}s/step  eta {eta_h:.2f}h", flush=True)
        if step % args.save_freq == 0 or step == args.steps:
            save(step)
            if args.sample_decode and sample_batch is not None:
                policy.eval()
                try:
                    for s in policy.generate_reasoning(sample_batch):
                        print(f"  decode: {s}")
                finally:
                    policy.train()

    print(f"done in {(time.time() - t0) / 3600:.2f}h")


if __name__ == "__main__":
    main()
