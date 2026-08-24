#!/bin/bash
# Continue v21b for another 40K steps on the 3090. Mixed-data run at 2.0 epochs
# had learned ring driving but diluted parking (effective parking exposure 1.3
# epochs vs the 3.0 that produced v2b's clean 8/8). Doubling steps lifts parking
# to ~2.5 without a new dataset.
set -x
cd ~/sunhong/nav-vla
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./venv/bin/lerobot-train \
  --config_path=runs/navvla_smolvla_v21b/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps=80000 \
  2>&1 | tee -a logs/navvla_smolvla_v21b_resume.log
echo RESUME21B_EXITED
