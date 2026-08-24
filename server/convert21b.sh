#!/bin/bash
set -x
cd ~/sunhong/nav-vla
rm -rf data/lerobot/v21b runs/navvla_smolvla_v21b
./venv/bin/python code/to_lerobot.py data/packed_v2 \
  --repo-id sunhong/navvla_sim_v21b --out data/lerobot/v21b --hold-frames 30
echo CONVERT21B_DONE
GPU=0 BATCH=8 STEPS=40000 WORKERS=8 SAVE_FREQ=5000 \
  bash code/train_smolvla.sh \
    /home/autolab_sw/sunhong/nav-vla/data/lerobot/v21b \
    sunhong/navvla_sim_v21b \
    navvla_smolvla_v21b
echo TRAIN21B_EXITED
