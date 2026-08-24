#!/bin/bash
# v2c: parking-only + hold 45 + terminal x2 — the anti-creep dataset.
set -x
cd ~/sunhong/nav-vla
rm -rf data/lerobot/v2c runs/navvla_smolvla_v2c
./venv/bin/python code/to_lerobot.py data/packed_v2_parking \
  --repo-id sunhong/navvla_sim_v2c --out data/lerobot/v2c \
  --hold-frames 45 --terminal-copies 2
echo CONVERT_V2C_DONE
GPU=0 BATCH=8 STEPS=40000 WORKERS=8 SAVE_FREQ=5000 \
  bash code/train_smolvla.sh \
    /home/autolab_sw/sunhong/nav-vla/data/lerobot/v2c \
    sunhong/navvla_sim_v2c \
    navvla_smolvla_v2c
echo TRAIN_V2C_EXITED
