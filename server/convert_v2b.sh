#!/bin/bash
set -x
cd ~/sunhong/nav-vla
rm -rf data/lerobot/v2b runs/navvla_smolvla_v2b
./venv/bin/python code/to_lerobot.py data/packed_v2_parking \
  --repo-id sunhong/navvla_sim_v2b --out data/lerobot/v2b --hold-frames 15
echo CONVERT_V2B_DONE
GPU=1 BATCH=8 STEPS=40000 WORKERS=8 SAVE_FREQ=5000 \
  bash code/train_smolvla.sh \
    /home/autolab_sw/sunhong/nav-vla/data/lerobot/v2b \
    sunhong/navvla_sim_v2b \
    navvla_smolvla_v2b
echo TRAIN_V2B_EXITED
