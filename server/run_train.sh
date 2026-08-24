#!/bin/bash
cd ~/sunhong/nav-vla
GPU=1 BATCH=8 STEPS=40000 WORKERS=8 SAVE_FREQ=5000 \
  bash code/train_smolvla.sh \
    /home/autolab_sw/sunhong/nav-vla/data/lerobot/v2 \
    sunhong/navvla_sim_v2 \
    navvla_smolvla_v2
echo TRAIN_EXITED
