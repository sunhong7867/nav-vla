#!/bin/bash
cd ~/sunhong/nav-vla
GPU=0 BATCH=8 STEPS=40000 WORKERS=8 SAVE_FREQ=5000 \
  bash code/train_smolvla.sh \
    /home/autolab_sw/sunhong/nav-vla/data/lerobot/v21 \
    sunhong/navvla_sim_v21 \
    navvla_smolvla_v21
echo TRAIN21_EXITED
