#!/bin/bash
set -x
cd ~/sunhong/nav-vla
rm -rf data/lerobot/v21
./venv/bin/python code/to_lerobot.py data/packed_v2 \
  --repo-id sunhong/navvla_sim_v21 --out data/lerobot/v21
echo CONVERT21_DONE
