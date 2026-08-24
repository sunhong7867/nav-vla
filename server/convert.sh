#!/bin/bash
set -x
cd ~/sunhong/nav-vla
rm -rf data/lerobot/v2
./venv/bin/python code/to_lerobot.py data/packed_v2 \
  --repo-id sunhong/navvla_sim_v2 --out data/lerobot/v2
echo CONVERT_DONE
