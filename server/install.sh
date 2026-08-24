#!/bin/bash
# lerobot environment for nav-vla, isolated from the system python.
#
# The system interpreter carries torch 1.8.0a0+unknown -- a 2021 build that is
# not a normal release and that lerobot (>=2.2) cannot use. It belongs to
# someone else's work on this shared account, so it is left untouched.
#
# Driver 570.195.03 supports CUDA 12.x, so cu124 wheels rather than the cu118
# toolkit that happens to be installed system-wide.
set -x
cd ~/sunhong/nav-vla || exit 1
V=./venv/bin/python

$V -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || exit 1
$V -m pip install lerobot || exit 1

$V -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
$V -c "import lerobot; print('lerobot', getattr(lerobot,'__version__','?'), lerobot.__file__)"
echo "INSTALL_DONE"
