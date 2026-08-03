#!/bin/bash
# SmolVLA fine-tune on the nav-vla sim corpus. Runs on the GPU server.
#
#   bash train_smolvla.sh <dataset_root> <repo_id> [job_name]
#
# GPU index is not a detail: on this box `nvidia-smi` reports the 4070 Ti SUPER
# at index 0 and the 3090 at index 1, which is the reverse of what the servers
# doc used to claim. The 3090's 24 GB is the one worth using.
#
# `--dataset.video_backend=pyav` is required, not a preference. LeRobot's default
# decoder (torchcodec) dlopen()s FFmpeg 5+ shared libraries; this machine is
# Ubuntu 22.04 with FFmpeg 4.4 (libavutil.so.56), so the default fails at load
# with a bare OSError and a perfectly good dataset looks corrupt.
set -euo pipefail

ROOT="${1:?dataset root}"
REPO="${2:?repo id}"
JOB="${3:-navvla_smolvla_$(date +%Y%m%d_%H%M)}"

BASE="$HOME/sunhong/nav-vla"
V="$BASE/venv/bin"
OUT="$BASE/runs/$JOB"

# 3090. Set explicitly so a concurrent user on the other card cannot push this
# run onto a 16 GB device mid-training.
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
# Measured, not projected. The plan quotes 3.82 GiB at batch 32, but that figure
# is for the frozen-trunk configuration; training the vision encoder too makes
# 403M parameters learnable, and `empty_cameras=2` sends THREE 512x512 images
# through the encoder per sample (one real, two blank padding for the cameras
# smolvla_base expects). batch 32 exhausted all 23.57 GiB of the 3090.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Do NOT create $OUT: lerobot-train refuses to start when the output directory
# already exists and --resume is false, and creating it here just to hold the
# log was enough to trip that. The log lives in logs/ instead.
mkdir -p "$BASE/logs"
LOG="$BASE/logs/$JOB.log"
echo "job      $JOB"
echo "dataset  $ROOT ($REPO)"
echo "output   $OUT"
echo "log      $LOG"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

# smolvla_base was pretrained with three cameras (camera1/2/3) and this rig has
# one. Two knobs, and they must be used together:
#   --rename_map        maps our `observation.images.front` onto `camera1`
#   --policy.empty_cameras=2  pads the other two with blanks
# Without the rename the run dies with "Missing features: camera1, camera2,
# camera3 / Extra features: front"; without empty_cameras the two it still
# expects are simply absent.
#
# CONSEQUENCE FOR SERVING: the trained checkpoint's config names the camera
# `observation.images.camera1`, not `front`. `vla_policy_server.py` reads the key
# out of the loaded config rather than assuming, so the two cannot drift apart.
#
# `--policy.push_to_hub=false` because it defaults to true and then demands a
# `policy.repo_id` to upload to. Checkpoints stay on this machine and are pulled
# over ssh; nothing here should be publishing to the Hub by default.
#
# `--policy.path` alone, never alongside `--policy.type`: lerobot rejects the
# pair outright ("Cannot specify both"). The path already determines the type,
# and it is what loads the pretrained weights.
#
# Starting from `lerobot/smolvla_base` rather than random init: the 450M trunk
# has seen far more vision-language than 250 driving episodes could teach it,
# and the plan budgets no time for pre-training.
#
# The vision encoder is NOT frozen. Freezing it plus train_expert_only is the
# configuration in which the trunk never adapts to the outdoor driving domain,
# and that is precisely the condition under which language becomes decoration —
# the failure this whole corpus exists to avoid. If VRAM forces it, that is a
# result to report, not a default to adopt.
"$V/lerobot-train" \
  --dataset.repo_id="$REPO" \
  --dataset.root="$ROOT" \
  --dataset.video_backend=pyav \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_amp="${AMP:-true}" \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --policy.empty_cameras=2 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --output_dir="$OUT" \
  --job_name="$JOB" \
  --batch_size="${BATCH:-8}" \
  --steps="${STEPS:-40000}" \
  --num_workers="${WORKERS:-8}" \
  --save_freq="${SAVE_FREQ:-5000}" \
  --log_freq=200 \
  --seed=1000 \
  2>&1 | tee -a "$LOG"
