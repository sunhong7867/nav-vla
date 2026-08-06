#!/bin/bash
# Poll the lab server until a training run's checkpoint appears or the
# trainer dies. Tolerates network outages (failed ssh = skip round).
# Usage: watch_ckpt.sh <run_name> <step_dir e.g. 060000> [max_rounds=60]
RUN=$1; STEP=$2; ROUNDS=${3:-60}
SRV=autolab_sw@115.145.211.157
for i in $(seq 1 $ROUNDS); do
  R=$(ssh -o ConnectTimeout=15 $SRV \
    "ls ~/sunhong/nav-vla/runs/$RUN/checkpoints 2>/dev/null | grep -c $STEP; pgrep -cf 'lerobot-trai[n]'" 2>/dev/null)
  CKPT=$(echo "$R" | sed -n 1p); ALIVE=$(echo "$R" | sed -n 2p)
  if [ "$CKPT" = "1" ]; then echo "${RUN}_${STEP}_READY"; exit 0; fi
  if [ "$ALIVE" = "0" ] && [ -n "$ALIVE" ]; then
    sleep 60
    C2=$(ssh -o ConnectTimeout=15 $SRV "ls ~/sunhong/nav-vla/runs/$RUN/checkpoints 2>/dev/null | grep -c $STEP" 2>/dev/null)
    [ "$C2" = "1" ] && { echo "${RUN}_${STEP}_READY"; exit 0; }
    echo "${RUN}_TRAIN_DIED"; exit 1
  fi
  sleep 600
done
echo "${RUN}_WATCH_TIMEOUT"; exit 2
