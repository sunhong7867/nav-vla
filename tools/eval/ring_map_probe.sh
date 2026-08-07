#!/bin/bash
# Full-ring mapping probe for a policy checkpoint.
# Usage: ring_map_probe.sh <checkpoint_dir> <out_prefix>
#   EXTRA_BRIDGE_ARGS="-p curv_boost:=1.2" 처럼 환경변수로 브리지 파라미터 추가 가능
# Brings up bare sim + server + bridge, runs the standard inner/outer 115 s
# counterfactual cruise from the canonical start, writes <out_prefix>_map.json,
# then tears the server/bridge down (sim left up for chained probes).
set -e
CKPT=$1
PFX=$2
[ -d "$CKPT" ] || { echo "usage: ring_map_probe.sh <ckpt_dir> <out_prefix>"; exit 2; }
WS=/home/sh/ROS2_project/nav-vla
OUT=$WS/eval_out
mkdir -p $OUT
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash

pat_g='gz si'; pkill -9 -f "${pat_g}m" || true; pat_l='driving_sim.launc'; pkill -f "${pat_l}h" || true
pat_s='vla_policy_serve'; pkill -f "${pat_s}r" || true; pat_b='vla_bridge_nod'; pkill -f "${pat_b}e" || true
pat_pb='parameter_bridg'; pkill -f "${pat_pb}e" || true
sleep 3
rm -rf ~/.gz/sim/log

setsid nohup ros2 launch simulation_pkg driving_sim.launch.py use_camera:=true \
  use_perception_pipeline:=false use_driver:=false use_policy:=false \
  > $OUT/${PFX}_sim.log 2>&1 < /dev/null &
sleep 25

setsid nohup /home/sh/venv/navvla/bin/python -u \
  $WS/src/nav_vla_pkg/scripts/vla_policy_server.py \
  --checkpoint $CKPT --endpoint ipc:///tmp/nav_vla.sock --warmup 4 \
  > $OUT/${PFX}_serve.log 2>&1 < /dev/null &
until grep -q 'serving on' $OUT/${PFX}_serve.log 2>/dev/null; do sleep 3; done

setsid nohup ros2 run nav_vla_pkg vla_bridge_node --ros-args -p use_sim_time:=true \
  -p max_speed:=3.2 -p image_topic:=/camera/image_raw -p speed_slew:=0.08 \
  ${EXTRA_BRIDGE_ARGS:-} \
  > $OUT/${PFX}_bridge.log 2>&1 < /dev/null &
sleep 4

NPUB=$(timeout 10 ros2 topic info /cmd_vel --verbose 2>/dev/null | grep -c 'Endpoint type: PUBLISHER')
echo "cmd_vel publishers: $NPUB"
[ "$NPUB" -le 2 ] || { echo GHOST; exit 20; }

python3 $WS/src/nav_vla_pkg/scripts/probe_policy_counterfactual.py \
  --x -1.49 --y 24.55 --yaw -1.571 --duration 115 \
  --say-a "Start driving in the inner lane, at a ${SPEED_WORD:-fast} speed." \
  --say-b "Start driving in the outer lane, at a ${SPEED_WORD:-fast} speed." \
  --out $OUT/${PFX}_map.json > $OUT/${PFX}_probe.log 2>&1

pat_s='vla_policy_serve'; pkill -f "${pat_s}r" || true; pat_b='vla_bridge_nod'; pkill -f "${pat_b}e" || true
echo "PROBE_DONE ${PFX} -> $OUT/${PFX}_map.json"
