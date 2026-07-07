#!/usr/bin/env bash
# Lane-change (crossover) data collection for nav-vla.
#
# Drives the zones in track order and flips the target lane every episode, so each
# short hop to the next zone carries exactly one lane change on that segment -- not
# a full lap between changes. The loop start zone ROTATES each round, so every
# segment sees BOTH change directions over rounds (a fixed order would only ever
# change lane one way per segment).
#
# Spawn is lane2, so the alternation is deterministic: 2->1, 1->2, 2->1, ...
# (never random, so current lane != target lane always -- no accidental no-op).
#   round0: Start(spawn) -> M2(1) -> T2(2) -> M3(1) -> T3(2) -> T4(1) -> Start(2)
#   round1: rotated start -> same segments, opposite directions, ...
#
# Run in THREE terminals (same as normal data collection), each after `navvla`:
#   # Terminal 1 -- bare sim + camera
#   ros2 launch simulation_pkg driving_sim.launch.py use_camera:=true
#   # Terminal 2 -- oracle navigator (performs the crossover)
#   ros2 run nav_vla navigator_node
#   # Terminal 3 -- this script
#   bash src/nav_vla/scripts/collect_lane_change.sh
#
# Output: a new src/nav_vla/data/session_<stamp>/ with lane-changing episodes.
#
# NOTE: train_stage_a.py drops lane-changing frames by DEFAULT. To actually learn
# the crossover, train the combined data with --keep-lane-changes.
#
# Note: no `set -u` -- ROS setup.bash references unset vars (AMENT_TRACE_SETUP_FILES).
set -eo pipefail

ROUNDS="${ROUNDS:-30}"
# Track order (counter-clockwise from the lane2 spawn at Start). Keep this order.
ZONES="${ZONES:-M2,T2,M3,T3,T4,Start}"

source /opt/ros/jazzy/setup.bash
source "$(dirname "$0")/../../../install/local_setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-32}"

echo "lane-change collection: goal_lane=alternate (2->1,1->2,...) order=sequential rounds=${ROUNDS} zones=${ZONES}"
ros2 run nav_vla data_engine_node --ros-args \
  -p nav_mode:=lane \
  -p goal_lane:=alternate \
  -p rounds:="${ROUNDS}" \
  -p shuffle:=false \
  -p zones:="${ZONES}"
