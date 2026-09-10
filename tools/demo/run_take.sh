#!/bin/bash
# Reasoning demo (option B): r11 drives+narrates, supervisor makes the
# discrete calls. Presets: pass (16m, full overtake+return) / watch (5.6m,
# close safety stop). See docs/reasoning_project_report.md 6.1.
WS=/home/sh/ROS2_project/nav-vla
PRESET=${1:-pass}
source /opt/ros/jazzy/setup.bash; source $WS/install/setup.bash
setsid nohup python3 $WS/tools/demo/avoid_supervisor.py --preset "$PRESET" \
  --seed-x 14.7 --seed-y -9.5 > $WS/eval_out/demo/supervisor.log 2>&1 < /dev/null &
sleep 3
python3 $WS/tools/demo/demo_take.py
