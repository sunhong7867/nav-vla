#!/bin/bash
# v9 장애물 대조쌍 — 파일럿 수집 (4그룹 12ep, ~15분) + finalize.
# 본수집은 collect_v9_full.sh (파일럿 통과 후).
#
# 그룹당 3변형: v0 장애물 없음 / v1 내 차선(스크립트 회피) / v2 반대 차선.
# 스택: 시뮬(카메라만) + route_oracle + episode_recorder + collector.
WS=/home/sh/ROS2_project/nav-vla
OUT=$WS/eval_out/v9_pilot
DATA=$WS/src/nav_vla_pkg/data_v9
GROUPS=${V9_GROUPS:-4}
PREFIX=${V9_PREFIX:-v9p_}
SEED=${V9_SEED:-20260904}
PACK=${V9_PACK:-$DATA/packed_v9_pilot}
mkdir -p "$OUT" "$DATA"
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash

# RAM preflight (스킬 규칙: 우회 금지)
AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge 3 ] || { echo "[v9] 가용 RAM ${AVAIL}G < 3G — 앱을 닫고 재시도"; exit 4; }

pat='gz si';             pkill -9 -f "${pat}m"        2>/dev/null
pat='driving_sim.launc'; pkill -f "${pat}h"           2>/dev/null
pat='route_oracle_nod';  pkill -f "${pat}e"           2>/dev/null
pat='episode_recorde';   pkill -f "${pat}r"           2>/dev/null
pat='collect_corpu';     pkill -f "${pat}s.py"        2>/dev/null
for _ in $(seq 1 10); do
  pgrep -f "gz si[m]|route_oracle_nod[e]|episode_recorde[r]" >/dev/null || break
  sleep 1
done
ros2 daemon stop >/dev/null 2>&1
sleep 3
rm -rf ~/.gz/sim/log
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /dev/shm/fastdds_* /dev/shm/sem.fastdds_* 2>/dev/null

echo "[v9] 시뮬 기동..."
setsid nohup ros2 launch simulation_pkg driving_sim.launch.py \
  use_camera:=true use_perception_pipeline:=false use_driver:=false \
  use_policy:=false use_debug_visualizers:=false use_vla_camera:=false \
  > "$OUT/sim.log" 2>&1 < /dev/null &
for _ in $(seq 1 24); do
  N=$(timeout 5 ros2 topic list 2>/dev/null | grep -c "camera/image_raw")
  [ "${N:-0}" -ge 1 ] && break
  sleep 5
done
[ "${N:-0}" -ge 1 ] || { echo "[v9] SIM_FAIL"; exit 3; }

echo "[v9] 오라클+레코더 기동..."
setsid nohup ros2 run nav_vla_pkg route_oracle_node --ros-args \
  -p use_sim_time:=true > "$OUT/oracle.log" 2>&1 < /dev/null &
setsid nohup ros2 run nav_vla_pkg episode_recorder_node --ros-args \
  -p out_dir:="$DATA" -p use_sim_time:=true > "$OUT/recorder.log" 2>&1 < /dev/null &
sleep 5

echo "[v9] 수집 시작 ($(date +%H:%M)) — ${GROUPS}그룹..."
python3 $WS/src/nav_vla_pkg/scripts/collect_corpus.py \
  --driver oracle --groups 0 --speed-groups 0 --floor-groups 0 \
  --ring-groups 0 --obstacle-groups "$GROUPS" \
  --group-prefix "$PREFIX" --seed "$SEED" \
  > "$OUT/collect.log" 2>&1
RC=$?
echo "[v9] 수집 종료 rc=$RC ($(date +%H:%M)) — 요약:"
tail -12 "$OUT/collect.log"

pat='gz si';             pkill -9 -f "${pat}m"        2>/dev/null
pat='driving_sim.launc'; pkill -f "${pat}h"           2>/dev/null
pat='route_oracle_nod';  pkill -f "${pat}e"           2>/dev/null
pat='episode_recorde';   pkill -f "${pat}r"           2>/dev/null
sleep 3
rm -rf ~/.gz/sim/log

SESS=$(ls -dt "$DATA"/session_* 2>/dev/null | head -1)
if [ -n "$SESS" ] && [ "$RC" -eq 0 ]; then
  echo "[v9] finalize: $SESS"
  PACK_OUT=$PACK bash $WS/src/nav_vla_pkg/scripts/finalize_corpus.sh "$SESS" \
    > "$OUT/finalize.log" 2>&1 \
    && echo "[v9] finalize 완료 -> $PACK" \
    || echo "[v9] finalize 실패 — $OUT/finalize.log 확인"
fi
echo "[v9] DONE"
