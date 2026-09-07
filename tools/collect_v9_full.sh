#!/bin/bash
# v9 장애물 대조쌍 — 본수집: train 40그룹 + heldout 8그룹 (~2h) + finalize.
# 파일럿(2026-09-04)에서 검증된 경로. 클록 500 Hz — 100 Hz는 pose 스탬프를
# 10 ms로 양자화해 액션 라벨에 p50 5.5 cm 보간 오차를 넣는다(게이트 2 cm).
# 그룹 수는 고정값: 파일럿에서 V9_GROUPS 환경변수 유입 사고(1000그룹)가
# 있어 환경 오버라이드를 제거했다.
WS=/home/sh/ROS2_project/nav-vla
OUT=$WS/eval_out/v9_full
DATA=$WS/src/nav_vla_pkg/data_v9
TRAIN_GROUPS=40
HELDOUT_GROUPS=8
mkdir -p "$OUT" "$DATA"
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash

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

echo "[v9] 시뮬 기동 (clock 500 Hz)..."
setsid nohup ros2 launch simulation_pkg driving_sim.launch.py \
  use_camera:=true use_perception_pipeline:=false use_driver:=false \
  use_policy:=false use_debug_visualizers:=false use_vla_camera:=false \
  clock_hz:=500.0 \
  > "$OUT/sim.log" 2>&1 < /dev/null &
for _ in $(seq 1 24); do
  N=$(timeout 5 ros2 topic list 2>/dev/null | grep -c "camera/image_raw")
  [ "${N:-0}" -ge 1 ] && break
  sleep 5
done
[ "${N:-0}" -ge 1 ] || { echo "[v9] SIM_FAIL"; exit 3; }

echo "[v9] 장애물 4대 상주 스폰 (파킹 스팟, 이후 teleport만)..."
python3 - <<'PYEOF' || { echo "[v9] OBSTACLE_SPAWN_FAIL"; exit 5; }
import sys
sys.path.insert(0, "/home/sh/ROS2_project/nav-vla/src/simulation_pkg")
from simulation_pkg import basic
models = ["hatchback_green", "hatchback_red", "hatchback_blue",
          "hatchback_yellow"]
for i, m in enumerate(models):
    basic.load_model(f"v9_{m}", m, (60.0 + 8.0 * i, 60.0, 0.01265,
                                    0.0, 0.0, 0.0), skip_if_exists=True)
print("spawned", len(models))
PYEOF

echo "[v9] 오라클 기동..."
setsid nohup ros2 run nav_vla_pkg route_oracle_node --ros-args \
  -p use_sim_time:=true > "$OUT/oracle.log" 2>&1 < /dev/null &
sleep 3

run_batch() {  # $1=groups $2=prefix $3=split $4=seed $5=pack_dir
  # 배치마다 레코더를 새로 띄운다: 레코더 세션은 프로세스당 하나라, 한
  # 세션에 두 배치를 섞으면 finalize가 train/heldout을 가르지 못한다.
  setsid nohup ros2 run nav_vla_pkg episode_recorder_node --ros-args \
    -p out_dir:="$DATA" -p use_sim_time:=true \
    > "$OUT/recorder_$2.log" 2>&1 < /dev/null &
  sleep 5
  echo "[v9] 수집: $2 ${1}그룹 split=$3 ($(date +%H:%M))..."
  python3 $WS/src/nav_vla_pkg/scripts/collect_corpus.py \
    --driver oracle --groups 0 --speed-groups 0 --floor-groups 0 \
    --ring-groups 0 --obstacle-groups "$1" \
    --group-prefix "$2" --split "$3" --seed "$4" --out-dir "$DATA" \
    > "$OUT/collect_$2.log" 2>&1
  RC=$?
  tail -6 "$OUT/collect_$2.log"
  pat='episode_recorde'; pkill -f "${pat}r" 2>/dev/null
  for _ in $(seq 1 8); do pgrep -f "episode_recorde[r]" >/dev/null || break; sleep 1; done
  # finalize 대상은 ep_* 를 담은 레코더 세션 (plan.json 쪽은 컬렉터 세션)
  SESS=$(ls -dt "$DATA"/session_*/ep_0000 2>/dev/null | head -1 | xargs dirname)
  if [ -n "$SESS" ] && [ "$RC" -eq 0 ]; then
    echo "[v9] finalize: $SESS -> $5"
    PACK_OUT=$5 bash $WS/src/nav_vla_pkg/scripts/finalize_corpus.sh "$SESS" \
      > "$OUT/finalize_$2.log" 2>&1 \
      && echo "[v9] finalize OK" || echo "[v9] finalize FAIL — $OUT/finalize_$2.log"
  fi
  return $RC
}

run_batch $TRAIN_GROUPS  v9t_ train   20260905 "$DATA/packed_v9"
run_batch $HELDOUT_GROUPS v9h_ heldout 20260906 "$DATA/packed_v9_heldout"

pat='gz si';             pkill -9 -f "${pat}m"        2>/dev/null
pat='driving_sim.launc'; pkill -f "${pat}h"           2>/dev/null
pat='route_oracle_nod';  pkill -f "${pat}e"           2>/dev/null
pat='episode_recorde';   pkill -f "${pat}r"           2>/dev/null
sleep 3
rm -rf ~/.gz/sim/log
echo "[v9] DONE ($(date +%H:%M))"
