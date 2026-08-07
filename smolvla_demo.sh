#!/bin/bash
# SmolVLA 데모 스택 원클릭 기동 — 터미널 하나로 전부 실행.
#
#   ./smolvla_demo.sh                # v6 데모 스택 전체 기동 (시뮬+서버+브리지+내비+내레이터+채팅GUI)
#   ./smolvla_demo.sh --ckpt models/ckpt_v7_60k   # 다른 체크포인트로
#   ./smolvla_demo.sh --no-gui       # 채팅 GUI 없이 (문장은 ros2 topic pub로)
#   ./smolvla_demo.sh down           # 전체 종료 + gz 로그 정리
#
# 로그: eval_out/demo/*.log
WS="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOGD=$WS/eval_out/demo
CKPT=$WS/models/ckpt_v6_60k
GUI=1

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

kill_stack() {
  # pkill 자기매칭 방지: 패턴을 쪼개서 자기 자신(cmdline에 전체 문자열 없음)만 피함
  pat='gz si';            pkill -9 -f "${pat}m"          2>/dev/null
  pat='driving_sim.launc';pkill -f "${pat}h"             2>/dev/null
  pat='vla_policy_serve'; pkill -f "${pat}r"             2>/dev/null
  pat='vla_bridge_nod';   pkill -f "${pat}e"             2>/dev/null
  pat='navigator_nod';    pkill -f "${pat}e"             2>/dev/null
  pat='vla_narrato';      pkill -f "${pat}r.py"          2>/dev/null
  pat='chat_gui_nod';     pkill -f "${pat}e"             2>/dev/null
  pat='parameter_bridg';  pkill -f "${pat}e"             2>/dev/null
  # Qt GUI는 이벤트 루프가 파이썬 시그널을 삼켜 SIGTERM으로 안 죽는 경우가
  # 있음 — 잠시 후에도 살아있으면 강제 종료
  sleep 2
  pat='chat_gui_nod';     pkill -9 -f "${pat}e"          2>/dev/null
  pat='vla_narrato';      pkill -9 -f "${pat}r.py"       2>/dev/null
}

if [ "${1:-}" = "down" ]; then
  echo "[demo] 전체 종료 중..."
  kill_stack
  sleep 3
  rm -rf ~/.gz/sim/log
  echo "[demo] 종료 완료 (gz 로그 정리됨)"
  exit 0
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --no-gui) GUI=0; shift ;;
    *) echo "알 수 없는 인자: $1"; exit 2 ;;
  esac
done
CKPT="$(readlink -f "$CKPT")"
[ -f "$CKPT/model.safetensors" ] || { echo "체크포인트 없음: $CKPT"; exit 2; }
mkdir -p "$LOGD"

echo "[demo] 이전 스택 정리 + DDS 초기화..."
kill_stack
ros2 daemon stop >/dev/null 2>&1
sleep 4
rm -rf ~/.gz/sim/log
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /dev/shm/fastdds_* /dev/shm/sem.fastdds_* /dev/shm/cyclonedds* 2>/dev/null

echo "[demo] 1/5 시뮬레이터 기동 (약 25초)..."
setsid nohup ros2 launch simulation_pkg driving_sim.launch.py use_camera:=true \
  use_perception_pipeline:=false use_driver:=false use_policy:=false \
  > "$LOGD/sim.log" 2>&1 < /dev/null &
sleep 25

echo "[demo] 2/5 정책 서버 기동: $CKPT"
setsid nohup /home/sh/venv/navvla/bin/python -u \
  "$WS/src/nav_vla_pkg/scripts/vla_policy_server.py" \
  --checkpoint "$CKPT" --endpoint ipc:///tmp/nav_vla.sock --warmup 4 \
  > "$LOGD/serve.log" 2>&1 < /dev/null &
for _ in $(seq 1 40); do
  grep -q 'serving on' "$LOGD/serve.log" 2>/dev/null && break
  sleep 3
done
grep -q 'serving on' "$LOGD/serve.log" || { echo "정책 서버 기동 실패 — $LOGD/serve.log 확인"; exit 3; }

echo "[demo] 3/5 브리지 기동..."
setsid nohup ros2 run nav_vla_pkg vla_bridge_node --ros-args -p use_sim_time:=true \
  -p image_topic:=/camera/image_raw -p max_speed:=3.2 -p speed_slew:=0.08 \
  > "$LOGD/bridge.log" 2>&1 < /dev/null &

echo "[demo] 4/5 내비게이터 + 내레이터 기동..."
setsid nohup ros2 run nav_vla_pkg navigator_node > "$LOGD/navigator.log" 2>&1 < /dev/null &
setsid nohup python3 "$WS/src/nav_vla_pkg/scripts/vla_narrator.py" > "$LOGD/narrator.log" 2>&1 < /dev/null &
sleep 6

NPUB=$(timeout 10 ros2 topic info /cmd_vel --verbose 2>/dev/null | grep -c 'Endpoint type: PUBLISHER')
echo "[demo] /cmd_vel 퍼블리셔 수: $NPUB (2 초과면 유령 노드 — down 후 재기동)"
[ "$NPUB" -le 2 ] || { echo "[demo] 유령 퍼블리셔 감지, 중단"; exit 4; }

if [ "$GUI" = 1 ]; then
  echo "[demo] 5/5 채팅 GUI 기동..."
  setsid nohup ros2 run nav_vla_pkg chat_gui_node --ros-args -p control_backend:=smolvla \
    > "$LOGD/chat_gui.log" 2>&1 < /dev/null &
else
  echo "[demo] 5/5 GUI 생략 — 문장 직접 발행 예:"
  echo "  ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable \\"
  echo "    /vla/instruction std_msgs/String \"{data: 'Start driving in the outer lane, at a fast speed.'}\""
fi

echo
echo "[demo] 스택 기동 완료. 예시 명령: \"start drive\" / \"천천히 안쪽 차선으로\" / \"go to T2\" / \"멈춰\""
echo "[demo] 종료: ./smolvla_demo.sh down"
