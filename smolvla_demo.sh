#!/bin/bash
# SmolVLA 데모 스택 원클릭 기동 — 터미널 하나로 전부 실행.
#
#   ./smolvla_demo.sh                # v6 데모 스택 전체 기동 (시뮬+서버+브리지+내비+내레이터+채팅GUI)
#   ./smolvla_demo.sh --ckpt models/ckpt_v7_60k   # 다른 체크포인트로
#   NAVVLA_VENV=~/venv/navvla ./smolvla_demo.sh    # 정책 서버 venv 지정
#   ./smolvla_demo.sh --no-gui       # 채팅 GUI 없이 (문장은 ros2 topic pub로)
#   ./smolvla_demo.sh check          # 실행 전 환경만 점검
#   ./smolvla_demo.sh down           # 전체 종료 + gz 로그 정리
#
# 로그: eval_out/demo/*.log
WS="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOGD=$WS/eval_out/demo
CKPT=${NAVVLA_CKPT:-$WS/models/ckpt_v6_60k}
VENV=${NAVVLA_VENV:-$HOME/venv/navvla}
POLICY_PYTHON=${NAVVLA_PYTHON:-$VENV/bin/python}
GUI=1

ROS_SETUP=/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash

preflight() {
  local failed=0
  [ -f "$ROS_SETUP" ] || { echo "[demo] ROS 환경 없음: $ROS_SETUP"; failed=1; }
  # gz는 Jazzy 벤더 패키지라 ROS를 source해야 PATH에 뜬다 — 서브셸에서 검사
  ( [ -f "$ROS_SETUP" ] && . "$ROS_SETUP" >/dev/null 2>&1
    command -v gz >/dev/null ) || {
    echo "[demo] Gazebo(gz) 없음: sudo apt install ros-jazzy-ros-gz"
    failed=1
  }
  [ -f "$WS/install/setup.bash" ] || {
    echo "[demo] 워크스페이스 미빌드: ./setup_smolvla_demo.sh 실행 필요"
    failed=1
  }
  [ -x "$POLICY_PYTHON" ] || {
    echo "[demo] 정책 Python 없음: $POLICY_PYTHON"
    echo "       NAVVLA_VENV 또는 NAVVLA_PYTHON으로 경로 지정 가능"
    failed=1
  }
  [ -f "$CKPT/model.safetensors" ] || {
    echo "[demo] 체크포인트 없음: $CKPT/model.safetensors"
    echo "       --ckpt 또는 NAVVLA_CKPT로 경로 지정 가능"
    failed=1
  }
  if [ "$failed" -eq 0 ]; then
    "$POLICY_PYTHON" -c 'import lerobot, msgpack, zmq, torch' 2>/dev/null || {
      echo "[demo] venv에 lerobot/msgpack/pyzmq/torch 중 일부가 없습니다."
      failed=1
    }
  fi
  [ "$failed" -eq 0 ]
}

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

if [ "${1:-}" = "check" ]; then
  CKPT="$(readlink -f "$CKPT")"
  preflight && echo "[demo] 실행 환경 정상"
  exit $?
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --no-gui) GUI=0; shift ;;
    *) echo "알 수 없는 인자: $1"; exit 2 ;;
  esac
done
CKPT="$(readlink -f "$CKPT")"
preflight || exit 2
source "$ROS_SETUP"
source "$WS/install/setup.bash"
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
setsid nohup "$POLICY_PYTHON" -u \
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
