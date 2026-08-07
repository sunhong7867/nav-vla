#!/bin/bash
# Thor 실차 VLA 스택 원클릭 기동 — smolvla_demo.sh의 실차판.
#
#   ./thor_car_demo.sh wheels-up     # 바퀴 든 검증 모드: 어댑터+시리얼만 기동
#                                    #   → 별도 안내에 따라 wheels_up_test 실행
#   STEER_SIGN=+1 ./thor_car_demo.sh # 전체 스택 (wheels-up 통과 후에만!)
#   ./thor_car_demo.sh check         # 실행 전 환경 점검
#   ./thor_car_demo.sh down          # 전체 종료
#
# 전체 모드는 STEER_SIGN(+1|-1)을 명시해야만 기동한다 — wheels-up 검증을
# 건너뛰고 주행하는 실수를 구조적으로 막기 위해서다 (마스터플랜 D3).
# 속도 상한은 CAP_SPEED_INT(기본 60 = 기존 스택 v_base의 60%)로 보수적.
#
# 로그: eval_out/car/*.log
WS="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOGD=$WS/eval_out/car
CKPT=${NAVVLA_CKPT:-$WS/models/ckpt_v6_60k}
VENV=${NAVVLA_VENV:-$HOME/venv/navvla}
POLICY_PYTHON=${NAVVLA_PYTHON:-$VENV/bin/python}
SERIAL_PORT=${SERIAL_PORT:-/dev/ttyACM0}
CAP=${CAP_SPEED_INT:-60}
ROS_SETUP=/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash

preflight() {
  local failed=0
  [ -f "$ROS_SETUP" ] || { echo "[car] ROS 없음: $ROS_SETUP"; failed=1; }
  [ -f "$WS/install/setup.bash" ] || { echo "[car] 미빌드 — colcon build 필요"; failed=1; }
  [ -e "$SERIAL_PORT" ] || echo "[car] ⚠ $SERIAL_PORT 없음 — 시리얼은 dry-run이 된다"
  [ -x "$POLICY_PYTHON" ] || { echo "[car] venv 없음: $POLICY_PYTHON"; failed=1; }
  [ -f "$CKPT/model.safetensors" ] || { echo "[car] 체크포인트 없음: $CKPT"; failed=1; }
  [ "$failed" -eq 0 ]
}

kill_stack() {
  pat='vla_policy_serve'; pkill -f "${pat}r"          2>/dev/null
  pat='vla_bridge_nod';   pkill -f "${pat}e"          2>/dev/null
  pat='thor_vla_bringup'; pkill -f "${pat}.launch"    2>/dev/null
  pat='camera_publisher_nod';       pkill -f "${pat}e" 2>/dev/null
  pat='cmd_vel_motion_adapter_nod'; pkill -f "${pat}e" 2>/dev/null
  pat='serial_sender_nod';          pkill -f "${pat}e" 2>/dev/null
  sleep 2
}

case "${1:-}" in
  down)
    echo "[car] 전체 종료..."
    kill_stack
    echo "[car] 종료 완료 (시리얼 워치독이 정지 프레임을 보냈고, 노드 종료 시 s0l0r0 송신됨)"
    exit 0 ;;
  check)
    preflight && echo "[car] 실행 환경 정상"
    exit $? ;;
  wheels-up)
    MODE=wheelsup ;;
  "")
    MODE=full ;;
  *)
    echo "사용법: $0 [wheels-up|check|down]"; exit 2 ;;
esac

source "$ROS_SETUP"
source "$WS/install/setup.bash"
mkdir -p "$LOGD"

if [ "$MODE" = wheelsup ]; then
  echo "[car] wheels-up 모드 — 어댑터+시리얼만 (카메라·정책 없음)"
  echo "[car] ⚠ 바퀴를 완전히 띄우고 E-stop을 준비할 것"
  kill_stack
  setsid nohup ros2 run thor_vehicle_pkg cmd_vel_motion_adapter_node --ros-args \
    -p cap_speed_int:=$CAP -p require_pose:=false \
    > "$LOGD/adapter.log" 2>&1 < /dev/null &
  setsid nohup ros2 run thor_vehicle_pkg serial_sender_node --ros-args \
    -p port:=$SERIAL_PORT \
    > "$LOGD/serial.log" 2>&1 < /dev/null &
  sleep 3
  echo "[car] 기동 완료. 검증 시작:"
  echo "    ros2 run thor_vehicle_pkg wheels_up_test"
  echo "[car] 통과하면 출력된 STEER_SIGN으로 전체 모드 기동. 종료: $0 down"
  exit 0
fi

# ---- full 모드 ----
case "${STEER_SIGN:-}" in
  +1|1|-1) ;;
  *)
    echo "[car] 거부: STEER_SIGN이 없다. wheels-up 검증을 먼저 통과하고"
    echo "      STEER_SIGN=+1 (또는 -1) $0  으로 기동할 것."
    exit 3 ;;
esac
preflight || exit 2
echo "[car] 전체 스택 — steer_sign=$STEER_SIGN, 속도 상한 $CAP/255"
echo "[car] ⚠ v6는 시뮬 학습 체크포인트 — 차선 추종을 기대하지 말 것."
echo "[car]   이 모드는 배선·지연 검증(빈 공간 저속)용이다."
kill_stack

echo "[car] 1/4 정책 서버..."
setsid nohup "$POLICY_PYTHON" -u "$WS/src/nav_vla_pkg/scripts/vla_policy_server.py" \
  --checkpoint "$CKPT" --endpoint ipc:///tmp/nav_vla.sock --warmup 4 \
  > "$LOGD/serve.log" 2>&1 < /dev/null &
for _ in $(seq 1 40); do
  grep -q 'serving on' "$LOGD/serve.log" 2>/dev/null && break; sleep 3
done
grep -q 'serving on' "$LOGD/serve.log" || { echo "서버 실패 — $LOGD/serve.log"; exit 4; }

echo "[car] 2/4 카메라+어댑터+시리얼 (steer_sign=$STEER_SIGN)..."
setsid nohup ros2 launch thor_vehicle_pkg thor_vla_bringup.launch.py \
  serial_port:=$SERIAL_PORT cap_speed_int:=$CAP require_pose:=false \
  steer_sign:=$STEER_SIGN \
  > "$LOGD/bringup.log" 2>&1 < /dev/null &
sleep 5

echo "[car] 3/4 브리지 (실차 카메라, 압축 직수신)..."
setsid nohup ros2 run nav_vla_pkg vla_bridge_node --ros-args \
  -p image_topic:=/image_raw/compressed -p compressed_image:=true \
  -p max_speed:=1.0 -p speed_slew:=0.08 \
  > "$LOGD/bridge.log" 2>&1 < /dev/null &
sleep 2

echo "[car] 4/4 준비 완료. 주행 시작 (빈 공간, 저속):"
echo "  ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable \\"
echo "    /vla/instruction std_msgs/String \"{data: 'Start driving in the inner lane, at a slow speed.'}\""
echo "  정지: 같은 명령으로 data: '' 발행 또는 $0 down"
