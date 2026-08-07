#!/bin/bash
# Prepare a clone for the SmolVLA Gazebo demo without machine-specific paths.
set -euo pipefail

WS="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
VENV=${NAVVLA_VENV:-$HOME/venv/navvla}
ROS_DISTRO=${ROS_DISTRO:-jazzy}
ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
INSTALL_ROS_DEPS=0

if [ "${1:-}" = "--install-ros-deps" ]; then
  INSTALL_ROS_DEPS=1
elif [ $# -gt 0 ]; then
  echo "사용법: $0 [--install-ros-deps]"
  exit 2
fi

[ -f "$ROS_SETUP" ] || {
  echo "ROS 2 $ROS_DISTRO가 없습니다: $ROS_SETUP"
  echo "ROS 2와 ros_gz를 먼저 설치한 뒤 다시 실행하세요."
  exit 2
}

if [ "$INSTALL_ROS_DEPS" -eq 1 ]; then
  command -v rosdep >/dev/null || {
    echo "rosdep이 없습니다: sudo apt install python3-rosdep"
    exit 2
  }
  source "$ROS_SETUP"
  rosdep install --from-paths "$WS/src" --ignore-src -r -y \
    --rosdistro "$ROS_DISTRO"
fi

command -v python3 >/dev/null || { echo "python3가 없습니다"; exit 2; }
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$WS/requirements-smolvla-demo.txt"

source "$ROS_SETUP"
cd "$WS"
colcon build --symlink-install

echo
echo "설정 완료"
echo "  venv: $VENV"
echo "  다음 단계: models/ckpt_v6_60k에 체크포인트를 복사"
echo "  점검: ./smolvla_demo.sh check"
echo "  실행: ./smolvla_demo.sh"
