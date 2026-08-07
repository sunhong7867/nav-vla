#!/usr/bin/env bash
# LiDAR BEV Studio 실행 — 어느 위치에서든 사용 가능.
#   ./lidar_gui.sh            기본 실행 (/lidar_points)
#   ./lidar_gui.sh --topic /다른토픽 등 모든 인자 그대로 전달
exec python3 "$(dirname "$(readlink -f "$0")")/tools/lidar_alignment_gui/lidar_bev_studio.py" "$@"
