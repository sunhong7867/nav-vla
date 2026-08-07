# 노트북(라이다) <-> Thor 트랙 링크 환경. 양쪽 머신에서 source 한다.
#
#   source track_link_env.sh <서버IP> [domain]
#
#   노트북(디스커버리 서버 역할):
#       source track_link_env.sh <노트북자기IP>
#       fastdds discovery -i 0 -l <노트북자기IP> -p 11811   # 별도 터미널, 켜둠
#       ros2 run ... track_pose_node ...
#   Thor(수신):
#       source track_link_env.sh <노트북IP>
#       python3 check_track_link.py
#
# 왜 Discovery Server인가: WiFi에서 Fast DDS 기본 멀티캐스트 디스커버리는
# 자주 깨진다 (시뮬에서도 "DDS 그래프 붕괴" 전력, ver/20260805_1640).
# 이 Thor에는 rmw_fastrtps만 설치돼 있어(CycloneDDS 없음) Fast DDS 네이티브
# 해법인 서버 방식을 쓴다. 서버는 트랙사이드 노트북(고정 IP 쪽)에 둔다.
#
# ROS_SUPER_CLIENT=TRUE: ros2 topic list/echo 같은 CLI 도구가 서버 경유
# 그래프 전체를 보게 한다. 없으면 노드는 통신되는데 CLI만 빈 목록이 나와
# 링크가 죽은 것처럼 보인다.

if [ -z "$1" ]; then
    echo "usage: source track_link_env.sh <discovery-server-ip> [domain]" >&2
else
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID="${2:-47}"     # Thor VLA_AD 스택과 동일 도메인
    export ROS_DISCOVERY_SERVER="$1:11811"
    export ROS_SUPER_CLIENT=TRUE
    echo "track link: domain=$ROS_DOMAIN_ID server=$ROS_DISCOVERY_SERVER"
fi
