#!/bin/bash
# 노트북<->Thor chrony 시간 동기 설정. sudo로 실행해야 하며, 이 스크립트는
# 준비물이다 — 세션에서 자동 실행하지 않는다 (시스템 설정 변경).
#
#   노트북(시간 서버):  sudo bash setup_time_sync.sh server <서브넷 예:10.221.71.0/24>
#   Thor(클라이언트):   sudo bash setup_time_sync.sh client <노트북IP>
#
# 게이트 (D2): chronyc tracking 의 System time 오프셋 < 30 ms.
# 정렬은 언제나 header.stamp 로만 한다 — 수신 시각 정렬 금지.
set -euo pipefail
ROLE="${1:?server|client}"
ARG="${2:?server=허용서브넷, client=서버IP}"

command -v chronyd >/dev/null || {
    echo "chrony 미설치 — 설치 시도"; apt-get update -qq && apt-get install -y chrony; }

CONF=/etc/chrony/conf.d/track_link.conf
mkdir -p /etc/chrony/conf.d
if [ "$ROLE" = server ]; then
    cat > "$CONF" <<EOF
# 트랙 링크 시간 서버 (노트북). 인터넷 없이도 자체 시계로 서빙한다.
allow $ARG
local stratum 10
EOF
else
    cat > "$CONF" <<EOF
# 트랙 링크 클라이언트 (Thor). 노트북을 유일 소스로, 공격적 폴링.
server $ARG iburst minpoll 2 maxpoll 4 prefer
makestep 0.2 3
EOF
fi
systemctl restart chrony || systemctl restart chronyd
sleep 3
chronyc tracking
echo "확인: 'System time' 오프셋 < 0.030 s 이면 D2 동기 게이트 통과"
