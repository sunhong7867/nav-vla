# 2026-08-07 10:44 — 노트북→Thor 트랙 링크 수신 준비 + lidar_perception_pkg 허사이 개조

라이다·노트북이 아직 연결되지 않은 상태에서, Thor 수신측과 노트북측 도구를
전부 만들어 로컬에서 검증했다. 실기 연결 날은 "코드 작성"이 아니라
"측정만 하는 날"이 되도록.

## 1) 전송 계층 — Discovery Server로 확정 (마스터플랜 §4.3 정정)

**Thor에는 rmw_fastrtps만 있다** (`ls /opt/ros/jazzy/lib | grep rmw` — 직접
확인). [real_car_master_plan.md](real_car_master_plan.md) §4.3의 "CycloneDDS
정적 peers 기본"은 설치 없이는 불가 → **Fast DDS Discovery Server**(노트북=
서버, Thor=클라이언트)로 정정한다. WiFi 멀티캐스트 회피 효과는 동일하고
추가 설치가 없다. `fastdds` CLI 존재 확인.

- [track_link_env.sh](../../src/track_localizer_pkg/scripts/track_link_env.sh)
  — 양쪽 공용 env (`ROS_DISCOVERY_SERVER`, `ROS_SUPER_CLIENT=TRUE`,
  domain 기본 47=Thor VLA_AD 스택과 동일)
- Thor 네트워크: WiFi 단일 `wlP1p1s0` = 10.221.71.245/24

## 2) 수신 검증 도구 + 루프백 실측

- [mock_track_publisher.py](../../src/track_localizer_pkg/scripts/mock_track_publisher.py)
  — track_pose_node와 동일 토픽·QoS 계약(BEST_EFFORT depth 1)의 합성 발행.
  라이다 없이 링크만 따로 시험하는 용도
- [check_track_link.py](../../src/track_localizer_pkg/scripts/check_track_link.py)
  — 수신율·최악 갭·stamp age 측정, D2 게이트(≥10 Hz, 갭<0.5 s) PASS/FAIL

Thor 단독 루프백 (디스커버리 서버 127.0.0.1 경유, 10 s):

| 토픽 | 수신 | 최악 갭 |
|---|---|---|
| /track/vehicle_pose (+map/status/estop) | **9.2 Hz** (게이트 0.9×10 통과) | **0.102 s** < 0.5 |

stamp age 중앙값 +1.5 ms (로컬이라 오프셋 0 기준) — chrony 후 이 수치가
D2 동기 게이트가 된다.

## 3) lidar_perception_pkg 허사이 개조 (사용자 지시)

기존 패키지는 차량 부착 RPLidar(2D, serial) 체인:
publisher → `/scan` → processor/obstacle_detector. 허사이는 PointCloud2라
**`hesai_ingest_node` 신설**로 잇는다: z-밴드 크롭 + 각도 빈별 최소거리 →
`/scan` 재생성 + `/lidar/health` 감시. **기존 processor/obstacle_detector는
무수정으로 허사이를 소비**하게 된다. LaserScan(~수 KB)은 원시 점군과 달리
WiFi로 Thor 전송 가능 — Thor측 장애물 게이트 후보.

합성 PointCloud2 검증 (전방 2 m 벽 + 밴드 밖 바닥 + 후방 5 m 기둥):

| 검사 | 결과 |
|---|---|
| 빈별 최소거리 | 전방 2.00 m·후방 5.00 m 정확 |
| z-밴드 크롭 | 바닥점 누출 0빈 |
| health | ok, 밴드 내 51점 (벽 50+기둥 1) 일치 |

## 4) 시간 동기 준비 (미적용 — sudo 필요)

chrony **미설치** 확인. [setup_time_sync.sh](../../src/track_localizer_pkg/scripts/setup_time_sync.sh)
작성 — 노트북 `server <서브넷>` / Thor `client <노트북IP>`로 각자 sudo 실행.
게이트: `chronyc tracking` 오프셋 < 30 ms.

## 재현

```bash
# 루프백 (Thor 단독)
bash <scratchpad>/loopback_test.sh          # 세션 스크래치패드에 있음
# 실기 연결 시 (노트북): source track_link_env.sh <노트북IP> && fastdds discovery -i 0 -p 11811
#                        ros2 run lidar_perception_pkg hesai_ingest_node
#                        ros2 run track_localizer_pkg track_pose_node ...
# (Thor):               source track_link_env.sh <노트북IP> && python3 check_track_link.py
```

## 미해결

- **실기 WiFi 경유 측정 0회** — 루프백은 도구 검증일 뿐. 노트북 연결 후
  check_track_link 재실행이 실제 D2 사전 점검
- chrony 적용은 sudo 대기. 적용 전 stamp age는 클럭 오프셋 포함 값
- `hesai_ingest_node`의 z-밴드 기본값(0.05~0.5)은 트랙사이드 기준 —
  실측 후 재조정 (사람 다리 대역과 겹침 주의, track_localizer.yaml과 동일 이슈)
- lidar_publisher_node(RPLidar)는 그대로 둠 — 차량 온보드 2D를 다시 쓸 경우 대비
