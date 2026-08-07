# 2026-08-07 11:03 — thor_vehicle_pkg 삽입: nav-vla 단독으로 Thor 실차 경로 완결

"토르 주행 시스템을 nav-vla로 삽입" 결정의 구현. VLA_AD 전체를 연결하는
대신 **하드웨어 I/O 3조각만 이식 + 어댑터 1개 신설**로 nav-vla가 자립한다.
VLA_AD 원 스택은 비교 베이스라인·데이터 시연자로 보존.

## 패키지 사용 감사 (질문에 대한 답)

`driving_sim.launch.py` 실측: 시뮬 파이프라인은 3개가 아니라 **5개**를 쓴다
— `simulation_pkg`·`nav_vla_pkg`·`interfaces_pkg` + 선생(욜로 차선추종)으로
`camera_perception_pkg`·`decision_making_pkg` 2개. 실차용으로
`track_localizer_pkg`·`lidar_perception_pkg`·`thor_vehicle_pkg`(신설)가 추가.

**나머지 VLA_AD 연결은 불필요** — 필요했던 것은 카메라 퍼블리셔, 시리얼
송신, `MotionCommand` 계약뿐이고, `MotionCommand.msg`는 nav-vla
`interfaces_pkg`에 이미 동일하게 존재(직접 확인).

## 신설: `src/thor_vehicle_pkg`

```
camera_publisher_node      VLA_AD image_publisher 이식 (MJPG 640x480@30,
                           JPEG q80 CompressedImage, 수동노출, 캡처 스레드)
serial_sender_node         VLA_AD serial_sender 이식 + 포트/보레이트 파라미터화
                           + 수신 워치독(0.5 s 침묵 → 정지 프레임 스트리밍)
cmd_vel_motion_adapter_node  ★신설 — /cmd_vel(Twist) → MotionCommand.
                           시뮬 계약을 그대로 두고 실차 끝단만 교체하는 지점
launch/thor_vla_bringup.launch.py
```

실차 경로: `camera → vla_bridge/policy_server(시뮬과 동일) → /cmd_vel →
어댑터(안전 게이트) → MotionCommand → serial → Arduino`.

어댑터 = 1세대 게이트웨이+먹스 최소형: cmd_vel 워치독·geofence estop·
operator estop·pose 유실(`require_pose:=true`, 트랙 주행 필수) 중 하나라도
걸리면 20 Hz로 정지 명령 스트리밍. 변환 파라미터(휠베이스 0.54,
5°/step ±7, speed_per_mps 62, cap 100)는 **전부 D0 실측 전 자리표시**이고
`steer_sign`은 wheels-up 부호 검증 전 신뢰 금지.

## 검증 (합성, VLA_AD 빌드의 MotionCommand로 — 계약 동일성 교차 확인 겸)

| 시나리오 | 결과 |
|---|---|
| v=1.0, w=0.5 → δ=atan(0.54·0.5)=15.1° | steering **3**, speed **62** 정확 |
| w=3.0 급조향 | ±7 클램프 동작 |
| geofence estop 중 | 비정지 명령 **0건** |
| estop 해제 | 즉시 복귀 |
| cmd_vel 중단 1.2 s | 워치독 0 명령 15건 (20 Hz 스트림) |
| serial dry-run | 미존재 포트에서 기동 + 자체 워치독 발동 확인 (실포트 미접촉) |

## 미달·미해결

- **nav-vla가 이 Thor에서 미빌드** — colcon build 후 `ros2 launch
  thor_vehicle_pkg thor_vla_bringup.launch.py` 실기동 확인 필요 (테스트는
  노드 직접 실행으로 대체함)
- 실제 시리얼·실차 부호/단위 검증은 D0·D3 (wheels-up) — 이 코드로 바로
  차를 굴리지 말 것
- `vla_bridge`의 CompressedImage 직수신(재인코딩 제거)은 미착수 — 현재
  브리지는 raw Image 구독이라 중간 디코더가 필요하거나 브리지 수정 필요.
  다음 작업 후보 1순위
- 후진(v<0)은 플랜트 미확인이라 0으로 게이트 — D0에서 확인 후 해제 검토
