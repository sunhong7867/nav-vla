# 2026-08-07 14:30 — 실차 첫날 준비 완료: 내일은 확인만 한다

내일 차량 작업이 "코드 작성"이 아니라 "체크리스트 실행"이 되도록 코드 3건을
오늘 완결·검증했다. 전부 dry-run/스텁으로 폐루프 확인 완료.

## 오늘 만든 것

1. **브리지 CompressedImage 직수신** (`vla_bridge_node.py`,
   `compressed_image:=true`) — 실차 카메라(JPEG)를 재인코딩 없이 서버로.
   시뮬(raw)은 기본값 false로 무변경. 스텁 서버 폐루프 검증:
   합성 CompressedImage → 브리지 → 스텁 추론 → `/cmd_vel` **121건 수신 PASS**
2. **`wheels_up_test`** (`ros2 run thor_vehicle_pkg wheels_up_test`) —
   정책 없이 결정적 명령으로 6항목 검증(무명령 정지·전진·좌/우 조향 부호·
   워치독·속도 단조). 끝나면 `STEER_SIGN` 확정값 출력.
   dry-run 전 플로우 PASS (speed_int 19/37/56 = 62/mps 스케일 정확)
3. **`thor_car_demo.sh`** — 실차판 원클릭. `wheels-up`(어댑터+시리얼만) /
   full(전체 스택) / check / down. **full 모드는 `STEER_SIGN` 명시 전 거부
   (exit 3 검증)** — wheels-up을 건너뛰는 실수를 구조적으로 차단.
   속도 상한 기본 60/255, 브리지 max_speed 1.0 m/s로 보수 설정

## 내일 절차 (차량 앞 체크리스트)

```bash
navvla                      # 폴더+소싱
# ── 1) 바퀴 완전히 띄우고, E-stop 손에 ──
./thor_car_demo.sh wheels-up
ros2 run thor_vehicle_pkg wheels_up_test     # 6항목 육안 검증, 부호 확정
# 실패 시: serial baud 의심 → SERIAL_BAUD 아두이노 소스 대조 (115200 vs 9600)
#          워치독 실패 → MCU command timeout 확인 전 주행 금지

# ── 2) D0 실측 (차량 옆에 있는 김에) ──
#   ±7 실각도: 조향 최대로 꺾고 각도기/폰 측정 → steer_deg_per_step 갱신
#   휠베이스 줄자 (54 cm 재확인) / 카메라 높이·피치

# ── 3) 빈 공간 저속 (사람·장애물 없는 곳, v6는 시뮬 학습이라 거동 무보장) ──
STEER_SIGN=+1 ./thor_car_demo.sh             # wheels-up이 알려준 부호로
ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable \
  /vla/instruction std_msgs/String "{data: 'Start driving in the inner lane, at a slow speed.'}"
# 관찰: 움직임 방향·속도 상한·브리지 지연 로그. 정지: ./thor_car_demo.sh down
```

**성공 기준 (D3 부분)**: wheels_up_test 6/6 PASS, 부호 확정, 저속에서
의도 방향으로 움직이고 down/E-stop으로 즉시 정지. **차선 추종은 판정하지
않는다** — v6는 시뮬 코퍼스 학습이라 실차 시야는 분포 밖.

## 미달·주의

- `speed_per_mps=62` 등 변환 스케일은 여전히 자리표시 — 실주행 속도가
  이상하면 여기부터 (D0에서 PWM-속도 곡선 실측으로 대체)
- 후진 게이트(v<0 → 0)는 유지 — 플랜트 후진 지원 미확인
- 카메라 점유 충돌 주의: VLA_AD 스택과 동시 기동 금지 (/dev/video0 단독 점유)
