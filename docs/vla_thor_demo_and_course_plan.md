# SmolVLA 기반 Thor 실차 데모 및 단기강좌 후속 계획서

> **스냅샷 주의 (2026-08-24 부기):** 이 문서는 2026-08-07 이전 시점 기준이다. 이후 변경 — v8/v8g 방향·직행 축, "직행은 좌표 내비 위임, v8g는 순항+반응형 전담" 결정(08-24) — 은 [ver/README.md](ver/README.md) 참조.

> **목표:** 먼저 1인이 로비 현수막 트랙에서 1/5 크기 유아용 전동차를 자연어로 주행시키는
> **VLA 직접 제어 데모**를 완성한다. 이후 검증된 모델·인터페이스·데이터를 동결한 뒤,
> NVIDIA Thor 없이 진행하는 단기강좌를 별도 설계한다.
>
> **상태:** 계획 및 인터페이스 확정 단계. 현재 SmolVLA 성과는 Gazebo 폐루프 결과이며,
> Thor 실차 SmolVLA 주행은 아직 검증되지 않았다.
>
> **우선순위:** 실차 데모 완성 → 데모 체크포인트/실행계약 동결 → 단기강좌 설계
>
> **작성일:** 2026-07-31
>
> **Thor 기준 저장소:** [sunhong7867/VLA_AD](https://github.com/sunhong7867/VLA_AD),
> `main@f410a873d870f66f0b33db3311f531951a045d91` 확인 기준

관련 문서:
[시뮬레이션 VLA 계획서](sim_vla_plan.md),
[SmolVLA v21b-120K 결과](ver/20260731_1226_120k-final-and-raw-speed.md),
[v3y 데이터·학습 기동 기록](ver/20260731_1731_v3y-yolo-corpus-trained.md),
[Hesai track localizer 기록](ver/20260727_2311_track-localizer.md),
[기존 Qwen 기반 Thor 통합안](vla_ad_nl_integration.md)

---

## 0. 결론과 이번 결정

1. **현재는 단기강좌를 만들지 않는다.** 실차 데모가 재현 가능해진 뒤 교육용으로 단순화한다.
2. 최종 데모의 정상 제어 경로는
   **카메라 + 차량 상태 + 트랙 상태 + 자연어 원문 → SmolVLA → 연속 행동 청크 → 실차**다.
3. Qwen/Ollama·정규식 파서·기존 VLM `VlaIR`·YOLO 차선 경로가 정상 주행 행동을 정하면
   이번 데모의 **VLA-only 기준을 통과하지 못한다.**
4. 기존 VLA_AD의 차선 주행 스택은 버리지 않는다. **실차 데이터 수집 시연자와 비교
   베이스라인**으로만 사용한다.
5. Hesai OT128 원본 점군은 노트북에서 처리한다. Thor에는 기본적으로 원시
   `PointCloud2`가 아니라 **track-frame pose, 품질, lap 상태, geofence stop**만 전송한다.
6. 최종 데모 추론은 연구실 서버가 아니라 **Thor 온디바이스**를 기본안으로 한다.
   연구실 서버는 학습과 개발 중 비교 실험에만 사용한다.

기존 [Qwen 기반 Thor 통합안](vla_ad_nl_integration.md)과
2026-07-31의 Qwen 정규화 기록은 당시 설계·데이터 수집 이력으로 보존한다. 이번 문서가
**최종 데모 제어 경로에 대해서는 우선한다.**

---

## 1. 두 작업의 범위

| 구분 | 지금 할 일 | 지금 하지 않을 일 | 종료 조건 |
|---|---|---|---|
| **A. 실차 데모** | 본인 1인이 로비 트랙에서 Thor 탑재 전동차를 텍스트·음성으로 VLA 주행 | 20팀 서버, 학생 설치 자동화, 강좌 교안 | D0~D7 게이트 통과 |
| **B. 단기강좌** | 데모 완료 후 학생용 데이터·체크포인트·차량 구조로 단순화 | 데모와 동시에 별도 플랫폼 개발 | C0~C4 게이트 통과 |

데모의 1차 기능은 다음 순서로 확장한다.

1. 저속 차선 유지 한 바퀴
2. 속도 증감과 안전 정지
3. 좌·우 차선변경
4. 차선을 유지하며 지정 기준선/목적지까지 이동 후 정지
5. 자연어로 요청한 한 바퀴·두 바퀴를 정확히 수행
6. 같은 목적지까지 차선을 무시하고 안전영역 안에서 최단 경로로 이동
7. 키보드 명령과 음성 명령의 동일 동작 확인

최단 경로는 정상 차선 주행보다 분포 이동과 안전 위험이 크므로 마지막에 연다.

---

## 2. 무엇을 VLA 주행으로 인정할 것인가

SmolVLA도 내부에는 vision-language backbone을 포함한다. 여기서 “LLM/VLM이 아닌 VLA”의
뜻은 모델 내부 명칭이 아니라 **학습 목표와 출력이 실제 행동인지**로 구분한다.

### 2.1 인정하는 정상 제어 경로

```text
camera + numeric state + raw instruction
        -> SmolVLA 한 번의 정책 추론
        -> action chunk [dx, dy, dyaw]
        -> 저수준 궤적 추종/단위 변환
        -> steering + motor command
```

- SmolVLA가 이미지·상태·원문 지시를 함께 보고 미래 ego-frame 행동 청크를 출력한다.
- 저수준 제어기는 SmolVLA가 만든 궤적을 추종할 뿐, 차선·목적지·경로를 새로 선택하지 않는다.
- 안전 계층은 속도·곡률을 제한하거나 정지시킬 수 있지만 다른 경로로 대신 주행하지 않는다.

### 2.2 허용하는 보조 기능

- `faster-whisper` 기반 로컬 STT: 음성을 **문자열로만** 바꾼다.
- Hesai 정합·측위·시간 동기·lap progress 계산
- 미터/라디안 행동을 실차 조향 단계와 PWM으로 변환하는 저수준 controller
- 물리 E-stop, RC override, geofence, pose/network/VLA timeout, command watchdog
- 데이터 수집과 비교 실험에서만 사용하는 YOLO/navigator/고전 제어기

### 2.3 최종 데모 정상 경로에서 금지하는 것

- Qwen/Ollama 또는 정규식이 자연어를 intent/zone/lane/speed 명령으로 변환
- VLM이 JSON/`VlaIR`만 출력하고 YOLO 차선 추종기가 실제 조향 수행
- navigator/route oracle이 런타임 경로를 만들고 VLA가 그 경로를 그대로 실행
- VLA 실패 시 고전 제어기로 바꾸어 계속 주행한 뒤 성공으로 기록

고전 제어 takeover가 안전을 위해 작동했다면 차량은 살릴 수 있지만, 해당 trial은
**VLA 데모 실패**로 기록한다.

---

## 3. 현재 확보된 것과 아직 없는 것

| 항목 | 현재 상태 | 판정 |
|---|---|---|
| GUI 직접 입력 | `control_backend:=smolvla`에서 Qwen/regex/planner를 우회하고 원문을 `/vla/instruction`에 발행 | **확보** |
| 음성 입력 | 로컬 `faster-whisper` CPU INT8 → 같은 원문 입력 경로 | **확보, 실차 미검증** |
| SmolVLA 서빙 | LeRobot venv의 policy server와 ROS bridge가 ZMQ/msgpack으로 분리됨 | **시뮬 확보** |
| VLA 행동 | 10 Hz에서 `[dx,dy,dyaw]` 청크를 소비하고 `/cmd_vel` 발행 | **시뮬 전용** |
| v21b-120K | 링의 언어별 형상 차이는 noise floor의 4.8배였으나 RR 0.14, 정지 0/8 | **부분 성공, 목표 미달** |
| 언어 일반화 | 학습 표현 8/8, 미학습 paraphrase 0/8 | **미달** |
| v3y | 173 train episode, 22 heldout, CMI 0.333 bit; 문서상 40K 학습 기동까지만 확인 | **폐루프 결과 미확인** |
| Hesai 측위 | `/track/vehicle_pose`, status, geofence 발행 코드 존재 | **합성 검증만 완료** |
| 트랙 재설치 | 현재 homography가 2026-07-25 설치 위치에 고정 | **매 설치 재정합 도구 없음** |
| Thor 기존 주행 | 카메라→YOLO→차선 경로→Pure Pursuit→`MotionCommand`→serial 경로 존재 | **시연자/베이스라인** |
| Thor 기존 VLA | Qwen3-VL-8B가 `VlaIR`을 만들고 차선 추종기가 조향 | **직접 행동 VLA가 아님** |
| Thor SmolVLA | 코드·체크포인트·LeRobot runtime이 VLA_AD에 없음 | **미착수** |
| 실차 상태 | VLA_AD에 SmolVLA가 요구하는 speed/yaw-rate/steer `Odometry` 발행기가 없음 | **신규 필요** |
| 요구 기능 데이터 | 정확한 1/2 lap, 실차 차선변경, shortest path, 도중 속도변경 데이터 없음 | **신규 필요** |

따라서 현재 상태를 정확히 표현하면 다음과 같다.

> **SmolVLA 기반 Gazebo 폐루프 PoC와 직접 명령 경로는 확보했지만, 언어 일반화·정지·전역
> 임무·Thor 실차 제어는 아직 데모-ready가 아니다.**

---

## 4. 목표 시스템 구조

```text
노트북(트랙사이드)
  Hesai OT128 /lidar_points
       -> 바닥 제거·차량 검출·track 정합·상태 추정
       -> /track/vehicle_pose      (Odometry + covariance)
       -> /track/vehicle_status    (quality/age/heading-valid)
       -> /track/lap_state         (progress/current_lap)
       -> /track/geofence_estop    (Bool)

  Mic -> faster-whisper(STT only) ─┐
  Chat GUI -> raw instruction ─────┴──────────────┐
                                                  │ ROS 2 / 유선망 우선
Thor(차상)                                        ▼
  front camera + vehicle state + track state + raw instruction
       -> SmolVLA policy server
       -> action chunk [dx, dy, dyaw]
       -> trajectory safety gateway
       -> 0.54 m급 실차 Ackermann/PWM adapter
       -> /topic_control_signal (MotionCommand)
       -> serial_sender -> Arduino -> steering/motor
```

### 4.1 노트북 역할

- OT128 드라이버와 원시 point cloud 처리를 담당한다.
- 현수막을 펼칠 때마다 `lidar/lobby -> track` 변환을 다시 구한다.
- raw cloud rosbag은 노트북에 저장하고, 네트워크에는 경량 상태만 보낸다.
- GUI와 음성 인식을 실행해 해석하지 않은 문자열을 Thor로 보낸다.
- 평가 시 실제 궤적, 차선 중심 오차, 목적지 오차, 경로 길이를 기록한다.

### 4.2 Thor 역할

- 전방 카메라와 차량 상태를 시간 정렬한다.
- SmolVLA를 온디바이스로 추론한다.
- VLA 행동 청크를 실차 조향·속도로 변환한다.
- 안전 gateway와 command mux의 **유일한 최종 출력**만 serial로 보낸다.
- 기존 Qwen TRT, 기존 `motion_planner`, YOLO 경로는 VLA-only launch에서 끈다.
  필요하면 shadow logging만 하되 actuator 토픽에는 연결하지 않는다.

### 4.3 왜 Hesai pose를 VLA 관측에 추가해야 하는가

현재 SmolVLA 계약은 `camera + [speed,yaw_rate,steer] + text`뿐이다. 이 입력만으로는
고정 landmark가 없는 긴 루프에서 다음을 신뢰성 있게 수행하기 어렵다.

- 현재 몇 바퀴째인지 기억
- 이름으로 지정한 기준선에서 정지
- 동일 목적지까지 lane-follow와 shortest-path를 구분

따라서 실차용 새 checkpoint의 state 후보는 다음 정보를 포함한다.

```text
[speed, yaw_rate, steering,
 track_x, track_y, sin(track_yaw), cos(track_yaw),
 pose_age, lap_progress, completed_laps]
```

`lap_monitor`는 위치로부터 진행률과 완료 횟수만 계산한다. 자연어에서 “두 바퀴”를 파싱하거나
두 번째 통과 때 강제로 정지하지 않는다. **몇 바퀴를 요구했는지는 SmolVLA가 원문과 state를
함께 보고 학습해 행동으로 결정한다.** 입력 차원이 바뀌므로 현재 sim checkpoint를 그대로
사용할 수 없고 재학습이 필요하다.

첫 데모에서 raw LiDAR를 SmolVLA의 별도 modality로 직접 넣지는 않는다. 그것은 모델 구조 변경과
전 데이터 재수집이 필요한 후속 연구다.

---

## 5. 기능별 추가 데이터와 판정

| 기능 | 현재 데이터로 가능한 주장 | 새로 필요한 데이터/상태 | 데모 판정 예시 |
|---|---|---|---|
| 저속 lane-follow | sim 링 추종만 부분 검증 | Thor 카메라, 실차 행동, Hesai pose로 한 바퀴 | 5회 연속 이탈·개입 없이 완주 |
| 1/2 lap | `Take a lap` 문구는 있으나 episode가 시간 종료 | 동일 시작점의 1 vs 2 lap paired data, lap state, 마지막 감속/정지 | 요청 횟수에서 정지, 각 5회 연속 성공 |
| 기준선/목적지 | 구 navigator는 가능하지만 VLA가 아님 | 여러 시작점·양 차선·여러 목적지와 정지 ramp | 목표선 오차 잠정 ≤0.30 m, lane 이탈 0 |
| shortest path | 구 direct oracle뿐이며 런타임 VLA 증거 없음 | 같은 관측의 lane-follow vs lane-ignore paired data | 안전영역 내 주행, lane 경로보다 길이 ≥10% 짧음 |
| 차선변경 | VLA_AD 자료구조는 있으나 현 path planner가 adjacent path를 소비하지 않음 | keep vs left/right change paired data, 변경 후 안정화 | 방향별 5회 연속 목표 차선 정착 |
| 속도 증감 | sim raw tier가 있으나 실차 PWM과 미정합 | 실차 slow/normal/fast 및 주행 중 up/down transition | 실측 중앙속도 `slow < normal < fast`, 이탈 0 |
| 음성 | STT→raw instruction 코드 존재 | 실제 마이크·소음·ASR 오인식 문장을 포함한 평가 세트 | typed와 별도로 end-to-end 성공률 보고 |

처음에는 기능 하나만 담은 원자 명령을 학습·검증한다. 복합 문장과 주행 중 추가 명령은
각 원자 기능이 통과한 뒤, 원문 대화 이력을 단순 연결한 입력으로 확장한다. 연결기는 의미를
해석하지 않는다.

---

## 6. 구현 단계와 Gate D0~D7

기간은 1인 개발, 차량·Thor·트랙을 필요할 때 사용할 수 있다는 가정의 **약 8~15주 계획 범위**다.
D0~D2 실측 전에는 납기로 확정하지 않는다.

### D0 — 제어 계약과 안전 경계 동결 (약 1주)

- Thor의 실제 ROS 2 환경, 카메라 topic/stamp, `MotionCommand`, serial, 펌웨어를 실기 확인한다.
- 휠베이스, 최대 조향각, steering `-7..7`의 실제 각도, PWM-속도 곡선을 측정한다.
- 저장소의 serial sender 115200 baud와 Arduino 소스 9600 baud 불일치를 실제 flash 기준으로 해소한다.
- MCU에 command timeout이 실제 있는지 확인하고, 없으면 마지막 명령 유지가 불가능하도록 추가한다.
- VLA-only launch의 단일 actuator publisher와 topic graph를 문서화한다.
- 데모 언어를 동결한다. 현재 checkpoint의 한국어 일반화는 근거가 없으므로, 한국어가 필수라면
  Qwen 번역을 끼우지 않고 bilingual corpus/backbone 검증 일정을 별도로 잡는다.

**PASS:** 알 수 없는 단위·프레임·baud가 0개이고, 물리 E-stop/RC override/command timeout이
실제로 모터를 정지시킨다.

### D1 — 기존 Thor 주행 기준선과 실차 레코더 (약 1주)

- 기존 YOLO+lane+Pure Pursuit로 저속 한 바퀴를 수행한다. 이것은 VLA 결과가 아니라
  하드웨어 기준선이자 데이터 시연자다.
- Thor 카메라, 제어 명령, Hesai pose, timestamp, 안전 개입을 하나의 episode로 기록한다.
- Gazebo 의존 recorder를 실차 `Odometry`/Hesai pose 기반으로 교체한다.

**PASS:** 기존 제어로 반복 가능한 한 바퀴와 재생 가능한 rosbag/episode가 생기고,
카메라-행동-pose 정렬이 검증된다.

### D2 — 현수막 재설치 정합과 Hesai 실측 (약 1~2주)

- 현수막 네 모서리 또는 정해진 기준점에 OT128이 볼 수 있는 **휴대형 수직/재귀반사 marker**를 둔다.
  인쇄된 평면 그림만으로는 3D/2D LiDAR 재정합 기준점이 되지 않는다.
  이 marker는 차량 검출용 마스트가 아니라 설치 정합 전용이며, 정합 후 제거하거나 vehicle ROI에서
  제외해 기존 차량 cluster와 섞이지 않게 한다.
- 고정 lobby frame과 이동 가능한 track frame을 분리하고, 매 설치 `lobby/lidar -> track`을 계산한다.
- 1/5 전동차 높이에 맞게 현재 0.03~0.35 m 검출 gate를 다시 측정한다.
- 사람 다리 오검출, 원거리, centroid 면 편향, 정지 시 heading invalid를 실차 rosbag으로 측정한다.
- 노트북↔Thor `chrony`와 header stamp 정렬을 설정한다.
- 현수막을 서로 다른 위치에 3회 다시 펼쳐 동일 절차를 반복한다.

잠정 수용 목표:

- 정적 위치 RMS ≤ 0.10 m, stretch goal ≤ 0.05 m
- 직선 heading RMS ≤ 3°
- direct/shortest 기능을 열려면 corner heading peak ≤ 5° 또는 Thor IMU/차량 모델 융합으로 동등 성능
- 시간 동기 잔차 < 30 ms, pose publish ≥ 10 Hz
- pose 또는 링크가 500 ms 이상 끊기면 주행 명령 0

**PASS:** 세 번의 재설치에서 모두 위치·시간·loss-stop 게이트를 통과한다. 코너 heading이
미달이면 lane-follow만 진행하고 shortest path는 열지 않는다.

### D3 — Thor SmolVLA runtime과 action adapter (약 1~2주)

- `nav-vla`의 policy server, pre/post processor, warm-up, ZMQ 계약을 Thor로 이식한다.
- Thor의 `/image_raw/compressed`를 받아 불필요한 JPEG 재인코딩을 제거한다.
- `[dx,dy,dyaw]` 청크를 실측 차량 기하에 맞는 steering/PWM으로 바꾸는 저수준 tracker를 만든다.
- 기존 sim 상수 `wheelbase=2.86 m`, `max_steer=0.6 rad`, Twist 출력을 제거한다.
- 기존 `VlaIR`용 LASA를 억지로 재사용하지 않고, action freshness·곡률·가속·pose age·geofence를
  검사하는 `trajectory_safety_gateway`를 만든다.
- 휠을 든 상태에서 부호·포화·timeout을 먼저 검증한 뒤 빈 공간 저속 직선/원으로 진행한다.
- Thor에서 eager/compiled/export 후보의 latency, memory, queue underrun을 실측한다.
  aarch64/LeRobot 호환이 미달이면 ONNX/TensorRT export를 다음 경로로 사용한다.

**PASS:** steering/speed 부호와 단위 test 전부 통과, 강제 장애마다 500 ms 이내 정지,
의도치 않은 재출발 0, 정상 run의 queue underrun <1%, watchdog 0회.

### D4 — 데모 기능용 paired data 수집 (약 2~4주)

- 기존 Thor 제어기, teleop, 수집 전용 route oracle을 **시연자**로 사용한다.
- 행동 라벨은 integer steering을 그대로 복사하지 않고, 시간 정렬된 Hesai ego-motion을
  오프라인 양방향 평활해 metric `[dx,dy,dyaw]` 청크로 만든다.
- 같은 시작 pose·카메라 관측에서 instruction만 다른 paired demonstration을 만든다.
  - 1 lap vs 2 lap
  - lane 유지 목적지 vs 같은 목적지 shortest
  - keep vs left/right lane change
  - slow/normal/fast와 도중 speed up/down
- 목적지 정지, 차선변경 후 안정화, off-nominal 복구 구간을 포함한다.
- typed 문장과 실제 STT transcript를 저장하고 train/heldout paraphrase를 물리 분리한다.
- 절대 lobby 좌표가 아니라 **재설치에도 동일한 track frame**으로 저장한다.

**PASS:** 기능×시작점×차선×속도 coverage 빈칸 0, 각 핵심 셀에 성공 matched pair ≥5,
timestamp/frame/action 검증 통과. 시연자 실패 샘플은 성공 BC에서 제외하되 실패율은 남긴다.

### D5 — 재학습과 open-loop/Gazebo gate (약 1~2주/반복)

- 기존 sim checkpoint는 warm-start로만 사용하고, driving-dominant 실차용 state 계약으로 재학습한다.
- 고정 관측에서 지시만 바꾸어 방향·속도·정지 출력이 올바르게 달라지는지 먼저 측정한다.
- 학습 표현뿐 아니라 미학습 paraphrase를 별도 평가한다.
- open-loop 통과 전 실차 폐루프에 올리지 않는다.
- 이후 Gazebo에서 각 기능 5회씩 폐루프 평가한다.

**PASS:** 핵심 instruction pair 방향 정확도 ≥80%, action limit 위반 0,
heldout paraphrase 성공률 ≥0.8×in-distribution, Gazebo 시나리오별 ≥4/5,
off-track/collision 0, classic takeover/non-safety override 0.

### D6 — 실차 단계 상승 (약 2~4주)

다음 순서를 바꾸지 않는다.

```text
wheels-up
  -> 빈 공간 저속 직선/원
  -> 저속 lane 1 lap
  -> 정확한 2 lap
  -> speed up/down
  -> lane change
  -> lane 유지 목적지 정지
  -> shortest path
  -> voice end-to-end
```

각 단계는 5회 연속 성공 전 다음 기능이나 속도를 열지 않는다. 안전 takeover가 동작해도 해당 trial은
실패이며 원인을 데이터·정합·정책·adapter 중 하나로 귀속한다.

**PASS:** 요구 기능별 5회 연속 성공, collision/off-envelope 0, 수동 개입 0,
watchdog 0, 요청 lap 수·차선·목적지·속도 순서가 모두 맞는다.

### D7 — 데모 패키징

- 단일 launch와 모드별 설정 파일을 만든다.
- `track calibration -> sensor health -> policy warm-up -> actuator enable` 순서의 preflight를 자동화한다.
- GUI에 pose age, policy latency, queue, geofence, active checkpoint, 최종 command source를 표시한다.
- 모든 run에 checkpoint hash, git SHA, calibration ID, rosbag을 남긴다.
- 실패 시 즉시 정지·수동 회수·재기동 절차를 체크리스트로 만든다.

**PASS:** 전원 OFF 상태에서 시작하는 독립 cold-start 리허설 3회 모두 성공하고,
요구 기능 전체를 정해진 순서로 재현한다.

---

## 7. 코드 이식 범위

| 분류 | 항목 | 조치 |
|---|---|---|
| nav-vla 재사용 | `chat_gui_node.py` SmolVLA direct/voice | 원문 topic과 상태 UI만 이식 |
| nav-vla 재사용 | `vla_policy_server.py` | checkpoint/pre-post/warm-up/ZMQ 계약 유지 |
| nav-vla 수정 | `vla_bridge_node.py` | compressed camera, 확장 state, 실차 action output으로 변경 |
| nav-vla 수정 | episode recorder/LeRobot 변환 | Gazebo pose 대신 Hesai+Thor timestamp 사용 |
| nav-vla 수정 | `track_localizer_pkg` | 재설치 정합, 차체 gate, orientation fusion, 실측 validator 추가 |
| VLA_AD 재사용 | camera publisher, `MotionCommand`, serial, Arduino, operator E-stop | 실차 I/O와 하드웨어 계층 유지 |
| VLA_AD 격리 | Qwen3-VL TRT, `VlaIR`, 기존 LASA | 기존 연구/비교 모드에만 유지 |
| VLA_AD 격리 | YOLO lane/path/motion planner | 데이터 시연자·shadow baseline에만 유지 |
| 신규 | `smolvla_runtime` launch | VLA-only node ownership을 명시 |
| 신규 | vehicle state builder | Hesai pose 변화량, last command, 가능하면 IMU/encoder 융합 |
| 신규 | trajectory tracker + safety gateway + command mux | VLA 행동을 유일한 정상 actuator source로 연결 |
| 신규 | track registration/calibration tool | 현수막 재설치마다 재생성·검증 |

두 저장소의 `interfaces_pkg`를 같은 shell에서 무작정 overlay하지 않는다. 노트북↔Thor에는 우선
표준 `Odometry/String/Bool`을 사용하고, Thor 내부의 신규 trajectory 메시지는 VLA_AD 쪽에서
버전 관리한다.

---

## 8. 현수막을 다시 펼칠 때의 운영 원칙

Hesai로 만든 기존 이미지는 **트랙 모양의 template**로는 재사용할 수 있지만,
당시 lobby 절대좌표를 그대로 사용할 수는 없다.

매 설치 절차:

1. 현수막 기준점에 휴대형 vertical marker 3~4개 설치
2. OT128에서 marker 검출
3. `lidar/lobby -> track` 변환 계산
4. track mask, 목표선, lap line, safe envelope를 새 변환으로 투영
5. 측량점 static check와 짧은 직선 heading check
6. calibration ID 저장 후 actuator enable

학습 데이터에는 lobby 절대위치가 아니라 track-frame 좌표를 넣는다. 따라서 현수막 위치가 바뀌어도
재정합만 성공하면 같은 정책 계약을 유지할 수 있다.

`shortest path`의 geofence는 차선/도로 mask가 아니라 **사람·벽·현수막 바깥을 제외한 물리적
safe envelope**여야 한다. road mask를 geofence로 쓰면 lane-ignore 명령과 안전 계층이 서로 모순된다.

---

## 9. 단기강좌는 데모 완료 후 별도 착수

### C0 — 시작 조건

D7을 통과한 checkpoint, 데이터 schema, observation/action 계약, 안전 절차가 동결되기 전에는
강좌용 서버·교안·20팀 계정을 만들지 않는다.

### 강좌 기본 방향

- Thor와 클라우드는 사용하지 않는다.
- 학생에게 검증된 dataset, baseline checkpoint, container/설치 스크립트를 제공한다.
- 수업 시간에 처음부터 대규모 학습하지 않고, 명령 데이터 수정·소규모 fine-tune·평가에 집중한다.
- 대표 학생 노트북에서 SmolVLA 양자화 추론이 replan budget을 만족하면 팀별 local inference를 우선한다.
- GPU가 부족한 팀만 연구실 서버를 사용한다.
- 20팀 전체 동시가 아니라 실제 운영 조건인 **3~4 active team + 여유 worker 1개**로 서버를
  부하시험한다. 측정 전 “문제없다”고 가정하지 않는다.
- raw LiDAR를 연구실 서버로 모으지 않는다. 각 차량/트랙 옆 컴퓨터에서 localize하고 pose만 사용한다.

### C1~C4 게이트

| Gate | 내용 | 통과 조건 |
|---|---|---|
| C1 | 대표 학생 노트북 3종 benchmark | 설치, memory, P50/P95, queue underrun 기록 |
| C2 | 연구실 서버 4팀 동시 inference | P95가 action horizon 이내, worker crash/queue starvation 0 |
| C3 | Thor 없는 reference car + A1M8 | 지정 코스 반복 완주와 E-stop 검증 |
| C4 | 20팀 수업 운영 리허설 | clean laptop 설치부터 첫 주행까지 수업 시간 내 재현 |

A1M8 2D LiDAR는 평평한 현수막 그림 자체를 landmark로 보기 어렵다. 강좌용 위치추정은
스캔 높이에 맞춘 휴대형 vertical landmark, 고정 벽 map, 또는 카메라 track perception 중 하나를
데모 이후 reference setup으로 확정한다. 학생에게는 Hesai로 만든 특정 lobby 절대맵이 아니라
track template과 매 설치 calibration 결과를 제공한다.

---

## 10. 핵심 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| 현재 SmolVLA가 조작용 기반 모델 | driving transfer가 약할 수 있음 | driving-dominant sim/real paired data, open-loop gate 선행 |
| 한국어·미학습 표현 | 현재 paraphrase 0/8 | bilingual/heldout corpus; Qwen 번역으로 숨기지 않음 |
| 정확한 1/2 lap의 장기 기억 | 현재 정책 state만으로 불가 | track pose 기반 lap progress를 관측으로 추가해 재학습 |
| 트랙 재설치 | 기존 homography/road mask 무효 | portable marker와 3회 재설치 gate |
| Hesai corner yaw 17° 합성 peak | shortest/direct 궤적 오류 | orientation/IMU 융합 전 shortest 차단 |
| Thor LeRobot/aarch64 | runtime 또는 compile 실패 | D3 benchmark, 필요 시 ONNX/TensorRT export |
| raw OT128 네트워크 | 지연·대역폭 증가 | 노트북 local processing, pose/status만 전송 |
| 기존 시연자 결함 모방 | 잘못된 차선/경로 학습 | Hesai 궤적 검수, 실패율/coverage 기록, paired data 검사 |
| actuator 마지막 명령 유지 | 통신 장애 시 runaway | MCU command timeout, hardware E-stop, final command mux |
| shortest와 road geofence 충돌 | 안전 계층이 정상 명령을 차단 | road mask와 physical safe envelope 분리 |

---

## 11. 바로 다음 작업 5개

1. Thor 실기에서 camera/topic/frame/unit/serial baud/MCU timeout을 확인하고 D0 계약을 동결한다.
2. 기존 Thor 제어기로 저속 한 바퀴를 재현하며 실차 camera-command-Hesai recorder를 만든다.
3. 현수막을 3회 다른 위치에 펼쳐 marker 기반 재정합과 pose 정확도를 실측한다.
4. SmolVLA server를 Thor에서 benchmark하고 wheels-up action adapter/safety gateway를 검증한다.
5. 여섯 기능의 paired-data coverage 표를 만든 뒤, 저속 lane-follow 파일럿부터 수집한다.

이 다섯 항목이 끝나야 전체 데이터량, Thor 추론 방식, 한국어 범위와 데모 일정을 근거 있게
다시 산정할 수 있다.
