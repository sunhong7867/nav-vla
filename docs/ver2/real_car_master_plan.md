# 실차 완성 마스터플랜 — v7 판정에서 Hesai→Thor 폐루프 데모까지

작성 2026-08-06 밤. v7 판정은 내일(08-07) 아침. 근거 코드:
`track_localizer_pkg/`(측위), `tools/lidar_alignment_gui/`(정합 GUI),
`tools/eval/`(프로브), `train_smolvla.sh`, VLA_AD `serial_sender_node.py` 등.
이 문서는 [vla_thor_demo_and_course_plan.md](../vla_thor_demo_and_course_plan.md)
(2026-07-31)의 D0~D7 게이트 체계와 §2 VLA-only 원칙을 **계승**하고,
2026-08-06 기준 현황·실행 순서·Hesai→Thor 연결 상세에 대해 우선한다.
[navvla_vs_vlaad_comparison.md](navvla_vs_vlaad_comparison.md) §8의 2~6주
순서를 완성 시점까지 확장한 것이 이 문서다.

---

## 0. 한 줄 답

```
주 0        v7 판정 → 시뮬 동결 (성공/실패 무관하게 실차 전환 시작)
주 1        D0  제어 계약·안전 경계 실측 동결 (Thor·Arduino·차량)
주 2        D1  기존 스택 기준선 + 실차 레코더    ← 3커맨드 주행은 이미 확인
주 2~3      D2  Hesai 재설치 정합 + pose 실측 게이트
주 3~4      D3  Thor SmolVLA 런타임 + 액션 어댑터     ┐ 병렬
주 3~5      시뮬 재정합 (실트랙 치수·0.54 m·15단계)   ┘
주 5~8      D4  실차 paired 코퍼스 수집 (Hesai 라벨)
주 8~10     D5  재학습 + open-loop/Gazebo 게이트  (사이클당 ~1주, 반복)
주 10~14    D6  실차 단계 상승 (wheels-up → … → voice)
주 14~16    D7  데모 패키징 + 논문 실험
```

총 **10~16주** (1인 기준). D0~D2 실측 전에는 납기로 확정하지 않는다
(데모 계획서 §6 원칙 유지). 세 가지 상위 판단:

1. **정상 제어 경로는 SmolVLA 직접 제어다** — Qwen/정규식/YOLO 차선추종이
   행동을 정하면 VLA 데모 실패로 기록 (데모 계획서 §2 그대로).
2. **Hesai는 Thor에 pose·상태만 보낸다.** 원시 점군은 노트북에서 종결 (§4).
3. **체크포인트는 2세대로 나눈다** — 1세대(현 계약, 기능 1~4) 먼저,
   2세대(pose 확장 state, 기능 5~6)는 그 뒤 (§5).

---

## 1. 완성의 정의

데모 계획서 §1의 기능 사다리를 그대로 쓴다. 순서를 바꾸지 않는다.

1. 저속 차선 유지 한 바퀴 → 2. 속도 증감·안전 정지 → 3. 좌·우 차선변경 →
4. 차선 유지 목적지 정지 → 5. 정확한 1/2바퀴 → 6. shortest path →
7. 키보드/음성 동일 동작

**완성 = D7 PASS**: 전원 OFF에서 시작하는 cold-start 리허설 3회 전부 성공 +
요구 기능 전체를 정해진 순서로 재현. 각 기능은 5회 연속 성공 전에 다음을
열지 않고, 안전 takeover가 살린 trial은 실패로 기록한다.

부수 산출물 (논문 원자료): ① 실차 반사실 실험 (같은 관측, 문장만 변경 →
궤적 발산 ≥ 0.5 m), ② 어블레이션 (A0 고전 스택 vs VLA-only), ③ 일반화 지도
(시뮬→실차 전이에서 무엇이 살고 무엇이 죽는가).

---

## 2. 현재 상태 (2026-08-06 밤)

데모 계획서 §3 표의 갱신판. 바뀐 행만 적는다.

| 항목 | 07-31 상태 | 현재 | 근거 |
|---|---|---|---|
| 시뮬 폐루프 | v21b RR 0.14, 목표 미달 | **v6-60K 이탈 0/8, 데모 확정**. v7(DAgger 44포즈) 60K 학습 중, 내일 아침 판정 | [ver/20260806_0140](../ver/20260806_0140_v6-eval-demo-final.md), [ver/20260806_1045](../ver/20260806_1045_dagger-cycle1-launch.md) |
| 언어 접지 | 링 형상 4.8× | 반사실 차선 분리 **2.51 m / 27×** 확보 | `ver/20260804_1056` |
| 언어 일반화 | paraphrase 0/8 | 동일 (미학습 표현 미전이 +0.36 m) — 파서 하이브리드 또는 D4 heldout 설계로 대응 | `ver/20260804_1056` |
| 트랙 재정합 도구 | **없음** | **LiDAR BEV Studio 완성** (~3,900줄, 4점 정합 + '최종 확인' 워크플로) — 단 실측 미검증 | [tools/lidar_alignment_gui/](../../tools/lidar_alignment_gui/) |
| Hesai 측위 | 합성 검증만 | 동일 (코드 존재, 실측 0회). heading 코너 피크 17°/1.3 s — 기능 6 차단 조건 유효 | [track_pose_node.py](../../src/track_localizer_pkg/track_localizer_pkg/track_pose_node.py), `track_localizer.yaml` 주석 |
| Thor 기존 주행 | 시연자/베이스라인 | **3커맨드로 트랙 주행 확인** (2026-08-06, 사용자) — D1 기준선 절반 확보 | 사용자 실주행 |
| Thor SmolVLA | 미착수 | 동일 | — |
| 실차 state 발행기 | 신규 필요 | 동일 | — |
| VLA_AD 저장소 | 단일 | 원본 `~/hoon/VLA_AD` / 작업 사본 `~/hong/VLA_AD` 분리, 사본은 분석 산출물 정리됨 | `navvla_vs_vlaad_comparison.md` 부록 |
| 학습 레시피 | 1회 검증 | `train_smolvla.sh` **6회 검증** (v3y→v6), SW서버 3090, 60K ≈ 8.5 h | [servers.md](../servers.md), `ver/` 연대기 |

미해소 실측 항목 (D0 대상): serial baud 115200(코드) vs 9600(Arduino 소스)
불일치, MCU command timeout 유무, steering ±7의 실제 각도, 휠베이스 54 cm
줄자 재확인, Thor 추론 지연 (README 0.74 Hz vs 문서 4.6 s, **3.4× 모순**).

---

## 3. 주 0 — v7 판정 (내일 아침, 가장 먼저)

```bash
# SW서버에서 완주 확인 후 로컬로 pull
ssh autolab_sw@115.145.211.157 'ls ~/sunhong/nav-vla/runs/navvla_smolvla_v7/checkpoints/'
scp -r autolab_sw@…:~/sunhong/nav-vla/runs/navvla_smolvla_v7/checkpoints/060000 $SP/ckpt_v7_60k

# 링 맵 프로브 2회 (주행 8회) → 구간별 비교
bash tools/eval/ring_map_probe.sh $SP/ckpt_v7_60k y7a
bash tools/eval/ring_map_probe.sh $SP/ckpt_v7_60k y7b
python3 tools/eval/compare_segments.py y6v_map.json v6 y7a_map.json v7
```

판정 기준 ([ver/20260806_1045](../ver/20260806_1045_dagger-cycle1-launch.md)):
DAgger 선별에 쓰인 치우침 구간들의 |편차| 감소 + 이탈 0/8 유지 + 양 차선
중앙값 비악화.

분기:

- **성공** → v7을 시뮬 최종본으로 동결. 데모 서빙 교체 + 실차 HG-DAgger
  설계 근거 확보.
- **부분/실패** → 사이클 2 (84포즈 증량) **1회만** 더 돌리고, 그래도 미달이면
  v6 동결. `ver/20260806_0140`의 결론대로 시뮬 정밀도 추격은 여기서 끝낸다.

어느 분기든 **실차 전환 일정은 밀리지 않는다.** 시뮬 체크포인트는 §7의
warm-start 재료일 뿐이고, D0~D2는 체크포인트와 무관한 하드웨어 작업이다.

병행 (이번 주 내, 하드웨어 불필요): ① `track_localizer.yaml`·
`ring_map_probe.sh`의 절대경로(`/home/sh/…`) 파라미터화, ② VLA_AD 작업
사본의 결함 수정 — `lasa_node`를 `pipeline_launch.py`에 추가 +
`was_discarded` 폴백 수정 (비교 실험 모드의 정직성; 데모 경로와 무관),
③ **트랙 자산 저작** — 존·기준선·중앙선·envelope를 템플릿 픽셀로
`track_assets.yaml` 저작 (§4.6), ④ BEV Studio N점 정합 확장 + pose
canonical 프레임 변환 (§4.6 결함 1·2).

---

## 4. 시스템 최종 구성 — Hesai → Thor 연결 상세

### 4.1 물리 배선과 대역폭

```
Hesai OT128 (트랙 긴 변 중앙 고정) ──GbE 직결·고정 IP──► 노트북 (트랙사이드)
    원시 점군 수십 Mbps UDP — 노트북에서 종결. 네트워크 재전송 금지.
    raw rosbag도 노트북 로컬 디스크에만.

노트북 ──WiFi 5 GHz (AP 전용 SSID)──► Thor (차상)
    /track/vehicle_pose    nav_msgs/Odometry   10 Hz   ~1 KB/msg
    /track/vehicle_status  std_msgs/String     10 Hz   JSON(quality/age/heading_valid)
    /track/lap_state       (신규 lap_monitor)  10 Hz   진행률·완료 횟수
    /track/geofence_estop  std_msgs/Bool       이벤트
    /vla/instruction       std_msgs/String     이벤트  (GUI·faster-whisper 원문)
    합계 ≪ 100 KB/s — 대역폭은 문제가 아니다. 문제는 DDS 디스커버리다(§4.3).

Thor ──USB serial 115200(확정 필요)──► Arduino ──► 조향 서보 + 모터
```

표준 메시지(`Odometry`/`String`/`Bool`)만 쓰는 이유: 두 저장소의
`interfaces_pkg` 필드가 달라 커스텀 메시지는 Thor에 중복·동기화 부담을
만든다 (`track_pose_node.py` 헤더에 명시된 설계 결정).

### 4.2 노트북 파이프라인 — 구현 완료분과 잔여

| 단계 | 상태 | 비고 |
|---|---|---|
| hesai_ros_driver → `/lidar_points` | 외부 드라이버 | 고정 IP, PTP 미사용 시 드라이버 스탬프 사용 (`use_sensor_stamp: true`) |
| BEV 크롭 + 바닥 RANSAC + 높이 게이트 | 구현 완료 | **게이트 0.03~0.35 m는 저상 RC 기준 — 1/5 전동차 실루엣으로 재측정 필수.** 이 대역은 서 있는 사람 다리와 겹친다 (`track_localizer.yaml` 주석) |
| 4점 정합 (현수막 ↔ LiDAR) | **BEV Studio로 도구화 완료** | `./lidar_gui.sh` → 4점 클릭 → '최종 확인' → homography 저장. 매 설치 재정합, calibration ID 기록 |
| CTRV UKF 추적 + heading | 구현 완료, 합성 검증만 | `min_speed 0.35 m/s` 이하 heading hold + invalid 플래그(`covariance[35]`), coast 0.5 s 후 발행 중단 + estop |
| 정확도 검증 | **실측 0회** | [validate_track_pose.py](../../src/track_localizer_pkg/scripts/validate_track_pose.py): static <5 cm / heading RMS <3° / 3 m 룩어헤드에서 5° = 횡 26 cm라는 근거까지 내장 |
| lap_monitor | **신규 필요** | pose → 진행률·완료 횟수만 계산. "두 바퀴" 해석 금지 (의미는 정책이 배운다, §5) |

### 4.3 전송 계층과 시간 동기

- **DDS**: WiFi에서 멀티캐스트 디스커버리가 깨진다 — nav-vla 시뮬에서 이미
  "DDS 그래프 붕괴"를 겪었다 (`ver/20260805_1640` §3). 기본값: CycloneDDS +
  정적 peers 리스트(노트북·Thor 고정 IP) + 멀티캐스트 off. 그래도 불안하면
  zenoh-bridge-ros2dds로 격리. `ROS_DOMAIN_ID`는 트랙 전용으로 분리.
- **QoS**: pose/status = BEST_EFFORT, depth 1 (낡은 pose는 무가치 — 최신만).
  estop = RELIABLE + TRANSIENT_LOCAL (늦게 붙은 노드도 마지막 상태 수신).
- **시간 동기**: chrony — 노트북이 서버, Thor가 클라이언트. 게이트: 잔차
  < 30 ms (D2). 정렬은 언제나 `header.stamp`로만 하고 수신 시각은 쓰지 않는다.

### 4.4 장애 모드와 안전 규칙

| 장애 | 검출 | 반응 |
|---|---|---|
| pose 유실/지연 | `pose_age > 0.5 s` | gateway가 주행 명령 0 (D2 게이트 그대로) |
| WiFi 단절 | 위와 동일 + Thor 측 토픽 무소식 | 동일 + **MCU command timeout이 최후 방어선** (D0에서 유무 확인, 없으면 추가) |
| 사람이 차량으로 오검출 | 높이 게이트 대역 겹침 | 게이트 재측정 + `confirm_frames 3` + `vehicle_track_area_only` + 정합 마커는 정합 후 제거 |
| 현수막 재설치 | homography 무효 | 마커 3~4개 → BEV Studio 재정합 → validate 게이트 → calibration ID 갱신 (데모 계획서 §8 절차) |
| 코너 heading 오차 | 17° 피크 (합성) | 기능 6(shortest) 차단 유지. 해소 후보: Thor IMU/차량 모델 융합 |

### 4.5 Thor 내부 최종 배선

```
front camera + [state] + /track/* + /vla/instruction (원문)
    → SmolVLA policy server (온디바이스, ZMQ 계약은 시뮬 그대로)
    → action chunk 30×[dx, dy, dyaw]
    → trajectory tracker      (미터 청크 → 실측 기하 기반 조향/PWM)
    → trajectory_safety_gateway  (freshness·곡률·가속·pose_age·geofence)
    → command mux (유일한 최종 출력) → MotionCommand → serial → Arduino
```

신규 4개: state builder / tracker / gateway / mux (데모 계획서 §7).
기존 LASA는 `VlaIR` 스칼라용 설계라 이 경로에 재사용하지 않는다 — 단
VLA_AD 원 스택(YOLO+pure-pursuit+LASA)은 **비교 베이스라인·데이터 시연자
모드로 온전히 보존**한다.

### 4.6 맵·존 자산 — 시뮬 zone_map의 실차 등가물

핵심 재정의: **맵은 라이다로 받는 것이 아니라 사전 지식이다.** 현수막은
인쇄물이라 기하가 고정이고, 도안(`track2.png` 5228×3594)이 곧 맵이다.
road mask도 라이다가 아니라 템플릿에서 추출한다(`extract_road_mask`).
라이다의 역할은 ① 설치당 1회 정합, ② 차량 pose — 두 가지뿐이므로,
원시 강도맵에서 차선이 희미한 것은 런타임 위험이 아니다 (정합점 수동
선택에만 영향 → 마커로 대체).

| 시뮬 | 실차 등가물 |
|---|---|
| `zone_map.yaml` + `track_paths.json` (world frame) | `track_assets.yaml` (신규) — 존·기준선·차선 중앙선·safe envelope를 **템플릿 픽셀 좌표로 1회 저작**, 설치마다 현재 homography로 자동 투영 |
| gz TF pose | `/track/vehicle_pose` — 단 **canonical 프레임 변환 필수** (아래 ②) |
| navigator 존 감독 | 로직 그대로, pose 소스만 교체 (`track_pose_node.py` 헤더에 (x,y,yaw) 호환 명시) |

현 구현의 결함 3개 — D2 전에 해소한다:

1. **4점 정합은 자기 오차를 잴 수 없다.** 대응점이 정확히 4개라
   homography가 정확히 결정 → 잔차 항상 0 (`inliers [1,1,1,1]` 직접 확인).
   → BEV Studio를 N≥8점 최소자승으로 확장 + 홀드아웃 검증점 2~3개로
   잔차 보고. 스케일은 줄자 기지 거리 1~2개와 대조 (오차 < 1%).
2. **pose가 센서 기준 프레임으로 발행된다.** 재설치 시 좌표계가 통째로
   바뀌어 존 자산·수집 데이터가 무효가 된다. `homography_bev_to_track`이
   이미 저장돼 있으므로 **템플릿 미터 프레임(canonical)으로 변환해 발행** —
   행렬곱 한 줄. 데모 계획서 §8 "track-frame 좌표 저장"의 실체.
3. **현수막 비강체성.** 주름·늘어짐은 평면 homography가 못 잡는다.
   홀드아웃 잔차로 측정, 국소 > 5 cm면 구간별 정합으로 승격.

오차 예산: 정합 ~5–10 cm + pose ~5–10 cm = 스택 **~10–20 cm** < 존 허용치
0.3 m (`zone_map.yaml tol.pos`). 성립하지만 여유가 얇아 결함 1을 고치지
않으면 초과를 검출하지 못한다.

검증 사다리: **V1** 존 모서리에 표적 정치 → pose vs 저작 좌표 대조 →
**V2** 텔레옵 1랩 궤적을 템플릿 오버레이, 차선 배정 안정성 → **V3** 존
도착 감독 dry-run (허용치 내 판정). V1~V3는 D2 PASS 조건에 포함.

**자산 저작과 결함 1·2 수정은 하드웨어가 필요 없다 — 주 0에 착수 가능.**

---

## 5. 체크포인트 2세대 전략 — 상태 계약의 분기

문서 간 충돌이 하나 있다. `navvla_vs_vlaad_comparison.md` §7은 "Hesai는
라벨·평가 전용, 학습 입력 진입 금지"(시뮬 철칙의 연장)라 했고, 데모 계획서
§4.3은 기능 5·6을 위해 pose를 관측 state에 추가하자고 했다. **기능별로 갈라
적용하면 둘 다 옳다:**

| 세대 | state | 커버 기능 | 근거 |
|---|---|---|---|
| 1세대 | `[speed, yaw_rate, steer]` — **v7과 동일 계약** | 1 차선유지 · 2 속도 · 3 차선변경 · 4 목적지 정지 | 시뮬에서 검증된 계약 그대로. 목적지 정지는 시뮬과 동일하게 pose **감독**(모델 밖)이 문장 ""+정지로 전환 — 학습 입력 아님 |
| 2세대 | + `[track_x, track_y, sin/cos(track_yaw), pose_age, lap_progress, laps]` | 5 정확한 N바퀴 · 6 shortest path | 시각 반복 트랙에서 lap 기억은 카메라만으로 원리적 불가. "몇 바퀴"의 의미는 lap_monitor가 아니라 정책이 원문+state로 배운다 |

**1세대 먼저.** 이유: ① 시뮬 레시피·코퍼스 구조 무변경으로 실차 첫 사이클의
변수를 줄인다, ② 관측 차원 변경은 전 코퍼스 재변환 + 전면 재학습이므로
늦출수록 싸다, ③ 기능 1~4만으로도 데모 가치가 성립한다. 2세대 착수 조건:
D6에서 기능 1~4가 각 5회 연속 통과. 코너 heading 미달이 지속되면 기능 6은
열지 않는다 (D2 규칙).

---

## 6. 단계별 실행 계획 (D0~D7 갱신판)

게이트 원문은 데모 계획서 §6이 기준이다. 여기는 갱신점과 순서만 적는다.

### 주 1 — D0 제어 계약·안전 경계 동결

실측 체크리스트 (전부 반나절~하루):

1. Thor 추론 P50/**P95** — `vla_fps` 30분 기록으로 0.74 Hz vs 4.6 s 모순 해소
2. serial baud 115200 vs 9600 — 실제 flash 기준 확정
3. MCU command timeout 유무 — 없으면 추가 (마지막 명령 유지 = runaway)
4. steering ±7의 실제 각도 + 펌웨어 한계인지 임의 클램프인지
5. 휠베이스·최대 조향각·PWM-속도 곡선·카메라 내부/외부 파라미터
6. 데모 언어 동결 (한국어 필수면 bilingual 코퍼스 일정 별도 — Qwen 번역으로
   숨기지 않음)

**PASS**: 알 수 없는 단위·프레임·baud 0개, E-stop/RC override/timeout이
실제로 모터를 정지.

### 주 2 — D1 기준선 + 실차 레코더

3커맨드 주행은 확인됐으므로 남은 것은 **레코더**: Thor 카메라 + 제어 명령 +
Hesai pose + stamp + 안전 개입을 하나의 episode로. Gazebo 의존 recorder를
실차 `Odometry`/Hesai 기반으로 교체.

**PASS**: 재생 가능한 episode + 카메라-행동-pose 정렬 검증.

### 주 2~3 — D2 Hesai 재설치 정합 + 실측

높이 게이트 재측정 → 마커 설치 → BEV Studio 정합 → `validate_track_pose.py`
static/heading → chrony. **현수막을 서로 다른 위치에 3회** 재설치해 반복.

**PASS**: 3회 모두 static < 0.10 m (목표 0.05), heading RMS < 3°(직선),
동기 잔차 < 30 ms, pose ≥ 10 Hz, 유실 0.5 s 규칙 동작. **추가 (§4.6)**:
정합 홀드아웃 잔차 < 0.05 m, 줄자 스케일 대조 < 1%, V1~V3 검증 사다리
통과 (존 좌표 대조·궤적 오버레이·도착 감독 dry-run).

### 주 3~4 — D3 Thor SmolVLA 런타임 + 어댑터

policy server 이식(ZMQ 계약 유지) → eager/compile/TRT export 지연 벤치마크
→ tracker(실측 기하) + gateway + mux → wheels-up 부호·포화·timeout 검증 →
빈 공간 저속 직선/원.

**PASS**: 부호·단위 테스트 전부, 강제 장애마다 500 ms 내 정지, 재출발 0,
queue underrun < 1%.

### 주 3~5 (병행) — 시뮬 재정합

Gazebo 월드를 실트랙 치수(~15×11 m)로, Ackermann을 실측 휠베이스로, 카메라
FOV·렌즈 왜곡 정합, **조향 15단계 양자화를 시뮬 플랜트에도 이식** (D5의
Gazebo 게이트가 실차를 예측하려면 필수).

**PASS**: 재정합 시뮬에서 욜로 선생 편차가 현 수준(0.16~0.30 m) 재현 +
같은 스크립트 경로의 시뮬/실차 종점 발산 < 0.3 m.

### 주 5~8 — D4 실차 paired 코퍼스

- 시연자 = 기존 YOLO 스택 / teleop / 수집 전용 route oracle
- **행동 라벨 = Hesai ego-motion 오프라인 양방향 평활 → 10 Hz SE(2) 청크.**
  integer steering 복사 금지 (시뮬 §10-4 교훈: 온라인 필터 라벨 오차는 ADE
  목표의 3.6배)
- 같은 시작 pose·관측에서 문장만 다른 paired 시연 (시뮬 counterfactual 설계
  이식), off-ring 가드 → off-track 가드 이식, 실패율 기록
- typed + STT transcript 저장, heldout paraphrase 물리 분리, track-frame 좌표

**PASS**: 기능×시작점×차선×속도 coverage 빈칸 0, 핵심 셀 pair ≥ 5,
stamp/frame/action 검증 통과.

### 주 8~10 — D5 재학습 + 게이트 (사이클 반복)

§7 사이클로 SW서버 재학습 → open-loop 반사실 (방향 정확도 ≥ 80%) →
재정합 Gazebo 폐루프 (기능별 ≥ 4/5) → 통과 시에만 실차.

### 주 10~14 — D6 실차 단계 상승

`wheels-up → 빈 공간 → 저속 1 lap → 속도 → 차선변경 → 목적지 정지` 순서
고정, 각 5회 연속. 기능 1~4 완료 시점에 §5의 2세대 착수 여부 결정.

### 주 14~16 — D7 패키징 + 논문 실험

preflight 자동화, cold-start 3회, 전 run에 ckpt hash + git SHA +
calibration ID + rosbag. 논문 실험: 실차 반사실 (≥ 0.5 m), A0 vs VLA-only
어블레이션, HG-DAgger 사이클 1회.

---

## 7. 학습 사이클 — 데이터 왕복

```
[실차 수집] Thor(카메라·명령) + 노트북(Hesai pose, raw bag 로컬)
    → 노트북에서 오프라인 평활 + resample + package + verify
    → scp → SW서버 ~/sunhong/nav-vla/data/<pack>
    → to_lerobot.py → train_smolvla.sh (3090, 60K ≈ 8.5 h, batch 8)
    → checkpoint scp → Thor $SP/ckpt_<tag>
    → wheels-up 스모크 → 실차 프로브 → docs/ver/ 기록
```

- 사이클당 약 1주 (수집 2~3일 + 학습 하룻밤 + 평가 1일).
- 시뮬 v6/v7 체크포인트는 **warm-start 재료로만** 쓴다. 1세대 계약이 동일해
  가능하고, 처음부터보다 표현이 앞서 있다는 가정은 첫 사이클에서 from-scratch
  대조 1회로 확인한다.
- 모든 사이클은 `docs/ver/`에 음성 결과 포함 기록 (기존 원칙).

---

## 8. 게이트 요약표

| 단계 | 게이트 | 수치 |
|---|---|---|
| 주 0 | v7 판정 | 치우침 구간 개선 + 이탈 0/8 + 중앙값 비악화 |
| D0 | 계약 동결 | 미확인 단위·baud 0개, E-stop/timeout 실동작 |
| D1 | 레코더 | 카메라-행동-pose 정렬 episode 재생 가능 |
| D2 | Hesai pose | 재설치 3회 모두: static <0.10 m, heading <3°, 동기 <30 ms, ≥10 Hz |
| D2+ | 맵·존 자산 | 정합 홀드아웃 잔차 <0.05 m, 스케일 <1%, V1~V3 통과 (§4.6) |
| D3 | 런타임 | 장애 시 500 ms 내 정지, 재출발 0, underrun <1% |
| 시뮬 재정합 | 충실도 | 선생 편차 0.16~0.30 m 재현, 시뮬/실차 종점 발산 <0.3 m |
| D4 | 코퍼스 | coverage 빈칸 0, 셀당 pair ≥5, 라벨=평활 ego-motion |
| D5 | 학습 | open-loop 방향 ≥80%, heldout ≥0.8×, Gazebo ≥4/5, 위반 0 |
| D6 | 실차 | 기능별 5회 연속, collision/개입/watchdog 0 |
| D7 | 데모 | cold-start 3회, 전 기능 순서 재현 |

---

## 9. 위험 (갱신분만 — 나머지는 데모 계획서 §10 유효)

| 위험 | 대응 |
|---|---|
| 조향 15단계가 청크 정책 이득을 소거 | D0에서 펌웨어 해제 가능성 최우선 확인. 불가면 D4 게이트에 "양자화 통과 후에도 반사실 분리 유지" 포함해 조기 검출 |
| Thor 지연 실측이 4.6 s 쪽이면 커밋 파탄 | D3에서 TRT export·청크 길이 단축 재학습을 예비 경로로. 0.6 s 커밋 붕괴 전력(`ver/20260730_0910`)이 하한 |
| v7 실패 | §3 분기 — 일정 영향 없음 |
| WiFi DDS 디스커버리 | §4.3 설정을 D2에서 리허설. 시뮬의 붕괴 처방전(전 노드 kill + `/dev/shm` 정리) 문서화돼 있음 |
| 코너 heading 17° | 기능 6 차단 규칙 유지. IMU 융합은 2세대와 함께 검토 |

---

## 10. 미해결·결정 필요

- **2세대 state 계약 착수 시점** — 기능 1~4 통과 후 사용자 결정 (§5)
- **한국어 범위** — D0에서 동결. bilingual 코퍼스는 일정에 별도 항목
- **논문 매핑** — 시뮬 결과(nav-vla)와 실차 결과가 한 편인지 두 편인지,
  VLA_AD Access 플랜과의 관계 미정
- v7 결과 (내일 아침) → 이 문서 §3 분기 실행 후 `docs/ver/`에 기록
- `validate_track_pose.py` 실측 0회 — D2가 첫 실행
- 현수막 비강체 왜곡의 실측 크기 미지 — 홀드아웃 잔차가 첫 데이터.
  국소 > 5 cm면 구간별 정합 승격 (§4.6 결함 3)

---

## 11. 관련 문서

- 게이트 원문·기능 정의: [vla_thor_demo_and_course_plan.md](../vla_thor_demo_and_course_plan.md)
- 두 저장소 비교·결함 목록: [navvla_vs_vlaad_comparison.md](navvla_vs_vlaad_comparison.md)
- 파이프라인·판정: [vla_training_comparison.md](../vla_training_comparison.md)
- 한계: [generalization_limits_and_questions.md](../generalization_limits_and_questions.md)
- 서버·격리 규칙: [servers.md](../servers.md)
- 측위 최초 기록: [ver/20260727_2311_track-localizer.md](../ver/20260727_2311_track-localizer.md)
- 연대기: [ver/README.md](../ver/README.md)

---

## 부록 — 근거 수준

**직접 검증 (이 세션에서 코드·설정 확인)**: `track_pose_node` 토픽·QoS 설계,
`track_localizer.yaml` 전 파라미터(높이 게이트 재측정 경고 포함),
`validate_track_pose.py` 게이트 수치, BEV Studio 존재·워크플로, `tools/eval/`
프로브 체인, `train_smolvla.sh`, 서버 접속 정보.

**문서 전사 (실측 아님)**: heading 코너 17°/1.3 s (합성 궤적), Thor 지연
0.74 Hz/4.6 s (상호 모순, D0에서 해소), baud 불일치·MCU timeout (데모 계획서
D0 목록), 휠베이스 54 cm.

**추정 (실측으로 대체 예정)**: 주차별 기간 전부 — D0~D2 실측 전 납기 아님.
OT128 대역폭 "수십 Mbps"는 자릿수 추정. WiFi 전송량 합계는 메시지 크기
계산값.
