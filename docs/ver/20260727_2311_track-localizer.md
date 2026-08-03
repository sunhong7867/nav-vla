# 2026-07-27 23:03 — `track_localizer_pkg` 신규 + 패키지 `_pkg` 리네임

> 1주차 목표(§Hesai pose 발행)의 코드 작업. 관련 문서:
> [vla_readiness_and_roadmap.md](../vla_readiness_and_roadmap.md) §6,
> [vla_ad_nl_integration.md](../vla_ad_nl_integration.md) §7

---

## 1. 왜 했는가

VLA_AD(실차 Thor)에는 **전역 좌표계가 없다.** pose·odom·맵을 소비하는 노드가 하나도 없어
"목적지로 이동"이 원천적으로 불가능하다. 트랙 긴 변 중앙에 **고정 설치된** Hesai OT128이
이 공백을 외부에서 메운다. 차량은 아무것도 안 보내고, 노트북이 차량 위치를 계산해 Thor로 넘긴다.

기존 자산 `0725_4점정합ver/live_bev_intensity_viewer.py`는 검출까지는 하지만
**CSV로만 기록하고 ROS로 발행하지 않으며, heading을 산출하지 않는다.** 이 두 개가 이번 작업이다.

---

## 2. 추가된 것 — `src/track_localizer_pkg/`

```
track_localizer_pkg/
  track_localizer_pkg/
    bev_detector.py     BEV 크롭 → 바닥평면 상대 높이 게이트 → 클러스터 → centroid
    heading.py          CTRV 무향칼만필터, heading 추정
    track_pose_node.py  ROS 노드
  config/
    track_localizer.yaml
    alignment/          track2.png, homography, 4점 (정합 자산 저장소에 편입)
  scripts/
    validate_track_pose.py   정확도 게이트 측정 도구
```

### 발행 토픽

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/track/vehicle_pose` | `nav_msgs/Odometry` | x=forward_m, y=lateral_m, yaw=heading, twist=속도 |
| `/track/vehicle_status` | `std_msgs/String` | JSON: status, 점 개수, heading_valid, 속도 |
| `/track/geofence_estop` | `std_msgs/Bool` | OFF_ROAD 또는 추적 소실 시 True (fail-closed) |

### 설계 결정 3개

**① 커스텀 메시지를 쓰지 않는다.** 두 저장소 모두 `interfaces_pkg`를 갖고 있는데 필드가 다르다
(통합 설계서 §9). 커스텀 메시지를 만들면 Thor 쪽에 복제하고 동기화해야 한다.
`Odometry` + `String` + `Bool`은 인터페이스 패키지 없이 어디서나 빌드된다.

**② heading 유효성을 공분산에 실어 보낸다.** `pose.covariance[35]`가 heading이 실측인지
홀드값인지 구분한다(유효 1e-4, 홀드 1e6). 이 플래그를 무시하는 소비자는 차가 설 때마다
낡은 heading으로 조향하게 된다.

**③ pose는 BEST_EFFORT로 보낸다.** WiFi로 Thor에 가는데, 유실된 프레임의 재전송이 더 최신
프레임 뒤에 쌓이면 안 된다. 소비자가 마지막 값을 홀드하는 구조에서는 **늦은 pose가 없는
pose보다 나쁘다.**

### 기존 검출 로직은 그대로 이식했다

`live_bev_intensity_viewer.py`의 기하를 **비트 단위로 동일하게** 옮겼다. 정합 homography가
그 기하에서 찍혔기 때문이다. 캔버스 크기가 다르면 노드는 도로 마스크를 **로드하지 않고 거부한다**
(어긋난 마스크를 조용히 그리는 것보다 낫다).

검증: 캔버스 240×320 (homography가 기대하는 값과 일치), world→px→world 왕복 오차 ±0.025 m
(= 0.05 m/px 격자의 반 픽셀, 즉 양자화 한계).

---

## 3. heading — 측정 결과와 **미달 사실**

heading은 이 프로젝트에 없던 기능이라 새로 만들었고, **목표(RMS < 3°)를 부분적으로만 달성했다.**

### 3.1 등속도(CV) 모델은 폐기했다

처음에 [x, y, vx, vy] 등속도 칼만필터로 만들고 `atan2(vy, vx)`로 heading을 뽑았다. 합성 궤적 결과:

| 조건 | heading RMS | 위치 오차 |
|---|---|---|
| 직선 1.5 m/s | 2.77° | 3.4 cm |
| 직선 0.8 m/s | 5.79° | 3.4 cm |
| 직선 0.3 m/s | 15.57° | 3.0 cm |
| **선회 1.5 m/s** | **64.03°** | **269 cm** |

선회에서 완전히 발산했다. 원인 두 가지:
- **버그**: 프로세스 잡음 Q를 `outer(G, G)`로 만들어 rank-1이 되었다. x/y 가속 잡음이 완전
  상관인 모델, 즉 "가속은 항상 45° 방향으로만 일어난다"는 뜻이 되어 선회를 표현 못 한다.
- **모델 부적합**: 차량은 등속직선이 아니라 **선회율을 갖는** 물체다.

### 3.2 CTRV 무향칼만필터로 교체

상태를 `[px, py, v, psi, psi_dot]`로 바꿔 heading과 선회율을 **상태에 직접 넣었다.**
코너가 필터가 싸워야 할 외란이 아니라 추종하는 상태가 된다.

구현 중 만난 함정 2개(둘 다 수정):
- UKF 스케일 파라미터 α=0.3이 `w_c[0] = -7.2`라는 큰 음수 가중치를 만들어 공분산이 PSD를
  잃고 Cholesky가 런타임에 실패했다. → α=1, κ=0 (λ=0)으로 모든 가중치를 비음수로.
- 초기 heading 분산을 π²로 주면 시그마 포인트가 psi 방향으로 몇 radian씩 퍼져 wrap이 일어나고
  공분산이 무의미해진다. → **0.3 m 이동할 때까지 측정을 버퍼링**했다가 변위 방향으로 heading을
  시딩한다. 그 전까지는 위치만 발행하고 heading은 invalid로 표시.

### 3.3 프로세스 잡음 스윕 — 정상상태 vs 과도응답이 상충한다

centroid 잡음 σ=6 cm, 10 Hz 기준:

| accel_std / yaw_accel_std | 직선1.5 | 직선0.8 | 코너1.5 | 코너0.8 | 위치 |
|---|---|---|---|---|---|
| 1.00 / 2.00 | 5.39° | 7.47° | 6.07° | 8.51° | 4.8 cm |
| 0.50 / 1.00 | 3.86° | 5.31° | 4.43° | 5.96° | 4.3 cm |
| **0.30 / 0.50** | **2.75°** | 3.80° | **3.15°** | 4.13° | **3.8 cm** |
| 0.20 / 0.30 | 2.14° | 2.97° | 2.46° | 3.20° | 3.5 cm |
| 0.10 / 0.15 | 1.54° | 2.13° | 1.83° | 2.35° | 3.1 cm |
| 0.05 / 0.08 | 1.14° | 1.56° | 1.47° | 1.83° | 2.7 cm |

정상상태만 보면 조일수록 좋다. **그런데 코너 진입 과도응답은 정반대다** (0.6 rad/s 스텝, 1.2 m/s):

| accel/yaw_accel | 진입 피크 | 3° 복귀까지 | 코너 직후 RMS |
|---|---|---|---|
| **0.30 / 0.50** | **17.33°** | **1.3 s** | **5.33°** |
| 0.20 / 0.30 | 18.98° | 1.7 s | 7.19° |
| 0.10 / 0.15 | 21.72° | 2.2 s | 10.83° |
| 0.05 / 0.08 | 25.86° | 2.9 s | 15.97° |

→ **`accel_std=0.3, yaw_accel_std=0.5`로 확정.** 직선은 이미 목표를 통과하므로 과도응답을 우선했다.

### 3.4 결론 — 정직하게

> **heading RMS < 3°는 직선에서만 성립한다. 코너 진입에서는 17° 피크가 나고 1.3초 걸려 회복한다.
> 재튜닝으로 해결되지 않는다 — 조일수록 과도응답이 악화된다.**

위치로부터 유추하는 heading은 선회율 스텝을 원리적으로 추종할 수 없다. 이걸 메우려면
**실제 방향 관측**(직사각/L-shape 피팅을 2차 업데이트로 융합)이 필요하고, 이번 버전에는 **의도적으로
넣지 않았다.**

**실무적 함의 — 이게 중요하다:**
- **lane-follow 존 내비게이션은 이 pose로 충분하다.** navigator의 lane-follow 모드는 스티어링을
  계산하지 않는다. 존 도착 판정(위치)만 하고 조향은 카메라 lane follower가 한다.
  위치는 3.8 cm로 목표를 통과한다.
- **direct 모드(pose 기반 pure-pursuit)는 코너에서 쓰면 안 된다.** heading에 조향 루프를 닫는
  유일한 모드이고, 정확히 그 지점이 취약하다.

즉 1주차 산출물은 **Tier 2 lane-follow를 지원하기에 충분하고, direct 추종에는 부족하다.**

---

## 4. 패키지 리네임 — `_pkg` 접미사 통일

저장소의 다른 패키지는 전부 `_pkg`인데 두 개만 예외였다.

| 이전 | 이후 |
|---|---|
| `src/nav_vla/nav_vla/` | `src/nav_vla_pkg/nav_vla_pkg/` |
| `src/track_localizer/track_localizer/` | `src/track_localizer_pkg/track_localizer_pkg/` |

- `git mv`로 옮겨 이력 보존.
- 치환은 **단어 경계 기준**(`sed -E 's/\bnav_vla\b/nav_vla_pkg/g'`)으로 했다.
  ROS 노드 이름 `nav_vla_chat_gui_node`와 환경변수 `NAV_VLA_WHISPER_MODEL`은 의도적으로
  **건드리지 않았다** — 패키지 이름이 아니다.
- 영향: 28개 파일 / 124곳. `package.xml`, `setup.py`, `setup.cfg`, `resource/`, entry point,
  import, `simulation_pkg`의 launch 파일과 의존성, docs 경로.
- 리네임 전 바이트코드가 `bad marshal data`를 일으켜 `__pycache__`와 `*.pyc`, 그리고
  `build/nav_vla`·`install/nav_vla`를 정리했다.

**검증**
```
colcon build --packages-select track_localizer_pkg nav_vla_pkg --symlink-install
  -> 2 packages finished [2.02s]
ros2 pkg executables track_localizer_pkg
  -> track_localizer_pkg track_pose_node
```

---

## 5. 실행 방법

```bash
# 라이다 드라이버
source ~/ROS2_project/hesai_ws/install/setup.bash
ros2 launch hesai_ros_driver start.py

# pose 노드
cd ~/ROS2_project/nav-vla && source install/setup.bash
ros2 run track_localizer_pkg track_pose_node \
    --ros-args --params-file src/track_localizer_pkg/config/track_localizer.yaml

# 확인
ros2 topic echo /track/vehicle_status
ros2 topic hz /track/vehicle_pose        # 라이다 프레임레이트와 같아야 함
```

정확도 게이트 측정 (트랙 실측 필요):
```bash
# 측량한 지점에 차를 세우고
python3 src/track_localizer_pkg/scripts/validate_track_pose.py static --truth 5.00 1.20 --label P1
# 측량한 직선을 천천히 주행
python3 src/track_localizer_pkg/scripts/validate_track_pose.py heading --from 3.0 0.0 --to 10.0 0.0
# 종합
python3 src/track_localizer_pkg/scripts/validate_track_pose.py report
```

---

## 6. 미해결 / 다음

1. **실측 검증을 아직 안 했다.** §3의 수치는 전부 **합성 궤적**이다. 실제 centroid 잡음이
   6 cm인지 15 cm인지 모른다. rosbag과 실차로 `validate_track_pose.py`를 돌려야 한다.
   이게 게이트 통과의 실제 근거가 된다.
2. **높이 게이트를 차량 실루엣에 맞춰야 한다.** 현재 바닥 기준 0.03~0.35 m는 낮은 RC 섀시용이고,
   어제 사람이 검출된 것도 사실상 **다리**를 잡은 것이다. 1/5 전동차 실측 후 조정.
   조정 안 하면 트랙 주변 사람이 후보로 잡힌다.
3. **centroid 편향 미측정.** 단면 반사의 중심점은 차량이 회전하면 센서를 향한 면 쪽으로
   최대 차폭의 절반까지 이동한다. static 측정에서 bias와 jitter를 분리해 봐야 한다.
4. **시간 동기 미설정.** 노트북↔Thor chrony, `header.stamp` 정렬, flash-and-maneuver 검증,
   잔차 < 30 ms (1.5 m/s에서 4.5 cm).
5. **원거리 성능 미확인.** 어제 이미지에서 링 간격이 원거리로 갈수록 눈에 띄게 벌어진다.
   3 / 8 / 13 m에서 각각 측정해야 한다.
6. **Tier 2 나머지** — `nl_command_node`(헤드리스 파서), `nl_bridge_node`, 존 좌표 재측량,
   navigator 이식. 이번 작업에는 포함되지 않았다.

---

## 7. 이번에 버린 것과 이유

| 버린 것 | 이유 |
|---|---|
| 등속도(CV) 필터 | 선회에서 heading 64° / 위치 269 cm 발산 (§3.1) |
| UKF 스케일 형식(α<1) | 음수 가중치 → 공분산 PSD 상실 → 런타임 Cholesky 실패 |
| 넓은 초기 heading 분산 | 시그마 포인트 wrap. 변위 기반 시딩으로 대체 |
| 프로세스 잡음 조이기 | 정상상태는 좋아지나 코너 진입이 26°/2.9 s로 악화 |
| 커스텀 메시지 | 두 저장소 `interfaces_pkg` 필드 불일치. 표준 타입으로 회피 |
| 형상(L-shape) 피팅 | 원거리에서 한 프레임 점 개수 부족. 단, **차후 2차 업데이트로 융합할 가치는 있음** |
