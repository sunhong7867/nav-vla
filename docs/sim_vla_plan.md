# 시뮬레이션 VLA 계획서

> 목표: nav-vla 시뮬레이션을 **정규식 파서 + 목표를 못 보는 BC 정책**에서 **진짜 VLA**로 전환하고,
> 그 결과를 실차로 이식한다.
> 1차 범위: 시뮬레이션. 실차 이식은 §0.1 결정 5에 따라 본론에 포함(후속 단계).
>
> 작성일: 2026-07-28. 관련 문서: [vla_readiness_and_roadmap.md](vla_readiness_and_roadmap.md),
> [ver/20260727_2311_track-localizer.md](ver/20260727_2311_track-localizer.md)

---

## 0. 한 문장 요약

> **학습을 시작하기 전에, "동일한 관측 + 다른 instruction → 실질적으로 다른 oracle 행동"이
> 데이터 안에 존재함을 수치로 증명하는 게이트(G2)를 통과시켜라. 아키텍처는 그다음이다.**

지금까지의 실패는 전부 이 게이트가 없어서 생겼다. 모델을 키우는 것으로는 고쳐지지 않는다.

---

## 0.1 확정된 결정 (2026-07-28)

| # | 결정 | 결과 |
|---|---|---|
| 1 | **명령은 영어 전용** | SmolVLA 유지. §6의 모든 실측치 유효. 한국어 요구가 재진입하면 트렁크 교체 + §6 전량 무효 |
| 2 | **반사실 축 = 서수 + 경로분기 + 속도** | 서수가 load-bearing. §3.2 그대로 |
| 3 | **일정 = 10~14주 논문 방어 규모** | 코퍼스 v3(2,000 ep / 90k decision frame)까지 범위 안. 4주 압박 해제 |
| 4 | **학습은 연구실 GPU 서버** | 노트북을 24/7 수집기로 전용. *서버 사양 확인 필요* |
| 5 | **논문에 실차 이식 포함** | **track_localizer_pkg가 크리티컬 패스로 복귀** → §0.2 |
| 6 | **시뮬레이터 결정론화 수행** | `use_sim_time` + stepped. R6 함정(pause 시 ROS 타이머 정지)을 Week 1 최우선 |
| 7 | **기존 파서·정책은 어블레이션 암으로 동결** | A-parser / A-classifier / A-stage_a. 제어 경로에서만 제거, 코드는 보존 |
| 8 | **구현·검증을 Claude가 수행** | Gazebo 실행이 필요한 부분만 명령어를 전달하고 결과를 받아 반영 |

### 0.2 결정 5의 파급 — 행동 라벨과 heading

실차 이식이 본론에 들어가면 §10-4의 경고가 **경고가 아니라 요구사항**이 된다.
행동 라벨 `[dx, dy, dyaw]`는 GT pose 유한차분이고, 실차에서 그 pose는 Hesai가 만든다.

[ver/20260727_2311_track-localizer.md](ver/20260727_2311_track-localizer.md) §3의 측정에 따르면
온라인 CTRV 필터의 heading은 직선 2.8°, **코너 진입 17° 피크**다. 3 m 지평에서 17°는 **0.9 m
라벨 오차**로, ADE 목표(0.25 m)의 3.6배다. 그대로 두면 실차 라벨이 코너에서 무의미해진다.

**해결책은 있고, 온라인 필터를 개선하는 것이 아니다.** 라벨링은 **오프라인**이므로 인과성 제약이
없다. 에피소드 전체를 확보한 뒤 **양방향 평활(RTS smoother)** 을 걸면 미래 관측이 과거 heading을
교정한다. 코너 진입의 17° 피크는 본질적으로 "아직 회전을 못 봤다"는 인과 지연이므로, 양방향
패스에서는 사라진다.

→ **작업 항목 추가**: `track_localizer_pkg`에 오프라인 RTS 평활 경로를 만든다(온라인 노드와 별개).
실차 라벨은 반드시 이쪽을 쓴다. 온라인 필터는 주행 중 게이팅·geofence 용도로만 남긴다.
시뮬 단계에서는 GT pose가 정확하므로 무관하지만, **동일한 라벨 파이프라인을 시뮬에서 먼저
검증해 두면 실차에서 새로 만들 것이 없다.**

---

## 1. 목표 — 무엇을 "시뮬레이션 VLA"라 부를 것인가

### 1.1 출하 주장 (단 하나)

> 동일한 world pose에서 시뮬레이터를 리셋하고 카메라 관측이 동일한 상태에서 **instruction
> 문자열만** 바꾸면, 정책이 출력하는 3.0 s SE(2) 경로의 **형상**이 oracle이 만드는 형상 차이의
> **70% 이상**만큼 달라진다. 그 변화는 sampler 노이즈 바닥의 6배 이상이고, **학습에서 본 적 없는
> paraphrase**에서도 유지되며, 제어 경로에는 watchdog 하나를 제외한 어떤 규칙도 없다.

"언어가 행동을 바꾼다" 정도의 약한 주장은 **스칼라 게인 하나로도 통과**한다. 그래서 채택하지 않는다.

### 1.2 4개 기준 → 산출물

| 기준 | 산출물 | 판정 |
|---|---|---|
| (a) 언어+이미지가 한 forward | `lerobot/smolvla_base` | 아키텍처로 자동 충족이지만 **증거로는 공허** — §3의 non-separable 축이 있어야 의미가 생긴다 |
| (b) 연속 multi-DOF 행동 | `action = [dx, dy, dyaw]` (ego-frame SE(2) delta, GT pose 유한차분) | `dyaw` distinct 값 ≥ 10⁴ (**현재 9개**) |
| (c) 언어만 바꿔 행동 변화 | `D_shape`(형상) / `D_sched`(스케줄) 분해 지표 | §7.3 G4 표 |
| (d) 루프 내 규칙 없음 | `vla_bridge_node.py` — watchdog 1개, 전량 계측 | non-watchdog override > 0 이면 **런 무효** |

### 1.3 명시적 범위 밖

- **한국어 instruction** — SmolVLM2 트렁크에서는 불가 (§6.4). 결정 1
- `track.png` 텍스처 편집 — 영구 보류 (§3.4)
- `sim.pt`(YOLOv8 차선 분할) 재학습 — 새 스택에서 YOLO는 제어 경로 밖
- DAgger 규모 복구 데이터, 다중 에이전트/교통, 역주행 코퍼스

---

## 2. 지금 시뮬이 VLA가 아닌 정확한 메커니즘

세 개의 독립 고장이며 각각 다른 기준을 깬다.

**(1) lane 세션 540/619개에서 goal이 노드 밖으로 나가지 않는다.**
`navigator_node.py`는 293–360행에서 stop_x/stop_y/travel_yaw를 전부 계산해 놓고, 362–363행에서
`_publish_lane("lane2")`와 `_publish_motion("start")` **두 개만** 발행한다. lane 이름의 유일한
사용처는 `lane_info_extractor_node.py:176`의 YOLO 클래스 마스크 선택이다. 즉 instruction →
액추에이터 채널 용량이 **lane 1비트 + stop 이벤트 1개**. 조향은 전적으로
`motion_planner_node.py:246-247`이 카메라 스플라인 기울기로 계산한다. → **기준 (c) 실패**

**(2) 기록되는 행동이 연속이 아니라 폐쇄 테이블이다.** (직접 측정)
```
motion_planner_node.py:37
  command = round(7 / (max_target_angle ** 3) * (target_angle ** 3))
```
lane1 세션 3,706 프레임 실측: `cmd.angular` **distinct 값 9개** (정확히 0.0923 rad/s의 배수),
`cmd.linear` **distinct 값 6개**. 전체 코퍼스에서도 k∈[−7,7]의 15개를 넘지 않는다.
→ **기준 (b)는 데이터를 아무리 더 모아도 실패한다.** 기록 대상 자체를 바꿔야 고쳐진다.

**(3) direct 모드는 goal-conditioned이지만 vision-blind이고 네트워크를 우회한다.**
`policy_node.py:455-457`이 `task_type=="direct"`일 때 `_control_direct`(483–561)로 분기해
`/cmd_vel`을 직접 발행 — **모델을 완전히 건너뛴다.** `_arrived()`, `_stop_line_state()`,
`limit_twist_to_raw_speed()`도 전부 제어 경로 안이다. → **기준 (d) 실패**

### 2.1 기존 619 에피소드는 재사용 불가

두 가지 독립적 이유로, goal-conditioning과 **무관하게** 재수집이 강제된다.

1. **타임스탬프가 없고 레이트가 계통적으로 틀렸다.** (직접 측정)
   `data_engine_node.py:239-241`이 `time.monotonic()`으로 게이팅한다. 목표 5 Hz(0.200 s)인데
   실측 프레임 간격 중앙값 **0.2307 s = 4.33 Hz, +15.4% 계통 오차**, 최대 0.3263 s.
   `steps.jsonl`에 타임스탬프 필드가 없어 **사후 복구가 불가능**하다.
   LeRobot v0.6.1은 `dataset_writer.py:225`에서 `timestamp = frame_index / fps`로 **스스로
   지어내므로** 이 오류를 절대 잡아주지 않는다 — 조용히 평탄화된다.
2. **시뮬레이터 캘리브레이션이 바뀌었다.** 커밋 `a2de7d4`가 `simulation_sender_node.py:46`의
   각속도 매핑을 속도 무관 → 속도 의존으로 바꿨다. 619 에피소드는 전부 그 이전(≤2026-07-07)
   데이터라 **오늘의 시뮬레이터와 yaw 캘리브레이션이 어긋나 있다**(1.37 m/s에서 약 −49%).

> 기존 코퍼스의 용도: **차선유지 warm-start**와 **음성 결과 어블레이션 표**. 그 외에는 없다.

---

## 3. 반사실 축 — 무엇이 진짜로 언어를 필요로 하는가

### 3.1 설계 규칙 (이 절의 나머지를 전부 결정한다)

> **어떤 언어 축이 기준 (c)에 기여하는 것은, 그 instruction이 관측 이력으로부터 복원 불가능한
> 프레임에서뿐이다.**

"행동이 달라지는 프레임 비율"을 density라고 부르면 안 된다. 올바른 지표는 **행동이 달라지면서
동시에 관측이 어떤 instruction이었는지를 폭로하지 않는 프레임 비율**이다.

| 축 | 관측/상태로 복원 가능? | 언어가 실제로 load-bearing인 구간 |
|---|---|---|
| 속도 | **가능** — `observation.state[0]=speed`. BC 최소손실 해가 `v_target(t) ← v_measured(t)` | 가감속 램프 1~2 s |
| 차선 | **가능** — 3.5 m 횡 오프셋은 차선 카메라에서 가장 두드러진 특징 | 명령 프레임 1개 |
| 경로(분기 후) | **가능** — 광장 안에 있으면 이미 들어온 것 | 진입 접근 2~3 s |
| **서수(ordinal)** | **불가능** — 첫 개구부 앞 이미지는 두 지시에서 동일 | 결정 프레임 + 접근 전체 |

이것이 "속도는 100% 프레임에서 다르다"가 위험한 이유다. **프레임 기준으로 참이고 정보 기준으로
거짓이다.**

### 3.2 채택 축

**Tier 1-A. 서수 / 개수 — 기하가 아니라 기억** ← *유일하게 scene-derivable하지 않은 축*

트랙은 하나의 폐루프이고, 광장 경계에 개구부가 **2개** 있다(x≈−2, 북측 y≈+21.6 / 남측 y≈−20.4;
**좌표는 미검증, 폴리라인 저작 전 재확인 필요**). CCW 주행 순서상 북측이 첫 번째다.

`"take the first opening"` vs `"take the second opening"` → **북측 개구부 앞에서 관측은 픽셀
단위로 동일**하고 oracle 행동은 좌회전 진입 vs 통과로 갈린다. **이 트랙에서 얻을 수 있는 유일한
완전한 반사실이다.** 여기에 기준 (a)의 증거가 걸린다 — 이미지에서 개구부를 찾는 **동시에** 지시를
읽어야만 풀리므로 **non-separable**하다. 비용 0, 텍스처 편집 불필요, 호길이 기반 카운터 5~10줄.

**Tier 1-B. 경로 분기 — 형상 divergence**
- 광장 통과 42 m ≈ 무조향 vs 루프 우회 64–74 m ≈ 연속 ±0.17 rad/s. 4초 안에 10 m 이상 벌어진다.
- 광장 안 **BayGroupA(x≈+3.3, Slot1–4)** vs **BayGroupB(x≈−8.8)** — 약 12 m 떨어진 좌/우 분기.
- **Slot1 vs Slot2(3.6 m 간격)는 절대 쓰지 않는다.** goal_zone 병리를 3.6 m 스케일로 재현할 뿐이다.
- `track.world`에 물리 장벽이 없다 — 도로·광장·잔디가 하나의 평면이다. 따라서 **분기는 텍스처
  편집 없이 오늘 이미 존재한다.**

**Tier 2. 속도 — 스케줄 divergence** `{0.7, 1.2, 1.8} m/s`
경로가 동일한 채 행동만 달라지는 유일한 축이라 (c)의 가장 깨끗한 시연이다. 현 코퍼스에 속도값이
6개뿐이라 정책이 속도를 *표현조차 못 하는* 병리도 함께 고친다.
**단독 출하 절대 금지.** §3.1 누출을 셋으로 완화: ① 에피소드 초기 속도를 지시 속도와 **독립적으로**
U[0.5, 2.0] 무작위화, ② 속도 축의 증거는 정상상태가 아니라 **전이 램프**에서만 인정,
③ `observation.state`만으로 행동을 예측하는 **state-only baseline(A6)** 결과를 반드시 병기.
상한 1.8 m/s인 이유: 3.0 s 지평 = 5.4 m로 카메라 가시 원뿔 안. 기존 2.75 m/s로 수집하면 3초
라벨(8.25 m)이 관측 밖으로 말려 나가 비가역적 손실 바닥을 만든다.

**Tier 3. 차선(inner/outer)** — 공짜지만 논거가 아니다. 첫 슬립 때 컷(§8.3).

### 3.3 기각 축

| 축 | 기각 사유 |
|---|---|
| **goal_zone (현 주축)** | 단방향 폐루프에서 모든 zone은 같은 경로로 도달된다. 목표는 *어디서 멈출지*만 정한다. 측정값 **0.063 m = 0.13 LSB** |
| **주행 방향(CW/CCW)** | 숫자는 가장 크지만(0.33 rad/s) 1초 뒤부터 이미지가 방향을 폭로한다 → 속도와 같은 병리에 코퍼스 2배 비용 |
| **direct vs lane 모드** | 언어 축이 아니라 **컨트롤러 스위치**. 지시가 "어느 프로그램을 돌릴지" 고르는 것 = **기준 (d) 정면 위반** |
| **Slot1–4 선택** | 3.6 m 간격 한 줄, 같은 방향 진입 → 마지막 몇 미터에서만 분기 |
| **1차선 vs 2차선** | 과도 전용. 정착하면 두 지시의 행동이 동일 |
| **"교차로에서 어느 쪽"** | **이 트랙에 교차로는 없다.** T자·로터리를 전제로 계획하지 말 것 |

### 3.4 텍스처 편집: 영구 보류

`sim.pt`(YOLOv8 차선 분할)가 540/619 에피소드의 행동을 만들었다. 텍스처를 바꾸면 perception이
OOD가 되고 기존 자산 전체가 원자적으로 무효화된다. 가독성 목적 외에는 하지 않는다.

---

## 4. 오라클 — 기록된 행동을 goal-conditioned로 만드는 패치

### 4.1 원칙

**기존 제어 경로를 고치지 않는다. 대체한다.** 새 노드가 `/cmd_vel`을 단독 발행하며
`motion_planner_node.py:37`의 `round()`를 **경유하지 않는다.** 이것이 기준 (b)를 데이터 수준에서
고치는 유일한 조치다. `navigator_node.py` / `policy_node.py` / `motion_planner_node.py` /
perception 파이프라인은 **그대로 둔다** — 실차용 15단계 경로는 배포용으로 온전히 보존된다.

### 4.2 `route_oracle_node.py` (~250 LOC, 약 95는 `simple_track_driver_node.py`에서 이식)

| 방향 | 토픽 | 타입 | 내용 |
|---|---|---|---|
| sub | `/oracle_goal` | String(JSON) | `{"branch":"first"|"second"|"none", "bay_group":"A"|"B"|null, "target_speed":0.7|1.2|1.8, "lane":"inner"|"outer"}` |
| pub | `/cmd_vel` | Twist | **단독 컨트롤러. 연속값. 재양자화 없음** |
| pub | `/oracle_action` | String(JSON) | **양자화 이전 float 전량**: curvature, steer_angle_rad, yaw_rate, target_speed, lateral_err, heading_err, s_remaining, openings_passed, route_id |
| pub | `/nav_status` | String | `"arrived: <id>"` 유지 → `data_engine_node.py:214-219` 무수정 |

**폴리라인 6개, route graph 아님.** 그래프에 attach index·spur·법선 오프셋을 계산하는 설계는
R=16.8 m 코너에서 자기교차하고 기하 디버깅에 3일이 더 든다. 대신 손으로 저작한 폴리라인 6개 +
최근접점 조인: `loop_inner`, `loop_outer`, `gateN→bayA`, `gateN→bayB`, `gateS→bayA`, `gateS→bayB`.
0.35 m 간격 densify.

**상태 기계**: `IDLE → TRACK → ARRIVED → IDLE`, 이탈 시 `ABORT(timeout|off_polyline|stuck)`.
- `TRACK` @ 20 Hz. pose는 **반드시 `WorldPoseStream`**(gz CLI ground truth).
  ⚠️ `/odom`은 DiffDrive 휠 적분이라 **텔레포트를 따라오지 않는다.**
  `simple_track_driver_node.py:70-75`가 `/odom`을 쓰므로 **그대로 이식하면 조용히 틀린다.**
- pure pursuit: `simple_track_driver_node.py:232-263`(closest/lookahead)과 `:168-209`(곡률
  `2·local_y/L²`, 속도, 가속 제한) 이식. 목표속도만 파라미터로 대체.
- **개구부 카운터**: 루프 호길이 위 gateN/gateS 마커 2개. `openings_passed` 증가,
  `branch=="first"` → 첫 마커에서 spur 전환, `"second"` → 첫 마커 통과 후 두 번째에서 전환.
  **이 5~10줄이 Tier 1-A 축 전부를 만든다.**
- pure pursuit은 후진하지 않는다. bay 진입은 전진 전용 설계.

> ⚠️ **함정**: `zone_map.yaml`의 `pose.yaw`는 heading이 아니라 **게이트/정지선의 방향**이며
> 실제 진행방향과 모든 zone에서 약 90° 어긋난다. 폴리라인 저작 시 zone yaw를 목표 heading으로
> 읽으면 안 된다.

### 4.3 `gz_reset.py` (~45 LOC)

`/world/default/set_pose` 서비스 사용 가능(`track.world:12-15`가 `UserCommands` 로드).
```
pause → /cmd_vel에 Twist() 발행(set_pose는 body twist를 0으로 만들지 않는다)
      → set_pose → unpause → query_world_pose() 1회(스트림 stale 방지)
      → /camera/image_raw 콜백 3회 이상 신규 수신 후 기록 시작
```
pause 코드는 `zone_capture_gui_node.py:84-103`에 이미 있다.
**금지**: `/world/default/control reset:{all:true}`(동적 스폰된 `ego_vehicle` 파괴),
`load_ego_car_node`/`basic.reset_model`(모델 제거+재생성, 호출당 수 초, 고정 pose 1개만 지원).

### 4.4 오라클의 두 번째 역할

1. **베이스라인** — 정책 성능은 항상 oracle 대비로 보고한다(G4 주지표가 `D_shape/D_oracle_shape`인 이유).
2. **허용성 필터** — 반사실 쌍은 **oracle 궤적이 지평 안에서 2 m 이상 벌어질 때만** 평가에 포함한다.
   그렇지 않은 쌍을 채점하는 것은 **반증 불가능한 것을 채점하는 것**이다. 필터 적용/미적용 둘 다 보고.

---

## 5. 데이터

### 5.1 Recorder v2 — 설계 규칙

**기록 시점에 rate limiting을 하지 않는다.** 소스 레이트로 전부 기록하고 정확한 격자로는
오프라인 리샘플한다. ROS 콜백을 정확히 10.000 Hz로 때리려는 시도가 +15.4% 오류를 만들었다.
카메라는 이미 30 Hz/640×480이라 10 Hz 목표에 3배 오버샘플 → 최악 정합오차 16.7 ms(1.5 m/s에서 2.5 cm).

- **시계**: 이미지 메시지의 `header.stamp`(sim time), float64 초. `time.monotonic()` **금지**.
  `use_sim_time:=true` 전역 적용 (현재 `src/` 전체에 **0회** 등장).
- **스레딩**: 콜백은 `queue.put((stamp, frame))`만. JPEG 인코딩·디스크 I/O는 writer 스레드.
- **반올림 금지**: 전 필드 float64 원본. 현재의 `round(cmd, 4)`가 양자화 이전 신호를 파괴한다.

### 5.2 온디스크 스키마

```
<out>/<session>/ep_XXXX/
    meta.json        # 에피소드 단위
    frames/          # 000000.jpg …  소스 레이트(~30 Hz), 데시메이션 없음
    frames.jsonl     # {t, seq, file, enc, w, h}
    poses.jsonl      # {t, x, y, yaw(unwrapped), z, roll, pitch}
    control.jsonl    # 아래
    events.jsonl     # {t, kind, data}
```

**세 개의 독립 dense 스트림 + 각자의 타임스탬프 → 오프라인 조인.** 오늘의 `_img_cb`는
image/pose/cmd를 타임스탬프 없이 한 순간에 융합하므로 어긋남이 복구 불가능하다.
**이것이 가장 중요한 구조 변경이다.**

**`control.jsonl`** (오늘 전혀 없는 스트림) — 세 계층을 전부 기록해 action space 논쟁을 다시 하지 않는다:

| 필드 | 의미 |
|---|---|
| `oracle_curvature`, `oracle_v`, `oracle_steer_rad` | oracle 연속 출력 (**1계층**) |
| `steer_float` | `motion_planner_node.py:37`의 `round()`에 들어가는 값 (**2계층**, 실차 이식 비교용) |
| `steer_cmd`, `speed_cmd` | 실제 발행된 정수 (**3계층**) |
| `v_cmd`, `steer_angle_rad` | 실행된 물리량 |
| `source` | `route_oracle` \| `lane_follower` \| `teleop` |
| `override` | `none` \| `watchdog` \| `stop_line` \| `timeout_stop` \| `clamp_speed` … |

`override`는 **기준 (d)의 감사 기록**이다. 프레임 단위 기록 없이 "규칙이 개입하지 않았다"고
주장할 수 없다.

**`meta.json` 핵심 필드**: `schema_version=2`, `instruction`, `intent_id`(paraphrase 불변),
`intent_slots`, **`termination ∈ {success, timeout, collision, off_track, stuck, operator_abort}`**
(bool 대체), `cf_group_id`, `cf_variant_id`, `start_pose_key`(양자화 해시), `seed`, `world`,
`git_sha`, `sim_rtf`, `camera`.

**실패를 반드시 기록한다.** 현재 619/619가 success라 정책은 "선을 벗어났을 때 무엇을 할지"를
한 번도 본 적이 없다. 초기 BC 손실에서는 제외하되 **버리지 않는다.**

### 5.3 오프라인 리샘플과 수용 게이트

`fps=10`(int), `t0` = `settled` 이벤트 후 첫 프레임, `grid[k] = t0 + k/10`.
프레임은 최근접, pose는 선형보간(yaw는 unwrap 후), control은 zero-order hold.

**에피소드 폐기 조건 — 조용히 넘어가지 말고 큰 소리로 실패시킨다:**
- 소스 프레임 간격 > 0.2 s 가 하나라도 존재
- grid 점의 5% 초과가 최근접 오차 > 0.05 s
- pose 보간 구간 > 0.1 s
- `median |t_nearest − grid[k]| > 0.017 s`

세션별 `resample_report.json` 산출. 생략하면 LeRobot이 조용히 평탄화한다. v0.6.1에는
`check_timestamps_sync`가 **존재하지 않는다**(grep 무결과).

### 5.4 행동 표현

```
action : float32 (3,) = [dx, dy, dyaw]        # ego frame @ t, REP-103 (x 전방, y 좌측)
dx   =  Δx·cos(yaw_t) + Δy·sin(yaw_t)         # m
dy   = -Δx·sin(yaw_t) + Δy·cos(yaw_t)         # m
dyaw = wrap_pi(yaw_{t+1} - yaw_t)             # rad
```

float64 보간된 **GT pose**에서 계산. **`cmd_vel`(양자화 심볼)에서 절대 계산하지 않는다.**
속도는 `√(dx²+dy²)·fps`, 곡률은 `dyaw/√(dx²+dy²)`로 복원되므로 **행동 표현과 평가 지표가
동일한 객체**가 된다.

`chunk_size=30`은 LeRobot이 로드 시점에 만든다(`resolve_delta_timestamps`, `action_is_pad` 마스크
포함) — **미리 펼치지 말 것.** SmolVLA에는 학습된 chunk-position embedding이 없으므로
`chunk_size` 변경은 안전하다.

> **`observation.state = [speed_mps, yaw_rate_radps, steer_angle_rad]` (3,), 32로 zero-pad.**
>
> `train_stage_a.py`의 `state_features()`(goal의 `dx, dy, dist, sin/cos yaw`)를 **절대 이식하지
> 않는다.** 그건 goal을 dense float로 넘겨주는 **언어 우회로**이고, 현재의 0.0049 rad/s를 만든
> 주범이며, 진짜 VLA 안에 같은 실패를 재현하는 가장 흔한 방법이다.
> GT world pose는 `nav.world_pose`(정책에 비가시, 평가자에게만 가시)로 기록한다.
> `observation.state` 자체를 생략할 수는 없다 — `SmolVLAPolicy.prepare_state`가 `KeyError`를 낸다.

**LeRobot 변환 하드룰** (v0.6.1, 소스 확인):
- `task`(str)는 **매 `add_frame`마다 필수.** 이것이 언어 경로 전부다.
- `timestamp`/`frame_index`/`episode_index`/`index`/`task_index`를 넘기면 `Extra features` 예외.
- 분석 컬럼은 반드시 **`nav.*`** 접두사. `action.*`/`observation.*`로 쓰면 `startswith` 매칭에
  걸려 정규화가 깨진다.
- `ds.finalize()` 누락 시 parquet footer 손상.
- 모든 `task` 문자열은 SmolVLM2 토크나이저 기준 **≤ 48 토큰**.

### 5.5 반사실 쌍 프로토콜과 물량

에피소드는 **junction-centred**로 만든다. 개구부 15~20 m 전 스폰, 결정이 해소된 뒤 5 s 종료.
전체 랩(60 s)을 수집하면 유용 프레임이 ~10%로 떨어지고 수집 시간이 3배가 된다.

수집 루프: `reset(pose_p)` → `settled` 대기 → `/oracle_goal` 발행 → 기록 → `termination` 라벨
→ **동일 `pose_p`에 대해 다음 instruction으로 반복.** 같은 `start_pose_key`를 공유하는 변형들이
정확한 반사실 쌍이다. `cf_group_id`/`cf_variant_id`는 **수집 시점에 기록**해야 하며 사후 복원이
불가능하다.

**코퍼스 v2 (파일럿, 4주 안)**

| 세트 | 셀 | ep/셀 | ep | 벽시계 |
|---|---|---|---|---|
| junction 단축 | `{none,first,second} × {A,B} × {0.7,1.2,1.8}` = 15 | 30 | 450 | ~4 h |
| ordinal 완주 | `{first,second} × {A,B}`, 1.8 m/s만 | 30 | 120 | ~2 h |
| 실패/복구 주입 | 위의 10%, 시작 pose 섭동 | — | ~57 | ~0.5 h |
| **합계** | | | **~570** | **~7 h** |

decision-relevant 프레임(= 쌍의 oracle 행동이 1 LSB 이상 차이 나는 프레임) ≈ **20k. 이건 파일럿이다.**

**코퍼스 v3 (방어 가능 규모, +2주)**: junction 2,000 ep, decision-relevant ≈ 90k, 무인 ~24 h.
렌탈 GPU에서 학습이 도는 동안 노트북이 병렬 수집한다.
**물량은 항상 에피소드 수나 GB가 아니라 decision-relevant 프레임 수로 예산하고 보고한다.**

### 5.6 GATE G2 — 학습 전에 통과해야 할 수치 (GPU 시간 0)

**oracle 데이터에 대해 측정한다. 정책이 아니다.** oracle이 하지 않은 것을 정책이 배울 수 없다.

| 항목 | 임계 | 구 스택 | **오라클 실측** (2026-07-28) |
|---|---|---|---|
| 리샘플 통과 에피소드 | ≥ 400 / 570 | — | 수집 중 (264 계획) |
| `start_pose_key`당 instruction 수 | ≥ 3, 정확 CF 쌍 ≥ 200 | 0 | 그룹당 2~4, **쌍 316 계획** |
| **ordinal 쌍 `D_shape`** | **≥ 1.5 m** | 해당 없음 | **6.36 m** ✅ |
| route 쌍 `D_shape` (차선) | ≥ 2.0 m | 0.063 m (goal_zone) | **5.02 m** ✅ |
| speed 쌍 `D_shape` (직교성) | **≤ 0.3 m** | 속도값 6개뿐 | **0.138 m** ✅ |
| speed 쌍 소요시간 비 | (신설) ≥ 1.25× | — | **1.66×** ✅ |
| 공유 접두 (ordinal) | (신설) ≥ 8 m | — | **15.0 m** ✅ |
| 노이즈 바닥 (같은 요청 2회) | — | — | **0.158 m** (G1은 0.095) |
| ~~`I(action;instr\|obs)` 중앙값~~ → **peak / 고CMI 프레임 비율** | **peak ≥ 0.8 b, ≥0.5b 프레임 ≥ 20%** | ≈ 0 (예상) | **0.973 b / 33%** ✅ |
| `dyaw` distinct 값 수 | ≥ 10⁴ | **9** (실측) | 113/115 (레코더 v2) |
| `median \|t_nearest − grid\|` | ≤ 0.017 s | +15.4% 계통오차 | 리샘플 시 측정 |

> 수치는 `scripts/verify_corpus.py`가 세션 디렉터리를 다시 읽어 산출한다. 위 값은
> 11 에피소드 스모크(`smoke_wide`)의 것이고, 본 코퍼스 수집 후 갱신한다.
>
> **`D_shape` 임계의 분모가 바뀌었다.** G1의 9.5 cm는 목표 없이 자유주행시킨 물리 노이즈였다.
> 수집 조건에서 같은 요청을 두 번 실행해 재니 **15.8 cm**다. 배율은 이 값 기준으로 읽는다.
>
> **조건부 상호정보만 남았다.** route/ordinal/speed 세 축은 오라클 수준에서 통과했다.

#### CMI 기준을 중앙값에서 층화로 바꾼 이유 (2026-07-29 확정)

원래 기준은 `median I(action;instruction|obs) ≥ 0.8 bit`였다. 264 에피소드 실측에서
세 축 모두 FAIL했고, 분포를 보니 **지표가 코퍼스 설계와 논리적으로 양립 불가능**했다.

```
ordinal:  p50 0.000   p75 0.102   p90 0.895   peak 0.973   (상한 log2(4)=2.00)
```

쌍봉이다. 서수 축이 성립하는 **이유**가 네 칸이 통로 14.4 m를 공유한다는 것이고, 그 구간에서는
어떤 명령이든 같은 행동이 나오므로 CMI가 0인 것이 **설계대로**다. 버킷 대다수가 공유 구간에
있으니 중앙값은 공유 구간을 재게 된다.

중앙값 0.8 bit를 만족하려면 **궤적이 즉시 갈라지는 코퍼스**여야 하는데, 그건 "같은 관측,
다른 문장"을 보여줄 수 없는 코퍼스다. **기준을 만족시키는 것과 주장을 증명하는 것이 상충한다.**

§7.3이 자매 지표 `S_lang`에 대해 이미 같은 층화를 요구하고 있다 —
*"공유 경로에서는 올바른 정책도 언어 불변이어야 하므로, 전체 중앙값에 임계를 걸면
'직선에서 텍스트를 바꾸면 차가 흔들리는' 것을 보상하게 된다."* CMI 항목만 평평한 중앙값으로
남아 있었다.

대체 기준 (`measure_cmi.py`가 산출):

| 축 | peak | ≥0.5b 프레임 | 판정 |
|---|---|---|---|
| **ordinal** | **0.973 b** / 상한 2.00 | 33% | PASS |
| lane | 1.036 b / 상한 2.58 | 53% | PASS |
| speed | 0.907 b / 상한 2.00 | 44% | PASS |

읽는 법: *통로 프레임의 33%에서 명령이 행동을 0.5 bit 이상 설명하고, 회전 지점에서는
가용한 2 bit 중 약 1.0을 설명한다.*

> **결과를 본 뒤 기준을 바꾼 것은 사실이므로 그대로 기록한다.** 정당화는 두 가지다 —
> (1) 계획 자체가 다른 지표에 같은 층화를 이미 요구하고 있고, (2) 원 기준은 코퍼스가
> 증명하려는 성질과 동시에 만족될 수 없다. 평평한 중앙값도 계속 병기한다.

---

## 6. 모델 + 학습

### 6.1 모델과 변경 사항

`lerobot/smolvla_base` (SmolVLM2-500M trunk + action expert, cross-attn). 총 ~450M, 학습 대상 107.7M.

| 필드 | 값 | 이유 |
|---|---|---|
| `chunk_size` | **30** (3.0 s @10 Hz) | 2.1 s 측정창이 라벨 안에 43% 여유로 들어가야 함 |
| `n_action_steps` | **5** (0.5 s 실행, 2 Hz replan) | 컴파일 지연 36 ms 대비 14배 여유 |
| `max_action_dim`/`max_state_dim` | **32 유지** | `action_in/out_proj`가 32로 사전학습됨. resize 금지 |
| `num_steps` (flow matching) | 10 유지 | 컴파일 시 sampler 14.4 ms |
| `resize_imgs_with_padding` | **(512,512)**, 카메라 1대 | 256 px는 16 tok → 차선·정지선 소실 |
| `pad_language_to` | **`"max_length"`** | `"longest"`면 prefix 길이가 지시마다 바뀌어 주행 중 재컴파일 |
| `compile_model`/`compile_mode` | `true` / **`"reduce-overhead"`** | 실측 expert 86.8 ms → 14.4 ms (5.8×). lerobot 기본 `max-autotune` 아님 |
| `normalization_mapping` | `{VISUAL: IDENTITY, STATE: MIN_MAX, ACTION: MIN_MAX}` | 조향은 대칭·0 편중·heavy-tail |
| `use_amp` | **`true`** | 기본 false. fp32는 1.5–2× 느리고 bs=64에 안 들어감 |

**LoRA는 쓰지 않는다.** 기본 레시피(`freeze_vision_encoder=true, train_expert_only=true`)가 이미
450M 중 107.7M만 학습하는 부분 파인튜닝이다. 얼린 VLM 위의 LoRA는 no-op.

### 6.2 하드웨어·비용 (RTX 4060 Laptop 8 GB 실측)

| 카메라 | 해상도 | batch | peak VRAM | ms/step | 20k steps |
|---|---|---|---|---|---|
| 1 | 512 | 16 | 3.08 GiB | 410 | 2.3 h |
| 1 | 512 | **32** | **3.82 GiB** | **781** | **4.3 h** |
| 1 | 512 | 64 | 5.37 GiB | 1560 | 8.7 h |
| 1 | 512 | 16 (VLM 해동) | 5.96 GiB | 573 | 3.2 h |

**run 1은 bs=32.** bs=64는 VRAM은 맞지만 RAM이 14 GB뿐이라 dataloader가 병목이 된다.
**동시에 Gazebo를 못 돌리는 것이 VRAM보다 큰 제약이며, 이것이 클라우드의 진짜 논거다.**

**클라우드: A100이 아니라 RTX 4090(24 GB), ~$0.4/h.** 60k step 최종 런 = 7–9 h ≈ **$3–4**.
$50이면 풀런 10회 + 스윕. 450M 모델에 A100($1.3–2/h)은 낭비.
**렌탈의 목적은 성능이 아니라 노트북을 24/7 수집기로 비워두는 것.**

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/navvla-sim-v2 \
  --policy.chunk_size=30 --policy.n_action_steps=5 \
  --policy.resize_imgs_with_padding="[512,512]" \
  --policy.tokenizer_max_length=48 --policy.pad_language_to=max_length \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
  --policy.compile_model=true --policy.compile_mode=reduce-overhead \
  --policy.optimizer_lr=1e-4 --policy.scheduler_decay_steps=60000 \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=32 --steps=20000 --num_workers=6 --wandb.enable=true
```

### 6.3 한국어 — 불가 (SmolVLA 유지 시)

실측: SmolVLM2 토크나이저 49,152 vocab 중 **한글로 디코드되는 항목 1개**(`이`).
`주차 슬롯 2번에 후진으로 천천히 주차해`(22자) = **49 토큰**, 한도는 48이고
`truncation=True`(`lerobot/processor/tokenizer_processor.py:83`) → **음절 중간에서 조용히 잘린다.**
한국어는 동사(=행동)가 문장 끝에 온다. 게다가 사전학습 semantic이 0이므로 A3(미학습 paraphrase)를
**구조적으로** 통과할 수 없다.

같은 문장이 Gemma-3 계열 17 토큰, Qwen2.5-VL 19 토큰 — 둘 다 실제 한국어 subword.
**한국어가 하드 요구라면 트렁크를 바꿔야 하고 §6의 모든 수치가 무효가 된다.**
→ **이번 주에 English-only 서면 확정을 받는다. 10분 대화로 한 달을 지킨다.**

### 6.4 폴백 (사전 승인 순서)

1. G3에서 `S_lang/S_noise < 3` → **VLM 마지막 4개 레이어 해동, bs=16**(5.96 GiB 실측). +1일.
   `freeze_vision_encoder + train_expert_only`는 트렁크가 옥외 주행 도메인에 전혀 적응하지
   않는다는 뜻이고, **그것이 정확히 언어가 장식이 되는 조건**이므로 기본적으로 필요할 가능성이 있다.
2. 두 번째 실패 → **from-scratch 베이스라인**(frozen text encoder + ResNet18, 동일 데이터셋).
   SmolVLA가 이걸 못 이기면 그 자체가 보고 가능한 결과이며 출하 가능. 추가 일정 0일.

---

## 7. 폐루프 서빙 + 평가

### 7.1 서빙 구조 — 프로세스 2개

ROS2 Jazzy의 시스템 Python 3.12(apt numpy/opencv)와 lerobot의 torch/transformers/av를 섞지 않는다.

- **`tools/vla_policy_server.py`** (~120 LOC) — lerobot venv, `rclpy` 없음. 체크포인트 로드 →
  컴파일 → **워밍업(reduce-overhead CUDA graph 30–120 s, 하네스가 world를 스텝하기 *전에* 완료)**
  → ZMQ REQ/REP `ipc:///tmp/nav_vla.sock`. msgpack `{jpeg, state:f32[3], task:str, seed:int}` →
  `{actions:f32[30,3]}`. **`seed`를 명시적으로 받는 것이 `D_same` 측정을 가능하게 한다.**
- **`nav_vla_pkg/vla_bridge_node.py`** (~300 LOC) — ROS2 노드, `MultiThreadedExecutor(4)`:
  - `sensor_cg`(Reentrant): 이미지/상태 → mutex 보호 `LatestObservation`. torch·인코딩 금지
  - `control_cg`(MutuallyExclusive): 10 Hz, `ActionQueue`에서 1개 pop → 발행. O(µs). GPU 접촉 금지
  - `instruction_cg`: `/vla/instruction` (RELIABLE + **TRANSIENT_LOCAL** depth 1 — 늦게 붙는
    평가 하네스도 latched 지시를 봄). 지시 변경 시 큐 flush(반사실 테스트의 정직한 설정)
  - **executor 밖 일반 스레드**: 큐 점유율 < 0.7·chunk일 때 추론 요청. 50 ms 왕복이 10 Hz
    타이머를 막을 수 없다

**chunk splicing**: 새 chunk를 큐 tail과 겹치는 구간에서 선형 가중 평균. 생략하면
`n_action_steps`마다 조향 twitch가 생기고 그게 `D_same` 노이즈 바닥을 부풀려 **평가 전체를 망친다.**

**지연** (실측, idle GPU, bf16, batch 1): eager 108 ms(9.2 Hz) → compiled **36 ms(~27 Hz)**.
Gazebo 렌더러 경합으로 1.3–1.6배 팽창 예상 → 실전 50–60 ms 계획.
**컴파일은 선택이 아니라 필수**(eager 9.2 Hz는 10 Hz 루프에서 매 tick underrun).
추론 VRAM ~1.5 GiB → Gazebo에 ~6 GiB 남음.

### 7.2 제어 경로에서 반드시 제거되는 것 (기준 d)

`policy_node.py`의 `_control_direct()`(483–561), `_arrived()`(702–739), `_stop_line_state()`(776–790),
`_advance_cruise_goal()`(563–578), `_state_features()`(741–754), `limit_twist_to_raw_speed()`(475–480).

**정지는 학습된 행동이 되어야 한다** — 학습 데이터가 감속 램프와 정지 프레임을 포함하고,
행동 공간이 `speed=0`을 표현해야 한다. 말단 램프는 전체의 ~1%이므로 학습에서 **~10% 가중으로
oversample**한다. 그렇지 않으면 "작은 정지 보조"가 온화한 이름의 파라미터로 다시 기어들어온다.

**허용되는 규칙은 정확히 하나**: 500 ms chunk 미도착 또는 큐 underrun 시 `speed=0, steering=hold`
watchdog. 전량 계측. **유효한 eval 런 = watchdog 발동 0회 + underrun < 1%**, 두 수치를 모든 결과에
병기한다. `control.jsonl`의 `override`가 **하드 게이트**: non-watchdog override > 0 이면 런 무효.

### 7.3 반사실을 증명하는 테스트

> ⚠️ **`D = max_t‖Δposition‖`와 `LSR = median(D_diff)/median(D_same)`는 둘 다 폐기한다.**
> 전자는 **속도만으로 통과된다**(동일 곡선에서 0.7 vs 1.8 m/s면 6초 뒤 ~6.6 m 차이 → 항법은 0).
> 후자는 **분모가 0으로 간다**(결정론 시뮬에서 `D_same`은 cm 단위 → `LSR ≥ 6`이 `D_diff = 0.3 m`로
> 만족). 분모가 0에 가까운 비율은 지표가 아니다.

**대체 지표 — 행동 표현과 같은 분해:**
- **`D_shape`** = 두 궤적을 *시간을 버린 점열*로 본 discrete Fréchet 거리 [m]. **속도 불변**
- **`D_sched`** = 6 s 호길이 차 [m]. **형상 불변**
- **`D_same_*`** = 같은 지시·같은 시작·**다른 sampler seed** → 재현성 바닥
- **주지표 `RR = D_shape / D_oracle_shape`** ≥ 0.7. 게임 불가능, 물리적 의미 있음, 애초에 분리
  불가능했던 쌍을 자동으로 드러냄

**Level 0 — open-loop 단일 프레임** (체크포인트마다 30분)
2,000개 held-out 프레임에서 `S_lang`(지시 간 std) / `S_noise`(seed 8개) / `S_img`(다른 프레임 8개).
**층화가 중요**: `S_lang`은 **decision-relevant 프레임에서만** 측정한다. 공유 경로에서는 올바른
정책도 언어 불변이어야 하므로, 전체 중앙값에 임계를 걸면 "직선에서 텍스트를 바꾸면 차가 흔들리는"
것을 보상하게 된다. 공유 경로에는 반대로 **`S_lang ≤ 1 LSB`를 별도 assert**한다.
통과: `S_lang/S_noise ≥ 10` **AND** `S_lang/S_img ≥ 0.3` (**현재 0.0049/0.1702 = 0.029**).

**Level 1 — 폐루프 반사실** (실제 증명)
40개 seed pose(모든 zone 접근, 양 lane, 양 개구부, 양 bay group, off-nominal 8개) × 쌍 ≥ 3 =
≥ 120 trial/side. reset(paused, stepped) → 지시 발행 → 60 tick(6.0 s) → τ 기록.
허용성 필터 적용/미적용 **둘 다 보고**(유리한 부분집합 선택이기도 하므로).

**반-governor 테스트 2개** (Level-0 데이터에서 공짜)
- **rank 테스트**: 관측 고정 시 언어 기인 행동 델타를 PCA. **PC1 분산비 ≤ 0.85** 요구
  (유효 언어 차원 ≥ 2). speed-only는 > 0.97이 나온다
- **분리가능성 잔차**: 최적 rank-1 곱 모델 `â ≈ f(img)·s(text)` 적합 후 **미설명 분산 ≥ 25%**
  요구. speed-only는 ~0 → 그것이 곧 "governor"의 정량화다

### 7.4 Ablation (같은 40 pose, 같은 seed, 체크포인트당 1개 표)

| # | 내용 | 통과 기준 |
|---|---|---|
| **A2** | **셔플된 유효 지시** — **주 ablation** | 지정-경로 성공률 ≤ chance+0.05, 차는 셔플된 목적지로 가야 함. **commanded × reached 혼동행렬 전체 보고** |
| A1 | 빈 지시 `""` | ≤ chance+0.05. **부차 지표로 강등** — `pad_language_to="max_length"`에서 `""`는 학습에서 본 적 없는 패턴이라 성능 저하가 언어 사용이 아니라 분포 이동을 증명할 수 있음 |
| **A3** | **미학습 paraphrase** — goal당 5개, 별도 LLM 생성, **학습 전 git 커밋** | 성공률 ≥ 0.8 × in-dist, `median D_shape(para, canonical) ≤ D_same + 0.5 m`. **현행 regex 파서가 구조적으로 통과 불가능한 유일한 테스트** |
| A4 | 조합 holdout — 15셀 중 3셀을 **학습 시점에만** 제외 | held-out ≥ 0.6 × seen, slow/fast 평균속도비 ≤ 0.7 (p<0.01) |
| A5 | **text-only baseline** (이미지 0) | branch 정확도 ≤ 0.8 × full. 초과하면 "VLA"가 아니라 비전 장식 달린 텍스트 분류기 |
| A6 | **state-only baseline** | 결과에 병기 (§3.2 속도 누출의 정직한 공시) |

> **모든 divergence 수치는 미터와 LSB를 병기한다.** 1 조향 LSB = 0.6/7 = 0.0857 rad →
> 2.1 s 횡변위 ≈ 0.15 m(@1.5 m/s). 기존 goal_zone **0.063 m = 0.13 LSB**, goal_lane 0.602 m ≈ 1.2 LSB.
> 실차의 15단계 `MotionCommand`가 **물리적으로 실행할 수 없는 차이**를 "16배 개선"으로 출하하지
> 않기 위한 장치다.

---

## 8. 일정 (기준일 2026-07-28 월)

인프라가 ~14/22 작업일이고 축 선택은 일정을 4일밖에 움직이지 않는다. 따라서 순서의 원칙은
**"축이 아직 싸게 바뀔 수 있는 동안 무엇을 만들 것인가"**이다.

### Week 1 (7/28–8/3) — 결정론 + 레코더. **ML 0**
- `gz_reset.py` (1일)
- `use_sim_time:=true` 전역 + stepped world. **타임박스 1.5일**
  > ⚠️ **알려진 함정**: `use_sim_time` + world paused ⇒ `/clock`이 멈추므로 **`create_timer`가
  > 영원히 발화하지 않는다.** 제어 루프·레코더 게이팅·watchdog이 전부 정지한다. 제어 경로를
  > `step(N) → observe → infer → act → step(N)`으로 **ROS 타이머 없이** 재구성해야 한다.
  > **이걸 다른 것보다 먼저 한다.** 폴백: free-run으로 가고 그 결과 `D_same` 바닥을 *측정*해 임계 상향
- Recorder v2 (3일) + 오프라인 리샘플러·수용 게이트 (1일)
- `route_oracle_node.py` v0: 단일 폴리라인 pure pursuit, GT pose (1일)

**GATE G1 — reset 결정론.** 동일 pose 20회 리셋, 동일 지시, 6 s open loop.
**PASS** 최종위치 쌍간 산포 ≤ 0.10 m / **WARN** 0.10–0.50 m(진행하되 모든 `D_diff` 임계를 측정
바닥의 3배 상향) / **STOP** > 0.50 m(**다른 무엇보다 먼저 고친다** — 하류의 모든 숫자가 노이즈다)

### Week 2 (8/4–8/10) — 분기, 코퍼스, 컨버터
- 폴리라인 6개 저작 + 개구부 카운터 (2일)
- 수집 드라이버(그리드 열거 → reset → 지시 → 기록 → termination → `cf_group_id` 기록) (1일)
- **A3 paraphrase 뱅크를 지금 동결하고 git 커밋. 사후 소급 불가**
- 코퍼스 v2 무인 수집 ~7 h
- 수집 중 `convert_to_lerobot.py` 작성 + 첫 20 에피소드 검증 (2일).
  lerobot은 현재 **미설치**이므로 venv 구성에 실측 여유를 둔다

**GATE G2 — §5.6 표.** route/ordinal 쌍 실패 시 **학습 진입 금지**

### Week 3 (8/11–8/17) — 학습 + 서빙
- lerobot venv + 500-step sanity. **타임박스 1일**
- run 1 야간: bs=32, 20k steps, ~4.3 h
- Level-0 하네스 + rank/separability 테스트 (2일)
- `vla_policy_server.py` + `vla_bridge_node.py` + splicing + 계측 watchdog (3일)

**GATE G3** — 3 s 말단 waypoint의 지시 간 std ≥ 0.5 m; `S_lang/S_noise ≥ 10`;
`S_lang/S_img ≥ 0.3`; 폐루프 1개 경로 완주 with watchdog 0회, underrun < 1%.
`S_lang/S_noise < 3`이면 §6.4 폴백 실행

### Week 4 (8/18–8/24) — 반사실 평가 + ablation + 데모
- `eval_harness_node.py` (2일), Level 1(분해 지표 + 허용성 필터), Level 2(A1–A6), 데모 녹화 (1일)

**GATE G4 — 출하 주장**

| 지표 | 임계 | 현재 |
|---|---|---|
| median `D_same_shape` | ≤ 0.4 m | — |
| ordinal/route `D_shape` @6.0 s | ≥ 3.0 m | — |
| ordinal/route `D_shape` @2.1 s | ≥ 1.0 m (**2.0 LSB**) | 0.063 m (0.13 LSB) |
| **`RR = D_shape / D_oracle_shape`** | **≥ 0.7** | — |
| `D_shape` diff/same 비 | ≥ 6 | — |
| speed 쌍 `D_sched` | ≥ 4.0 m | — |
| speed 쌍 `D_shape` | **≤ 0.6 m** ← 직교성 증명 | — |
| 언어 델타 PCA PC1 분산비 | ≤ 0.85 ← **반-governor** | — |
| rank-1 분리가능성 잔차 | ≥ 25% | — |
| Mann–Whitney p / Cliff's δ | < 1e-3 / ≥ 0.8 | — |
| A2 혼동행렬 대각 | ≥ 0.70 | — |
| A3 성공률 | ≥ 0.80 × in-dist | 파서는 구조적으로 0 |
| A5 text-only baseline | ≤ 0.8 × full | — |
| 지정 경로 성공률 | ≥ 0.80 | — |
| off-track / collision | ≤ 0.05 | — |
| watchdog / underrun | **0 / < 1%** (아니면 런 무효) | — |

### 버퍼 (8/25–8/29) — 초과분, 그다음 코퍼스 v3 수집 시작

### 8.3 슬립 시 컷 리스트 (엄격한 순서)

1. 텍스처 편집 — 영구 컷
2. **차선 축** — 셀이 절반이 된다. 논거가 아니라 보너스였다
3. `nav.path_ego` 보조 컬럼 — `nav.world_pose`에서 오프라인 유도 가능
4. bay group B → 단일 분기
5. stepped 결정론 → free-run + 측정된 `D_same` 바닥 (아깝게 컷)
6. A4 조합 holdout
7. 실패/복구 에피소드
8. 15k step 1회 런으로 축소

**어떤 시나리오에서도 컷 금지** (전부 수집 후 소급 불가):
sim-time 타임스탬프 · GT pose 기반 SE(2) 행동 · 수집 시점의 `cf_group_id`/`start_pose_key` ·
`D_shape`/`D_sched` 분해 · **A2 + A3** · `control.jsonl` 3계층 기록

### 8.4 일정에 대한 정직한 진술

"하루짜리"로 추정된 항목들의 현실 비용을 합산하면 **여유 없이 ~10–14주**다(레코더+sim-time 이관
1–2주, LeRobot 변환 3–5일, compile+serving 1주, eval 하네스 2–3주, 코퍼스 반복 3회 2–3주).

> **위 4주 계획은 파일럿 규모(코퍼스 v2, ~20k decision frame)의 시뮬레이션 결과를 내는
> 일정이며 논문 규모가 아니다. 1개월 실차 데모는 이 경로로 도달 불가능하다.**
> 도달 가능한 것은 **시뮬레이션 결과**이고, 그것도 §8.3의 컷을 실제로 실행할 때만이다.

---

## 9. 리스크

| # | 리스크 | P | 영향 | 최조기 신호 | 완화 |
|---|---|---|---|---|---|
| R1 | **모든 축이 self-revealing이라 (c)가 프레임의 85–95%에서 실패** | 0.85 | 치명 | **이번 주**: 기존 코퍼스에 `I(action; instruction \| obs-bucket)` 계산(~50 LOC). 예상 ≈ 0 bit | ordinal 축을 load-bearing으로 승격. 속도 누출은 초기속도 무작위화 + A6 공시 |
| R2 | **G1 실패(reset 산포 > 0.5 m)** — 하류 모든 숫자를 조용히 무효화 | 0.25 | 치명 | Day 5, G1 그 자체 | 그래서 Week 4가 아니라 Day 5의 게이트다. free-run + 측정 바닥으로 폴백 |
| R3 | **route oracle의 공유 구간이 정확히 goal-blind** | 0.9 | 높음(완화됨) | 수집 전, 200 pose에서 폴리라인 쌍간 divergence를 호길이에 플롯 → 첫 30–60 m가 평평하게 0 | **junction-centred 에피소드**로 이미 완화. 랩 전체 수집 금지 |
| R4 | **데이터 물량이 10–100× 부족** | 0.75 | 높음 | G2에서 decision-relevant 프레임을 세는 순간 | 파일럿임을 명시. v3 2,000 ep을 학습과 병렬 수집 |
| R5 | **SmolVLA 사전학습 expert가 옥외 주행 + SE(2) chunk로 잘 전이 안 됨** (베이스는 탁상 조작) | 0.5 | 높음 | G3의 `S_lang/S_noise` | VLM 마지막 4 레이어 해동 → from-scratch ResNet18. 못 이기면 그것도 결과 |
| R6 | **`use_sim_time` + paused world에서 ROS 타이머 전면 정지** | 0.85 | 중 | Week 1 Day 1 | 제어 루프를 step-완료 구동으로 재구성. 3–5일 예산 |
| R7 | **재수집이 "몇 시간"이 아니라 반복당 3–5일 기계시간** | 0.8 | 중상 | Week 2 첫 수집 실측 처리량 | 수집은 free-run, eval만 stepped. 학습을 렌탈 4090으로 |
| R8 | **평가가 4가지 방식으로 오도된 PASS를 낸다** | 0.7 | 높음 | 첫 G4 dry-run | `D_shape`/`D_sched` 분해 + `RR` 주지표 + A2 주 ablation + Level-0 층화. **이미 반영됨** |
| R9 | **정지 행동이 (d)를 위반하고 아무도 눈치채지 못한다** | 0.6 | 중 | eval 리포트 `override` 카운트 | `override > 0` = 런 무효 하드 게이트. 말단 램프 10% oversample |
| R10 | **한국어가 데모 시점에 요구로 재진입** → §6 전량 무효 | 0.5 | 중 | 이번 주 서면 확인 없음 | 10분 대화. 재진입 시 Qwen2.5-VL / Gemma-3로 트렁크 교체 |
| R11 | **`sim.pt` 의존성이 남아 텍스처/데이터가 원자적으로 OOD** | 0.4 | 중 | 새 스택 첫 수집에서 perception을 아예 끄고 돌려본다 | oracle이 YOLO에 의존하지 않도록 완전히 커밋 |

---

## 10. 이 결과가 증명하는 것과 증명하지 못하는 것

### 증명하는 것 (G4 통과 시)

- **(a)** 언어와 이미지가 한 forward pass에서 처리되고, **그 결합이 실제로 필요한 과제**
  (ordinal 개구부 선택)에서 동작한다. separable한 속도 축만으로는 이 문장을 쓸 수 없다.
- **(b)** 행동이 폐쇄 테이블이 아니라 float64 GT pose에서 유한차분된 연속 3-DOF SE(2) 경로다.
- **(c)** 동일 pose·동일 관측에서 지시만 바꾸면 궤적 **형상**이 oracle 대비 70% 이상 달라지고,
  형상과 스케줄이 **서로 독립적으로** 움직이며, PCA PC1 ≤ 0.85로 유효 언어 차원이 2개 이상이고,
  **미학습 paraphrase**에서도 유지된다. 마지막 항목이 현행 regex 파서가 구조적으로 통과할 수 없는 지점.
- **(d)** 제어 경로에 watchdog 하나뿐이고, 발동 횟수가 결과 헤드라인에 인쇄되며, 다른 override가
  1회라도 있으면 런이 무효 처리된다.

### 증명하지 못하는 것

1. **파일럿 규모다.** v2는 ~570 에피소드 / ~20k decision-relevant 프레임. 비교 대상
   (LMDrive ~64k 지시 클립, CALVIN ~2.4M 프레임, RT-1 ~130k 에피소드)에 **두 자릿수 못 미친다.**
   인용할 때 **decision-relevant 프레임 수를 같은 문장에 반드시 병기.**
2. **1비트짜리 언어라도 통과할 수 있는 테스트가 있다.** 채택 축이 `{branch:3} × {bay:2} × {speed:3}`
   = 15 행동 클래스다. A3를 통과하는 모델이 "10개 표현 → 15개 클래스" 분류기일 가능성은
   배제되지 않는다. **A5(text-only baseline)를 반드시 병기**해야 이 반론이 닫힌다.
3. **실차가 물리적으로 실행할 수 없는 차이는 무의미하다.** 실차 `MotionCommand`는 15단계
   (1 LSB = 0.0857 rad). 1.5 m/s에서 2.1 s 횡변위 1 LSB ≈ 0.15 m.
   **0.15 m 미만의 sim divergence는 실차에서 표현 불가능하다.**
4. **행동 라벨이 GT pose에 의존한다.** 실차에서 10 Hz GT pose는 Hesai 외부 측위에서 나오는데
   그 작업은 파킹 상태다. 의도적 선택이지만 **실차 재현에 하드 의존성을 하나 만든다.**
   반대로 `observation.state = [speed, yaw_rate, steer_angle]`은 실차가 휠 엔코더 + IMU + 조향
   피드백만으로 아는 값이고 goal 정보가 전혀 없다 — **이식 가능성을 지키는 유일한 설계 결정이며,
   "수렴을 돕기 위해" 여기에 `rel_goal`을 다시 넣는 것은 진짜 VLA 안에 0.0049 rad/s 실패를
   재건하는 것이다.**
5. **도로를 벗어나는 데 시뮬에서 아무 대가가 없다.** `track.world`는 도로·광장·잔디가 하나의
   평면이다. off-track은 물리가 아니라 소프트웨어 규칙이다. **여기서 학습된 정책은 도로 이탈에
   대해 한 번도 벌을 받은 적이 없다.**
6. **복구 데이터가 없다.** 619/619가 success이고 v2에서 10% 실패를 주입해도 여전히 on-manifold에
   가깝다. 평가 fixture의 off-nominal pose 8개는 실패할 가능성이 높고 **조용히 fixture에서 빼려는
   유혹이 클 것이다. fixture는 동결하고 실패를 보고한다.**
7. **센서 현실성 없음**: 640×480 단일 고정 카메라, 롤링 셔터·노출 제어·모션 블러·기하 그림자·
   타 에이전트 없음, 전 코퍼스 단방향 주행(양의 조향 사전확률 65%).

---

## 부록 — 근거 수준

- **직접 검증(이 저장소에서 실행)**: `cmd.angular` distinct 9개 / `cmd.linear` distinct 6개
  (lane1 세션 3,706 프레임) · 프레임 간격 중앙값 0.2307 s = 4.33 Hz(+15.4% 계통오차) ·
  트랙이 평면+텍스처(55.08×36.02 m)이고 물리 분기 없음 · 619/619 success · 고유 명령 182개 ·
  `grep instruction train/*.py` = 0
- **에이전트 검증(코드 판독)**: `navigator_node.py` 362–363행의 2개 토픽 발행 ·
  `motion_planner_node.py:37`의 `round()` · `policy_node.py:455-457` 분기 ·
  커밋 `a2de7d4`의 각속도 매핑 변경 · LeRobot v0.6.1 소스 동작 · SmolVLM2 토크나이저 한글 1개
- **미검증(코드 반영 전 재확인 필요)**: 개구부 좌표(북 y≈+21.6 / 남 y≈−20.4) ·
  BayGroupB 위치(x≈−8.8) · `track.world` 콜리전 블록 개수 · VRAM/지연 실측치의 재현
