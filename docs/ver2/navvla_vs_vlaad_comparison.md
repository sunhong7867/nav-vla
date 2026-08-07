# nav-vla vs VLA_AD — 공통점·차이점·보완 관계와 실차 이행 계획

작성 2026-08-06. 근거 코드: nav-vla `vla_bridge_node.py`, `collect_corpus.py`,
`to_lerobot.py`, `train_smolvla.sh`, `tools/eval/` / VLA_AD `vla_trt_node.cpp`,
`lasa_node.py`, `motion_planner_node.py`, `pipeline_launch.py`.
측정치는 docs/ver/ 해당 일자 문서와 VLA_AD `TASKS.md`·`paper/tables/`에서 가져옴.
VLA_AD는 선배가 진행한 프로젝트를 인수인계받아 새로 시작하는 것이며, 원본은
`~/hoon/VLA_AD`, 작업 사본은 `~/hong/VLA_AD`다 (부록 참조).
이 문서는 [vla_readiness_and_roadmap.md](../vla_readiness_and_roadmap.md)(2026-07-27)의
§0~§2 판정을 현재 상태로 **정정**한다 (§1). 정정 방식은
[vla_thor_demo_and_course_plan.md](../vla_thor_demo_and_course_plan.md):40-42의
선례를 따라, 이전 문서를 지우지 않고 범위만 명시해 우선한다.

---

## 0. 한 줄 답

**nav-vla는 학습된 액션층이다 — 시뮬에서 검증 완료, 실차 이력 0.**
**VLA_AD는 실차에 배포된 고전 제어층 + 의미층이다 — 학습된 액션 0.**
두 스택은 경쟁 관계가 아니라 같은 시스템의 아래층과 그 위층이다. 합치는 방향은
"한쪽이 다른 쪽을 흡수"가 아니라 **정책은 nav-vla, 권한 중재는 VLA_AD**다.

그리고 v7 가중치는 실차로 이식되지 않는다. 휠베이스 하나만으로 5.3배가
어긋난다(§5). **이식되는 것은 가중치가 아니라 파이프라인이다.**

---

## 1. 2026-07-27 판정 정정

`vla_readiness_and_roadmap.md` §0은 "nav-vla는 VLM이 아니다"라고 판정했다.
그 판정은 stage_a(ResNet18 + `nn.EmbeddingBag`, v0 베이스라인) 기준이고, 해당
코드는 브랜치 `agent-smolvla-integration`에서 삭제됐다. 같은 문서 §2의
반증가능한 VLA 정의 (a)~(d)를 그대로 써서 현재 상태로 다시 채점한다.

| 기준 | 07-27 판정 (nav-vla) | 2026-08-06 현재 | 근거 |
|---|---|---|---|
| (a) 비전+언어 통합 forward pass | ❌ 정책 이전에 언어를 `int`로 이산화 | ✅ SmolVLM2 트렁크가 이미지+문장+상태를 함께 인코딩 | [train_smolvla.sh](../../src/nav_vla_pkg/scripts/train_smolvla.sh), [to_lerobot.py](../../src/nav_vla_pkg/scripts/to_lerobot.py) |
| (b) 학습 헤드의 연속 다자유도 액션 | ❌ 2-DoF지만 목표 무시 | ✅ 30스텝 × [dx, dy, dyaw] flow matching 청크 | [vla_training_comparison.md](../vla_training_comparison.md) §2(c) |
| (c) 언어 반사실 민감도 | ❌ 0.0049 rad/s | ✅ **차선 분리 2.51 m**, 동일 문장 0.05 m 바닥 대비 27× | [ver/20260804_1056](../ver/20260804_1056_y5j-junction-pack-a3-driving.md), [ver/20260805_0331](../ver/20260805_0331_v5-clean-stable.md) |
| (d) 루프 내 규칙 미개입 | ❌ regex가 LLM보다 최종 권한 | ⚠️ 존 정차만 navigator 좌표 감독, 주행 자체는 정책 단독 | [generalization_limits_and_questions.md](../generalization_limits_and_questions.md) §5 |

**VLA_AD 쪽 판정은 바뀌지 않았다.** 07-27의 "진짜 VLM이지만 속도 스칼라 1개만
내는 거버너"는 여전히 정확하고, §4에서 그보다 더 낮다는 것이 확인된다 — 현
프로덕션 런치 구성에서는 그 속도 스칼라조차 제어에 도달하지 않는다.

정정 범위: 이 문서는 **두 스택의 현재 상태 판정과 이행 방향**에 대해 우선한다.
로드맵 §5(P0~P6 일정)·§8(모델 후보)은 그대로 유효하며 재작성하지 않는다.

---

## 2. 공통점

| 항목 | 내용 | 근거 |
|---|---|---|
| 같은 뿌리 | `interfaces_pkg` · `camera_perception_pkg` · `decision_making_pkg` 계보를 공유. 포크 후 각자 분화 | `vla_ad_nl_integration.md:15-16` |
| 런타임 | 둘 다 ROS 2 Jazzy, 단일 전방 카메라, 온보드 GPU 추론 | 양쪽 `package.xml` |
| YOLO 의존 | 양쪽 다 YOLO를 쓰지만 **역할이 정반대** — nav-vla는 코퍼스를 만드는 **선생**, VLA_AD는 실제로 주행하는 **본체** | `ver/20260731_1453`, `motion_planner_node.py:174` |
| 학습 계열 | 둘 다 모방학습(BC) 기반. 시연 분포 밖에서 무능한 것이 공통 한계 | `generalization_limits_and_questions.md` §1~§2 |
| 액추에이션 계약 | `MotionCommand.msg`가 양쪽 동일 (`steering`, `left_speed`, `right_speed`) | `vla_ad_nl_integration.md:235-245` |
| 안전 설계 철학 | 둘 다 "안전은 학습이 아니라 규칙 레이어"에 합의. VLA_AD는 LASA로 구현했고 nav-vla는 governor로 **설계만** 해둠 | `lasa_node.py`, `generalization_limits_and_questions.md` §5 |
| 언어 인터페이스 | 둘 다 자연어를 입력으로 받지만 들어가는 위치가 다름 (§3.2) | — |

---

## 3. 차이점 — 제어 권한이 어디에 있는가

### 3.1 제어 경로

nav-vla (시뮬):

```
카메라 → vla_bridge (JPEG + 상태 + 문장, 10 Hz)
      → vla_policy_server (SmolVLA 추론, ~1.2 s/청크)
      → 액션 청크 30×[dx, dy, dyaw]     ← 학습된 모델이 궤적 전체를 결정
bridge: v = dx/dt, 곡률 k = w/v 보존, max_speed 클램프, speed_slew 0.08 평활
      → /cmd_vel → 차량
        (존 정차만 navigator가 좌표 감독으로 개입)
```

VLA_AD (실차):

```
카메라 30 Hz → YOLO 세그 → 차선 BEV → 2차 다항식 x = f(y)
            → motion_planner 50 Hz: L_D=120 px 룩어헤드, atan(dx/dy),
              steering = max(min(int(deg/5), 7), -7)  ← 고전 제어가 조향 결정
            → MotionCommand → 시리얼 "s{st}l{l}r{r}\n"

  (병렬) Qwen3-VL-8B INT4 → VlaIR 5키 JSON (~0.7 Hz)
       → LASA 중재 → IrApplied(Δy, α) → motion_planner가 α로 속도만 변조
```

### 3.2 한눈 비교

| 축 | nav-vla | VLA_AD |
|---|---|---|
| 시연자 | 욜로 차선추종 스택 + navigator | 없음 (학습된 주행 정책 자체가 없음) |
| 언어의 위치 | **모델 입력** — 토큰이 행동과 같은 forward pass | 모델 입력이지만 출력이 텍스트 JSON (텍스트 병목) |
| 학습된 출력 | 30스텝 SE(2) 청크, 미터 단위 | 5키 JSON — 실질 제어값은 `speed_scale` **1-DoF** |
| 조향 결정 주체 | **학습된 정책** | **고전 pure-pursuit** (모델 미개입) |
| 제어 주기 | 정책이 곧 제어 (10 Hz 재추론, 2.1 s 커밋) | 제어 50 Hz ≠ 모델 ~0.7 Hz (비동기) |
| 모델을 빼면 | **차가 안 간다** | **그대로 간다** (A0 ≈ A4, §4) |
| 실차 배포 | 이력 0 | Thor 온보드 배포 완료, 트랙 주행 확인 |
| 안전 레이어 | 설계만 (governor 미구현) | LASA 8단계 구현 (단 프로덕션 런치 미포함, §4) |
| 액션 라벨 소스 | gz TF ground truth (학습 입력 아님, 라벨·평가 전용) | 없음 — 애초에 액션 라벨 학습을 하지 않음 |
| 개집합 시나리오 | 시연 0건 → 불가 | 신호등·수신호·공사구간 어휘 보유 |

핵심 대비 한 줄: **모델을 빼면 nav-vla는 차가 안 가고, VLA_AD는 그대로 간다.**
이것이 두 스택의 성격을 가르는 유일한 질문이다.

### 3.3 관찰과 코드의 일치

"nav-vla는 이미지 보고 학습 모델이 속도나 주행을 하지만, VLA_AD는 YOLO로 차선
만들고 pure-pursuit로 주행한다"는 관찰은 코드와 정확히 일치한다. nav-vla는
`vla_bridge_node.py`가 정책 청크를 그대로 `/cmd_vel`로 변환하므로 모델이
조향·속도를 모두 낸다. VLA_AD는 `motion_planner_node.py:212-213`이 조향을
계산하며, 이 계산에 모델 출력은 들어가지 않는다 — 모델이 넣는 값은
`dy_offset_px`(항상 0)와 `alpha`뿐이다.

---

## 4. VLA_AD 현 배포 구성 실측 — VLM이 제어에 미치는 영향은 0

개선 지점을 특정하기 위한 목록이다. 전부 VLA_AD 자체 기록이나 코드로 확인했다.

| # | 항목 | 실제 | 근거 |
|---|---|---|---|
| 1 | 횡방향 액션 | `SYSTEM_PROMPT_POLICY`가 요구하는 키가 5개뿐이라 `waypoint_offset_px`가 기본값으로 떨어짐 → **항상 0** | `vla_trt_node.cpp:40`, `:411` (직접 확인). 자체 기록 `TASKS.md:3856` "회피=VLA 기능 아님" |
| 2 | confidence 게이트 | `confidence`도 프롬프트에 없어 항상 기본 `0.8f`. `0.8 > c_min=0.5`이므로 LASA Step 4a **미발동** | `vla_trt_node.cpp:412`, `lasa_node.py:113` (직접 확인) |
| 3 | LASA 기동 | **`lasa_node`가 `pipeline_launch.py`에 없다** → `/vla/ir_applied` 미발행 → `motion_planner`가 영구히 `dy=0, α=1.0` | `grep -c lasa pipeline_launch.py` → **0** (직접 확인). 자체 기록 `TASKS.md:3855` |
| 4 | 권한 우회 | `behavior_manager_node`가 LASA 이전의 `vla/ir_raw`를 구독 → LASA가 유일 권한이 아님 | `behavior_manager_node.py:56` (직접 확인) |
| 5 | 폐기 시 거동 | `motion_planner`가 `was_discarded=True`를 "IR 없음"으로 처리해 LASA의 보수적 폴백을 버리고 **α=1.0으로 점프** | `motion_planner_node.py:122-125` (직접 확인) |
| 6 | 어블레이션 | **A0(VLA 없음) ≈ A4(전체)** — CTE 49.8 vs 50.2 px, GT 차선편차 34.4 px로 A0=A4, 조향 ±1 일치율 0.11~0.33에서 양 팔 차이 ≤5 pp | `vla_readiness_and_roadmap.md:37`, `TASKS.md:3382`, `:3337-3341` |
| 7 | 플랜트 양자화 | 조향 15단계 = **3.9 bit**가 하류 전체의 상한. 실측 휠베이스 54 cm는 코드에 한 번도 등장하지 않음 (`atan(dx/dy)/5°` 단순화) | `motion_planner_node.py:213` (직접 확인), `TASKS.md:2206` |
| 8 | 음성 결과 | S6 OOD에서 GT 정지 프레임 **47/47 전부 NORMAL, STOP 발동 0건** | `TASKS.md:3336` |

2026-06-02에 실험 하네스(`run_replay_sweep.sh` 등)에는 `lasa_node` 기동이
추가됐지만 프로덕션 런치에는 여전히 없다. 즉 항목 3은 "논문 실험은 고쳤고
배포는 안 고친" 상태다.

**이 목록이 결함 목록인 이유는 고칠 자리를 특정해 두기 위해서다.** §6의 보완
방향과 §8의 실행 순서가 전부 여기서 파생된다. 특히 항목 1이 §6.2의 근거이고
(학습된 액션층이 횡방향 0의 유일한 근본 해결책), 항목 3·5는 §8에서 측정과
무관하게 지금 고칠 수 있는 항목으로 분류된다.

---

## 5. 왜 v7 가중치는 실차로 이식되지 않는가

| 항목 | 시뮬 | 실차 | 이식 시 결과 |
|---|---|---|---|
| 휠베이스 | 2.86 m (`ackermann_cmd_adapter_node.py:23`) | **54 cm** | κ = tan δ / L. 같은 `steer_norm=0.3`이 반경 14.6 m vs 2.75 m — **5.3× 오차**가 학습된 시각↔곡률 연관 전체에 박힘 |
| 조향 표현 | 연속 | 15단계 (3.9 bit) | 0.15 m 미만의 청크 정밀도는 물리적으로 표현 불가 |
| 트랙 | ~40 m 실외 링 | ~15×11 m 실내 | 곡률 분포·시야 스케일이 다른 별개 과제 |
| 렌즈 | 왜곡 0 핀홀 | k1=−0.41, k2=−0.74, k3=1.29 | ±3° 외부파라미터 지터로는 못 덮음 |
| 액션 라벨 소스 | gz TF ground truth | **없음** | Hesai OT128 고정 인프라 없이는 `[dx, dy, dyaw]` 라벨 생성 불가 |
| 속도역 | 2.4~3.2 m/s | `v_base=100` × α ∈ [0.3, 1.0] | 정규화 기준이 달라 그대로 매핑 불가 |

> **이식되는 것은 가중치가 아니라 파이프라인이다.** `collect_corpus.py` →
> `resample` → `package` → [to_lerobot.py](../../src/nav_vla_pkg/scripts/to_lerobot.py)
> → `lerobot-train` → [ring_map_probe.sh](../../tools/eval/ring_map_probe.sh) →
> [compare_segments.py](../../tools/eval/compare_segments.py)는 전부 재사용된다.
> 재사용되지 않는 것은 v6/v7 체크포인트 자신뿐이다.

실차 주행을 돌려 다시 학습해야 한다는 판단은 맞다. 다만 "다시"는 처음부터가
아니라 **같은 레시피의 두 번째 도메인 적용**이고, 시뮬에서 여섯 번(v3y→v6)
돌려 디버깅을 끝낸 것이 그 비용을 이미 대부분 지불해 뒀다.

---

## 6. 보완 관계

### 6.1 VLA_AD → nav-vla

- **LASA 중재 골격** — TTL 폐기, 클램프, slew, safety override. nav-vla가
  `generalization_limits_and_questions.md` §5에서 스스로 필요하다고 적어둔
  governor와 같은 것이다. 새로 만들 필요 없이 가져오면 된다.
  단 **§4 항목 5는 이식 전에 고쳐야 한다** — 폐기 시 α=1.0 점프는 nav-vla의
  `speed_slew 0.08`(2.4 m/s² 제한) 설계와 정면으로 충돌한다.
- **실차 배포 경험** — Thor 온보드, TRT INT4 양자화 레시피, 조향 양자화·시리얼
  계약, Arduino 인터페이스. nav-vla에는 하나도 없다.
- **개집합 시나리오 어휘** — 신호등·수신호·공사구간·정지차량. nav-vla는 시연
  0건이라 학습으로는 못 얻는 능력이고, 규칙+VLM 레이어로만 가능하다.

### 6.2 nav-vla → VLA_AD

- **학습된 액션층** — §4 항목 1(횡방향 0)의 유일한 근본 해결책. 6키 코퍼스로
  `Δy`를 가르치려던 시도는 teacher 라벨링이 막혀 중단됐지만, 애초에 스칼라
  knob 하나를 회귀하는 것보다 청크 정책이 구조적으로 낫다.
- **반사실 평가 방법론** — `probe_policy_counterfactual.py` +
  `compare_segments.py`. 같은 출발점에서 문장만 바꿔 궤적 발산을 재는 방식.
  VLA_AD의 "lexical grounding ≠ control actuation"(`TASKS.md:3395`)을 수치로
  증명하거나 반증할 수 있는 도구다.
- **오염 감사 습관** — off-ring 가드가 찾아낸 라벨 오염 16 eps(전체의 2%)가
  이탈 4/4를 일으켰던 사례. "시간종료 = success"로 통과하는 라벨을 의심하는
  절차 자체.
- **append-only 음성결과 기록** — `docs/ver/` 33개 문서. 기각된 가설을 지우지
  않는다.
- **자동 수집 + DAgger** — 정책의 오차 상태에서 선생 복귀를 수집하는
  `--start-poses-file` 경로. 실차에서는 HG-DAgger(사람 개입 교정을 라벨로)로
  확장된다.

### 6.3 어느 쪽으로도 옮기면 안 되는 것

**아키텍처 자체의 통째 교체.** 각 스택의 설계는 자기 제약에 대해 옳은 답이다 —
시뮬은 ground truth가 있으니 액션 라벨을 자동 생성해 정책을 학습하는 것이 맞고,
실차는 ground truth가 없으니 VLM에 개집합 의미를 맡기고 기하는 고전 제어에 두는
것이 맞다. 한쪽이 다른 쪽을 흡수하면 그 이유가 사라진다.

---

## 7. 목표 아키텍처

```
              Qwen3-VL-8B INT4 ──► VlaIR (5키, ~0.7 Hz)
                                        │
카메라 ──► SmolVLA (액션 청크 30스텝) ──► /cmd_nav ──► LASA (권한 중재)
                                                          │
                                            MotionCommand ──► 시리얼 ──► 차량

Hesai OT128 (트랙 고정 인프라) ──► /track/vehicle_pose
   └─ 액션 라벨 생성 + 평가 기준선 전용. 제어 경로·학습 입력 진입 금지.
```

`vla_readiness_and_roadmap.md:131-138`의 "확정 결정" 4행 표와 대조하면:

| 역할 | 07-27 결정 | 현재 |
|---|---|---|
| 배포 정책 | 소형 VLM 트렁크 + 비자기회귀 액션청크 헤드 → 미터 단위 ego-frame waypoint | **충족** — SmolVLA 450M + flow matching 청크, `[dx, dy, dyaw]` 미터 단위 |
| 의미 거버너 | Qwen3-VL-8B INT4 유지 → `VlaIR` | 유효 — 단 §4 결함 해소 전제 |
| 교사 | Alpamayo-1.5 오프라인 전용 | 미착수 (욜로 스택이 선생 역할 대체) |
| 철칙 | 액션 라벨은 재생된 ego-motion에서만, VLM에서 절대 금지 | 유효 — 시뮬에서 지켜졌고 실차에서는 Hesai가 그 역할 |

Hesai를 **라벨·평가 전용**으로 못박는 것이 VLA_AD의 camera-only 주장
(`paper/sections/03_system.tex:26-40`)과 충돌하지 않는 유일한 방법이다.
시뮬에서 gz TF를 학습 입력에 넣지 않고 라벨·평가에만 쓴 것과 같은 규칙이며,
이 구분을 문서와 코드 양쪽에 명시해 둬야 나중에 심사에서 흔들리지 않는다.

---

## 8. 근시일 실행 순서 (2~6주)

### 지금 바로 (측정과 무관, 병행)

§4 항목 3(`lasa_node`를 `pipeline_launch.py`에 추가)과 항목 5(폐기 시 α=1.0
점프를 LASA가 발행한 보수값 사용으로 수정). 둘 다 코드 수정이고, 고치기 전에는
"LASA가 제어 경로를 보호한다"는 주장 자체가 성립하지 않는다.

### 주 1 — P0 측정 (전부 반나절짜리)

1. **Thor 온디바이스 P50/P95 실측** — `vla_fps` 토픽을 30분 기록. README의
   0.74 Hz와 `paper/tables/t9_deployment.tex:25-28`의 4.6 s가 **3.4× 모순**이고
   아직 아무도 재지 않았다(`TASKS.md:3443`). 청크 정책의 커밋 시간이 실차에서
   성립하는지가 여기서 갈린다.
2. **`steering ∈ [-7,7]`이 펌웨어 한계인지 임의 클램프인지** — Arduino 스케치가
   저장소에 없으므로 실측 필요. 임의라면 푸는 것이 최우선이다(3.9 bit 상한 해제).
3. **차량·카메라 실측** — 휠베이스, 최대 조향각, 카메라 높이/피치/내부파라미터.
   `bev_px_per_cm` 3중 모순(2.0 / 6.0 / 10.0) 확정, 1/5 vs 1/10 표기 통일.

**게이트**: 세 값이 모두 단일 확정되기 전에는 시뮬 재정합을 시작하지 않는다.

### 주 2~3 — 측위·라벨 인프라

LiDAR BEV Studio([tools/lidar_alignment_gui/](../../tools/lidar_alignment_gui/),
이번 브랜치 신규 ~3,900줄) 완주 → `/track/vehicle_pose` 발행.

**게이트**: 같은 스크립트 경로를 시뮬·실차에서 주행했을 때 궤적 종점 발산
< 0.3 m (`vla_readiness_and_roadmap.md:186`).

### 주 3~4 — 시뮬 재정합

Gazebo 월드를 실트랙 치수(~15×11 m 실내)로 재구축, Ackermann 휠베이스를
실측값으로 교체, 카메라 FOV·렌즈 왜곡 정합.

**게이트**: 재정합된 시뮬에서 욜로 선생을 다시 돌렸을 때 `compare_segments.py`
기준으로 선생 편차가 현재 수준(0.16~0.30 m)으로 재현.

### 주 5~6 — 실차 수집·재학습 1주기

Hesai-in-the-loop 오라클로 반사실 쌍 수집(사람 조종 불필요) → `resample` →
`package` → `to_lerobot.py` → `lerobot-train` 60K.

**게이트**: 실차 반사실 차선 분리 ≥ 0.5 m, 링 이탈 0, **그리고 조향 15단계
양자화를 통과한 뒤에도 분리가 유지**될 것. 마지막 조건이 §4 항목 7의 상한을
실측으로 확인하는 지점이다.

---

## 9. 미해결

- Sim2Real 격차를 계속 논외로 둘지 재확인 필요. `ver/20260806_1045`의 방침
  ("시뮬에서 검증 → 실차는 그때 해결")은 §5의 5.3× 오차 앞에서는 유지하기 어렵다.
- 조향 15단계 양자화가 청크 정책의 이득을 얼마나 깎는지 **미측정**. 주 5~6
  게이트가 이것을 처음 재는 자리다.
- 교차 트랙 제로샷 미실시 — `generalization_limits_and_questions.md` §3의
  계획된 실험 1~3이 그대로 남아 있다.
- VLA_AD 논문 노선이 두 번 선회해 현재 `paper/access/ACCESS_PLAN.md`가
  "STOP — evidence gates open" 상태다. 이 비교가 어느 논문의 어느 절에 들어갈지
  미정.
- v7 학습 결과 미도착 (2026-08-06 ~20시 예정). 도착 후 §1 (c) 행 수치 갱신 필요.

---

## 10. 관련 문서

- 파이프라인 대조: [vla_training_comparison.md](../vla_training_comparison.md)
- 한계·열린 질문:
  [generalization_limits_and_questions.md](../generalization_limits_and_questions.md)
- 로드맵 (§0~§2 정정 대상, §5·§8은 유효):
  [vla_readiness_and_roadmap.md](../vla_readiness_and_roadmap.md)
- 2026-07-25 통합 설계 이력:
  [vla_ad_nl_integration.md](../vla_ad_nl_integration.md)
- 데모·코스 계획:
  [vla_thor_demo_and_course_plan.md](../vla_thor_demo_and_course_plan.md)
- 시행착오 연대기: [ver/README.md](../ver/README.md)
- VLA_AD 측: `TASKS.md`, `BUGFIX_LOG.md`, `paper/access/ACCESS_PLAN.md`

---

## 부록 — 근거 수준

**직접 검증 (2026-08-06 양쪽 체크아웃에서 코드 확인)**
`waypoint_offset_px` 기본값 0 (`vla_trt_node.cpp:411`), `confidence` 기본값 0.8과
`c_min=0.5` (`:412`, `lasa_node.py:113`), `lasa_node`의 `pipeline_launch.py` 부재
(`grep -c` → 0), 조향 15단계 클램프 (`motion_planner_node.py:213`),
`was_discarded` 폴백 (`:122-125`), `behavior_manager`의 `ir_raw` 구독 (`:56`),
nav-vla `speed_slew` 구현 (`vla_bridge_node.py:355-383`), `train_smolvla.sh` 인자.

**문서 전사 (VLA_AD 기록에서 인용, 원자료 미확인)**
CTE 49.8 / 50.2 px, GT 차선편차 34.4 px, 조향 ±1 일치율, A0/A4 지연,
S6 47/47 NORMAL, Stage-1·v6 학습 설정. 위 수치는 전부 `TASKS.md`·
`BUGFIX_LOG.md`·`paper/tables/`에서 옮긴 것이고 raw 데이터로 재확인하지 않았다.

작업 사본 `~/hong/VLA_AD`는 새로 시작하기 위해 분석 산출물·로스백을 정리한
상태이므로, 재확인이 필요하면 **선배 원본 `~/hoon/VLA_AD`**를 본다 (같은 커밋
`c39e239`, `experiments/` 전체 보유). LoRA 학습 설정과 TRT 엔진도 같은 계층의
`~/hoon/LlamaFactory`·`~/hoon/TensorRT-Edge-LLM`에 있다. 논문에 이 수치를 쓰기
전 그쪽에서 한 번 재생성하는 것이 안전하다.

**미검증 (재확인 필요)**
Thor 온디바이스 P95 — **저장소 어디에도 존재하지 않는다**. 실차 휠베이스 54 cm는
문서 값이며 줄자 재확인 필요. 1/5 vs 1/10 스케일 표기 모순(`README.md:4` vs
`paper/sections/03_system.tex:34`) 미해소. 추론 주기 0.74 Hz vs 4.6 s 모순 미해소.
