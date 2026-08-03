# nav-vla / VLA_AD — VLA 성숙도 진단 및 실차 VLA 로드맵

> 질문: (1) 지금은 VLM인가, 단순 LLM 명령 파서인가? (2) VLA까지 가려면 무엇이 필요한가?
> (3) 알파마요 말고 다른 VLA 모델을 쓰는 게 나은가? (4) 아니면 지금 학습데이터를 VLA용으로
> 만드는 게 나은가?
>
> 대상: nav-vla(시뮬) + [VLA_AD](https://github.com/sunhong7867/VLA_AD)(실차 Thor).
> 최종 목표: 1/5 크기 유아용 전동차 + Jetson AGX Thor(온보드) + Hesai OT128(트랙 고정 인프라)
> 로 실제 트랙에서 자연어 기반 주행.
>
> 작성일: 2026-07-27. 관련 문서: [vla_ad_nl_integration.md](vla_ad_nl_integration.md)

---

## 0. 한 줄 답

**nav-vla는 "정규식 우선 명령 파서 + 목표를 못 보는 차선추종 BC 정책"이다. VLM이 아니다.**
**VLA_AD는 "진짜 VLM이지만 속도 스칼라 1개만 내는 거버너"다. Action은 이름일 뿐이다.**
두 시스템 어디에도 **픽셀과 언어가 같은 forward pass를 타고 연속 액션이 나오는 모델은 없다.**

그리고 가장 중요한 발견은 모델이 아니라 **데이터**에 있다 → §3.

---

## 1. 현재 상태 판정 (근거 검증 완료)

| 구성요소 | 표방 | 실제 정체 | 근거 |
|---|---|---|---|
| `chat_gui_node.py` | LLM 언어층 | **정규식 파서 + LLM 폴백.** 결정 경로가 LLM보다 **먼저** 실행되고(`:556-566`), 이후 4개 후처리기가 LLM 플랜을 버리고 원문 regex로 재작성(`:1339-1352`, `:1430-1468`) | Ollama 페이로드에 `images` 키 없음(`:607-624`) → qwen3:4b는 **픽셀을 본 적이 없다**. 출력은 8동사 × 15존 × 3차선으로 폐쇄(`:576-606`) |
| `action_policy_model.py` | 학습된 행동 정책 | **순서 무시 bag-of-words 분류기.** 40.5k 파라미터, 137토큰 어휘. `_rule_override`가 자기 학습셋의 85.9%를 텐서 생성 전에 결정 | `nn.EmbeddingBag(mode="mean")`(`:132`). `"avoid the child and pull over"` → **`start`** (fail-open 안전 역전) |
| `policy_node.py` / `stage_a.pt` | VLA 정책 | **목표를 못 보는 ResNet18 차선추종기.** 언어는 정수 → `nn.Embedding`: 11,348,142 중 **536 파라미터 = 0.0047%** | 어블레이션: goal zone 셔플 시 조향 변화 **0.0049 rad/s**, **이미지** 셔플 시 **0.1702 rad/s** (35배). pose/goal 전체 0으로 → 0.0051 |
| `train_stage_a.py` | VLA 학습 | **언어 조건부 학습이 아예 아님** | `grep -rn instruction src/nav_vla_pkg/train/*.py` → **0회** (직접 확인). 83k 프레임의 언어를 읽는 모델이 없음 |
| Alpamayo 연동 | VLA teacher | **쓰기 전용 사이드카.** 실제 10B 추론은 되지만 결과는 Tk 위젯 + JSONL로만 감 | 기본 비활성(`alpamayo_endpoint=""`, `:300-305`). `grep -rni alpamayo train/*.py` → **0회** (직접 확인) |
| `obstacle_vla_node.py` | 반응형 VLA | **ResNet18 → 로짓 1개 → `std_msgs/Bool`.** 프레임당 1비트, 언어 없음 | `:79-88` |
| **VLA_AD** `vla_trt_node.cpp` | VLA 정책 | **JSON 5키만 내는 VLM. 횡방향 액션 0** | `SYSTEM_PROMPT_POLICY`(`:40`)가 요구하는 키는 `safety_mode, speed_scale, scenario, current_lane, reasoning`뿐. `waypoint_offset_px`는 `j.value(..., 0.0f)`(`:411`)로 **항상 0** (직접 확인) |
| **VLA_AD** LASA confidence 게이트 | 논문 기여 3 | **사실상 死코드.** `confidence = j.value("confidence", 0.8f)`이고 `0.8 > c_min=0.5` → JSON 파싱 실패 시에만 발동 | `vla_trt_node.cpp:412` |
| **VLA_AD** 구동 | End-to-end | **100% 고전 제어.** YOLO 다항식 위 pure-pursuit → `max(min(int(angle/5),7),-7)` = **조향 15단계** | `motion_planner_node.py:213` (직접 확인). 자체 어블레이션: A0(VLA 없음) CTE 49.8px vs A4 50.2px |

**요약:** VLA_AD는 언어층이 더 성숙하고, 액션층은 nav-vla보다 **덜** 학습되어 있다.
nav-vla 정책은 그래도 픽셀에서 연속 `(linear, angular)`를 회귀하지만, VLA_AD의 신경망은 조향을
아예 건드리지 않는다.

---

## 2. VLA의 기술적 기준선

논문에 넣고 조항별로 방어할 수 있는 **반증 가능한** 정의:

> 시스템이 **VLA**이려면 **(a)** 자유형 언어 토큰과 이미지 패치가 **같은 forward pass**를 타고,
> **(b)** *학습된* 헤드가 **연속·다자유도 액션**을 낸다(폐쇄 테이블에서 심볼 선택 ✕),
> **(c)** 비전을 고정하고 **언어만 바꿨을 때 액션이 실제로·올바르게 달라지며**,
> **(d)** 제어 경로 안에서 손으로 쓴 규칙이 모델 의도를 덮어쓰지 않는다.

| 기준 | nav-vla | VLA_AD | 실패 이유 |
|---|---|---|---|
| (a) 비전+언어 통합 forward pass | ❌ | ✅ | nav-vla는 정책 **이전에** 언어를 `int`로 이산화 |
| (b) 학습 헤드의 연속 다자유도 액션 | ❌ (2-DoF지만 목표 무시) | ❌ (**1-DoF**: `speed_scale`) | VLA_AD 모델은 `V_BASE=100`에 곱할 스칼라만 냄 |
| (c) 언어 반사실 민감도 | ❌ **0.0049 rad/s** | ❌ (VLM 제거 → `dy=0, α=1.0` 정상주행) | **진짜 실패 지점. 그리고 이건 데이터 문제다** |
| (d) 루프 내 규칙 미개입 | ❌ (`_apply_*`, `_rule_override` 85.9%) | ⚠️ (LASA는 정당한 *안전 포락선*) | nav-vla는 regex가 LLM보다 최종 권한 보유 |

---

## 3. 핵심 진단 — 막고 있는 건 모델이 아니라 데이터다

83,350 프레임 전부를 만든 오라클(`simple_track_driver_node.py`)은 **명령을 입력으로 받지 않는
맵루프 pure-pursuit 추종기**다. 따라서 데이터 안에서 이미 `p(action | image, instruction) = p(action | image)`
이다. **이 코퍼스로는 OpenVLA·π0·Alpamayo 무엇을 파인튜닝해도 파라미터만 100~300배 큰
같은 차선추종기가 나온다.**

측정값 — pose를 (1 m × 22.5°) 셀로 묶고, 명령만 다른 쌍끼리 2.1초 후 횡방향 위치 차이:

| 쌍 | 매칭 수 | 중앙값 발산 | >0.5 m |
|---|---|---|---|
| `goal_zone` 다름 | 5,523 | **0.063 m** | 24.2% |
| `goal_lane` 다름 | **150** | 0.602 m | 58.0% |

목표 조건화가 **6 cm**. 차선 반사실 쌍은 전 코퍼스에 **150개**뿐.

직접 확인한 코퍼스 실태:
- 619 에피소드 / **619개 전부 `success: true`** → **복구(recovery) 데이터 0.** 10Hz 실차에서
  covariate shift가 바로 실패 모드가 된다.
- **고유 명령 문자열 182개**, 전부 영어 템플릿(`data_engine_node.py:99-111`의 `random.choice`).
  **한국어 학습데이터 0** (GUI는 한국어 음성을 받는데도).
- 전체 프레임의 31%가 cruise 에피소드 4개.
- `steps.jsonl`에 **타임스탬프 필드 없음** (프레임 mtime 간격 0.201~0.265 s, ±15% 변동).

> **결론: 데이터가 병목이다. 나머지는 전부 부차적이다.**

---

## 4. 권고 — 4개 질문에 대한 확정 답

### 질문 (1) VLM인가 LLM 파서인가?
**LLM 파서다.** 그것도 regex가 최종 권한을 갖는 파서. (VLA_AD 쪽은 진짜 VLM 맞다.)

### 질문 (2) VLA까지 가려면?
§2의 (a)~(d)를 전부 만족시켜야 하고, 그중 **(c)를 막는 것은 모델이 아니라 데이터**다.
→ 목표 조건부 오라클 구축 → 반사실 데이터 재수집 → 액션청크 헤드 학습 (§5).

### 질문 (3)+(4) 알파마요 vs 다른 VLA vs 내 데이터로 자체 학습

**결론: 자체 소형 VLA를 만들고(비-자기회귀 액션청크 헤드), Alpamayo-1.5는 오프라인
reasoning 라벨러 + 평가 심판으로 강등한다. 어떤 파운데이션 VLA도 차에 올리지 않는다.**

근거 4가지:

1. **Thor는 출력 형태에 따라 성능이 극단적으로 갈린다.**
   - *액션 헤드* VLA: GR00T-N1.6-3B가 TensorRT BF16으로 **92 ms / 10.9 Hz** (NVIDIA 공식).
     Cosmos-3 Edge(4B)는 640×360에서 15 Hz, 추론 1회당 32액션.
   - *자기회귀 텍스트 생성* VLM: 같은 보드에서 디코드 **37.3 tok/s** → 약 0.2~0.3 Hz.
     (VLA_AD의 실측 57 ms/token과 정확히 일치한다.)
   → **액션이 토큰 스트림에서 나오는 설계는 전부 1 Hz 근방에서 막힌다.** "Qwen3-VL이 액션
   토큰을 뱉게 한다" 안과 AutoVLA 안이 이 한 줄로 동시에 탈락.
   ⚠️ 커뮤니티에 도는 "π0.5가 Thor에서 23 Hz" 수치는 **검증 불가** — 자작 CUDA 커널을 쓴
   포럼 글 1건이다. 인용하지 말 것. 방어 가능한 상한은 **GR00T-N1.6-3B의 10.9 Hz**.

2. **NVIDIA 자신의 Alpamayo 가이드가 "배포"가 아니라 "증류"다.** 모델 카드 스펙:
   RGB 카메라 **4대 @10Hz**, VRAM **24 GB+**, H100 검증, **비상업 라이선스**(코드는 Apache-2.0),
   그리고 실제 크기는 10B이 아니라 **11B**(Cosmos-Reason2 백본 8.2B + 디퓨전 액션 전문가 2.3B).
   NVIDIA는 "개발자가 풀 모델을 차량에 올리는 일은 드물 것"이라 명시한다.
   → 질문 (3)은 NVIDIA 기준으로도 틀린 형태, **질문 (4)가 NVIDIA가 권장하는 경로다.**

3. **채택해도 액션 헤드는 어차피 버려야 한다.** 매니퓰레이션 VLA(OpenVLA/π0/GR00T)는
   7-DoF 그리퍼 델타를 낸다 — 무용. Alpamayo는 궤적을 내지만 **6.4 s / 64 waypoint** 지평이고
   카메라 높이 1.5 m, 속도 10~30 m/s 실차 고속도로 데이터로 학습됐다. 우리 차는 ~1.5 m/s,
   카메라 ~30 cm, 트랙 ~15 m. **6.4초면 거의 반 바퀴다.** 사전학습 prior가 역전이(anti-transfer)한다.

4. **당신이 가진 희소 자산은 고정형 OT128이다.** 실차 ego 궤적을 사람 라벨링 없이,
   심지어 사람 조종 없이 자동으로 라벨링할 수 있다. 어떤 사전학습 체크포인트보다 가치가 크다.

**확정 결정:**

| 역할 | 무엇 | 비고 |
|---|---|---|
| 배포 정책 | **신규 제작**: 소형 VLM 트렁크 + **비자기회귀 액션청크 헤드** → **미터 단위 ego-frame waypoint** | 개발 반복은 SmolVLM2-500M(Apache-2.0, 8GB 노트북 가능), 최종 탑재는 3B급 |
| 의미 거버너 | **Qwen3-VL-8B INT4 그대로 유지** → `VlaIR` | ~0.2~1 Hz. 이미 SFT·양자화·TRT 완료. **학습된 정책을 감싸는 순간 진짜 기여가 된다** |
| 교사 | **Alpamayo-1.5: 오프라인 전용, 노트북 전용.** reasoning 라벨 + 평가 심판 | 게이트: 실트랙 프레임 200개 사람 평가, **사실정확도 <70%면 폐기**하고 범용 VLM으로 대체 |
| 철칙 | **액션 라벨은 재생된 ego-motion(Hesai/Gazebo pose)에서만. VLM에서 절대 금지** | 위반 시 환각을 증류하게 됨 — VLA_AD 로그에 이미 증거 있음(빈 `detections: []`와 함께 나온 물체 속도 목록) |

### 액션 공간은 **미터 단위 ego-frame waypoint**로 확정 (BEV 픽셀 ✕, 정규화 조향 ✕)

- *정규화 조향은 전이되지 않는다.* κ = tan δ / L. 시뮬 `wheel_base=2.86`
  ([ackermann_cmd_adapter_node.py:23](../src/simulation_pkg/simulation_pkg/ackermann_cmd_adapter_node.py#L23)),
  실차 **54 cm**. 같은 `steer_norm=0.3`이 시뮬에선 반경 14.6 m, 실차에선 **2.75 m**.
  학습된 시각-곡률 연관 전체에 5.3배 오차가 박힌다.
- *BEV 픽셀은 교정 전까지 사용 불가.* 살아있는 캘리브 파일 3개가 `bev_px_per_cm`를
  **2.0 / 6.0 / 10.0**으로 서로 다르게 말한다. 2 px/cm면 path frame이 3.2×2.4 m밖에 안 되고
  `L_D=120px`는 **0.60 m** 예측거리 — 2 m/s에서 0.30 s로 **재계획 주기보다 짧다.**

> **정직하게 말할 것: `motion_planner_node.py`는 그대로 두지 못한다.** 미터 단위 waypoint를
> 소비하는 pure-pursuit 추종기(L_D ≈ 0.8~1.2 m, 실측 54 cm 휠베이스 기반 속도적응)를 새로
> 써야 한다. 이게 진짜 VLA의 대가이고, 계획에 숨기지 말고 명시해야 한다.

---

## 5. 로드맵 — 1인 기준, 보수적 추정

**현실적 총량: 방어 가능한 VLA까지 30~36주.** 논문 방어 가능 결과는 ~22주,
명령 일반화 주장까지는 ~32주.

### P0 — 근거 수치 확보 (1~3주)
- **Thor 온디바이스 P50/P95 실측.** 한 번도 안 했는데(`TASKS.md:3443`) `README.md:7`은 0.74 Hz,
  논문은 ~4.6 s(0.22 Hz)라고 한다. 반나절이면 되고, 모든 지연 주장이 여기 걸려 있다.
- `bev_px_per_cm`를 **하나로** 확정하고 나머지 2개 파일 삭제.
- 차량 실측: 휠베이스, 최대 조향각, 카메라 높이/피치/내부파라미터.
  **1/5 vs 1/10 모순 해소** (`README.md:4` vs `PORTFOLIO_HANDOFF.md:5,632` "Traxxas 1/10").
- `steering ∈ [-7,7]`이 펌웨어 한계인지 임의 클램프인지 확인
  (`motion_planner_node.py:213`). **15단계 = 3.9비트가 하류 전체의 상한이다.**

### P1 — 오라클을 목표 조건부로 (3~7주) ← *프로젝트 최고 ROI*
`simple_track_driver_node.py`가 `/oracle_goal` = `{goal, lane, stop_or_pass, speed}`를 받도록
패치하고 lookahead를 그에 맞게 재구성. **VLA가 대체할 대상인 목표 인식 플래너를 먼저 만들어야
한다.** 표준 BC 부트스트랩이니 이름 붙이고 일정에 넣을 것 — L1 베이스라인 겸용이다.
- `train/measure_counterfactual_divergence.py` 제작. 지표 **CD** = 매칭된
  (pose셀, 명령상이) 쌍의 2초 후 횡방향 차이 중앙값.
- **게이트: 4개 기동 클래스(분기 선택 / 차선변경 / 정지 vs 통과 / 속도)에서 매칭쌍 ≥5,000개,
  CD ≥ 0.5 m. 현재 5,523쌍 @ 0.063 m. 통과 전에는 학습 시작 금지.**

### P2 — 시뮬을 실트랙의 미터 단위 복제본으로 (5~9주, P1과 병행)
`zone_map.yaml`은 T존이 있는 **~40 m 야외** Gazebo 트랙이고 실트랙은 T존이 없는
**~15 × 11 m 실내**다(homography에서 유도 — *줄자로 검증할 것*). 두 개를 co-train하는 건
라벨 공간이 다른 두 과제를 섞는 것이다.
- Gazebo 월드를 실트랙 치수로 재구축, Prius(L=2.86 m)를 실측 휠베이스 Ackermann 모델로 교체,
  카메라 높이/FOV 정합, **렌즈 왜곡 랜덤화**(실카메라 k1=−0.41, k2=−0.74, k3=1.29 —
  시뮬 카메라는 왜곡 0 핀홀이고 ±3° 외부파라미터 지터로는 못 덮는다).
- **검증:** 같은 스크립트 경로를 시뮬/실차에서 주행 → Hesai 궤적과 Gazebo 궤적 오버레이 →
  종점 발산 < 0.3 m.

### P3 — 데이터 엔진 v2 + 수집 (7~15주)
`data_engine_node.py` 변경:
- **모든** `steps.jsonl` 행에 `instruction`, **`t`(ROS stamp — 현재 없음)**,
  `action_chunk`(미래 ego-frame pose 8~16개, 미터), `phase`, `event` 기록(`:256-270`).
  이중 인코딩된 `lane` JSON 문자열 파싱.
- `record_fps` 5 → 10.
- 15개 템플릿 `random.choice`를 **≥2,000개 명령 패러프레이즈 뱅크, 한/영 50:50**으로 교체.
  홀드아웃은 표현만 바꾼 게 아니라 **조합 홀드아웃**(존 × 차선 × 속도 조합 전체를 미학습).
  패러프레이즈 홀드아웃은 어휘 강건성만 보지, 접지(grounding)를 보지 않는다.
- **off-nominal 복구 데이터 추가.** 액션 노이즈 주입 + perturb-and-recover 에피소드,
  DAgger 1라운드 계획.
- **Hesai-in-the-loop 오라클로 실트랙 반사실 수집.** `/ego_pose_track`을 제어 루프에 넣어
  목표 조건부 오라클이 **실차를 직접** 반사실 쌍으로 주행 — 사람 조종 불필요.
  VLA는 카메라 단독으로 학습·평가한다. **실트랙 데이터를 감당 가능하게 만드는 핵심 수.**
- **게이트:** 시뮬·실차 **양쪽** ≥5k 쌍에서 CD ≥ 0.5 m; 실주행 ≥3시간(50분 ✕);
  **절대** 곡률 임계(rad/m) 초과 프레임 ≥8k (백분위 기준 ✕ — "75퍼센타일 초과"는 아무것도
  안 해도 만족된다).

### P4 — 학습 (13~22주)
- 트렁크 + ACT식 청크 헤드, 지평 감쇠 가중 L1, 보조 **`arrived` 로짓**.
  *멈출 때를 스스로 결정 못 하는 VLA는 루프를 닫은 게 아니다* — 지금은 GT pose 기하 340줄
  ([policy_node.py:702-739](../src/nav_vla_pkg/nav_vla_pkg/policy_node.py#L702-L739))이 그 판단을 한다.
- P2 이후 두 도메인이 공유하는 명령 축에 한해 sim:real co-train.
- **컴퓨트는 빌려라.** 개발 노트북은 RTX 4060 Laptop(VRAM 8 GB) — 3B 모델을 448px로
  LoRA 학습 불가이고, 같은 GPU에서 Gazebo 폐루프와 3B 추론서버를 함께 돌리면 OOM.
  3B / 100k 프레임 / 3 epoch ≈ 2.2e18 FLOPs → **8×H100 노드에서 하루 미만.
  전체 프로젝트 $500~1,500.** 싸고, 선택사항이 아니다.
- **게이트:** open-loop ADE@2s < 0.25 m. 단 플랜트가 조향을 15단계로 양자화하므로
  **양자화 반 스텝 미만의 게이트는 액추에이터가 표현 못 하는 걸 재는 것**임에 유의.
  반사실 게이트: 홀드아웃 매칭 프레임에서 "차선변경 명령" 궤적이 "차선유지" 궤적과
  2초에 >0.5 m 발산해야 함.

### P5 — 배포·통합 (20~30주)
- 트렁크+헤드 ONNX → TensorRT. **2주가 아니라 3주 잡을 것** (VLM export는 상습 초과).
- 신규 `TrajectoryProposal.msg`(`traj_x_m[]`, `traj_y_m[]`, `dt_s`, `confidence`, `ttl_ms`).
  **`VlaIR`를 확장하지 말 것** — LASA 로그 스키마와 모든 오프라인 스크립트가 깨진다.
- 미터 단위 pure-pursuit 추종기 50 Hz 신규 작성.
- **LASA 궤적 게이트:** 실측 54 cm 휠베이스 기준 κ ≤ κ_max; 코리도 포함(모든 waypoint가
  발행된 `LaneCorridorPolygon` 내부); waypoint별 slew; TTL 만료 시 0이 아니라 **YOLO 차선 경로로**
  폴백.
- **죽은 confidence 게이트를 살릴 것.** 액션 헤드에서 실제 confidence 산출(앙상블/드롭아웃
  불일치, 또는 flow-matching 샘플 분산 — 결정론적 L1 헤드는 엔트로피가 없으니 메커니즘을
  의도적으로 고를 것). **이래야 논문 기여 3이 실제로 참이 된다.**
- **불일치 게이트는 기동 조건부 레퍼런스를 써야 한다.** 차선추종 pure-pursuit 기준으로 게이팅하면
  VLA를 VLA답게 만드는 차선변경·회전·정지를 정확히 거부한다. P1에서 만든 *명령된 목표*
  pure-pursuit과 비교할 것.

### P6 — 어블레이션 + 집필 (28~36주)
A-parser(regex) / A-classifier(`action_policy`) / A-stage_a(목표 무시, **음성 결과**) /
A-governor-only(현 VLA_AD) / A-full. 학습본 vs 미학습 패러프레이즈 vs 조합 홀드아웃을
**Hesai 외부 측정**으로 보고.

### nav-vla → VLA_AD 이식 / 폐기

| 이식 | 이유 |
|---|---|
| 에피소드 레코더(`data_engine_node.py`) 스키마 v2 | 최대 자산. 논문 전체가 여기 얹힘 |
| Gazebo 월드 + 트랙 기하, **미터 단위 재구축** + 축척 맞춘 Ackermann 모델 | 시뮬은 데이터 공장. 실트랙과 일치해야 함 |
| `simple_track_driver_node.py` — **목표 조건화 패치 이후** | 수집 오라클 + L1 베이스라인 + 기동 조건부 안전 레퍼런스 |
| 음성/텍스트 입력면만 (mic → STT → `/operator_text`) | VLA가 **원문 문자열**을 소비. **경로에 파서 없음** |
| `speed_control.py` 0..250 매핑 | 이미 `MotionCommand`와 호환 |
| 존 개념 → 트랙 프레임 미터 `track_map.yaml` | 사람용 이름만. 모델은 목표를 **문장에 렌더된 연속 ego-frame 벡터**로 받지, 클래스 라벨로 받지 않는다 |

| 폐기 | 이유 |
|---|---|
| `action_policy_model.py` / `action_policy_node.py` / `action_sentence_generator.py` | 순서 무시, 규칙이 85.9% 발동, fail-open 안전 역전. 논문 표 한 줄 이상의 가치 없음 |
| `chat_gui_node.py:675-735`, `:1339-1468` (`_deterministic_drive_plan`, `_apply_*`) | regex가 모델 의도를 재작성하는 건 실격 사유. `baseline_parser.py`로 옮겨 A-parser 어블레이션으로 |
| `policy_node.py` / `stage_a.pt`의 forward 경로 | **문서화된 음성 결과로 동결.** 지금 보유한 가장 출판 가치 높은 산출물이 이 어블레이션 표다 |
| `gz_pose.py` (`gz model -m ego_vehicle -p` 서브프로세스) | 실차에서 `None` 반환. sim/real 공용 `/ego_pose` 토픽 추상화로 교체 |
| `alpamayo_teacher_server.py` (MOCK) | 조작된 "reasoning"으로 로그 오염 |
| "VLA Judgment (CoC-lite)" 패널 | 손으로 쓴 f-string + 5분기 if/elif(`:742-777`, `:1120-1134`). "State Monitor"로 개명하거나 모델로 뒷받침할 것. 아키텍처와 무관하게 방어 취약점 |
| 619 에피소드 코퍼스를 *명령 조건부* 학습데이터로 쓰는 것 | 독립 pose+heading 셀 ~1,128개(셀당 68.5회 재방문), 전 프레임의 31%가 cruise 4개, 619/619 성공, 영어 템플릿 182개. **차선유지 warm-start와 음성 결과 어블레이션 용도로만** |

### 방향과 무관하게 고칠 버그
- [policy_node.py:455](../src/nav_vla_pkg/nav_vla_pkg/policy_node.py#L455) — `if self.task_type == "direct" and pose:`
  가 pose가 `None`일 때 **학습 안 된 신경망 분기로 낙하**한다. **fail-closed**(0 발행)로.
- `action_policy_model.py` OOV — 파싱 실패 명령은 반드시 `none`/`stop`, 절대 `start` 금지.
- `alpamayo_real_server.py:179-189` — 짧은 시퀀스를 frame 0 복제로 패딩. 진짜 4프레임
  시간 스택(t−1.5/−1.0/−0.5/0)을 먹이기 전에는 출력을 신뢰하지 말 것.

---

## 6. 하드웨어 배치

| 위치 | 무엇 | 주기 | 비고 |
|---|---|---|---|
| **Thor (차상)** | 카메라 | 30 Hz | |
| | YOLOv8-seg 차선 인지 + BEV | 50 Hz | **유지.** 이미 랩을 완주하는 L1 폴백 |
| | **소형 VLA 정책**(VLM 트렁크 + 비자기회귀 액션 헤드) → `TrajectoryProposal` | **10 Hz** | NVIDIA 실측 GR00T-N1.6-3B = 92 ms / 10.9 Hz TRT BF16 기준. 청크 지평 ≥ 재계획 주기 ×5 |
| | Qwen3-VL-8B INT4 → `VlaIR` 의미 거버너 | **~0.2~1 Hz** | **P50/P95 먼저 실측.** README 0.74 Hz와 논문 4.6 s가 동시에 참일 수 없음 |
| | LASA(스키마+TTL+곡률+코리도+slew+실 confidence) | 50 Hz | |
| | **미터 단위** pure-pursuit 추종기 (L_D ≈ 0.8~1.2 m) | 50 Hz | 재작성. "motion_planner 무변경"이라고 주장하지 말 것 |
| | `/operator_text` 안전 키워드 반사 → e-stop | **< 50 ms** | 모든 모델 우회. *파서가 아니라 반사(reflex)*로 규정 |
| | serial → 모터 | | 펌웨어가 아니라면 조향 **15단계** |
| **노트북 (트랙사이드)** | Hesai 드라이버 `HesaiLidar_ROS_2.0` (Ubuntu 24.04 / Jazzy 공식 지원 확인) | 10 Hz | |
| | 지면 RANSAC — **1회 적합 후 고정**(센서 정지) | 1회 | 차량탑재 LiDAR 대비 진짜 이점 |
| | 클러스터 → **직사각/L-shape 피팅 + 치수 prior** → CV 칼만 → `/ego_pose_track` | 10 Hz | 현 검출기는 `connectedComponentsWithStats` 중심점으로 **위치만, yaw 없음**. 단면 반사의 중심점은 heading에 따라 흘러가므로 교체 필수 |
| | `/track_geofence_estop` (Bool, 상시 무장, 500 ms 부재 시 fail-closed) | 10 Hz | |
| | STT → `/operator_text` | 사람 속도 | |
| | **오프라인:** 액션 라벨 공장, Alpamayo 교사, 평가 지표 | — | RTX 5090 Laptop 24 GB는 Alpamayo 최소 요구(24 GB) **정확히 하한** — 오프로딩/카메라 축소 예상 |
| **클라우드** | SFT 학습 | — | 8×H100 몇 시간. 총 $500~1,500 |

**Hesai의 역할은 정확히 3개, 금지 1개:**
1. **오프라인 액션 라벨 엔진** — 미래 ego pose → ego-frame waypoint 청크. 시뮬과 동일 스키마. 사람 라벨링 0.
2. **수집 전용 in-the-loop 오라클** — 목표 조건부 오라클이 실차를 반사실 쌍으로 자율 주행.
3. **독립 평가자 + 무장된 geofence** — 측량된 중심선 대비 **미터 단위** 횡오차.
   주행에 쓰는 바로 그 카메라에서 유도한 BEV 픽셀로 자기 채점하는 것 — 심사위원이 가장 먼저
   공격할 지점이고, 이게 그걸 막는다. 챕터 하나 값어치.

**금지: 평가 시점에 Hesai pose를 정책 입력으로 쓰는 것.** 정책은 카메라 단독 주행.
"결국 GT pose 쓴 거 아니냐"는 반론을 봉쇄하고, nav-vla의 GT pose 목발을 재현하지 않는다.

**측위 관련 정정:**
- 트랙은 **~15 × 11 m 실내**(homography 유도 — 줄자 검증 필요). 장변 중앙 마운트면 최대 거리
  ~13 m. 13 m에서 0.1°(H) × 0.315°(V, 균일 — **0.125°는 ROI 밴드 전용**이므로 ROI를 트랙면에
  조준할 것)이면 ~1.2×0.6×0.5 m 차량에 대략 **200~350 포인트**. **heading 추정 가능하다.**
  ⚠️ 이 구성의 OT128 yaw 정확도 공개 벤치마크는 **없음 — 직접 측정할 것.**
- **재귀반사 마스트 아이디어는 폐기.** 13 m에서 없는 문제를 풀고, 기존
  `--vehicle-height-min 0.03 --vehicle-height-max 0.35` 게이트를 깨뜨린다.
  (단 이 게이트는 차량 실측 실루엣으로 넓힐 것.)
- **yaw 전략:** v > 0.2 m/s에서는 경로접선 yaw(1.5 m/s 비홀로노믹 차량은 슬립 무시 가능,
  형상 피팅보다 낫다), 그 이하는 치수 prior 직사각 피팅. **저속 구간이 곧 "정지 명령" 시나리오** —
  가장 중요한 반사실이 가장 나쁜 라벨을 받으므로 명시적으로 게이팅할 것.
  **요구: heading RMS < 3°.** 3 m 지평에서 yaw 5° 오차 = 26 cm 라벨 오차로, ADE 목표보다 크다.
- **시간 동기는 1주차 항목.** 노트북↔Thor chrony, `header.stamp`로만 정렬,
  flash-and-maneuver 테스트로 검증, 잔차 **< 30 ms**(1.5 m/s에서 4.5 cm).
- OT128은 **차량탑재** 자동차용 센서로 마케팅된다(ISO 26262/21434). 우리 용법은 **용도전용**이므로
  "자동차급 라이다를 노변 인프라로 repurpose한다"고 쓰고 roadside-lidar 측위 문헌을 인용할 것.
- Thor의 Ollama qwen3:4b는 동작하지만 **NVIDIA 컨테이너를 쓸 것**
  (`ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04`) — 네이티브 설치는 CPU로 폴백한다.
  메모리 공존은 비이슈(122 GB 중 ~9 GB), 진짜 비용은 iGPU SM 경합. 기존 뮤텍스가 옳고 MIG가 더 깔끔.
  **더 나은 답: Thor에 올리지 말 것.** 런타임 구성요소가 아니라 베이스라인 arm이다.

---

## 7. 리스크 + 데모 폴백 사다리

각 레벨은 별도 launch 인자이자 별도 결과표 행. **어느 층이 실패해도 데모는 남는다.**

| 레벨 | 실행 내용 | 현 상태 | 실패 시 폴백 |
|---|---|---|---|
| **L0** | 하드웨어 e-stop + 수동 RC 오버라이드 | 자율주행 전 필수 | — |
| **L1** | 고전 제어만: YOLO 차선 + pure-pursuit | **오늘 동작** (A0 완주, CTE 49.8 px) | L0 |
| **L2** | L1 + Qwen3-VL-8B `VlaIR` 거버너(속도/정지/시나리오) | **오늘 동작** ~0.2~1 Hz. 말이 속도를 바꾸고 정지를 유발함 | L1 |
| **L3** | L2 + 소형 VLA 궤적, LASA 게이팅, `w·traj_vla + (1−w)·traj_perception` 블렌드(0.5 s 램프) | ~26주 목표 | `w→0` 으로 L2 |
| **L4** | VLA 완전 권한(`w=1`), 동일 시작 pose 반사실 A/B | ~30주 목표 | L3 |
| **L5** | 미학습 패러프레이즈 + 조합 홀드아웃 일반화 | ~34주 목표 | L4 |

**리스크 순위:**

1. **반사실 데이터 수집은 "VLA가 대체할 플래너를 먼저 만들어야 하는" 수 주짜리 작업.**
   일정 최대 위협. → P1으로 명명, 4~5주 배정. 이 오라클은 L1 겸 안전 레퍼런스라 3중으로 회수됨.
2. **실트랙 반사실 민감도가 ≈0으로 나와 하드웨어에서 차선추종으로 퇴화** — P0 결함의 도메인 재현.
   **프로젝트 전체에서 가장 가능성 높은 실패.** → §P3 Hesai-in-the-loop 오라클,
   **학습 전에** 실데이터에서 CD ≥ 0.5 m 게이트.
3. **Sim→real 갭.** 다른 트랙 시뮬 6시간 + 실주행 50분으로는 못 메운다. → P2 미터 복제본,
   왜곡·외부파라미터 랜덤화, 실주행 ≥3시간, co-train.
4. **yaw 라벨 오차가 모든 궤적 라벨을 회전시킨다.** → 수집 전 <3° RMS 게이트.
5. **플랜트 양자화(조향 15단계).** 모델 품질과 무관하게 CTE 개선 상한이 될 수 있다.
   1주차에 펌웨어 확인; 고정이면 플랜트 한계로 명시하고 모든 게이트를 그 위에 설정.
6. **당신의 트렁크+헤드에 대한 Thor 처리량은 미검증.** 앵커: GR00T-N1.6-3B 92 ms(10.9 Hz),
   Cosmos-3 Edge 4B 15 Hz. 주의: Thor의 TRT 가속비는 NVIDIA 표에서 **가장 낮다**(1.27× vs
   Orin 1.73×, RTX 5090 1.86×) — TRT가 지연을 구해줄 거라 가정하지 말 것.
   폴백: SmolVLM2-500M 트렁크.
7. **ONNX/TRT export 초과.** 3주 배정 + PyTorch eager 5 Hz 폴백을 L3-minus로.
8. **라이선스** → §8.
9. **Alpamayo 교사 품질.** 사전 확약 폴백: 실트랙 출력 200개 사람 평가에서 사실정확도 <70%면
   **완전 폐기**, 범용 VLM으로 reasoning 라벨. 기여는 *오프라인 VLM 교사를 통한 증류*이지
   Alpamayo 자체가 아니다.

**심사에서 반드시 답할 수 있어야 하는 질문:**
> *"NVIDIA가 reasoning과 궤적을 함께 내는 오픈 10B VLA를 공개했는데, 왜 더 작은 액션 헤드를 새로 만들었나?"*

답: 1.5 m 카메라·10~30 m/s에서 학습된 6.4 s 지평은 30 cm 카메라·1.5 m/s의 15 m 실내 트랙에서
반 바퀴에 해당한다. 11B 비상업 가중치는 Edge-LLM/TRT 경로에 들어가지 않고, NVIDIA 자신의
가이드가 배포가 아니라 증류다. **단, 이 논리는 reasoning trace에도 똑같이 적용된다** —
그러니 도메인 갭으로 액션을 실격시키면서 말은 신뢰하는 게 아니라, **교사도 경험적으로 게이팅**할 것(리스크 9).

---

## 8. 모델 후보 · 라이선스 (사실검증 완료)

| 모델 | 실제 크기 | 라이선스 | 액션 공간 | 이 프로젝트 적합성 |
|---|---|---|---|---|
| `nvidia/Alpamayo-1.5-10B` | **11B** (8.2B Cosmos-Reason2 + 2.3B 디퓨전 액션) | **비상업**(코드는 Apache-2.0) | **진짜 VLA**: 6.4 s / 64 waypoint 궤적 + CoC reasoning | **오프라인 교사로만.** 1.5는 nav guidance·가변 카메라 지원(R1은 미지원). 파인튜닝 레시피: `NVlabs/alpamayo-recipes` |
| `nvidia/Alpamayo-R1-10B` | 11B | 비상업 | 궤적 + reasoning | **✕ — 명시적으로 route/waypoint 입력 미지원** |
| `openvla/openvla-7b` | 7.54B | **MIT** | 7-DoF 그리퍼 델타 | ✕ 액션 의미 무용. 아키텍처만 참고 |
| OpenVLA-OFT | (레시피, 베이스 아님) | MIT | L1 회귀 + 청킹 + 병렬 디코드 | **○ 레시피로 채택 가치 높음** |
| openpi π0 / π0.5 | 3.50B / 3.62B | Apache-2.0(가중치는 Gemma 약관 확인) | 팔 관절 + **x-y base 속도** | ✕ Ackermann 비홀로노믹 표현 불가. 아키텍처 재사용만 |
| `nvidia/GR00T-N1.7-3B` | ~2.7B급 | **NVIDIA Open Model(상업 허용)** | 휴머노이드 모터 벡터 | ✕ 주행 사전학습 없음. **단 Thor 실측 앵커로 인용 가치** (N1.5는 비상업, **N2는 미공개 — 2026년말 예정**) |
| `lerobot/smolvla_base` | 450M | Apache-2.0(상속) | flow-matching 연속 청크 | **◎ 개발 반복용 최적. 8GB 노트북 가능** |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 3.75B | ⚠️ **`qwen-research` 비상업** | (백본) | 트렁크 후보지만 **라이선스 함정** |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 8.29B | **Apache-2.0** | (백본) | 상업 여지 필요 시 이쪽. 단 Thor 지연 재검증 필요 |
| `OpenDriveVLA/OpenDriveVLA-0.5B` | 733M | **Apache-2.0** | 궤적 | 라이선스 최상. **단 학습 스크립트 미공개** → 자체 데이터 파인튜닝 불가 |
| AutoVLA | Qwen2.5-VL-3B 기반 | ⚠️ Academic + qwen-research **이중 제약** | 이산 drive token(자기회귀) | ✕ 자기회귀 = Thor에서 ~1 Hz 상한 |
| Impromptu VLA | Qwen2-VL 3B/7B | CC-BY-SA-4.0(**카피레프트**) | 궤적 | 데이터셋(80k+ 코너케이스)이 유용 |

> Alpamayo·GR00T의 **추론/레시피 코드는 전부 Apache-2.0**이다. 가중치만 제약되므로
> 파이프라인 코드는 자유롭게 공개할 수 있다. 학위논문 + 비수익 데모는 "research or evaluation"에
> 해당하여 모두 사용 가능하나, 상업화 경로가 생기면 별도 라이선스가 필요하다.

---

## 9. 이번 주 할 일 (5개)

1. **논문의 모든 주장을 떠받치는 두 수치 측정.**
   (a) Thor 온디바이스 v6 Qwen3-VL-8B P50/P95 — 한 번도 안 했고(`TASKS.md:3443`),
   같은 레포 안에서 `README.md:7`(0.74 Hz)와 논문(~4.6 s = 0.22 Hz)이 모순.
   (b) 차량 실측: 휠베이스, 최대 조향, 카메라 높이/피치/내부파라미터,
   그리고 `motion_planner_node.py:213`의 `[-7,7]`이 펌웨어인지 임의인지.
   `README.md:4`(1/5) vs `PORTFOLIO_HANDOFF.md:5,632`(Traxxas 1/10) 모순 해소.
2. **BEV 스케일 확정.** 캘리브 파일 3개가 2.0 / 6.0 / 10.0 px/cm. 줄자로 재고 1개만 남기고 삭제.
   **미터 단위 작업 전체가 여기 막혀 있다.**
3. **`simple_track_driver_node.py`에 `/oracle_goal` 수용 패치.** 프로젝트 최고 ROI.
   이게 생기기 전까지 추가로 기록하는 모든 프레임은 목표 무시 데이터라 VLA 학습에 무가치하다.
4. **`train/measure_counterfactual_divergence.py` 작성 후 기존 619 에피소드에 실행.**
   예상: goal-zone 쌍 ~0.063 m(n=5,523), lane 쌍 ~0.60 m(n=150).
   이 수치는 동시에 **동기 부여 음성 결과 + 데이터 수용 게이트 + 현재 보유한 가장 출판 가치 높은
   산출물**이다(goal 셔플 어블레이션 표 Δang 0.0049 vs 이미지 셔플 0.1702와 함께).
5. **Hesai 측위에 heading 부여.** 치수 prior 직사각/L-shape 피팅 + 0.2 m/s 이상 경로접선 yaw +
   CV 칼만 → `/ego_pose_track`에 `nav_msgs/Odometry` 10 Hz 발행.
   3 / 8 / 13 m 정지 격자에서 줄자 검증. **게이트: 위치 < 5 cm, heading RMS < 3°.**
   같은 세션에서 chrony + flash-and-maneuver 동기 테스트(< 30 ms)까지.

**문서 손보는 김에 정정할 것:** Thor는 **128 GB**(T5000), 64 GB 아님
(`BUGFIX_LOG.md:1241,1475`, `TASKS.md:3247`). **sm_110 / CC 11.0 Blackwell iGPU이고 GB200이 아님**
(`bugfix_log.md:341,376`). 2070 TFLOPS는 **sparse FP4**(dense는 1035).
JetPack 7.1은 의도적 고정임을 명시(7.2가 2026-06-02 출시) — 안 그러면 구버전으로 읽힌다.

---

## 부록 — 이 문서의 근거 수준

- **직접 검증(이 저장소에서 실행):** 619 에피소드 / 619개 전부 `success:true` / 고유 명령 182개 /
  `grep instruction src/nav_vla_pkg/train/*.py` = 0 / `grep alpamayo train/*.py` = 0 /
  VLA_AD `SYSTEM_PROMPT_POLICY` 5키 / `waypoint_offset_px` 기본값 0.0f /
  `steering = max(min(int(angle/5),7),-7)`.
- **웹 출처 검증:** Thor 사양·GR00T/Cosmos Thor 벤치마크·Alpamayo 모델카드/라이선스·
  OT128 사양·HesaiLidar_ROS_2.0 Jazzy 지원·각 모델 라이선스 태그.
- **검증 불가로 표시:** π0.5 Thor 23 Hz(포럼 1건, 자작 커널) / 이 구성의 OT128 yaw 정확도 /
  Thor에서 Ollama와 TRT 엔진 동시 구동 벤치마크.
- **어블레이션 수치**(0.0049 vs 0.1702, CD 0.063 m)는 분석 중 산출된 값으로,
  §9-4의 스크립트로 재현 가능하게 만들 것.
