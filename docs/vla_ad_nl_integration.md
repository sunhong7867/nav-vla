# nav-vla 자연어 주행 로직 → VLA_AD(Thor) 통합 설계서

> **스냅샷 주의 (2026-08-24 부기):** 이 문서는 2026-08-07 이전 시점 기준이다. 이후 변경 — v8/v8g 방향·직행 축, "직행은 좌표 내비 위임, v8g는 순항+반응형 전담" 결정(08-24) — 은 [ver/README.md](ver/README.md) 참조.

> 목표: nav-vla 시뮬에서 검증된 **자연어 기반 주행 명령**(차선변경 · 속도 증감 ·
> 정지/출발 · 기준선 따라/무시 · 목적지(존) 이동)을, 실차 Thor의 **VLA_AD** 스택에
> 이식해 실제 트랙에서 구동한다. 글로벌 측위는 **트랙 긴 변 중앙 고정 인프라 Hesai
> 라이다**가 담당한다(차량 온보드 SLAM 없음).
>
> 상태: 설계 단계 (코드 미착수). LLM 결정 = **Ollama qwen3:4b 별도 구동**.
> 작성일: 2026-07-25.

---

## 1. 핵심 전제 — 왜 이 통합이 자연스러운가

두 저장소는 **같은 뿌리**다. 둘 다 `interfaces_pkg`, `decision_making_pkg`,
`camera_perception_pkg` 계열을 공유한다. 차이는:

| | nav-vla (시뮬) | VLA_AD (실차 Thor) |
|---|---|---|
| 자연어 층 | ✅ `chat_gui_node.py` + `speed_control.py` | ❌ 없음(=이식 대상) |
| 실행/구동 | Gazebo `cmd_vel`/Ackermann | YOLO 인지 → VLM/VlaIR → LASA → motion_planner → serial |
| 측위(pose) | Gazebo GT pose (`gz_pose.py`) | **전무** (전부 카메라 상대 BEV 픽셀) |
| 글로벌 좌표계 | world frame (`zone_map.yaml`) | **없음** |

**결정적 관찰 두 가지:**

1. **nav-vla의 자연어 층은 완전히 독립적이다.** `chat_gui_node.py`는 순수
   `std_msgs`(String/Int32) 5개 토픽만 발행한다. 커스텀 메시지·Gazebo 의존이 **0**.
   → 파싱/플래닝 코어를 그대로 들어낼 수 있다.

2. **VLA_AD는 글로벌 좌표계가 전혀 없다.** pose/odom/맵을 소비하는 노드가 하나도
   없다. 순수 반응형 주행이라 "목적지로 이동"이 원천적으로 불가능하다.
   → **인프라 라이다가 이 공백을 정확히 메운다.** 라이다가 차량의 트랙-프레임 위치를
   외부에서 계산해 Thor로 넘겨주면, VLA_AD에 없던 글로벌 측위가 생긴다.

이 두 관찰이 **2계층 통합**으로 이어진다.

---

## 2. 2계층 아키텍처

### Tier 1 — 반응형 명령 (라이다 불필요, 카메라만으로 데모 가능)
차선변경 · 속도 증감 · 정지/출발 · 횡방향 오프셋(기준선 따라/무시 근사).
VLA_AD의 기존 진입점에 직접 매핑 + 차선변경 실행기 신규.

### Tier 2 — 목적지/존 이동 (인프라 라이다 pose 필요)
`(forward_m, lateral_m, heading)`을 라이다에서 DDS로 Thor에 전달 → 포팅한
navigator가 존 도착판정/direct 추종 → VLA_AD 구동부 연결.

> 데모 전략: Tier 1을 먼저 완성해 라이다 네트워크와 무관하게 "말로 움직이는" 데모를
> 확보하고, Tier 2를 얹어 "목적지 이동"까지 확장한다. 라이다 링크가 불안정해도
> Tier 1은 항상 동작 → 데모 리스크 분리.

---

## 3. VLA_AD 명령 진입점 (seam) 정리

이식 대상이 물릴 자리. (출처: `src/decision_making_pkg`, `src/interfaces_pkg`)

| 진입점 | 타입 | 의미 | LASA 경유 |
|---|---|---|---|
| `vla/text_command` | `std_msgs/String` | 자연어 → VLM이 VlaIR 생성 (기존 유일 NL 경로) | VLM→LASA |
| `vla/ir_raw` | `interfaces_pkg/VlaIR` | **경계 의도**: Δy + α + 안전/차선 라벨 | ✅ LASA |
| `operator/speed_override` | `std_msgs/Float32` (0~1, `-1`=해제) | 즉시 속도 override | ✕ 직결 |
| `operator/estop` | `std_msgs/Bool` | 비상정지(behavior_manager ESTOP) | ✕ |
| `lane/manual_override` | `std_msgs/String` (`lane1\|lane2\|auto`) | ego-lane **라벨만** 변경(경로 재타겟 아님) | ✕ |

**`VlaIR.msg` 핵심 필드**
```
float32 waypoint_offset_px  # Δy (BEV-ROI px). +=우측 회피, -=좌측. LASA에서 ±160 clamp
float32 speed_scale         # α. 1.0=유지, 0.0=정지 (motion_planner에서 [0.3,1.0] clamp)
string  safety_mode         # NORMAL | CAUTION | STOP
string  current_lane        # lane1(좌) | lane2(우) | unknown
string  scenario            # none|obstacle_on_path|traffic_signal|...
float32 ttl_ms  float32 confidence  string reasoning
```

**motion_planner 소비 방식** (`decision_making_pkg/motion_planner_node.py`, 50Hz)
```python
# ir_applied(=LASA 승인본) 사용
dy_offset_px = ir_applied.waypoint_offset_px
alpha        = clip(ir_applied.speed_scale, 0.3, 1.0)
if speed_override >= 0.0: alpha = clip(speed_override, 0.0, 1.0)  # override 우선
x_shifted = [x + dy_offset_px for x in x_points]   # 횡방향
speed     = int(v_base(100) * alpha)               # 종방향
# pure-pursuit → steering = clip(int(deg/5), -7, +7)
```
**최종 구동**: `MotionCommand{steering[-7..7], left_speed, right_speed[0..255]}` →
`topic_control_signal` → serial `s{steering}l{left}r{right}\n` (/dev/ttyACM0, 115200).

**차선변경 실행기 부재(중요):** `LaneInfo.adjacent_target_points`(옆차선 기하)가
발행되지만 **아무도 소비하지 않는다.** `lane/manual_override`는 라벨만 바꾼다.
→ 진짜 차선변경은 신규 구현 필요(§6).

---

## 4. nav-vla 자연어 층 정리 (이식 원본)

`chat_gui_node.py` 파싱 순서: **regex 우선 → qwen3:4b 폴백 → regex 후처리 교정**.
LLM 출력을 절대 raw로 신뢰하지 않고 `_apply_*` 교정기가 원문 regex로 덮어씀.

**LLM 호출**: Ollama `http://localhost:11434/api/chat`, `model="qwen3:4b"`,
structured `format`(JSON 스키마), `temperature=0`. 플랜 스키마:
`{"steps":[{action, zone, lane, speed}], "reason":...}`,
`action ∈ {drive_to_zone, drive_direct, change_lane, keep_lane, stop, start, set_speed, none}`.

**5개 명령 토픽 (전부 std_msgs)**

| action | 토픽 | 타입 | 페이로드 | 자연어 예 |
|---|---|---|---|---|
| drive_to_zone | `/nav_goal` | String | `{"zone":"T2"}` (+`"lane"` 선택) | "1차선 따라 T2까지 가" |
| drive_direct | `/direct_nav_goal` | String | `"T2"` | "차선 무시하고 M3로 가" |
| change_lane / keep_lane | `/lane_mode_command` | String | `"lane1"`/`"lane2"` | "2차선으로 변경" |
| stop | `/motion_control_command` | String | `"stop"` | "정지", "멈춰" |
| start | `/motion_control_command` | String | `"start"` | "출발", "resume" |
| set_speed | `/speed_command` | Int32 | `0..250` | "속도 100으로", "빠르게" |

**speed_control.py** (순수·이식 가능): 0~250 raw, `DEFAULT=150`,
`parse_speed_raw`(명시 숫자 → 상대 ±20 → 명명 slow70/normal150/fast200).
`*_mps` 변환 2개만 실차 스케일로 조정.

**navigator_node.py** (실행기, 시뮬 pose 결합):
- lane-follow 모드: 스티어링 계산 안 함. 목표 차선을 `/lane_mode_command`, `"start"`를
  `/motion_control_command`에 발행하고 GT pose로 존 도착판정 → `"stop"` + `/nav_status`에
  `arrived: <zone>`. **실제 조향은 카메라 lane follower가 담당.**
- direct 모드: GT pose 기반 자체 pure-pursuit → `/direct_motion_command`(MotionCommand).
- 차선변경 = `/lane_mode_command`에 lane1/lane2 재발행.
- **모든 pose = `gz_pose.py`(Gazebo CLI).** → Tier 2에서 라이다 pose로 교체 대상.

**존(zone)**: `config/zone_map.yaml`의 명명 위치. 각 존 = world-frame `pose:{x,y,yaw}`,
`tol`, `stop_offset`, `arrival_mode`(stop-line/area) 등. 15개(Start,M2,T2,M3,횡단보도,
T4,T3,... , Slot1-4). Slot/IN/OUT 등 7개는 direct-only.

---

## 5. 통합 구조 — VLA_AD에 추가할 신규 패키지

```
VLA_AD/src/nl_command_pkg/            (신규)
  nl_command_node        chat_gui 파싱/플래닝 코어 이식 (tkinter/odom/Alpamayo 제거, 헤드리스)
                         → 기존 5개 std_msgs 토픽 그대로 발행
  nl_bridge_node         5개 명령 토픽 → VLA_AD seam 변환 (§3 매핑) + 차선변경 실행기
  track_pose_node        (Tier 2) /track/vehicle_pose 구독 → 존 도착판정/모드 게이팅
  config/zone_map.yaml   존 좌표 (라이다/트랙 프레임으로 재측량)
```
+ `speed_control.py`는 `vla_common` 또는 `nl_command_pkg`에 거의 그대로 복사.

### 전체 데이터 흐름
```
[자연어 텍스트/음성]
   │
   ▼  (Ollama qwen3:4b @ localhost:11434, regex 우선)
nl_command_node ──► /nav_goal /direct_nav_goal /lane_mode_command
   │                /motion_control_command /speed_command
   ▼
nl_bridge_node ─┬─ /speed_command(0..250) ─────► operator/speed_override (α=raw/250)
                ├─ /motion_control_command stop ─► operator/estop=True / override=0
                ├─ /motion_control_command start ► operator/estop=False, override=-1
                ├─ /lane_mode_command lane1/2 ───► [차선변경 실행기] (§6)
                └─ /nav_goal /direct_nav_goal ───► track_pose_node (Tier 2)
                                                        │
   [인프라 라이다]                                      │
   live_bev_intensity_viewer(+publisher) ─► /track/vehicle_pose ─┘
        (forward_m, lateral_m, heading, status)   (DDS, 노트북→Thor WiFi)
                                                        │
                              lane-follow: start/stop + lane 게이팅 ─► nl_bridge
                              direct: pure-pursuit ─► VLA_AD 구동부
   ▼
VLA_AD 기존 스택: (VlaIR/override) → LASA → motion_planner → serial → 모터
```

---

## 6. 차선변경 실행기 설계 (net-new)

VLA_AD엔 진짜 차선변경이 없다. 두 안:

**안 A (부트스트랩, 빠름):** `/lane_mode_command`를 지속적 `VlaIR.waypoint_offset_px`
(적절 부호·크기 Δy)로 변환해 차량을 옆으로 밀고, `current_lane` 라벨이 바뀌면 Δy를 0으로.
- 장점: 기존 Δy 경로 재사용, 신규 코드 최소. LASA 안전 로직 그대로 적용.
- 단점: 조향 해상도 ±7(=14 step)라 크로스 궤적이 거칠다. 오픈루프 근사.

**안 B (정식, 권장 최종):** 신규 노드가 `yolov8_lane_info`(LaneInfo, `adjacent_target_points`
포함)를 구독. 차선변경 명령 시 `path_planner`가 `adjacent_target_points`로 경로를
재타겟하도록 전환 → 실제 차선변경 실행 후 복귀.
- 장점: 옆차선 기하 데이터가 이미 발행됨(소비자만 없음). 폐루프.
- 단점: `path_planner_node`/`motion_planner`에 차선변경 상태머신 추가 필요.

> 권장: **A로 데모 부트스트랩 → B로 승격.** 두 안 모두 `lane/manual_override`로 라벨
> 일관성 동기화.

---

## 7. Tier 2 — 인프라 라이다 목적지 이동

### 7.1 노트북(라이다) 측
`live_bev_intensity_viewer.py`에 ROS publisher 추가 (현재 CSV만 기록):
- 토픽 `/track/vehicle_pose`, 페이로드 `forward_m, lateral_m, heading, status`.
  메시지: `geometry_msgs/PoseStamped`(+상태는 별도 String) 또는 커스텀
  `TrackVehicle.msg{float32 forward_m, lateral_m, heading; string status}`.
- `heading`: 현재 미산출. `self.trajectory`(연속 프레임 이동방향)에서 추정, 저속 시
  클러스터 주축 폴백.
- 전송: 노트북↔Thor 동일 `ROS_DOMAIN_ID`, WiFi DDS. `PointCloud2`가 아니라 경량
  pose만 보내므로 대역폭 문제 없음.

### 7.2 좌표 프레임 정합 (이미 확보된 자산 재사용)
존 좌표는 **라이다/트랙 프레임(m)** 으로 재측량해야 한다. 이미 만든 4점 homography
파이프라인(`0725_4점정합ver/`)을 그대로 재사용:
```
트랙 이미지에서 존 클릭 → homography_track_to_bev → pixel_to_world → (forward_m, lateral_m)
```
→ 새 `zone_map.yaml`(라이다 프레임) 생성. `pixel_to_world`는
`live_bev_intensity_viewer.py`에 이미 있음.

### 7.3 Thor 측 `track_pose_node`
`navigator_node.py`의 도착판정 + direct pure-pursuit를 이식하되, `gz_pose.WorldPoseStream`
소비부를 `/track/vehicle_pose` 구독으로 **교체**. 매핑 에이전트 결론: `stream.latest`가
`(x,y,yaw)`를 라이다 프레임으로 채워지면 navigator 로직은 거의 그대로 동작.
- lane-follow 존: 도착 게이팅(start/stop + lane) → `nl_bridge_node`
- direct 존: pure-pursuit steering/speed → VLA_AD 구동부(`topic_control_signal` 주입 또는
  VlaIR 경로)

---

## 8. LLM 구동 — Ollama qwen3:4b (Thor)

- Thor에 Ollama 설치, `ollama pull qwen3:4b`, `http://localhost:11434` 유지.
- nav-vla의 시스템 프롬프트·structured `format` 스키마·regex를 **그대로 재사용**
  (검증된 자산). 변경점: 호스트 파라미터, (필요시) 존 목록만 실트랙용으로 교체.
- 메모리: qwen3:4b ≈ 3~4GB. 기존 TRT VLM 엔진(~87GB)과 Thor 122GB 통합메모리에서 병행 여유.
  단 동시 추론 시 대역/전력 경합은 데모 중 모니터링.

---

## 9. 메시지 호환성 주의

두 저장소 모두 `interfaces_pkg`가 있으나 필드가 다르다:
- `LaneInfo`: nav-vla=`{slope, target_points, is_lane_changing}` vs
  VLA_AD=`{slope, target_points, current_lane, *_lane_width_px, adjacent_target_points}`.
- `PathPlanningResult`: nav-vla=`{x_points, y_points, is_lane_changing}` vs
  VLA_AD=`{x_points, y_points}`.
- `MotionCommand`: **동일** `{steering, left_speed, right_speed}`.

→ **nl_command_node는 std_msgs만 쓰므로 충돌 없음.** bridge/차선변경/track_pose 노드는
**VLA_AD의 `interfaces_pkg`만** 사용(nav-vla 것을 가져오지 말 것).

---

## 10. 구현 단계 (설계 승인 후)

| Phase | 내용 | 검증 | 라이다 |
|---|---|---|---|
| 0 | `nl_command_pkg` 골격 + `speed_control` 복사 + `nl_command_node`(헤드리스 파싱) | 타이핑 텍스트 → 5개 토픽 echo 확인 | ✕ |
| 1 | `nl_bridge_node`: 속도 + 정지/출발 매핑 | 실차가 말에 반응(정지/속도) | ✕ |
| 2 | 차선변경 실행기 (안 A → B) | 실 lane follower에서 차선변경 | ✕ |
| 3 | 라이다 pose publisher + 존 재측량 + `track_pose_node` | 목적지 이동/도착 정지 | ○ |

---

## 11. 오픈 이슈 / 리스크

1. **heading 추정 안정성** — 저속·정지 시 이동방향 부정확. 클러스터 주축 폴백 검증 필요.
2. **라이다→Thor DDS(WiFi)** — pose만 보내 경량이지만 지연/유실 시 도착판정 흔들림.
   워치독 + 최근 pose 홀드 필요.
3. **존 재측량 정확도** — homography 정합 오차가 도착판정에 직접 전파.
4. **조향 해상도 ±7** — 차선변경/direct 추종 궤적 거칠 수 있음. v_base·게인 재튜닝.
5. **LASA/estop과 자연어 정지의 관계** — "정지"를 estop(하드) vs speed_override 0(소프트)
   중 무엇으로? 안전상 estop 권장하나 재출발 흐름 정의 필요.
6. **VLA_AD lane follower 실트랙 품질** — 시뮬 대비 실제 차선 인식 성능이 lane-follow 존
   전체의 전제. 별도 검증 필요.

---

## 부록 — 참조 파일

- VLA_AD: `src/decision_making_pkg/{lasa_node,motion_planner_node,behavior_manager_node,path_planner_node}.py`,
  `src/interfaces_pkg/msg/{VlaIR,IrApplied,MotionCommand,LaneInfo}.msg`,
  `src/vision_language_action_pkg/.../vla_control_node.py`, `src/serial_communication_pkg/...`
- nav-vla: `src/nav_vla_pkg/nav_vla_pkg/{chat_gui_node,navigator_node,policy_node,speed_control,gz_pose}.py`,
  `config/zone_map.yaml`
- 라이다 정합: `0725_4점정합ver/{live_bev_intensity_viewer,track_map_align,pick_alignment_points}.py`,
  `track_map_aligned_homography.json`
