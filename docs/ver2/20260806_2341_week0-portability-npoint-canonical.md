# 2026-08-06 23:41 — 주 0 병행 작업: 경로 이식성, N점 정합, canonical 프레임, 구역 변환, VLA_AD LASA 배선

[real_car_master_plan.md](real_car_master_plan.md) §3의 "하드웨어 불필요"
항목 4건 + §4.6 결함 1·2의 구현. 전부 이 세션(Thor)에서 실행 검증.

## 1) 절대경로 파라미터화

`ring_map_probe.sh`(WS·venv를 스크립트 위치/env에서 유도),
`compare_segments.py`(`NAVVLA_WS` env 또는 `__file__` 기준),
`track_localizer.yaml`(상대경로) + `track_pose_node._resolve_config_path`
(share → 소스 config 순 해석). 검증: 이 체크아웃에서
`alignment/track2.png` 실해석 확인.

## 2) BEV Studio N점 정합 + 잔차 (§4.6 결함 1)

`findHomography(method=0)`은 원래 N점 최소자승 — GUI 캡(4)만 문제였다.
최소 4·최대 12점으로 확장, `alignment_residuals()` 신설, homography JSON에
`n_points`·`residuals_bev_px`·`residual_rms_m` 저장, 적용 시 상태바에 표시.

측정 (합성, 1 px 찍기 노이즈 주입):

| 조건 | 재투영 RMS |
|---|---|
| 8점 | **0.602 px ≈ 0.030 m** — 오차가 보인다 |
| 4점 | 7.3e-06 px — **구조적 0 재확인** (자기 오차 측정 불가) |

## 3) canonical 템플릿 프레임 (§4.6 결함 2)

`track_pose_node`에 `TemplateFrame` 추가: 저장된 `homography_bev_to_track`
으로 pose를 템플릿 미터 프레임(+y up)으로 변환해
`/track/vehicle_pose_map` 추가 발행. 스케일은 homography 국소 야코비안에서
자동 유도, `template_m_per_px` 파라미터로 줄자 실측값 대체 (D2 게이트).

실측 (0725 정합 데이터로):

- 유도 스케일 **3.041 mm/px** → 템플릿 = **15.90 × 10.93 m**.
  로드맵의 "~15×11 m 실내" 추정과 일치 — 체인 전체의 교차 검증
- 센서 1 m 이동 → 맵 0.9928 m (**0.72% 국소 스케일 오차** — 4점 정합의
  실제 품질. 줄자 검증이 이 값을 확정한다)
- yaw 변환 vs 이동 방향: 차이 0.000°

## 4) 구역 ↔ 템플릿 변환 (`zones_to_track_frame.py` 신설)

스튜디오 구역 편집기는 센서 프레임으로 저장 → 재설치마다 무효.
`to-assets`(센서 → `track_assets.yaml` 캐노니컬) / `to-sensor`(재설치 후
새 homography로 역투영). 왕복 검증: 합성 구역 2개 **최대 오차 0.00 mm**
(mm 반올림 하 무손실), 1.5 m 기준선 길이 1.498 m로 보존.

## 5) VLA_AD 결함 수정 (작업 사본 `~/hong/VLA_AD`)

- `pipeline_launch.py`에 `lasa_node` 추가 (누락으로 모든 A4 replay가
  사실상 A0였던 것 — `TASKS.md:3855`)
- `motion_planner_node.py`: `was_discarded=True`여도 LASA의 slew된 폴백
  값을 사용 (기존엔 무시하고 α=1.0 점프 — one-cycle nominal jump)
- **부수 발견: install 트리의 심링크 26개가 원본 `~/hoon/VLA_AD`를
  가리키고 있었다** (20:16 생성, 20:45 재빌드가 못 고침). hong에서
  `ros2 launch` 하면 hoon의 launch/config가 로드되는 상태. 26개 전부
  hong/build로 재지향, 잔여 0 확인. 실제 ROS Jazzy에서 런치 서술 파싱
  → 노드 10개(기존 9 + lasa) 확인

## 재현

```bash
cd ~/hong/nav-vla && python3 -m py_compile tools/eval/compare_segments.py \
  src/track_localizer_pkg/track_localizer_pkg/track_pose_node.py \
  tools/lidar_alignment_gui/{bev_pipeline,lidar_bev_studio,zones_to_track_frame}.py
# canonical 자기일관성 (스케일·yaw)
python3 -c "import sys;sys.path.insert(0,'src/track_localizer_pkg');\
from track_localizer_pkg.track_pose_node import TemplateFrame;\
from track_localizer_pkg.bev_detector import DetectorConfig;\
t=TemplateFrame('src/track_localizer_pkg/config/alignment/track_map_aligned_homography.json',DetectorConfig());\
print(t.m_per_px*1000, 5228*t.m_per_px, 3594*t.m_per_px)"
cd ~/hong/VLA_AD && find install -type l -lname '/home/autolab/hoon/*' | wc -l  # 0
```

## 미달·미해결

- **ROS 런타임 폐루프 검증 없음** — 노드 기동·토픽 발행은 라이다/시뮬
  연결 후. 스튜디오 GUI 변경도 headless라 `py_compile`만 (기동 1회 확인
  필요)
- 스케일 0.72% 오차의 원인(찍기 오차 vs 현수막 비강체)은 N점 재정합 +
  줄자 전까지 분리 불가
- VLA_AD launch의 `debug_visualizer`는 조건부라 `debug:=true`에서만 뜸 —
  lasa와의 rqt 확인은 실기에서
- 스튜디오 구역 편집기의 저장 프레임 자체를 template로 바꾸는 것은
  보류 — 변환 유틸로 충분한지 D2에서 판단
