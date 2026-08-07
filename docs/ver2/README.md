# docs/ver2 — 서브 세션(Thor) 변경 이력

메인 노트북에서 작성하는 [ver/](../ver/README.md)와 **작업 장소를 구분**하기
위한 폴더다. Jetson AGX Thor(`~/hong/`) 세션에서 만든 문서·변경 기록은
여기에 쌓는다. 파일명 규칙·필수 항목·append-only 원칙은
[ver/README.md](../ver/README.md)를 그대로 따른다.

## 계획 문서 (날짜 접두사 없음)

| 문서 | 요약 |
|---|---|
| [real_car_master_plan.md](real_car_master_plan.md) | v7 판정부터 Hesai→Thor 폐루프 데모까지 실행 계획 (D0~D7 갱신판, §4.6 맵·존 자산) |
| [navvla_vs_vlaad_comparison.md](navvla_vs_vlaad_comparison.md) | nav-vla vs VLA_AD 공통점·차이점·보완 관계, 2026-07-27 판정 정정 |

## 목록

| 날짜 | 문서 | 요약 |
|---|---|---|
| 2026-08-07 13:17 | [시뮬 데모 Thor 완주](20260807_1317_sim-demo-on-thor.md) | **회색 화면 진범 = GUI ogre1 (aarch64)** — 카메라 프레임 캡처로 서버측 무혐의 입증 후 ogre2 전환(런치가 아키텍처별 자동 선택, 노트북 무변경). 하드코딩 `~/ROS2_project` 8곳+는 심링크로 우회. **폐루프 주행 확인**: cmd_vel 8.1 Hz, 동시부하 지연 289~299 ms(+25%), underrun 1~3%, 내레이터 정상 |
| 2026-08-07 12:52 | [Thor 추론 실측 233 ms](20260807_1252_thor-inference-benchmark.md) | **P50 233 / P95 243 ms (4.3 Hz), 지터 ±7%** — 시뮬 4060 대비 ~5×, "커밋 파탄" 위험 기각, 청크 단축 여지 확보. `lerobot==0.4.4` 핀이 결정적(0.6.1은 hub 충돌), **`num2words` 누락 발견**→requirements 추가, 데모 프리플라이트에 gz 검사 추가. 잔여: Gazebo(sudo)·compile·동시부하 재측정 |
| 2026-08-07 11:03 | [thor_vehicle_pkg 삽입](20260807_1103_thor-vehicle-pkg-insertion.md) | **nav-vla 단독 실차 경로 완결** — 카메라·시리얼 이식 + `/cmd_vel→MotionCommand` 어댑터(게이트웨이 겸) 신설. 합성 검증 5종 PASS (δ=15.1°→steer 3, ±7 클램프, estop 0건, 워치독 20 Hz). 변환 파라미터는 전부 **D0 실측 전 자리표시**. 시뮬 파이프라인 실사용 패키지는 3개가 아니라 **5개**(선생 스택 포함) 확인 |
| 2026-08-07 10:44 | [트랙 링크 수신 준비 + 허사이 개조](20260807_1044_track-link-and-hesai-ingest.md) | 전송층을 **Fast DDS Discovery Server로 정정**(Thor에 rmw_fastrtps만 존재), mock/check 도구 + Thor 루프백 **9.2 Hz·갭 0.102 s PASS**. `lidar_perception_pkg`에 `hesai_ingest_node` 신설 — PointCloud2→`/scan` 변환으로 기존 2D 체인 무수정 재사용, 합성 검증(벽 2.00 m·바닥 누출 0). chrony는 스크립트만(sudo 대기) |
| 2026-08-06 23:41 | [주0 병행: 이식성·N점 정합·canonical 프레임](20260806_2341_week0-portability-npoint-canonical.md) | 절대경로 제거, 스튜디오 4→N점(4점 잔차=**구조적 0** 재확인, 8점 0.60 px), `/track/vehicle_pose_map` 신설(유도 스케일 3.041 mm/px → 템플릿 **15.90×10.93 m**, 로드맵 추정과 일치), 구역↔템플릿 왕복 무손실. VLA_AD: lasa 런치 추가·α점프 수정 + **install 심링크 26개가 hoon 원본을 가리키던 것 발견·재지향** |
