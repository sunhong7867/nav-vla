# 2026-07-31 14:53 — 시연자 교체: route_oracle → 욜로 스택 + navigator (사용자 지시)

> 앞선 문서: [12:26 120K 최종 + raw 속도 전환](20260731_1226_120k-final-and-raw-speed.md)

## 1. 발단 — "오라클이 차선을 밟고 달린다" (사용자 관찰)

route_oracle의 차선 경로는 [extract_track_paths.py](../../src/nav_vla_pkg/scripts/extract_track_paths.py)가
도로 경계의 기하 중심선을 **±도로폭/4로 offset**해 만든 것이지, 실제 차선 표시
기준이 아니다. 욜로 스택을 실주행시켜 측정한 결과 (양 차선 2,340 이동 포즈):

- lane1: 욜로 실주행 중앙값 **+1.70 m** (IQR +1.20~+2.02) vs 저장 경로 +1.58 m
  → 저장 경로가 **~0.45 m 중앙선 쪽**에 있고, 코너 컷팅까지 겹치면 점선을 밟는다
- 결론: **v21b 모델의 "점선 끼고 주행"은 혼합 실패가 아니라 이 선생을 충실히
  배운 결과**다. 학습으로 고칠 수 없는 종류의 결함(데이터 결).

**오늘 오전 수집한 v3 링/순항 데이터(잘못된 경로 선생) 전량 학습 제외.**
어휘 확대·raw 티어·순항 intent·hold 게이트 등 인프라는 전부 유지.

## 2. 교체 (사용자: "욜로랑 맵 좌표 기반으로, 몇 달 맞춰둔 것 재사용")

시연 스택 = 사용자의 데모 구성 그대로:

```bash
ros2 launch simulation_pkg driving_sim.launch.py   # 기본값에 욜로 파이프라인 포함
ros2 run nav_vla_pkg navigator_node                 # 존 정차 감독 (튜닝된 stop_offset)
ros2 run nav_vla_pkg episode_recorder_node          # 기록 (운전자 무관)
python3 .../collect_corpus.py --driver yolo ...     # 채팅 대신 반사실 그룹 오케스트레이션
```

collect_corpus에 `--driver yolo` 추가 — 채팅 GUI와 **동일한 공개 인터페이스**로 명령:
- 존 주행: `/nav_goal` `{"zone": "T2", "lane": "lane1"}` → navigator가 튜닝된
  정차 로직 수행, `/nav_status` "arrived:"로 성공 판정 (기존 루프 그대로 매칭)
- 순항: `/lane_mode_command` + 지속시간 종료 (nav_goal 없음)
- 속도: `/speed_command` (Int32 raw **상한** — motion_planner speed_limit_raw)

## 3. 속도 티어 재조정: raw {70, 110, 150}

speed_command는 상한 클램프라서 내부 목표(직선 150)를 넘길 수 없다. 200으로
넘기려면 target_speed_raw 기동 파라미터를 바꿔야 하는데 이는 **튜닝된 포락선을
벗어나므로** 보류 (사용자와 결정할 것). 150 = 2.94 m/s = 사용자가 "잘 달린다"고
한 그 속도이고 v2 fast(1.8)의 1.63배.

실측: 직선 raw 150(2.94 m/s), 코너에서 140→100 모듈레이션 — 사용자가 기억한
"140"은 이 코너 순항 대역과 정확히 일치.

## 4. 언어 아키텍처 (사용자 문제 제기 → 하이브리드 합의안)

"명령 자체를 학습시키면 새 명령을 못 하지 않나" — 맞고, A3(미학습 표현 0/8)이
그 증거다. 대응: **데모 입력단에 기존 Qwen 파서(chat_gui)를 유지**해 임의 문장을
정규 intent로 변환 → 정규 문장을 VLA에 입력. 새 표현 일반화는 LLM이, 카메라
제어는 학습 정책이 담당. VLA 원문장 일반화는 한계/ablation으로 정직하게 보고.

## 5. 미검증/다음

- `--driver yolo` 스모크 미실행 (심 내려둔 상태 — 사용자 지시). 검증 항목:
  리셋 직후 욜로 파이프라인 회복, navigator 정차 품질, 순항 종료 레이스
- lane2 측정치는 전환/S-커브 구간이 섞여 통계가 흐림 (중앙값 −1.46, IQR −3.71~+2.15)
  — 재수집 검증 때 차선별 정착 구간만으로 재측정할 것
- 서버: 링 전용 v3r 변환은 중단해둠 (재수집 후 병합 변환 1회로)
