# 2026-08-07 14:52 — 대시보드 재작업: VLA_AD 원본 충실 이식으로 교체

[14:45 문서](20260807_1445_driving-dashboard.md)의 미니멀 대시보드는 사용자
요구("VLA_AD GUI 양식 그대로, 좌측 하단만 라이다 맵")를 잘못 읽은 결과였다
— 폐기하고 `operator_gui_pkg/dashboard_node.py`(866줄, 선배 스택)를 충실
이식했다. 디자인 토큰·QSS·레이아웃·위젯 클래스·15 Hz 틱 구조 전부 원본
그대로, 교체·매핑만 다음과 같이:

| 원본 (VLA_AD) | 이식판 (nav-vla) |
|---|---|
| 좌하단 모션 라벨+채팅 | ★**트랙맵**(도안+차량점+궤적, 캐노니컬 프레임) + 모션 라벨 / 채팅 유지 |
| 카메라 3패널 (Raw·BEV·Lane+Path) | 그대로 — 선생 스택 병행 시 BEV/Lane도 살아남 |
| VlaIR 패널 | VLA Policy — instruction + `/vla/status` 지연/큐/underrun + cmd_vel |
| TTL Gate 롤링바 | **Pose Age Gate** — pose 나이 vs 0.5 s 유실 임계 (동일 시각화) |
| E2E Latency 롤링바 | Policy Latency — 기준선 300 ms (Thor 실측 233+여유) |
| BehaviorState 배지 | DRIVING / IDLE / POSE LOST / ESTOP |
| 채팅 → `vla/text_command` | → `/vla/instruction` 직발행 (학습 문장) |
| 퀵버튼 STOP/Resume/… | STOP=**estop+빈 문장 이중 정지**, Resume=estop 해제만(자동 재주행 금지), Slow/Normal/Fast=학습 문장 프리셋 |
| Lane Detection + Force 버튼 | 유지 (nav-vla `LaneInfo`, 시뮬 선생 스택에서 동작) |

검증: 모의 피드 실기동 + 스크린샷 육안 — 원본과 동일한 외관으로 트랙맵
차량점·궤적, Age Gate 바([VALID] 52 ms), Policy Latency 바(291 ms), Node
Health 8점, 퀵버튼 렌더 확인. cmd_vel 15 Hz 틱 vs 10 Hz 발행 깜빡임은
1 s 캐시로 수정.

## 미해결

- VlaIR·LoopTiming·lasa_stats 등 VLA_AD 전용 데이터는 미표시 — 비교 모드
  (VLA_AD 스택 병행)에서 원본 dashboard를 그대로 쓰면 된다
- 트랙맵의 존/기준선 오버레이는 track_assets.yaml 저작 후 추가 후보
