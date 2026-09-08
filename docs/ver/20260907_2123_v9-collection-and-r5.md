# 2026-09-07 21:23 — v9 수집 완료 + r5/r6 (obstacle 접지)

## 수집 여정 (run1~run6, 함정 6개)

관찰-후-회피 설계(사용자 결정: 앞차가 멈춘 걸 모델은 모른다 → 감속·정지
2.5 s 관찰 후 차선 변경)로 v9 대조쌍 수집을 완료하기까지:

1. **V9_GROUPS 환경변수 유입**으로 4그룹이 1000그룹으로(run1) — 스크립트
   하드코딩으로 재발 방지. (부산물 96ep은 구식 즉시회피라 v1 제외 후 활용)
2. **spawn/despawn 반복이 gz create 서비스를 wedging**(run3, 2 h 행) —
   해치백 4색 상주 스폰 + set_pose teleport로 전환.
3. **정지 마진이 중심거리 기준**이라 차체(합 4.6 m)가 겹침 = 충돌(run4,
   사용자 목격) — `OB_STOP_MARGIN_M=8`(범퍼 ~3.4 m) + 충돌 가드 추가.
4. **500 Hz 클록 실험 역효과**(run5): CPU 경합으로 tf에 2 s 구멍(66/120
   손실). 100 Hz 복귀.
5. **레코더 취약점 2개 수정**: dynamic_pose 배열 인덱스 추적 → 근접 매칭
   (장애물 상주로 배열 순서가 흔들림), 노드클록 스탬프 → 브리지 헤더 스탬프.
6. 그래도 남는 gz 발행 버스트 드랍(양 스트림 공통, 오늘 머신 상태) —
   리샘플 보간 게이트를 v9 한정 `--max-interp-err-m 0.06`으로 완화
   (finalize `RESAMPLE_EXTRA` 관통). v3y급 코퍼스는 기본 2 cm 유지.

최종 v9: train 64 + heldout 9 + 파일럿 v0/v2 25 = **153ep** (전 에피소드
watch-then-avoid 사이클 완주, 충돌 0). 라벨: "A car 8.4 m ahead in our
outer lane; slowing…" → "Stopped 4.2 m behind the car ahead, watching
whether it moves." → "The car ahead has not moved — treating it as parked
and passing in the inner lane." (거리는 범퍼 간격 기준.)

## v9r 데이터셋 + r5

- v9r = v3y(173) + v9(64) + 파일럿 v0/v2(25) = 262ep, 66.5k 프레임,
  사이드카 5,566세그(장애물 프레임 10.2%). 서버 `data/lerobot/v9r`.
  주의: 서버 packed_v3y엔 라벨 없는 옛 세션 167ep가 섞여 있어
  **reasoning.jsonl 보유를 멤버십 조건**으로 스테이징(convert_v9r.sh).
- r5 (25k, λ_CE 0.2): action loss 0.06~0.08 무회귀(회피 액션 학습 포함).
  v3y heldout: lane 72/speed 54/trend 35/zone 31 — r4와 등락 혼재.
  obstacle-無 프레임에서 차량 환각 0 (obstacle 항목 100%).
- **핵심 실패**: 장애물 창(전방 14 m) 집중 평가에서 차량 언급 **7%(2/28)**
  — 라벨은 학습에 포함됐으나(검증됨) 자연 빈도(10%)로는 모델이 차를
  무시하는 게 CE상 저렴했다.

## r6 결과 (obstacle-boost 4배, 25k, 2.66 h)

- **장애물 창 언급률 7% → 100%(28/28)**, 차 없는 v3y 프레임 환각 0(40/40)
  — 가중 샘플링이 검출 자체를 해결. v3y 4항목은 등락 혼재(붕괴 없음:
  lane 65/speed 64/trend 38/zone 35).
- 남은 거친 부분(다음 사다리 칸): ① 내/반대 차선 관계 혼동(rel=ours에
  beside 템플릿), ② watching 프레임(v=0)에서 정지 문장 미출력,
  ③ 거리 수치가 티어 110과 얽힌 garble("A car 110, 1.3 m").
- 서빙 후보: `runs/navvla_reasoning_r6/checkpoints/last` — 회피 액션 +
  장애물 인지 발화 동시 탑재. 데모는 `--reasoning-every` 경로 그대로.
- 다음 레버 후보: phase별 세분 부스트(watching/passing만 추가 가중),
  거리 수치를 5 m 버킷 단어로("a few meters ahead"), v9 데이터 증량.

## 기타

- 로컬 `~/venv/navvla` 소실(사용자 정리 추정) — 패러프레이즈는 시스템
  python 폴백(transformers 없이 토큰 캡 근사)으로 동작, 서빙은 서버 venv.
- pkill 자기매치를 원격 ssh 명령에서도 밟음(동일 명령 뒤쪽의 스크립트
  경로 문자열) — 원격도 킬과 재기동을 별도 ssh로 분리할 것.
- 장시간 원격 작업은 ssh 직결 금지(Broken pipe로 동사) — setsid nohup
  분리 후 로그 폴링.
