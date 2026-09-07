# 2026-08-31 21:48 — Reasoning 라벨 파이프라인 (마일스톤 1)

## 배경

SmolVLA가 "이미지를 보고 주행 근거를 말하는" reasoning VLA로 가려면 액션+언어
co-training이 필요하고, 그 전제가 **장면에 사실적으로 근거한 프레임별 reasoning
라벨**이다. 별도 VLM(Qwen)의 사후 해설은 정책의 근거가 아니므로(충실성 문제),
라벨은 시뮬 ground truth에서 규칙으로 뽑은 사실 골격 위에서만 만든다. 이번
마일스톤은 **라벨 파이프라인 구축·검수까지** — 학습(subclass co-training,
3h 예산)은 라벨 품질 확인 후 별도 결정.

## 만든 것

1. `src/nav_vla_pkg/scripts/label_reasoning.py` — packed 에피소드의
   `resampled_10hz.jsonl` + `track_paths.json`/`zone_map.yaml`에서 프레임별
   사실(곡률 클래스, 차선 arc-length 기준 goal 거리, 5프레임 중앙값 속도와
   추세, 턴, 지시 문맥)을 추출해 결정론 영어 골격 문장을 세그먼트 단위로
   합성. 출력은 에피소드별 `reasoning.jsonl` 사이드카(원본 불변).
2. `src/nav_vla_pkg/scripts/paraphrase_reasoning.py` — 골격당 3~5개 영어
   변형을 Ollama로 생성, **사실 보존 검증**(수치 일치, 존/차선/턴 단어 보존,
   추세 반전 금지, 환각 객체 블랙리스트) 통과분만 유지. 고유 문장당 1회만
   호출(캐시).
3. `tools/review_reasoning.py` — 중복률·장소별 문장 다양성·토큰 길이·사실
   커버리지 통계 + 무작위 세그먼트(이미지+facts+문장) HTML 검수 페이지.

## 결과 (v3y 본선 173 + heldout 40 = 213 에피소드)

- 3,928 세그먼트(18.4/ep), 고유 문장 2,968 → **중복률 24.4%** (목표 <30% 통과)
- 패러프레이즈 9,321개 유지 / 15,712 시도 (**40.7% 거부** — 검증기가 실제로
  일함: 추세 반전, 수치 변형, 새 객체 환각이 걸러진 사유)
- 골격 토큰 길이(SmolVLM2 토크나이저) p50=32, p90=40, max=56 → 다음 단계
  `reasoning_max_length=64` 예산 안
- goal arc 거리가 12.0→1.7 m 단조 감소로 success 종료와 일치(검증됨)
- 검수 페이지: `src/nav_vla_pkg/data_v3y/reasoning_review.html` (1.4 MB,
  무작위 60 세그먼트)

## 밟은 함정 (재발 방지)

- **존 `s_m`은 ring-center 파라미터**(index×0.35)라 lane 자체 arc 길이
  (lane1 loop 128.3 m ≠ ring 142.0 m)와 단위가 다름 → goal 거리가 11 m
  틀렸었다. 존의 **index**로 해당 lane 누적길이 배열을 조회해야 한다.
- `resampled_10hz.jsonl`의 `state[0]`(속도)은 프레임 간 0.5→4.7→1.9처럼
  튄다 → 5프레임 중앙값 없이 문장에 인용하면 안 됨. `action`은 꼬리 행에서
  null.
- **qwen3:4b는 이 Ollama 빌드에서 `think:false`도 `/no_think`도 무시** —
  호출당 90 s를 사고에 쓰고 content가 빈다. 패러프레이즈는
  qwen2.5:3b-instruct(비사고)로. 모델은 지시해도 줄번호("1. ")를 붙이므로
  숫자 검증 전에 접두 제거 필요.
- 티어 단어 "slow pace"는 감속 검증과 어휘 충돌 → "leisurely"로 변경.

## 다음 단계 (보류 중)

- 사용자 육안 검수(HTML) 통과 → co-training 진행 여부 결정.
- 학습 측 설계는 확정돼 있음: pip lerobot subclass(vendoring 없음), 독립
  언어 헤드(`lm_head` 초기화·unfreeze), `ckpt_v8g_60k` 시작 + 비전 동결로
  3h 예산, 서빙은 KV 캐시 복제 후 사이드 스레드 디코드, counterfactual
  프로브에 텍스트 발산·행동 일치 지표. 상세: plan 파일 및 대화 기록.
- 장애물 근거 설명은 v9 수집에서 obstacle snapshot을 라벨러 입력에 추가.
