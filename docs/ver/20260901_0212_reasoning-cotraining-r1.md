# 2026-09-01 02:12 — Reasoning co-training r1 (마일스톤 2)

## 무엇을 했나

라벨 파이프라인(전 노트 20260831_2148) 위에서 SmolVLA에 언어 헤드를 붙여
액션+reasoning **co-training 1차 런**을 완주하고, heldout 프레임으로 접지
(grounding)를 정량 평가했다. 목표였던 "3시간 예산" 안에서 완료(2.66 h).

## 구조 (pip lerobot 0.4.4 subclass, fork 없음)

- `src/nav_vla_pkg/reasoning_vla/` — `ReasoningSmolVLAPolicy`:
  - 액션 경로는 스톡 그대로(상속·바이트 동일). reasoning은 `embed_prefix`
    출력을 공유하는 **두 번째 trunk 패스** `[image|lang|state|reasoning]`,
    토큰별 att-flag 1 = causal. 액션 expert는 reasoning을 못 봄(마스크 수술
    불필요 — 다른 패스라서).
  - 헤드는 체크포인트에 살아있던 frozen `lm_head`(47M) 복사 후 unfreeze.
  - 함정: cross-attn 레이어는 expert 스트림 None을 못 받음 →
    `fill_kv_cache=True`로 호출해 전 레이어를 self-attn 경로로 강제(캐시 폐기).
  - `generate_reasoning()` — naive 재전방 greedy (평가/데모용, ~2 s/60tok).
- `server/code/train_reasoning.py` — 커스텀 루프. 비전 인코더 동결, trunk
  lr 1e-5 / 헤드 5e-5, λ_CE=0.1, bf16 autocast. lerobot 프로세서 재사용
  (정규화·태스크 토크나이즈). 함정: config는 `PreTrainedConfig.from_pretrained`
  (draccus `type` 필드), FFmpeg4 박스라 `video_backend="pyav"`.
- 사이드카: `build_reasoning_sidecar.py` — lerobot 에피소드 인덱스로 조인,
  3,271 세그먼트. 학습 시 프레임마다 골격+패러프레이즈 중 무작위 1개 샘플
  (앵무새 방지 증강).

## 런 r1 (서버 3090, runs/navvla_reasoning_r1)

- 시작점 `navvla_smolvla_v3y/last`(40k, v3y 전용) → 25k steps, batch 8,
  0.38 s/step, **2.66 h**. 체크포인트 5k마다.
- CE 23.2 → ~1.1 (유니폼 10.8을 600 step에 돌파). **action loss 0.04~0.06
  전 구간 무회귀** — 독립 헤드 설계가 의도대로 작동.

## Heldout 평가 (40 ep에서 프레임 1장씩, 모순 감지 채점)

`server/code/eval_reasoning_decode.py` — jpeg→preproc→generate 풀 경로.

| 항목 | greedy | 의미 |
|---|---|---|
| 고유 문장 | 37/40 | 앵무새 아님 (프레임별로 다른 문장) |
| lane | 68% | 지시 차선 언급 + 반대 차선 부재 |
| speed | 69% | 인용 수치가 실측/계획/목표 속도 ±0.35 안 |
| zone | 38% | 근접 존 이름 언급 |
| trend | 28% | 가감속 주장 무모순 |

- **repetition penalty는 역효과**: 1.15에서 lane 68→18%, speed 69→47%.
  절 반복을 환각 다양성으로 바꿀 뿐. greedy가 기본값.
- 남은 병목: 절 반복 퇴화("...; slowing from 1.4 m/s" 연쇄)와 zone/trend
  시각 접지. lane/speed는 지시문·state에서 읽을 수 있는 사실이라 높고,
  zone/trend는 이미지를 읽어야 해서 낮다 — 접지의 실측 경계가 그대로 보임.

## 서빙·평가 통합 (구현 완료, 데모 미실행)

- `vla_policy_server.py --reasoning-every N [--reasoning-log f.jsonl]` —
  헤드 있는 체크포인트 자동 감지, N번째 요청마다 사이드 스레드 디코드,
  응답 `"reasoning"` 키(액션 경로 무지연, 구형 브리지 호환).
- `vla_bridge_node.py` → `/vla/reasoning` 재발행.
- `probe_policy_counterfactual.py` — reasoning 스트림 수집 + 쌍 간 텍스트
  발산(같은 문장 바닥 대비) `text_divergence`.

데모: 서버에서 `--checkpoint runs/navvla_reasoning_r1/checkpoints/last/`
`pretrained_model --reasoning-every 10` 으로 기존 터널 경로 그대로.

## 다음 레버 (r2 후보, 우선순위순)

1. 절 반복: 라벨을 한 문장으로 강제했으니 eos 학습이 약한 것 — CE 가중
   λ 0.1→0.3 또는 eos 토큰 가중, 그리고 steps 연장(CE 1.1은 아직 하강 중).
2. zone/trend 접지: v9 수집의 대조쌍(장애물 유/무, 같은 지점 다른 조건)이
   정공법. 라벨 facts에 이미 스키마 예약됨.
3. counterfactual 프로브 실측(시뮬 필요): 텍스트 발산 vs 바닥, 말한 행동
   vs 실제 궤적 일치 — 시뮬 스택 기동 시 `sim-stack-bringup` 절차 준수.
