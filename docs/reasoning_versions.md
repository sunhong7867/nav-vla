# Reasoning VLA 버전 대장 (코퍼스 · 데이터셋 · 학습 런 · 체크포인트)

> 2026-09-08 기준. reasoning co-training 계보(r1~r8)와 그 데이터의 단일
> 참조 문서. 각 결정의 상세 근거는 `docs/ver/`의 해당 노트 참조.
> 서버 = `autolab_sw@115.145.211.157:~/sunhong/nav-vla`.

## 1. 패킹 코퍼스 (라벨이 붙는 단위)

| 이름 | 위치(로컬 `src/nav_vla_pkg/`) | 에피소드 | 내용 | 비고 |
|---|---|---|---|---|
| packed_v3y | `data_v3y/packed_v3y` | 173 | ring_goal(111)+cruise(62), cf축 lane/speed/floor | 7월 수집, mm급 순도. **서버 사본에는 라벨 없는 옛 세션 167ep가 섞여 있음** — 합본 시 reasoning.jsonl 보유로 필터 |
| packed_v3y_heldout(+supp) | `data_v3y/packed_v3y_*heldout` | 22+18 | v3y heldout | 언어 평가 기본셋(40프레임) |
| packed_v9_pilot | `data_v9/packed_v9_pilot` | 34 | 장애물 3변형, **즉시 회피(구식)** | v1(내 차선)은 행동 모순으로 학습 제외, v0/v2만 합류 |
| packed_v9 | `data_v9/packed_v9` | 64 | 장애물 3변형, **관찰-후-회피** (감속→3.4m 범퍼갭 정지→2.5s 관찰→통과→복귀) | run6, 게이트 6cm 완화 회수분 포함 |
| packed_v9_heldout | `data_v9/packed_v9_heldout` | 9 | 관찰-후-회피 heldout | `--obstacle-frames` 평가용 |

라벨: 각 에피소드의 `reasoning.jsonl` (label_reasoning.py 골격 +
paraphrase_reasoning.py 변형). 속도는 티어만 인용(70/110/150), 거리는
범퍼 간격, phase = ahead/watching/passing/returning/beside.

## 2. LeRobot 데이터셋 (서버 `data/lerobot/`)

| 이름 | 구성 | ep/frames | state | 사이드카 | 만든 스크립트 |
|---|---|---|---|---|---|
| v3y | packed_v3y | 173 / 39.6k | [v, w, steer] (3) | reasoning_labels.jsonl (r4 시절 티어 라벨) | 7월 convert |
| v9r | v3y + v9 + pilot(v0/v2) | 262 / 66.5k | 3 | 5,566세그 (장애물 프레임 10.2%) | `code/convert_v9r.sh` |
| **v9r4** | v9r 구성과 동일 | 262 / 66.5k | **[v, w, steer, standstill_s] (4)** | v9r과 동일 라벨 | `code/convert_v9r4.sh` (`--standstill-state`) |

`standstill_s` = 정지 경과 초(5s 캡). GO 트리거(정지 2.5s 후 통과)와
"watching" 문장을 관찰 가능하게 만드는 채널. 구형(v6/v8g) 데이터셋은
이 문서 범위 밖(직행 내비 계보, v8g는 state 5 = +goal bearing/dist).

## 3. 학습 런 (서버 `runs/`, 전부 3090 · 25k steps · ~2.7h · λ_CE 0.2*)

| 런 | 데이터 | 라벨 속도 표기 | 특이 설정 | heldout 핵심 결과 | 상태 |
|---|---|---|---|---|---|
| r1 | v3y | m/s | λ_CE 0.1 | lane68/speed69†/trend28/zone38, 라이브 검증 | 보존(last) |
| r2 | v3y | raw 정밀 정수 | λ 0.2 | speed 26% — 숫자 모드 붕괴("110" 남발) | 보존 |
| r3 | v3y | raw 10단위 반올림 | λ 0.2 | speed 13% — "speed 10" 고정 | 보존 |
| r4 | v3y | **티어만(70/110/150)** | λ 0.2 | speed 72%(인용 시)/trend42 — **속도 표기 확정** | 보존 |
| r5 | v9r | 티어 | 자연 빈도 | **장애물 창 언급 7%** — 희소 신호 무시 | 보존 |
| r6 | v9r | 티어 | `--obstacle-boost 4` | **장애물 창 100%, 환각 0**, v3y 무붕괴. 라이브: 3.6m 자율 정지(충돌0), 통과 자가 개시 실패 | 보존 · 데모 후보 |
| r7 | v9r | 티어 | + phase 프레임 ×2 | greedy 무효 — T=0.7에서 phase 문장 등장 = **첫-토큰 "A car" 고착** 진단 | 보존 |
| **r8** | **v9r4** | 티어 | + standstill_s 채널 | (진행 예정 — GO 트리거·watching 결속 겨냥) | 준비 중 |

† r1의 speed 69%는 코퍼스가 거의 1.4 m/s라 부풀려진 수치.
\* 체크포인트는 평가 후 중간본 즉시 삭제, `checkpoints/last`만 보존
(디스크 98% 상시 — v8h_server_watch.sh 관례).

## 4. 서빙 요건 (체크포인트 ↔ 브리지/서버 궁합)

| 체크포인트 | state 차원 | 브리지 파라미터 | 비고 |
|---|---|---|---|
| navvla_smolvla_v3y, r1~r7 | 3 | (기본) | |
| v8g 계열 | 5 | `goal_conditioning:=true` | 직행 내비 계보 |
| **r8+** | 4 | `standstill_state:=true` | 브리지가 odom으로 카운터 실계산 |

- 서버는 state 차원을 체크포인트 normalizer 통계에서 자동 감지.
- reasoning 출력: `vla_policy_server.py --reasoning-every N
  [--reasoning-log f.jsonl]` → 응답 `"reasoning"` 키 → 브리지가
  `/vla/reasoning` 재발행. repetition penalty 금지(greedy 기본),
  탐색 실험용 온도는 평가기 `--temp`.

## 5. 평가 아티팩트 (서버 `logs/`)

- `reasoning_eval_r{1..7}*.jsonl` — heldout 채점(m/s·raw·티어 3표기 인식,
  모순 감지, obstacle 언급/환각 항목).
- `--obstacle-frames`: 장애물 창(전방 14 m) 프레임만 샘플 — v9 접지의
  실측은 반드시 이 모드로(무작위 샘플은 신호 희석: r5가 무작위론 26%,
  창 집중으론 7%였음).
- counterfactual 프로브: `eval_out/policy_cf_reasoning_r4.json`
  (텍스트 발산 2.0× 바닥).

## 6. 미해결 계보 (다음 결정 지점)

- r8 결과에 따라: watching 결속·GO 트리거 해소 여부 판정.
- 첫-토큰 고착의 라벨측 해법(모든 obstacle 문장 "A car X m ahead;"로
  시작 통일) — r8이 부족하면 r9 후보.
- zone 접지(31~38% 정체), 거리 수치 garble("A car 110, 1.3 m"),
  티어 단어-숫자 불일치("leisurely tier of 150") — 라벨 어휘 분리 검토.
- 움직이는 장애물 변형: 현재 불가(모델에 구동 플러그인 없음) — 별도 작업.
