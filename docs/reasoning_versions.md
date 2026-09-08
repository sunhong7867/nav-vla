# nav-vla 버전 대장 (코퍼스 · 데이터셋 · 학습 런 · 체크포인트, v0~v9 + r1~r8)

> 2026-09-08 기준 전체 계보의 단일 참조 문서. 상세 근거는 `docs/ver/`의
> 해당 노트. 서버 = `autolab_sw@115.145.211.157:~/sunhong/nav-vla`.

## 0. 번호 체계 읽는 법 (혼동의 근원)

- **코퍼스 번호 = 레코더/수집 세대, 런 이름 = 데이터셋 변형.** v2 코퍼스
  하나에서 런 5개(v2/v21/v2b/v21b/v2c — hold-frames·데이터 슬라이스만
  다름), v3y+supp에서 런 3개(v4/v4b/v4c — step 수만 다름), packed_v8에서
  런 3개(v8/v8g/v8h — goal 라벨 변환만 다름)가 나왔다. 1:1 대응은
  v5·v6·v7뿐.
- **"v1"은 존재한 적 없다.** 레코더가 v2로 재작성되며(`ver/20260728_1518`)
  번호가 2에서 시작. v9 그룹 내부의 변형 표기 v0/v1/v2(장애물 없음/내
  차선/반대 차선)는 코퍼스 버전과 무관하니 혼동 주의.
- **y-접두 시리즈(y5j·y6o·y7d·y8a)는 버전이 아니라 수집 증분 관례**:
  `collect_corpus.py --group-prefix`로 찍혀 `packed_y*`로 패킹된 뒤 다음
  숫자 코퍼스에 합류한다(y8a는 자체 팩 없이 packed_v8로 직행).

## 0.5 전체 연대기 (v0 → v9)

| 세대 | 시기 | 무엇 | 왜/결과 | 현재 위치·상태 |
|---|---|---|---|---|
| v0 (stage_a) | 7월 초 | data_engine 619ep, 언어 신호 0, zone ID+상대좌표 CNN | 실차 이식 불가(전역좌표) → 폐기, git에서 삭제 | 서버 `archive_june_stage_a` |
| — | 07-27~29 | 레코더 v2·리샘플러·route_oracle·리셋 결정론·CMI·서빙 브리지 구축 | 코퍼스 없는 기반기 | — |
| **v2** | 07-29~31 | oracle 시연 264ep(cf쌍 329) — 언어가 조향을 바꾸는 첫 코퍼스 | 런 5종 파생. v2=첫 폐루프(단, 정지 못함), **v2b=hold-15 패딩으로 4베이 대각 8/8**, v21 중도폐기, v21b=첫 링 주행이나 주차 회귀·120k에도 미회복, v2c=anti-creep 실패(폐기) | `packed_v2(_parking)`, raw는 `raw_30hz` |
| A3 어블레이션 | 07-30 | 학습 표현 8/8 vs 미학습 표현 0/8 | 표면형 암기 판명 → v3 어휘 확장 동기 | `ver/20260730_2131` |
| **v3** | 07-31 오전 | 주행-우선 재편, raw 티어 70/150/200, cruise 인텐트 신설 | **당일 전량 폐기** — oracle 선생이 차선 위를 달림(경로가 중앙선 쪽 0.45m 치우침). 티어도 70/110/150으로 재절단(/speed_command는 상한 클램프) | `packed_v3`·`raw_v3_30hz` 보존·학습 금지, `lerobot/v3r`은 변환 중단 |
| **v3y** | 07-31~08-03 | 선생을 YOLO 스택으로 교체, 173ep(+heldout 22) | 카메라 토픽 불일치·유령 /cmd_vel 수정 후 차선중심 0.19-0.26m, cf 20-27×, **장수 베이스라인**. 오늘까지 v9r 합본의 뼈대 | `packed_v3y` (서버 사본에 라벨 없는 옛 167ep 혼입 주의) |
| v3y_supp → v4/v4b/v4c | 08-04 | supp 167ep 합본 재학습 3종(60k/40k/80k) | **전멸** — step↑=이탈↑ 단조. 원인: 링을 이탈하고도 success로 찍힌 cruise 16ep(2%)의 오염. 대책=수집기 off-ring 가드 | v4 라인 폐기, `lerobot/v4` 없음(런만 존재) |
| y5j | 08-04 | 정션 통과 강제 48ep | v5 재료. 주행판 A3 재확인(+2.51 vs +0.36m) | `packed_y5j` |
| **v5** | 08-05 | v3y+supp+y5j−오염16 = 372ep, 60k | 이탈 0/8 — 첫 "클린" 코퍼스 | `packed_v5` |
| y6o → **v6** | 08-05~06 | lane2 재균형 34ep 합류=406ep, 60k | 이탈 0/8·최저 최대이탈, 그러나 동분포 추가데이터 한계효용 0 판명. **데모 확정 모델** | `packed_y6o/v6`, 로컬 `models/ckpt_v6_60k` |
| y7d → v7 | 08-06~07 | DAgger 사이클1: v6 오류상태 44포즈 회복시연 | 표적 구간 개선, 전반 중심유지 0.1-0.2m 회귀 → **데모는 v6 유지**, 루프 자체는 검증 | `packed_y7d/v7` |
| 서빙의 날 | 08-07 | 모델 동결, preview 트래킹 모드 | **차선 접촉 0 달성**(2/2 재현). 고속은 코퍼스 문제로 결론(raw-250 티어 미수집) | `ver/20260807_1517` |
| (공백) | 08-08~20 | 대회 임무 | 수집·학습 없음 | — |
| **v8** | 08-21~22 | 축소 스케일, CCW/CW 순항+존+직행 158ep(y8a 야간수집, 1차분은 use_sim_time 누락으로 전량 재수집) | CW 주행 성공, **직행 4/4 실패** | `packed_v8`·`raw_v8_30hz`, `models/ckpt_v8_60k` |
| v8g | 08-24 | 재변환만: state 5D(+goal bearing/dist) | 근거리 cf 54×(goal 채널 도달 증명)·원거리 붕괴 → **직행은 좌표 내비로 위임 결정**. ring_goal에도 goal 라벨을 붙인 변환 모순이 유력 진범 | `lerobot/v8g`, `models/ckpt_v8g_60k` (r1의 초기화 출처) |
| v8h | 08-24 | 재변환: goal 라벨을 direct_zone만 | **결과 미기록(공백)** — 스크립트·데이터셋·런만 존재, 08-28 프리즈/회피 피벗에 묻힌 듯 | `lerobot/v8h`, `runs/..._v8h` |
| **v9** | 09-04~07 | 장애물 대조쌍(관찰-후-회피) 153ep — §1 상세 | reasoning 접지용. 수집 함정 6개는 `ver/20260907_2123` | `data_v9/packed_v9*` |

미확인 아티팩트: `packed_smoke`(07-29 파이프라인 스모크 3ep, 학습 안
함), `packed_eval`(출처 불명), `packed_v3_heldout`(v3 폐기분인지 v3y
heldout인지 모호), `packed_v3y_supp`는 수집 스크립트명이 `y4_supp.sh`로
한 세대 어긋남.

---

# Reasoning 계열 상세 (2026-08-31 이후)

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
| **r8** | **v9r4** | 티어 | + standstill_s 채널 | **언어 절반 성공**: greedy에서 phase 문장 등장(7/28), v=0 프레임이 "Stopped/has not moved" 발화 — 채널이 phase를 결속. 언급률 100→79, 숫자 garble 잔존. **라이브: 액션 GO 여전히 미개시**(3.6m 정지 유지) — GO 전이 프레임 희소 + still=5 캡 구간 외삽이 유력 | 보존 |

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

## 5.5 r8 학습-저장 함정 (재발 방지)

train_reasoning.py의 데이터셋-통계 정규화 오버라이드는 메모리에만 있었고
체크포인트 저장은 베이스(3차원) 프로세서 파일을 복사했다 → r8 서빙/평가가
차원 불일치로 즉사. 수정: save()가 라이브 preproc를 `save_pretrained`로
저장(2026-09-08). 기존 r8 체크포인트는 `code/fix_r8_processor.py`로 교정.

## 6. 미해결 계보 (다음 결정 지점)

- **GO 전이(액션) 미해결 — r9 후보 레버**: ① GO 전이 프레임(정지→출발
  ~10프레임/ep) 전용 초고배 부스트, ② watch 시간 고정 2.5s→still 캡을
  3.5로 낮춰 외삽 구간 제거, ③ 시연 단순화(감속-즉시-회피 재수집),
  ④ 데모 실용해 = 감독(chat_gui) GO 외부화 + r8 내레이션.
- 언어 phase 결속은 r8로 입증(정지 프레임 "Stopped" 발화) — 남은 건
  live 캡 구간(still=5)에서의 템플릿 회귀와 숫자 garble.
- 첫-토큰 고착의 라벨측 해법(모든 obstacle 문장 "A car X m ahead;"로
  시작 통일) — r8이 부족하면 r9 후보.
- zone 접지(31~38% 정체), 거리 수치 garble("A car 110, 1.3 m"),
  티어 단어-숫자 불일치("leisurely tier of 150") — 라벨 어휘 분리 검토.
- 움직이는 장애물 변형: 현재 불가(모델에 구동 플러그인 없음) — 별도 작업.
