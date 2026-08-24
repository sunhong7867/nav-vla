# 2026-08-24 11:38 — 직행(최단거리) 내비: 순수 VLA vs 좌표 위임, 결정 경로

배경: v8g 축 프로브(20260824_1106)에서 목표 채널은 54× 반응하지만 도달은
실패. 사용자 질문: "이미지만 보는 순수 VLA로 최단거리가 가능한가? 언어만
VLA에 맡기고 이동은 좌표 기반으로 해야 하나?"

## 판단

1. **이미지-only VLA 최단거리는 이 트랙에서 원리적으로 불가.**
   트랙이 시각적으로 균질(T2/T3/M2/M3 구분 불가) → 자기위치 추정 불가.
   2026-06-24에 실험으로 확정된 결론(카메라→존 내비 배회, 피벗 기록).

2. **v8g는 이미 순수가 아니고, 그 구도가 맞다.** 상태에 goal_bearing/
   goal_dist(2D 목표 벡터)를 넣는 목표조건화 BC — 이론상 이 둘만으로
   조준-직진 가능(오라클 직행 모드가 그 증거). 실차에서도 Hesai LiDAR
   외부 로컬라이제이션으로 같은 채널 공급 가능 → 시뮬 전용 치트 아님.
   실패 원인 후보는 구도 자체가 아니라 (a) bearing 부호/좌표계가 학습
   라벨과 서빙에서 불일치, (b) 2D 목표 벡터가 시각 입력에 묻힘.

3. **시스템(데모)은 하이브리드가 정답, 연구는 v8g 라인 유지.**
   - 데모: 언어 이해 = VLA/LLM, 최단거리 이동 = 좌표 내비게이터(완성돼
     있음), 차선 주행·반응형(장애물/신호등) = VLA. zero lane-touch 때의
     서빙 계층 조합 철학.
   - 연구: "목표 벡터 조건 단일 정책이 문장으로 순항/직행을 전환"이
     성립하면 그 자체가 기여. 버리려면 증거를 갖추고 버린다.

## 결정된 진행 경로 (2026-08-24, 사용자 확정)

1. **① bearing 부호/좌표계 검증** — 브리지 서빙 계산(`_goal_state_dims`)
   vs 학습 라벨 정의(수집/변환 파이프라인) 대조. 반나절 이하, 진행 중.
2. **② ①로도 실패 지속 시**: 직행은 좌표 내비에 위임, v8g는
   순항+반응형 전담으로 역할 확정.

관련: 직행 저속(0.7 m/s 평형점) 문제와 직행 템플릿 속도 절 부재는
20260824_1106 참조 — ①과 별개의 데이터 이슈.

## ① 검증 결과 (2026-08-24 11:50): 부호/좌표계 버그 없음 — 통과

4중 대조 전부 일치:

1. **존 좌표 파일**: 로컬 config/track_paths.json == 서버 code/track_paths.json
   (md5 동일).
2. **공식**: 학습(to_lerobot) `b = atan2(g−p) − heading`, heading =
   `yaw_u + (−90°)`(resample_episodes, v8 체인 `--yaw-offset-deg -90
   --pose-source tf`) vs 서빙(브리지) `b = atan2(g−p) − (yaw + (−90°))`
   — 동일 구조, 동일 오프셋.
3. **학습 라벨 실측 건전성**: packed_v8 직행 8 에피소드에서 bearing이
   출발 시 크고(0.1~2.6) 접근 중 |b| 중앙값 0.04~0.08 rad로 수렴,
   d→0.5 m. "목표를 향해 달릴 때 bearing≈0" 성립 — 라벨 정의 자체는 옳다.
4. **서빙 헤딩 관례 실측**: 프로브 리셋 raw yaw ±1.571에 대해 기록된
   실주행 초기 방향(동/서)이 `raw + (−90°)` 모션 헤딩과 3방향 모두 일치.

**부수 발견 (실패의 유력 원인, 데이터 조성 문제):** v8g 변환은
`ring_goal`(차선 따라 존까지 주행, 34 eps)에도 goal 라벨을 붙였다.
ring_goal에서는 목표 벡터가 조향 방향과 무관(차가 차선을 따라 돌므로
bearing이 옆·뒤를 가리키는 구간이 대부분) → "bearing을 향해 조향"을
가르치는 직행 41 eps와 정면 모순되는 34 eps가 섞여 그라운딩을 희석.
재시도한다면: goal 라벨을 direct_zone에만 붙이거나 mode 플래그 차원 추가.

### ① 4중 대조 재현 명령

```bash
# 1. 존 좌표 파일 동일성 (로컬 vs 학습 서버 사본)
md5sum src/nav_vla_pkg/config/track_paths.json
# 서버 쪽: ssh <학습서버> md5sum ~/nav-vla/code/track_paths.json  (서버 목록: docs/servers.md)

# 2. 공식 대조 — 학습 라벨 vs 서빙 계산이 같은 구조·오프셋인지 눈으로 확인
grep -n 'atan2' src/nav_vla_pkg/scripts/to_lerobot.py            # 라벨: b = atan2(g-p) - heading (L183 부근)
sed -n '414,435p' src/nav_vla_pkg/nav_vla_pkg/vla_bridge_node.py  # 서빙: _goal_state_dims, yaw + yaw_to_heading(-90°)

# 3. 학습 라벨 실측 건전성 — packed_v8 직행 에피소드에서 접근 중 |bearing|→0, dist→0.5 수렴 확인
#    (packed_v8은 학습 서버에 있음. observation.state[3]=bearing, [4]=dist)
python3 - <<'EOF'
import pandas as pd, glob, numpy as np
for f in sorted(glob.glob('data_v8/packed_v8/**/episode_*.parquet', recursive=True))[:8]:
    s = np.stack(pd.read_parquet(f)['observation.state'])
    print(f.split('/')[-1], 'b0=%.2f' % s[0,3], '|b|med(후반)=%.3f' % np.median(abs(s[len(s)//2:,3])), 'd_end=%.2f' % s[-1,4])
EOF

# 4. 서빙 헤딩 관례 실측 — 프로브 맵 JSON에서 리셋 raw yaw(±1.571)와 초기 진행 방향(동/서) 대조
python3 -c "import json; d=json.load(open('eval_out/v8g_direct_map.json')); print(d.keys())"
```

## ② 발동 (경로 확정)

①에서 버그가 없었으므로 합의대로: **직행은 좌표 내비게이터에 위임,
v8g는 순항+반응형 전담.** 연구 쪽 재도전은 위 ring_goal 라벨 분리
아이디어가 1순위 후보(변환+재학습 1회로 검증 가능)로 남겨둔다.
