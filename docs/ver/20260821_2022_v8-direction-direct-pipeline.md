# 2026-08-21 20:22 — 코퍼스 v8 착수: 방향(시계) + 최단거리(직행) 축, 파일럿 통과, 야간 수집 개시

사용자 결정: "차선 완전 무접촉은 장기전이니, **목표지점 최단거리**와
**반대방향 주행**을 되게 하자" — 둘 다 서빙으로 불가능한(정책이 새 행동을
배워야 하는) 능력이라 코퍼스 확장 + 재학습으로 진행. 축소 차(0.85×) 스케일로
수집하므로 보닛 편향도 함께 해소된다.

## 방향 축의 배경 (같은 날 실증)

차량을 반대로 두면 반시계 학습 편향대로 좌조향해 이탈 — 코퍼스가 CCW 단일
방향이라 예상된 일반화 한계. v8이 이 축을 분포 안으로 넣는다.

## 구현 (전부 파일럿으로 검증)

### 1. 시계방향 순항 — `--cw-cruise-groups`
- **선생의 차선 추종은 시각 기반이라 방향 무관**: 시작 요(yaw)만 π 반전하면
  시계방향 시연이 그대로 나온다 (드라이버 수정 0)
- 문장: 신규 `cruise_cw` 템플릿 6종 + 방향 어휘(`direction.cw`: "going
  clockwise" 등 4+2종). 차선 단어는 **기하 전용**(`lane_geo`: inner/outer만) —
  "the left lane"은 시계방향에서 의미가 뒤집히므로 배제
- **파일럿 4/4 통과**: 전부 CW 주행, 명령 차선↔실주행 차선 일치(72~100% 점유).
  lane_mode의 차선 의미도 기하적(안/바깥)이라 뒤집히지 않음을 확인

### 2. 최단거리 직행 — `--direct-groups`
- 내비게이터의 direct 모드(존 좌표로 직선 pure pursuit, `direct arrived:`)를
  수집기에 연결: `/direct_nav_goal` 발행 + off-ring 가드 면제(인필드 횡단이
  정의상 정상) + 같은 출발점에서 서로 다른 존 2개 = 반사실 쌍(cf_axis=zone)
- 문장: 신규 `direct_zone` 템플릿 6+2종 ("Take the shortest path to {zone}."
  등). **속도 슬롯 없음** — direct 모드는 자체 속도 프로파일이라 속도 단어는
  가르칠 수 없는 거짓이 된다
- **파일럿 1차 전멸(stuck) → 원인 발견**: 모션 플래너의 `/speed_command`는
  direct override까지 클램프하는 **전역 상한**인데, 수집기 `_halt_driver`가
  0으로 닫아둔 채였다. 라이브 추적(`/direct_motion_command` 35 →
  `/topic_control_signal` 0)으로 확정, direct 디스패치에서 상한 재개방(150)
  후 **재파일럿 4/4 성공**, 도착 오차 0.46~0.48 m (tol 0.6)
- 경로/직선비 1.6~3.0: 출발 헤딩이 목표 반대면 유턴 기동 포함 — 링 우회
  (~138 m) 대비 충분히 "최단"이며 시연 의미 정확

## 야간 본 수집 (진행 중, tools/collect_v8_overnight.sh)

| 유형 | 그룹×변형 | eps |
|---|---|---|
| CCW 순항 (스케일 리프레시) | 30×2 | 60 |
| **CW 순항 (신규)** | 30×2 | 60 |
| 존 이동 | 20×2 | 40 |
| **직행 (신규)** | 25×2 | 50 |
| 노이즈 바닥 | 6×2 | 12 |
| 합계 | | **222** |

수집(~2.5 h) 후 finalize(verify→10 Hz→CMI→package, PACK_OUT=data_v8/packed_v8)
자동. 내일: 서버 반입 → `lerobot/v8` 변환 → `navvla_smolvla_v8` 60K 학습(8.6 h)
→ 평가(신규: 방향 프로브, 직행 도착률 / 회귀: 링 프로브 무접촉 유지).

## 같은 날 잡은 운영 버그 3건 (데모 "갑자기 차선 밟음" 규명 과정)

1. **이중 vla_bridge**: pkill 비동기 → 이전 세션 브리지 생존 → 컨트롤러 2개가
   /cmd_vel 교대 점유. 처방: 4개 런처 전부 kill 후 reap-wait 루프
2. **GUI qwen 모델 로딩 GPU 경합**: 첫 명령 시점에 ~2.6 GB VRAM 로딩 →
   정책 지연 287→460 ms, 언더런 24% → 청크 앞부분만 실행 → 차선 결정 붕괴
   (outer 명령에 lane1 안착). 처방: GUI 시작 시 워밍업 + keep_alive=-1 상주
3. **GUI 기본 fast 문장**: `_vla_speed_raw` 150 = fast — normal(110)로 변경.
   fast 계획 + 상한 실행의 불일치는 차선 유지를 무너뜨림 (fast 프로브는
   관성 크리프로 리셋 검증 실패 2회 — 그 자체가 불안정 방증)

## 재현

```bash
bash tools/collect_v8_overnight.sh                # 수집+finalize 전체
# 파일럿만: collect_corpus.py --driver yolo --cw-cruise-groups 2 (또는 --direct-groups 2)
```
