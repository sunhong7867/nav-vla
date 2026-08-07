# 2026-08-07 12:52 — Thor 온디바이스 SmolVLA 추론 실측: P50 233 ms (4.3 Hz)

마스터플랜 D0 1번 항목(Thor 추론 P50/P95)의 SmolVLA 경로 측정.
시뮬 데모 준비(사용자 커밋 `b4144b2` 기반) 중 체크포인트가 도착해
Gazebo 없이 먼저 측정했다.

## 환경 구축 (사용자 setup 스크립트 + 보정 2건)

- venv `~/hong/venv/navvla` — **`lerobot==0.4.4` 핀**(사용자
  `requirements-smolvla-demo.txt`)이 결정적이었다. 최신 0.6.1은
  huggingface-hub 요구가 transformers와 충돌해 SmolVLA 임포트 불가
  (0.6.1: hub≥1.6 요구 vs transformers: hub<1.0 요구 — 직접 확인)
- 시스템 torch 2.9.0+CUDA 재사용(`--system-site-packages`) — 별도 다운 없음
- 보정 1: **`num2words` 누락** — lerobot 0.4.4 의존성 트리에 빠져 있어
  클린 venv에서 서버 기동 실패. requirements에 0.5.14 핀 추가
- 보정 2: `smolvla_demo.sh` 프리플라이트에 **gz 존재 검사** 추가
  (없으면 시뮬 기동 단계에서야 터지던 것을 사전 검출)
- venv 기본 경로 통일: `~/venv/navvla` → `~/hong/venv/navvla` 심링크

## 측정 (ckpt_v6_60k, 브리지와 동일 규격 요청 640×480 JPEG q88 + 상태 + 문장)

| 항목 | 값 |
|---|---|
| 첫 패스 (CUDA 정착) | 1033 ms |
| **P50 / P95** (40회, 앞 3회 제외) | **233 / 243 ms** |
| min / max | 215 / 248 ms — 지터 ±7% 이내 |
| 청크 | (30, 3) = 3 s 분량 |
| 환산 주기 | **4.29 Hz** |

## 판단

- **시뮬 노트북(RTX 4060, ~1.2 s/청크) 대비 약 5×** — Thor가 서빙 병목이
  아니라 여유다. 마스터플랜 위험표의 "실측이 4.6 s 쪽이면 커밋 파탄"
  시나리오 **기각** (단 4.6 s는 Qwen VlaIR 경로 수치 — 그쪽은 별도 측정)
- 2.1 s 커밋 대비 여유 9배 → **청크 단축·재추론 주기 단축 여지가 크다.**
  `generalization_limits` §4의 "청크 커밋의 구조적 긴장"(개루프 구간 편차)을
  실차에서 완화할 실물 근거. 단 과거 스윕에서 0.6 s 커밋은 차선 결정 붕괴
  전력이 있으므로 단축은 재학습과 함께 D5에서 실험
- 벤치 클라이언트는 세션 스크래치패드(`bench_client.py`) — 재사용 시
  저장소 이관 검토

## 미달·미해결

- **eager 기준** — `--compile` 미시도 (서버 자체 경고: 계획 예산 50~60 ms).
  컴파일로 더 내려갈 수 있으나 현 수치로도 충분해 후순위
- 합성 입력이라 카메라 파이프라인 지연(캡처→JPEG→ZMQ)은 미포함 —
  D3에서 end-to-end 재측정
- **GPU 동시부하 없음** — Gazebo+GUI 동시 구동(시뮬 데모) 및 Qwen VlaIR
  공존 시 재측정 필요
- Gazebo 설치 대기 (sudo) — 시뮬 데모의 마지막 전제
