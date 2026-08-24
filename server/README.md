# server/ — 학습 서버(~/sunhong/nav-vla) 스냅샷

랩 서버의 학습용 코드·스크립트 사본. 서버 목록·접속은 docs/servers.md 참조.

## 여기 있는 것
- `code/` — 서버에서 실행되는 것들: `train_smolvla.sh`(학습 진입점, GPU/배치/저장주기 env),
  `to_lerobot.py`·`vla_policy_server.py`(배포 사본 — **원본은 `src/nav_vla_pkg/scripts/`**,
  체인이 scp로 밀어넣음), `track_paths.json`(존 좌표), `v8h_server_watch.sh`(체크포인트 실시간 프루닝),
  `ckpt_probe.py`
- `*.sh` — 과거 변환/학습/재개 체인 기록 (v2~v21 시절; 현행 체인은 `tools/`)

## 버전 관리 안 하는 것 (서버에만 존재)
- `runs/` 체크포인트(런당 최종 060000만 유지, 각 865M) — 깃 부적합, 필요 시 HF hub
- `data/` 원본·패키징 코퍼스(31G) — 소스 오브 레코드는 서버
- `venv/`, `logs/`

## 재동기화
rsync -a --exclude=__pycache__ <SRV>:'~/sunhong/nav-vla/code/' server/code/
rsync -a <SRV>:'~/sunhong/nav-vla/' --include='*.sh' --exclude='*' server/
