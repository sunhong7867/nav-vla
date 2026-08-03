# GPU 서버 정리

> 학습용 원격 서버 요약. **비밀번호는 이 문서에 적지 않는다** — 이 저장소는 GitHub에 올라간다.
> 자격증명은 로컬 `~/.ssh/config` 와 비밀번호 관리자에만 둔다.
>
> 작성일: 2026-07-28

---

## 1. 어디에 무엇을 돌릴 것인가

| 용도 | 서버 | 이유 |
|---|---|---|
| **SmolVLA 학습 (주력)** | **SW서버 RTX 3090** | 24 GB면 충분(실측 bs=32 → 3.82 GiB). RAM 128 GB가 dataloader 병목을 없앤다 |
| **병렬 스윕** | SW서버 RTX 4070 Ti SUPER | 같은 서버 2번째 GPU. 하이퍼파라미터 스윕을 동시에 |
| **대형 실험 (필요 시)** | 반시공 PRO6000 | 450M 모델엔 과잉. 트렁크를 3B급으로 바꿀 때만 |
| **비권장** | 반시공 A100 ×4 | 다수 연구실 공용 + **스토리지 부족 명시**. 코퍼스 수십 GB를 올리면 민폐. GPU 중복 점유로 OOM 종료 사례 보고됨 |

---

## 2. SW서버 (주력)

| 항목 | 값 |
|---|---|
| 호스트 | `115.145.211.157` (기본 포트 22) |
| 계정 | `autolab_sw` |
| CPU | Intel Core i9-14900K |
| GPU | **RTX 3090 (24 GB)** ×1, RTX 4070 Ti SUPER (16 GB) ×1 |
| RAM | 128 GB |

```bash
# 접속 (8888 포트 포워딩 = Jupyter/TensorBoard용)
ssh -L 8888:localhost:8888 autolab_sw@115.145.211.157
```

**tmux 필수.** 일반 SSH에서 실행하면 연결이 끊길 때 학습도 죽는다.

```bash
tmux ls                      # 살아있는 세션 확인
tmux new -s navvla_train     # 새 세션
tmux attach -t navvla_train  # 기존 세션 접속
# Ctrl+b 누른 뒤 d           → 나가기 (프로그램은 계속 실행)
# Ctrl+c                     → 실행 중인 프로그램까지 종료
```

**GPU 지정** — 다른 사용자와 겹치지 않게 항상 확인 후 지정한다.
```bash
nvidia-smi                                   # 여유 GPU 확인
CUDA_VISIBLE_DEVICES=1 lerobot-train ...     # 3090 (index 1)
CUDA_VISIBLE_DEVICES=0 lerobot-train ...     # 4070 Ti SUPER (index 0)
```

---

## 2.1 nav-vla 작업 환경 (2026-07-29 구축)

공용 계정이므로 **모든 것을 `~/sunhong/nav-vla/` 아래에 격리**한다. 홈 디렉터리에는 아무것도
추가하지 않는다.

```
~/sunhong/nav-vla/
├── venv/                 python3.10 venv — torch 2.6.0+cu124, lerobot 0.4.4[smolvla]
├── data/packed_smoke/    노트북에서 rsync한 10 Hz 코퍼스
├── data/lerobot/         to_lerobot.py 출력
├── code/                 to_lerobot.py 등
├── runs/  logs/
```

### 시스템 파이썬을 건드리지 않은 이유

시스템 인터프리터에 **`torch 1.8.0a0+unknown`** 이 있다. 2021년 빌드이고 `a0+unknown`은
정상 릴리스가 아니다. lerobot은 torch ≥ 2.2를 요구하므로 못 쓰지만, **누구의 작업물인지 모르고
덮어쓰면 그 사람 환경이 깨진다.** venv로 분리했다.

### 실측 환경

| 항목 | 값 |
|---|---|
| 드라이버 | **570.195.03** → CUDA 12.x 가능 (시스템 nvcc는 11.8이지만 무관) |
| torch | 2.6.0+cu124, `cuda_available=True`, devices 2 |
| lerobot | 0.4.4 (`lerobot.datasets.lerobot_dataset`) |
| transformers | 4.57.6 |
| CPU / RAM | 32코어 / 125 GB |

### ⚠️ `video_backend="pyav"` 를 반드시 지정할 것

LeRobot 기본 디코더 `torchcodec`은 **FFmpeg 5+ 공유 라이브러리를 dlopen**한다.
이 서버는 Ubuntu 22.04라 FFmpeg 4.4(`libavutil.so.56`)뿐이다.

```
OSError: libavutil.so.57: cannot open shared object file
```

데이터셋은 멀쩡한데 **읽을 때만** 실패하므로 데이터가 깨진 것처럼 보인다. PyAV는 자체 FFmpeg를
번들하므로 시스템 패키지가 필요 없다. `to_lerobot.py`의 기본값을 `pyav`로 박아뒀고,
학습·평가에서도 같은 값을 넘겨야 한다.

### 변환 경로 검증 완료 (스모크 3 에피소드)

```
frames 1332, episodes 3, fps 10
observation.images.front  (3,480,640) float32
observation.state / action  원본 jsonl과 max|diff| = 0.0
task 'Park in the first bay on the right, slowly.'
```

행동 라벨이 비트 단위로 보존된다. 반사실 장부는 LeRobot 스키마에 자리가 없어
`nav_vla_index.jsonl`로 나란히 쓴다.

**마지막 격자점은 `action: null`이다** — SE(2) 델타에 `pose[k+1]`이 필요한데 없다.
에피소드 N개 격자점 → **N−1개 (관측, 행동) 쌍**. 변환기가 걸러낸다.

---

## 3. 반시공 서버 (공용)

| 항목 | 값 |
|---|---|
| 계정 | `autolab04` (정선홍) |
| PRO6000 | `115.145.135.218` **포트 1701** |
| A100 ×4 | 별도 주소 — 공용, 스토리지 부족 |
| 개인 데이터 | `/home/data/autolab/<ID>/` |

```bash
ssh -p 1701 autolab04@115.145.135.218
```

**운영 정책 (반드시 지킬 것)**
- 스토리지가 부족하므로 **대용량 코퍼스를 올리지 않는다**
- `nvidia-smi`로 가용 GPU를 확인하고 **지정해서** 사용 (중복 점유 시 상대방 OOM 유발)
- CPU/메모리 많이 쓰는 작업은 `htop`으로 모니터링
- 패키지는 개인 home에 설치하고 PATH를 잡아 쓴다

---

## 4. 데이터 전송 전략 — 이게 GPU보다 큰 제약이다

레코더 v2는 소스 레이트(~30 Hz)로 전부 기록한다. 추정:

| 코퍼스 | 프레임 | 30 Hz 원본 | **10 Hz 리샘플본** |
|---|---|---|---|
| v2 (570 ep × ~25 s) | ~427k | ~11 GB | **~3.6 GB** |
| v3 (2,000 ep) | ~1.5M | ~37 GB | **~12 GB** |

**원본을 서버에 올리지 않는다.**

| 위치 | 보관 | 이유 |
|---|---|---|
| 노트북 (로컬) | 30 Hz 원본 + 리샘플러 + 수용 게이트 | 리샘플 재실행에 필요. 학습에는 불필요 |
| 서버 | **10 Hz LeRobot 데이터셋만** | 학습에 필요한 건 10 Hz 격자뿐 |

전송량이 1/3로 줄고 서버 스토리지도 아낀다.

```bash
# 리샘플본만 전송 (rsync 권장 — 중단 후 재개 가능)
rsync -avP --partial \
  ~/ROS2_project/nav-vla/datasets/navvla-sim-v2/ \
  autolab_sw@115.145.211.157:~/data/navvla-sim-v2/
```

---

## 5. 사용 전 체크리스트

Week 3(학습 시작) **전에** 한 번은 확인해 둘 것. 그때 가서 막히면 일정이 밀린다.

```bash
nvidia-smi                          # GPU 여유 + 드라이버 버전
nvcc --version                      # CUDA 버전
python3 --version                   # 3.10+ 필요
df -h ~                             # 홈 여유 공간 (v3까지 최소 20 GB)
free -g                             # RAM
pip install --user lerobot          # 설치 가능한지
```
