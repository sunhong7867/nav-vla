# Thor에서 SmolVLA 시뮬레이션 실행

이 문서는 새로 clone한 Thor 환경에서 Gazebo SmolVLA 데모를 실행하는 최소 절차다.
ROS 2 Jazzy와 Gazebo Harmonic(`ros-jazzy-ros-gz`)이 설치돼 있다고 가정한다.

## 1. 저장소와 환경 준비

```bash
git clone https://github.com/sunhong7867/nav-vla.git
cd nav-vla
./setup_smolvla_demo.sh --install-ros-deps
```

설치 스크립트는 기본적으로 `$HOME/venv/navvla`에 Python venv를 만들고
`lerobot==0.4.4`, `msgpack`, `pyzmq`를 설치한 뒤 `colcon build`를 실행한다.
다른 위치를 쓰려면 실행 전 환경변수를 지정한다.

```bash
export NAVVLA_VENV=$HOME/venvs/nav-vla-policy
./setup_smolvla_demo.sh --install-ros-deps
```

NVIDIA Jetson/Thor에서 PyTorch를 JetPack 전용 wheel로 설치한 경우, 설치 스크립트의
venv는 `--system-site-packages`로 이를 재사용한다. 설치 후 다음 점검에서 `torch` import가
실패하면 해당 JetPack 버전에 맞는 PyTorch를 먼저 설치해야 한다.

## 2. 체크포인트 복사

모델 파일은 크기 때문에 GitHub에 포함되지 않는다. 체크포인트가 있는 컴퓨터에서
저장소의 `models/ckpt_v6_60k`로 디렉터리 전체를 전송한다.

```bash
mkdir -p models
rsync -av --info=progress2 /path/to/ckpt_v6_60k/ \
  THOR_USER@THOR_HOST:/path/to/nav-vla/models/ckpt_v6_60k/
```

전송 후 Thor에서 확인한다.

```bash
test -f models/ckpt_v6_60k/model.safetensors && echo OK
./smolvla_demo.sh check
```

다른 체크포인트 위치는 `--ckpt` 또는 `NAVVLA_CKPT`로 지정할 수 있다.

## 3. 실행과 종료

```bash
./smolvla_demo.sh
```

GUI 없이 실행:

```bash
./smolvla_demo.sh --no-gui
```

종료:

```bash
./smolvla_demo.sh down
```

로그는 `eval_out/demo/`에 기록된다. 정책 서버가 뜨지 않으면 먼저
`eval_out/demo/serve.log`와 `./smolvla_demo.sh check` 결과를 확인한다.
