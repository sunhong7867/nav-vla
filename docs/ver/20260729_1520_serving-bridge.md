# 2026-07-29 15:20 — 폐루프 서빙 브릿지 + 스텁 정책 서버 + 패키징

> 앞선 문서: [22:08 조건부 상호정보](20260728_2208_cmi-and-resampler.md)
> 계획: [sim_vla_plan.md](../sim_vla_plan.md) §7.1

코퍼스 v2 수집이 도는 3.5시간 동안, **체크포인트가 없어도 검증 가능한** 부분을 먼저 만들었다.
서빙 경로(ZMQ 왕복, 청크 스플라이싱, 10 Hz 제어 루프, 워치독, 지시 latching)는 평가 하네스에서
조용히 틀리기 가장 쉬운 곳이고, 학습이 끝난 뒤 **출력도 모르는 모델을 상대로** 디버깅하면
미지수가 둘이 된다.

| 파일 | 역할 |
|---|---|
| `nav_vla_pkg/vla_bridge_node.py` | ROS2측. 평가 중 `/cmd_vel`을 만지는 유일한 노드 |
| `scripts/stub_policy_server.py` | 같은 wire contract를 말하되 모델이 없는 서버 |
| `scripts/package_corpus.py` | 검증·리샘플된 세션 → 서버로 보낼 압축 트리 |
| `scripts/to_lerobot.py` | 패키지 → LeRobotDataset (lerobot이 있는 곳에서 실행) |

---

## 1. 프로세스 2개 — wire contract

```
->  {"jpeg": bytes, "state": [3]f32, "task": str, "seed": int, "req": int, "tick": int}
<-  {"actions": [[dx, dy, dyaw], ...]}
```

ROS2 Jazzy의 시스템 Python 3.12(apt numpy/opencv)와 lerobot의 torch/transformers/av를 섞지 않는다.

`seed`가 명시적으로 넘어가는 이유는 `D_same` — 모든 발산 수치의 분모 — 이 sampler를 호출측에서
고정할 수 있어야만 측정 가능하기 때문이다.

`tick`은 나중에 추가했다. §3.3 참조.

---

## 2. 스텁으로 시험한 것들

### 2.1 워치독이 큐가 가득 찬 상태에서 발동하고 있었다

```
q=24/30  chunks=24  lat=70.3ms  underrun=25.39%  watchdog=198
```

큐에 24개가 대기 중인데 워치독이 200번 넘게 발동했다. 조건이 틀렸다:

```python
stale = (now - self._last_chunk_t) * 1000.0 > self.watchdog_ms   # 500 ms
```

**청크 30개 @10 Hz = 3초 분량**이고, 리필 스레드는 70%가 소비될 때까지 기다렸다가 요청한다.
따라서 청크는 **정상적으로 1초에 한 번쯤** 도착한다. 도착 시각을 고정 500 ms와 비교하면
건강한 상태가 실패로 판정된다.

잡아야 하는 실패는 "청크가 안 온다"가 아니라 **"예측한 지평선을 넘어서 실행하고 있다"** 이다.

```python
horizon_s = self.chunk_len / self.rate_hz          # 3.0 s
stale = now - self._last_chunk_t > horizon_s * self.stale_factor    # 1.5x
```

### 2.2 그 워치독이 명령 스트림을 망가뜨리고 있었다

워치독은 `speed=0, steering=hold`를 낸다. 매 틱 발동하니 대부분의 틱이 홀드가 되고,
실제 액션이 나오는 틱에서 튀었다.

| | 수정 전 | 수정 후 |
|---|---|---|
| 워치독 발동 | 237 | **0** |
| underrun | 20.3% | **0.00%** |
| 최대 명령 스텝 | 0.121 rad/s | 0.039 |
| 2차 차분 최대 | 0.121 | 0.023 |
| 불연속 | 59 | 14 |

**증상은 스플라이싱 버그처럼 보였다.** 원인은 워치독이었다.

### 2.3 지시가 도착하지 않고 있었다 (QoS)

```
New publisher discovered on topic '/vla/instruction', offering incompatible QoS.
No messages will be received from it. Last incompatible policy: DURABILITY
```

계획대로 TRANSIENT_LOCAL로 구독했는데, DDS는 durability를 엄격히 매칭한다 —
**TRANSIENT_LOCAL 구독자는 VOLATILE 발행자를 아예 듣지 못한다.** 평범한 `ros2 topic pub`은
경고 한 줄만 남기고 무시된다.

같은 토픽에 **구독을 두 개** 만들었다. TRANSIENT_LOCAL(늦게 붙는 하네스가 latched 지시를 봄)
+ VOLATILE(아무 발행자나 통과). `_instr_cb`는 텍스트가 바뀔 때만 동작하므로 중복 수신은 무해하다.

### 2.4 카운터가 노드 시작부터 누적되고 있었다

"유효한 eval 런 = 워치독 0회 + underrun < 1%"는 **한 trial에 대한 진술**이다. 노드 시작 직후
지시가 없어 큐가 비어 있던 구간까지 누적하면 모든 런이 실격된다. 지시 변경(= trial 경계)에서
리셋한다.

### 2.5 `msgpack`이 ROS 노드 인터프리터에 없었다

`pip install`은 활성 venv(`~/venv/mujoco`)에 들어갔고 `ros2 run`은 `/usr/bin/python3`를 쓴다.
추론 스레드가 조용히 죽고 워치독만 발동해서 **"정책 서버가 죽었다"처럼 보였다.**
이제 `_infer_loop`가 ImportError를 잡아 정확한 설치 명령과 함께 로그를 남긴다.

---

## 3. 청크 스플라이싱 — 실제로 효과가 있는지

계획은 스플라이싱을 생략하면 "평가 전체를 망친다"고 적었다. 시험해봤다.

### 3.1 연속적인 스텁으로는 시험이 불가능하다

처음 스텁은 절대 틱 인덱스의 **완벽히 연속인 사인**을 냈다. 그러면 새 청크가 이전 청크가 끝난
자리에서 정확히 이어지므로 **스플라이싱이 매끄럽게 할 것이 없다.** overlap 0과 5의 최대 스텝이
소수점까지 같게 나왔다.

실제 정책의 연속된 두 예측은 **일치하지 않는다.** 그 불일치가 스플라이싱이 존재하는 이유다.
`--chunk-jitter`로 청크마다 ±0.08 rad/s를 주입했다.

### 3.2 결과

```
스텁이 청크 간 ±0.08 rad/s 불일치를 주입
사인 자체의 최대 스텝 = 0.01963 rad/s
```

| splice overlap | 최대 명령 스텝 | p99 | 개선 |
|---|---|---|---|
| 0 (끔) | **0.155** rad/s | 0.148 | — |
| **5 (기본)** | **0.041** | 0.041 | **3.8×** |
| 10 | 0.029 | 0.028 | 5.4× |

overlap 0에서 0.155는 불일치 전체(±0.08 → 최대 0.16)가 **한 틱에 몰린 값**이다.

overlap 10이 더 매끄럽지만 **새 관측이 1초 동안 이전 예측에 희석된다**(첫 새 액션의 가중치가
1/11). 주차 진입처럼 반응이 필요한 구간에서 손해다. 5(0.5초)를 기본으로 둔다.

**주의**: "nominal 초과 스텝 개수"는 스플라이싱을 켜면 **늘어난다**(14 → 30). 의도한 거래다 —
큰 점프 하나를 작은 편차 여러 개로 바꾼다. `D_same` 노이즈 바닥을 결정하는 것은 최댓값이다.

### 3.3 `tick`을 추가한 이유

스플라이싱을 켜고도 정확히 **정상 스텝의 2배**인 점프가 25초에 5번 남았다. 스텁이 소비된 틱 수를
`chunk_len * 0.3`으로 **추측**하고 있었고, 실제 소비량과 한 틱씩 어긋나면서 사인 위상이 밀렸다.

브릿지가 실행한 액션의 절대 개수를 요청에 실어 보내도록 했다. 결정론적 서버(replay 포함)가
위상을 맞추는 데 필요하다. **이걸 안 넣었으면 스텁의 버그를 브릿지의 스플라이싱 버그로
오진했을 것이다.**

---

## 4. 패키징 — 서버로 무엇을 보낼 것인가

로컬에 `lerobot`이 없다. LeRobot 온디스크 레이아웃(parquet 샤드, `meta/info.json`,
`meta/episodes_stats.jsonl`, AV1 비디오)은 버전(v2.0/v2.1/v3.0)마다 모양이 바뀌었고,
**이 기계에서 되읽어 검증할 수 없는 바이너리 형식을 손으로 쓰는 것**이 된다.
미묘하게 틀린 데이터셋은 변환에서 실패하지 않는다 — 학습이 끝난 뒤 틀린 것을 배운 모델로 실패한다.

그래서 둘로 나눴다.

**`package_corpus.py`** — 30 Hz 원본에서 10 Hz 격자가 참조하는 프레임만 추린다.
프레임은 **하드링크**라 즉시 끝나고 디스크를 쓰지 않는다(`tar`/`rsync`가 실체화한다).

```
원본 세션   ~24 MB/에피소드    281 MB (11 에피소드)
패키지      ~8 MB/에피소드     84 MB       → 30%
```

`meta.json`은 통째로 보존한다. `cf_group_id` / `cf_axis` / `start_pose_key` / `instruction`이
반사실 장부 전체이고 프레임에서 복원 불가능하다.

**`to_lerobot.py`** — `LeRobotDataset.create()` / `add_frame()` / `save_episode()`.
설치된 버전이 바이트를 정하게 둔다. import 경로 두 가지(`lerobot.datasets` /
`lerobot.common.datasets`)와 task 전달 방식 두 가지를 모두 시도한다.

반사실 장부는 LeRobot 스키마에 들어갈 자리가 없으므로 `nav_vla_index.jsonl`로 나란히 쓴다.

---

## 5. 변경된 파일

| 파일 | 변경 |
|---|---|
| `nav_vla_pkg/vla_bridge_node.py` | 신규 |
| `scripts/stub_policy_server.py` | 신규 |
| `scripts/package_corpus.py` | 신규 |
| `scripts/to_lerobot.py` | 신규 |
| `scripts/collect_corpus.py` | `--group-prefix` (세션 분할 수집 시 그룹 ID 충돌 방지) |
| `scripts/verify_corpus.py` | 세션 여러 개 동시 검증 + 그룹 ID 충돌 검사 |
| `scripts/measure_cmi.py` | 세션 여러 개 |
| `nav_vla_pkg/setup.py` | `vla_bridge_node` entry point |

---

## 6. 미해결

1. **`vla_policy_server.py` 미작성.** 스텁이 계약을 정의했으므로 체크포인트가 나오면 채운다.
   계획의 워밍업(reduce-overhead CUDA graph 30–120 s)이 여기 들어간다.
2. **시뮬 실주행 미검증.** 수집이 `/cmd_vel`을 점유 중이라 브릿지 출력을 `/vla/cmd_test`로
   리맵해서 시험했다. 차가 실제로 도는 것은 수집이 끝난 뒤.
3. **`to_lerobot.py` 미실행.** lerobot이 있는 기계가 없다. SSH 키/sshpass도 없어 서버 확인 불가.
4. **잔여 불연속 14건** (overlap 5, jitter 없음). 스텁 위상 문제를 고친 뒤 재측정 필요.

---

## 7. 교훈

**검증 하네스는 검증할 대상보다 먼저 만들 것.** 워치독·QoS·카운터·의존성 4건 모두 체크포인트
없이 드러났다. 학습 후에 발견했다면 "모델이 이상하다"로 며칠을 썼을 것이다.

**시험용 스텁이 너무 완벽하면 아무것도 시험하지 못한다.** 완벽히 연속인 사인은 스플라이싱을
무의미하게 만들었고, overlap 0과 5가 소수점까지 같은 값을 냈다. 실제 정책의 성질(연속 예측이
불일치함)을 스텁이 흉내내야 시험이 성립한다.

**증상이 가리키는 곳과 원인이 다르다.** 명령 스트림의 불연속 59건은 스플라이싱 버그로 보였고
실제로는 워치독이었다. 그리고 남은 5건은 브릿지가 아니라 스텁이었다.
