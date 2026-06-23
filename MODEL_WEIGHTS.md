# Model Weights

이 저장소는 연구 코드와 LC-Utt 발화/평가 자료를 관리합니다. YOLO/시뮬레이션 모델 가중치 파일은 GitHub에 업로드하지 않습니다.

`.gitignore`에서 다음 확장자를 제외합니다.

```text
*.pt
*.onnx
*.engine
```

현재 로컬에서 확인된 가중치 파일:

| 경로 | 크기 |
| --- | ---: |
| `best.pt` | 54.8 MB |
| `best_cap.pt` | 23.9 MB |
| `myunggyun_crosswalk.pt` | 6.0 MB |
| `myunggyun_track.pt` | 6.0 MB |
| `src/simulation_pkg/simulation_pkg/data/sim.pt` | 92.3 MB |
| `명균's pt파일/crosswalk.pt` | 6.0 MB |
| `명균's pt파일/parking_front.pt` | 6.0 MB |
| `명균's pt파일/parking_rear.pt` | 6.0 MB |
| `명균's pt파일/track.pt` | 6.0 MB |
| `명균's pt파일/traffic_light_sim.pt` | 5.5 MB |

다른 PC에서 이 repo를 clone한 뒤 시뮬레이션을 실행하려면 필요한 모델 파일을 위 경로에 직접 복사하세요.
