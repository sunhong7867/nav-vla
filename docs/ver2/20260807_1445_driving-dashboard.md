# 2026-08-07 14:45 — 주행 대시보드: VLA_AD operator dashboard의 nav-vla판

[14:30 실차 준비](20260807_1430_real-car-day-prep.md)의 확장. VLA_AD는
3커맨드(pipeline + vla_trt + dashboard)로 GUI 주행 확인을 하는데, nav-vla
실차 경로에는 운전자 GUI가 없었다. `thor_vehicle_pkg`에
**`driving_dashboard`** 신설 — 요청 사양(왼쪽 아래 라이다맵+차량 위치) 반영.

## 구성 (PySide6 — cv2 GUI는 이 Thor에서 헤드리스 확인)

```
좌상  카메라 라이브 (/image_raw/compressed)
좌하  ★트랙맵 + 차량 위치 + 궤적 트레일 (/track/vehicle_pose_map)
우측  명령 · 브리지 지연/큐/underrun(/vla/status) · /cmd_vel
      · MotionCommand · pose 나이(0.5 s 초과 시 ⚠유실) · geofence
      · E-STOP 버튼 (/operator/estop → 어댑터 게이트 직결)
```

- 트랙맵은 track_localizer의 **캐노니컬 프레임 재사용**: 배경 = 도안
  템플릿(track2.png), 좌표 변환 = `TemplateFrame`의 m_per_px — 라이다
  노트북이 붙으면 실위치가 찍히고, 없으면 "수신 없음" 표시
- `thor_car_demo.sh`의 wheels-up·full 모드 양쪽에 자동 기동
  (`NAVVLA_DASH=0`으로 끔). full 모드엔 채팅 GUI도 추가(`NAVVLA_CHAT=0`)

## 검증 (모의 데이터 실기동 + 스크린샷 육안)

mock_track_publisher(원 궤적) + 합성 카메라/status/cmd_vel/MotionCommand로
기동 — 트랙 도안 위에 궤적 트레일·차량점·진행방향 정확히 렌더, 우측 패널
전 항목 갱신(지연 291 ms·steer +2·pose 나이 0.1 s), E-STOP 버튼 표시 확인.

## VLA_AD 3커맨드와의 대응 (참고용 대조)

| VLA_AD | nav-vla |
|---|---|
| `pipeline_launch.py` + `vla_trt_launch.py` + `dashboard` (3커맨드) | `STEER_SIGN=±1 ./thor_car_demo.sh` (1커맨드 — 서버·카메라·어댑터·시리얼·대시보드·채팅 전부) |
| dashboard: 카메라·BEV·VlaIR·TTL·지연 | driving_dashboard: 카메라·**트랙맵+위치**·지연/큐·명령·E-STOP |

## 미해결

- 대시보드에서 E-STOP 외 조작(속도 상한 슬라이더 등)은 없음 — 의도적 최소
- 트랙맵의 존/기준선 오버레이(track_assets.yaml)는 미표시 — 자산 저작 후 추가 후보
- 시뮬에서는 /track/* 미발행이라 맵 패널이 비어 있는 게 정상 (Gazebo GUI가 그 역할)
