# 2026-08-07 15:07 — 좌하단을 정합 BEV로: 스튜디오 화면을 대시보드에 스트리밍

사용자 요구 확정판: 대시보드 좌하단에 도안 트랙맵이 아니라 **스튜디오의
'정합된 BEV'(라이브 점군 + 트랙 라인 오버레이)** 를 표시. 조향값 라벨은
제거하고 VLA Policy 패널로 병합.

## 구현

1. **`aligned_bev_publisher` 신설** (track_localizer_pkg, 노트북에서 실행) —
   점군 → BEV 강도 래스터(track_pose_node와 동일 캔버스 240×320) → 트랙
   라인 오버레이 → JPEG `/track/aligned_bev/compressed` (~8 Hz, ~20 KB/프레임).
   **원시 점군은 여전히 네트워크를 건너지 않는다** — 건너는 것은 JPEG뿐
   (마스터플랜 §4.1 유지)
2. 라인 추출·워프·블렌드는 **스튜디오 코드를 직접 임포트**
   (`tools/lidar_alignment_gui/layout_rendering.py`) — 첫 시도의 단순
   임계값(>60)은 풀컬러 도안(초록 배경·회색 노면, 평균 밝기 118)의 배경까지
   잡아 노란 덩어리가 됐다. HSV 기반 `extract_layout_line_mask`가 정답
3. 대시보드 좌하단 = QStacked: **BEV 신선(<2 s)이면 정합 BEV, 아니면 도안
   트랙맵 폴백** — 노트북 미연결 시에도 빈 화면이 아님. 조향 라벨은 제거,
   Steer/L/R은 VLA Policy 패널 3행째로 이동

## 검증 (합성 OT128 점군: 동심 스캔 링 + 고강도 페인트 호 + 이동 클러스터)

스크린샷 육안 — 스튜디오와 동일한 그림(검은 배경 + 스캔 링 + 노란 트랙
라인 + 차량 클러스터)이 대시보드 좌하단에 8 Hz로 렌더. 피드 중단 시 2 s 후
도안 맵 폴백 전환 확인. `bev_map` 헬스 점등 정상.

## 실기 연결 시 노트북 실행 세트 (갱신)

```bash
# 노트북: source track_link_env.sh <노트북IP> && fastdds discovery -i 0 -p 11811
ros2 run lidar_perception_pkg hesai_ingest_node
ros2 run track_localizer_pkg track_pose_node --ros-args --params-file ...
ros2 run track_localizer_pkg aligned_bev_publisher     # ★신규 — 대시보드 좌하단
```

## 정정 (15:12) — 도안 맵 폴백 제거

사용자 확인 결과 미수신 시 도안 트랙맵이 뜨는 폴백은 **"연결된 것처럼
보인다"는 혼동**을 유발 — 제거했다. 좌하단은 항상 정합 BEV 패널이고,
미수신이면 Waiting 표시. `TrackMapWidget`도 함께 삭제 (git 이력에 보존).

## 미해결

- 정합 BEV 위에 차량 pose 마커는 안 얹음 — 점군에 차량이 이미 보이므로
  중복. 필요해지면 옵션으로
- 240×320 세로 캔버스를 420×340 위젯에 KeepAspectRatio — 좌우 여백 있음.
  회전 표시가 나으면 90° 옵션 추가 후보
