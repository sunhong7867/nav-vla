# docs/ver2 — 서브 세션(Thor) 변경 이력

메인 노트북에서 작성하는 [ver/](../ver/README.md)와 **작업 장소를 구분**하기
위한 폴더다. Jetson AGX Thor(`~/hong/`) 세션에서 만든 문서·변경 기록은
여기에 쌓는다. 파일명 규칙·필수 항목·append-only 원칙은
[ver/README.md](../ver/README.md)를 그대로 따른다.

## 계획 문서 (날짜 접두사 없음)

| 문서 | 요약 |
|---|---|
| [real_car_master_plan.md](real_car_master_plan.md) | v7 판정부터 Hesai→Thor 폐루프 데모까지 실행 계획 (D0~D7 갱신판, §4.6 맵·존 자산) |
| [navvla_vs_vlaad_comparison.md](navvla_vs_vlaad_comparison.md) | nav-vla vs VLA_AD 공통점·차이점·보완 관계, 2026-07-27 판정 정정 |

## 목록

| 날짜 | 문서 | 요약 |
|---|---|---|
| 2026-08-06 23:41 | [주0 병행: 이식성·N점 정합·canonical 프레임](20260806_2341_week0-portability-npoint-canonical.md) | 절대경로 제거, 스튜디오 4→N점(4점 잔차=**구조적 0** 재확인, 8점 0.60 px), `/track/vehicle_pose_map` 신설(유도 스케일 3.041 mm/px → 템플릿 **15.90×10.93 m**, 로드맵 추정과 일치), 구역↔템플릿 왕복 무손실. VLA_AD: lasa 런치 추가·α점프 수정 + **install 심링크 26개가 hoon 원본을 가리키던 것 발견·재지향** |
