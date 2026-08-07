#!/usr/bin/env python3
"""Wheels-up 검증 커맨더 — 정책 없이 결정적 /cmd_vel로 배선·부호 확인.

바퀴를 완전히 띄운 상태에서 실행한다. 어댑터+시리얼이 떠 있어야 하며
(./thor_car_demo.sh wheels-up), 각 단계마다 Enter로 시작하고 바퀴를 눈으로
확인해 y/n로 답한다. 끝나면 steer_sign 등 확정 파라미터를 출력한다.

    ros2 run thor_vehicle_pkg wheels_up_test

검증 항목 (마스터플랜 D3 첫 게이트):
    ① 무명령 정지 (어댑터+시리얼 워치독)
    ② 전진 명령 → 바퀴 전진 회전
    ③ 좌회전 명령 → 앞바퀴 좌향 (아니면 steer_sign=-1)
    ④ 우회전 대칭
    ⑤ 명령 중단 → 0.5 s 내 정지 (워치독 체인)
    ⑥ 속도 3단 → 회전속도 단조 증가
"""

import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from interfaces_pkg.msg import MotionCommand
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)


class Commander(Node):
    def __init__(self):
        super().__init__("wheels_up_test")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.pub = self.create_publisher(Twist, "/cmd_vel", qos)
        self.last_mc = None
        self.create_subscription(
            MotionCommand, "topic_control_signal", self._mc, qos)

    def _mc(self, m):
        self.last_mc = (m.steering, m.left_speed)

    def drive(self, v, w, seconds):
        """10 Hz로 연속 발행 (안 하면 어댑터 워치독이 0으로 덮는다)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            t = Twist()
            t.linear.x = float(v)
            t.angular.z = float(w)
            self.pub.publish(t)
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_mc

    def idle(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_mc


def ask(q):
    while True:
        a = input(f"{q} [y/n] ").strip().lower()
        if a in ("y", "n"):
            return a == "y"


def main():
    print(__doc__)
    input("⚠ 바퀴가 완전히 떠 있고 E-stop이 손에 닿는지 확인 후 Enter…")
    rclpy.init()
    node = Commander()
    results = {}

    print("\n① 무명령 정지 — 3초간 아무 명령도 보내지 않습니다.")
    mc = node.idle(3.0)
    print(f"   어댑터 출력: {mc} (기대: (0, 0) 스트림)")
    results["idle_stop"] = ask("   바퀴가 완전히 정지해 있습니까?")

    print("\n② 전진 저속 (v=0.5) — 3초.")
    input("   Enter로 시작…")
    mc = node.drive(0.5, 0.0, 3.0)
    print(f"   어댑터 출력: steering={mc[0]}, speed={mc[1]}")
    results["forward"] = ask("   바퀴가 '앞으로' 돌고 조향은 중립입니까?")

    print("\n③ 좌회전 (v=0.5, w=+0.6) — 3초. ★핵심: steer_sign 판정")
    input("   Enter로 시작…")
    mc = node.drive(0.5, 0.6, 3.0)
    print(f"   어댑터 출력: steering={mc[0]} (양수 명령)")
    left_ok = ask("   앞바퀴가 '왼쪽'으로 꺾였습니까?")
    results["left"] = left_ok

    print("\n④ 우회전 (v=0.5, w=-0.6) — 3초.")
    input("   Enter로 시작…")
    mc = node.drive(0.5, -0.6, 3.0)
    print(f"   어댑터 출력: steering={mc[0]}")
    results["right"] = ask("   앞바퀴가 '오른쪽'으로 꺾였습니까 (③과 대칭)?")

    print("\n⑤ 명령 중단 → 워치독 — 발행을 끊습니다.")
    t0 = time.monotonic()
    node.idle(2.0)
    results["watchdog"] = ask("   끊은 뒤 ~0.5초 안에 바퀴가 멈췄습니까?")

    print("\n⑥ 속도 3단 (v=0.3 → 0.6 → 0.9, 각 2초).")
    input("   Enter로 시작…")
    for v in (0.3, 0.6, 0.9):
        mc = node.drive(v, 0.0, 2.0)
        print(f"   v={v}: speed_int={mc[1]}")
    results["speed_mono"] = ask("   회전 속도가 단계마다 빨라졌습니까?")

    print("\n===== 결과 =====")
    for k, v in results.items():
        print(f"  {k:12s} {'PASS' if v else 'FAIL'}")

    if results["left"] and results["right"]:
        print("\n조향 부호: steer_sign=+1 확정 (현 기본값 그대로)")
    elif not results["left"] and not results["right"]:
        print("\n조향 부호 반전: 내일부터 STEER_SIGN=-1 로 기동하십시오:")
        print("  STEER_SIGN=-1 ./thor_car_demo.sh")
    else:
        print("\n⚠ 좌/우 비대칭 — 배선·서보 점검 필요. 주행 진행 금지.")

    if not results["watchdog"] or not results["idle_stop"]:
        print("⚠ 워치독 실패 — MCU가 마지막 명령을 유지하는 펌웨어일 수 있음.")
        print("  D0 항목: 아두이노 소스에서 command timeout 확인/추가 전 주행 금지.")

    ok = all(results.values())
    print(f"\n종합: {'PASS — 빈 공간 저속 시험 진행 가능' if ok else 'FAIL — 위 항목 해소 전 주행 금지'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
