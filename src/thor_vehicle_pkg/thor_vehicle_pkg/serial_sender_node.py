#!/usr/bin/env python3
"""MotionCommand -> Arduino serial. VLA_AD serial_sender_node 이식판.

원본: VLA_AD/src/serial_communication_pkg/serial_sender_node.py (선배 스택).
프로토콜은 한 줄 그대로다: "s{steering}l{left}r{right}\\n".

이식하며 바꾼 것 두 가지:
1. 포트/보레이트 파라미터화. 저장소의 115200 과 Arduino 소스의 9600 이
   충돌한다는 기록(데모 계획서 D0)이 있어, 실제 flash 기준으로 맞출 때
   코드 수정 없이 -p baud:= 로 해소할 수 있게 했다.
2. 수신 워치독: MotionCommand 가 timeout 동안 없으면 정지 프레임을
   주기적으로 송신한다. MCU 에 command timeout 이 있는지 미확인(D0)인
   상태에서 마지막 명령 유지 = runaway 이므로 노드 측에서 먼저 방어한다.
"""

import time

import rclpy
from interfaces_pkg.msg import MotionCommand
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from .lib import protocol_convert_func_lib as PCFL

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


class SerialSenderNode(Node):
    def __init__(self):
        super().__init__("serial_sender_node")
        self.port = self.declare_parameter("port", "/dev/ttyACM0").value
        self.baud = int(self.declare_parameter("baud", 115200).value)
        self.sub_topic = self.declare_parameter(
            "sub_topic", "topic_control_signal").value
        self.watchdog_s = float(
            self.declare_parameter("watchdog_s", 0.5).value)

        self.ser = None
        if serial is not None:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=1)
                time.sleep(1)  # Arduino auto-reset 대기 (원본과 동일)
            except serial.SerialException as e:
                self.get_logger().warn(
                    f"{self.port} 사용 불가 ({e}) — dry-run 모드")
        else:
            self.get_logger().warn("pyserial 없음 — dry-run 모드")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(
            MotionCommand, self.sub_topic, self._on_cmd, qos)
        self.last_cmd_t = None
        self._stale_warned = False
        self.create_timer(0.2, self._watchdog)
        self.get_logger().info(
            f"serial {'OK' if self.ser else 'DRY-RUN'} — {self.port} "
            f"@{self.baud}, in={self.sub_topic}, watchdog {self.watchdog_s} s")

    def _write(self, steering, left, right):
        frame = PCFL.convert_serial_message(steering, left, right)
        if self.ser is not None:
            self.ser.write(frame.encode())

    def _on_cmd(self, msg):
        self.last_cmd_t = time.monotonic()
        self._write(msg.steering, msg.left_speed, msg.right_speed)

    def _watchdog(self):
        stale = (self.last_cmd_t is None
                 or time.monotonic() - self.last_cmd_t > self.watchdog_s)
        if stale:
            self._write(0, 0, 0)
            if not self._stale_warned and self.last_cmd_t is not None:
                self.get_logger().warn(
                    f"{self.sub_topic} {self.watchdog_s} s 침묵 — 정지 프레임 송신")
            self._stale_warned = True
        else:
            self._stale_warned = False

    def destroy_node(self):
        if self.ser is not None:
            try:
                self._write(0, 0, 0)
                self.ser.close()
            except Exception:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    node = SerialSenderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
