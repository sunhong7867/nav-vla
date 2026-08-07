#!/usr/bin/env python3
"""/cmd_vel (Twist) -> MotionCommand 어댑터 + 1세대 안전 게이트.

nav-vla의 서빙 스택(vla_bridge -> /cmd_vel)은 시뮬 계약 그대로 두고,
실차 끝단만 이 노드가 갈아끼운다:

    시뮬:  /cmd_vel -> Gazebo (ackermann adapter)
    실차:  /cmd_vel -> [이 노드] -> MotionCommand -> serial_sender -> Arduino

이 노드가 유일한 최종 액추에이터 출력이며(마스터플랜 §4.5의 gateway+mux
1세대 최소형), 다음이면 무조건 정지 명령을 스트리밍한다:

    - /cmd_vel 워치독 초과 (기본 0.5 s)
    - /track/geofence_estop == True (래치 아님 — False 복귀 시 해제)
    - require_pose=True 인데 /track/vehicle_pose_map 이 0.5 s 이상 침묵
    - /operator/estop == True

변환 (전부 D0 실측 대상 — 기본값은 자리표시):
    speed_int = round(v * speed_per_mps),  [0, cap_speed_int] 클램프
    delta     = atan(wheelbase * w / v)    # Ackermann, 곡률 k = w/v
    steer_int = clamp(round(steer_sign * delta_deg / steer_deg_per_step), ±7)

steer_sign 은 wheels-up 부호 검증 전까지 신뢰하지 말 것 (D3 첫 항목).
후진(v<0)은 플랜트 지원이 미확인이라 0으로 게이트한다.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from interfaces_pkg.msg import MotionCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool


class CmdVelMotionAdapter(Node):
    def __init__(self):
        super().__init__("cmd_vel_motion_adapter_node")
        # --- 변환 파라미터 (TODO-D0: 전부 실측으로 교체) ---
        self.wheelbase = float(self.declare_parameter("wheelbase", 0.54).value)
        self.steer_deg_per_step = float(
            self.declare_parameter("steer_deg_per_step", 5.0).value)
        self.max_steer_step = int(
            self.declare_parameter("max_steer_step", 7).value)
        self.steer_sign = int(self.declare_parameter("steer_sign", 1).value)
        self.speed_per_mps = float(
            self.declare_parameter("speed_per_mps", 62.0).value)
        # 첫 실차는 보수적으로: v_base=100 이 기존 스택의 정상 주행값
        self.cap_speed_int = int(
            self.declare_parameter("cap_speed_int", 100).value)
        self.max_speed_mps = float(
            self.declare_parameter("max_speed_mps", 3.2).value)

        # --- 안전 파라미터 ---
        self.cmd_timeout = float(
            self.declare_parameter("cmd_timeout_s", 0.5).value)
        self.pose_timeout = float(
            self.declare_parameter("pose_timeout_s", 0.5).value)
        # 벤치/wheels-up 에서는 False, 트랙 주행에서는 반드시 True
        self.require_pose = bool(
            self.declare_parameter("require_pose", False).value)

        self.out_topic = self.declare_parameter(
            "out_topic", "topic_control_signal").value

        qos_cmd = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.pub = self.create_publisher(MotionCommand, self.out_topic, qos_cmd)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, qos_cmd)
        self.create_subscription(
            Bool, "/track/geofence_estop", self._on_geofence, 10)
        self.create_subscription(Bool, "/operator/estop", self._on_operator, 10)
        self.create_subscription(
            Odometry, "/track/vehicle_pose_map", self._on_pose, sensor_qos)

        self.last_cmd_t = None
        self.last_pose_t = None
        self.geofence = False
        self.operator = False
        self._blocked_reason = None
        # 20 Hz 감시: 차단 상태에서는 정지 명령을 계속 흘려보낸다 — MCU가
        # 마지막 명령을 유지하는 펌웨어일 가능성(D0 미확인)에 대한 방어.
        self.create_timer(0.05, self._watchdog)
        self.get_logger().info(
            f"adapter ready — wheelbase {self.wheelbase} m, "
            f"{self.steer_deg_per_step}°/step ±{self.max_steer_step}, "
            f"speed {self.speed_per_mps}/mps cap {self.cap_speed_int}, "
            f"require_pose={self.require_pose} (전부 D0 실측 전 자리표시)")

    # ------------------------------------------------------------------
    def _blocked(self):
        now = time.monotonic()
        if self.operator:
            return "operator_estop"
        if self.geofence:
            return "geofence_estop"
        if self.last_cmd_t is None or now - self.last_cmd_t > self.cmd_timeout:
            return "cmd_vel_timeout"
        if self.require_pose and (
                self.last_pose_t is None
                or now - self.last_pose_t > self.pose_timeout):
            return "pose_timeout"
        return None

    def _publish(self, steer, speed):
        m = MotionCommand()
        m.steering = int(steer)
        m.left_speed = int(speed)
        m.right_speed = int(speed)
        self.pub.publish(m)

    def _on_cmd(self, msg):
        self.last_cmd_t = time.monotonic()
        reason = self._blocked()
        if reason:
            self._publish(0, 0)
            return
        v = min(max(msg.linear.x, 0.0), self.max_speed_mps)
        w = msg.angular.z
        speed = min(int(round(v * self.speed_per_mps)), self.cap_speed_int)
        if v > 1e-3:
            delta_deg = math.degrees(math.atan(self.wheelbase * w / v))
            steer = int(round(self.steer_sign * delta_deg
                              / self.steer_deg_per_step))
            steer = max(-self.max_steer_step, min(self.max_steer_step, steer))
        else:
            steer, speed = 0, 0
        self._publish(steer, speed)

    def _watchdog(self):
        reason = self._blocked()
        if reason:
            if reason != self._blocked_reason:
                self.get_logger().warn(f"output blocked — {reason}")
            self._publish(0, 0)
        elif self._blocked_reason:
            self.get_logger().info(f"unblocked ({self._blocked_reason} 해제)")
        self._blocked_reason = reason

    def _on_geofence(self, msg):
        self.geofence = bool(msg.data)

    def _on_operator(self, msg):
        self.operator = bool(msg.data)

    def _on_pose(self, msg):
        self.last_pose_t = time.monotonic()


def main():
    rclpy.init()
    node = CmdVelMotionAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
