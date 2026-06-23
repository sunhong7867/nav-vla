#!/usr/bin/env python3
"""
hz_monitor_node
----------------
여러 토픽의 실제 publish 주기(Hz)를 동시에 측정해서 주기적으로 표로 출력하고,
종료 시 요약 CSV를 남기는 가벼운 모니터 노드. `ros2 topic hz`를 여러 토픽에
대해 한 번에 돌리는 것과 같은 역할.

측정값은 wall-clock(실시간) 기준이다. Gazebo의 real-time factor(RTF)가 1보다
작으면, 시뮬레이션 안에서는 nominal 10/20 Hz로 돌아도 실시간 측정값은 그보다
낮게 나올 수 있다(예: RTF=0.5 → 약 5/10 Hz). nominal rate는 코드 타이머값이고,
여기서 찍는 건 "실제로 초당 몇 번 도착했는가"이다.

사용:
    ros2 run debug_pkg hz_monitor_node
    ros2 run debug_pkg hz_monitor_node --ros-args -p report_period:=1.0 -p window_sec:=5.0
"""
import os
import csv
import time
import atexit
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
)

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from interfaces_pkg.msg import (
    MotionCommand, DetectionArray, LaneInfo, PathPlanningResult,
)

# (표시이름, 토픽, 메시지타입, nominal Hz 또는 None)
DEFAULT_TOPICS = [
    ("cmd_vel",           "/cmd_vel",              Twist,              20.0),
    ("control_signal",    "/topic_control_signal", MotionCommand,      10.0),
    ("front_camera",      "/camera/image_raw",     Image,              15.0),
    ("top_camera",        "/top_camera/image_raw", Image,              10.0),
    ("detections",        "/detections",           DetectionArray,     None),
    ("lane_info",         "/yolov8_lane_info",     LaneInfo,           None),
    ("path_planning",     "/path_planning_result", PathPlanningResult, None),
]


class TopicMeter:
    def __init__(self, name, topic, nominal):
        self.name = name
        self.topic = topic
        self.nominal = nominal
        self.total = 0                 # 전체 누적 메시지 수
        self.first_t = None            # 첫 메시지 시각(monotonic)
        self.last_t = None             # 마지막 메시지 시각
        self.window = []               # 최근 윈도우 내 도착 시각들

    def tick(self, now):
        if self.first_t is None:
            self.first_t = now
        self.last_t = now
        self.total += 1
        self.window.append(now)

    def window_hz(self, now, window_sec):
        # 윈도우 밖 샘플 제거
        cutoff = now - window_sec
        self.window = [t for t in self.window if t >= cutoff]
        n = len(self.window)
        if n < 2:
            return 0.0
        span = self.window[-1] - self.window[0]
        return (n - 1) / span if span > 0 else 0.0

    def mean_hz(self):
        if self.first_t is None or self.last_t is None or self.total < 2:
            return 0.0
        span = self.last_t - self.first_t
        return (self.total - 1) / span if span > 0 else 0.0


class HzMonitor(Node):
    def __init__(self):
        super().__init__("hz_monitor")

        self.declare_parameter("report_period", 2.0)   # 표 출력 주기 [s]
        self.declare_parameter("window_sec", 5.0)       # 순간 Hz 측정 윈도우 [s]
        self.declare_parameter("save_root", os.path.expanduser("~/ros2_trajectory_logs"))
        self.declare_parameter("run_name", "")
        self.declare_parameter("csv_name", "")   # 명시하면 save_root/csv_name 으로 바로 저장

        self.report_period = float(self.get_parameter("report_period").value)
        self.window_sec = float(self.get_parameter("window_sec").value)
        save_root = self.get_parameter("save_root").value
        run_name = self.get_parameter("run_name").value
        csv_name = self.get_parameter("csv_name").value

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if csv_name:
            os.makedirs(save_root, exist_ok=True)
            self.out_dir = save_root
            self.csv_path = os.path.join(save_root, csv_name)
        else:
            if not run_name:
                run_name = f"hz_{ts}"
            self.out_dir = os.path.join(save_root, run_name)
            os.makedirs(self.out_dir, exist_ok=True)
            self.csv_path = os.path.join(self.out_dir, f"hz_{ts}.csv")

        # 비침투성 QoS (센서/제어 토픽 모두 호환되도록 BEST_EFFORT, depth 1)
        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.meters = {}
        for name, topic, msg_type, nominal in DEFAULT_TOPICS:
            meter = TopicMeter(name, topic, nominal)
            self.meters[name] = meter
            # 기본값 인자로 묶어 늦은 바인딩 회피
            self.create_subscription(
                msg_type, topic,
                lambda msg, m=meter: m.tick(time.monotonic()),
                qos_be,
            )

        self.timer = self.create_timer(self.report_period, self._report)
        self.get_logger().info(f"[hz_monitor] window={self.window_sec}s, report every {self.report_period}s")
        self.get_logger().info(f"[hz_monitor] csv -> {self.csv_path}")

        self._finalized = False
        atexit.register(self._finalize)

    def _report(self):
        now = time.monotonic()
        lines = []
        lines.append("")
        lines.append(f"{'topic':<18}{'win_hz':>9}{'mean_hz':>9}{'count':>8}{'nominal':>9}")
        lines.append("-" * 53)
        for name, topic, msg_type, nominal in DEFAULT_TOPICS:
            m = self.meters[name]
            whz = m.window_hz(now, self.window_sec)
            mhz = m.mean_hz()
            nom = f"{nominal:.0f}" if nominal is not None else "-"
            lines.append(f"{name:<18}{whz:>9.2f}{mhz:>9.2f}{m.total:>8d}{nom:>9}")
        print("\n".join(lines))

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        now = time.monotonic()
        try:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["topic", "window_hz", "mean_hz", "total_count", "nominal_hz"])
                for name, topic, msg_type, nominal in DEFAULT_TOPICS:
                    m = self.meters[name]
                    w.writerow([
                        name,
                        f"{m.window_hz(now, self.window_sec):.3f}",
                        f"{m.mean_hz():.3f}",
                        m.total,
                        nominal if nominal is not None else "",
                    ])
            print(f"\n[hz_monitor] summary saved -> {self.csv_path}")
        except Exception as e:
            print(f"[hz_monitor] save error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = HzMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
