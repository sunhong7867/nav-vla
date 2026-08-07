#!/usr/bin/env python3
"""nav-vla 주행 대시보드 — VLA_AD operator dashboard의 nav-vla판.

한 창에:
    좌상  카메라 라이브 (기본 /image_raw/compressed)
    좌하  ★트랙맵 + 차량 위치 (/track/vehicle_pose_map, 궤적 트레일 포함)
    우측  instruction · 브리지 지연/큐/underrun · /cmd_vel · MotionCommand
          · pose 나이 · geofence — 그리고 E-STOP 버튼 (/operator/estop)

트랙맵은 track_localizer의 캐노니컬 프레임을 그대로 쓴다: 배경은 도안
템플릿(track2.png), 차량점은 /track/vehicle_pose_map(미터, +y up)을
m_per_px로 픽셀 변환 — 라이다 노트북이 붙어 있으면 실시간 위치가 찍히고,
없으면 'pose 없음'으로 표시된다.

    ros2 run thor_vehicle_pkg driving_dashboard
"""

import json
import math
import os
import threading
import time
from collections import deque

import numpy as np

# opencv가 Qt5 플러그인 경로를 export해 PySide6(Qt6)를 깨뜨린다 —
# lidar_bev_studio와 동일한 예방 조치.
import cv2  # noqa: E402
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
    if "cv2" in os.environ.get(_var, ""):
        os.environ.pop(_var)

import rclpy
from geometry_msgs.msg import Twist
from interfaces_pkg.msg import MotionCommand
from nav_msgs.msg import Odometry
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

from track_localizer_pkg.bev_detector import DetectorConfig
from track_localizer_pkg.track_pose_node import TemplateFrame, _resolve_config_path

CAM_W, MAP_W = 640, 460


class DashNode(Node):
    """구독 전담 — GUI는 이 노드의 최신값을 10 Hz로 읽어 그린다."""

    def __init__(self):
        super().__init__("driving_dashboard")
        self.image_topic = self.declare_parameter(
            "image_topic", "/image_raw/compressed").value
        sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        rel = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1)

        self.lock = threading.Lock()
        self.jpeg = None
        self.pose = None          # (x, y, yaw, mono_t)
        self.trail = deque(maxlen=600)
        self.status = {}
        self.instruction = ""
        self.cmd = (0.0, 0.0)
        self.motion = (0, 0)
        self.estop = False
        # 카메라: 실차는 CompressedImage. (시뮬 raw는 Gazebo GUI가 있으므로 비지원)
        self.create_subscription(
            CompressedImage, self.image_topic, self._img, sensor)
        self.create_subscription(
            Odometry, "/track/vehicle_pose_map", self._pose, sensor)
        self.create_subscription(String, "/vla/status", self._status, 10)
        self.create_subscription(String, "/vla/instruction", self._instr, latched)
        self.create_subscription(Twist, "/cmd_vel", self._cmd, rel)
        self.create_subscription(
            MotionCommand, "topic_control_signal", self._motion, rel)
        self.create_subscription(
            Bool, "/track/geofence_estop", self._geo, 10)
        self.op_estop_pub = self.create_publisher(Bool, "/operator/estop", 10)

    def _img(self, m):
        with self.lock:
            self.jpeg = bytes(m.data)

    def _pose(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z), 1 - 2 * q.z * q.z)
        with self.lock:
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y,
                         yaw, time.monotonic())
            self.trail.append((self.pose[0], self.pose[1]))

    def _status(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        with self.lock:
            self.status.update(d)

    def _instr(self, m):
        with self.lock:
            self.instruction = m.data

    def _cmd(self, m):
        with self.lock:
            self.cmd = (m.linear.x, m.angular.z)

    def _motion(self, m):
        with self.lock:
            self.motion = (m.steering, m.left_speed)

    def _geo(self, m):
        with self.lock:
            self.estop = bool(m.data)


class Dashboard(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.op_estop = False
        self.setWindowTitle("nav-vla 주행 대시보드")
        self.setStyleSheet(
            "QWidget{background:#14171d;color:#e8ebf2;font-size:13px}"
            "QLabel#head{font-size:15px;font-weight:600;color:#9fb4d8}"
            "QPushButton{background:#20262f;border:1px solid #2c3442;"
            "border-radius:6px;padding:8px}"
            "QPushButton#estop{background:#8b1a1a;font-size:18px;font-weight:700}")

        # 트랙맵 배경 + 좌표 변환 (track_localizer 캐노니컬 프레임 재사용)
        self.template_px = None
        self.map_bg = None
        try:
            homo = _resolve_config_path("alignment/track_map_aligned_homography.json")
            tf = TemplateFrame(homo, DetectorConfig())
            img_path = _resolve_config_path("alignment/track2.png")
            bg = cv2.imread(img_path)
            if tf.ok and bg is not None:
                self.m_per_px = tf.m_per_px
                self.height_px = tf.height_px
                self.map_scale = MAP_W / bg.shape[1]
                bg = cv2.resize(
                    bg, (MAP_W, int(bg.shape[0] * self.map_scale)))
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                self.map_bg = QImage(
                    bg.data, bg.shape[1], bg.shape[0], bg.strides[0],
                    QImage.Format_RGB888).copy()
        except Exception as e:  # noqa: BLE001
            print(f"트랙맵 로드 실패 (pose 패널은 무지도 동작): {e}")

        self.cam_label = QLabel("카메라 대기…")
        self.cam_label.setFixedSize(CAM_W, 480)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.map_label = QLabel("트랙맵")
        map_h = self.map_bg.height() if self.map_bg else 320
        self.map_label.setFixedSize(MAP_W, map_h)

        left = QVBoxLayout()
        left.addWidget(self.cam_label)
        left.addWidget(self.map_label)   # ★ 요청 위치: 왼쪽 아래
        left.addStretch(1)

        self.rows = {}
        right = QVBoxLayout()
        for key, title in (("instr", "명령"), ("bridge", "브리지"),
                           ("cmd", "cmd_vel"), ("motion", "MotionCommand"),
                           ("pose", "pose"), ("safety", "안전")):
            h = QLabel(title)
            h.setObjectName("head")
            v = QLabel("—")
            v.setWordWrap(True)
            right.addWidget(h)
            right.addWidget(v)
            self.rows[key] = v
        self.estop_btn = QPushButton("E-STOP")
        self.estop_btn.setObjectName("estop")
        self.estop_btn.clicked.connect(self._toggle_estop)
        right.addWidget(self.estop_btn)
        right.addStretch(1)

        lay = QHBoxLayout(self)
        lay.addLayout(left)
        lay.addLayout(right, 1)

        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(100)

    def _toggle_estop(self):
        self.op_estop = not self.op_estop
        self.node.op_estop_pub.publish(Bool(data=self.op_estop))
        self.estop_btn.setText("E-STOP 해제" if self.op_estop else "E-STOP")

    def _tick(self):
        n = self.node
        with n.lock:
            jpeg, pose, status = n.jpeg, n.pose, dict(n.status)
            instr, cmd, motion = n.instruction, n.cmd, n.motion
            geo = n.estop
            trail = list(n.trail)

        if jpeg:
            pm = QPixmap()
            if pm.loadFromData(jpeg, "JPEG"):
                self.cam_label.setPixmap(pm.scaled(
                    CAM_W, 480, Qt.KeepAspectRatio))

        self._draw_map(pose, trail, geo)

        self.rows["instr"].setText(instr or "(없음)")
        self.rows["bridge"].setText(
            f"지연 {status.get('latency_ms', 0):.0f} ms · "
            f"큐 {status.get('queue', '—')} · "
            f"underrun {status.get('underrun_pct', 0):.1f}%"
            + (f" · override {status['override']}" if "override" in status else ""))
        self.rows["cmd"].setText(f"v {cmd[0]:+.2f} m/s   w {cmd[1]:+.2f} rad/s")
        self.rows["motion"].setText(f"steer {motion[0]:+d}   speed {motion[1]}")
        if pose:
            age = time.monotonic() - pose[3]
            self.rows["pose"].setText(
                f"x {pose[0]:.2f}  y {pose[1]:.2f}  yaw {math.degrees(pose[2]):.0f}°"
                f"  나이 {age:.1f} s" + ("  ⚠ 유실" if age > 0.5 else ""))
        else:
            self.rows["pose"].setText("수신 없음 (라이다 노트북 링크 확인)")
        self.rows["safety"].setText(
            ("GEOFENCE ESTOP " if geo else "") +
            ("OPERATOR ESTOP" if self.op_estop else "") or "정상")
        self.rows["safety"].setStyleSheet(
            "color:#ff6b6b;font-weight:700" if (geo or self.op_estop)
            else "color:#7bd88f")

    def _draw_map(self, pose, trail, geo):
        if self.map_bg is None:
            self.map_label.setText("트랙맵 없음 — pose: " +
                                   (f"{pose[0]:.2f}, {pose[1]:.2f}" if pose else "—"))
            return
        pm = QPixmap.fromImage(self.map_bg)
        p = QPainter(pm)
        s = self.map_scale

        def to_px(x, y):
            return (x / self.m_per_px * s,
                    (self.height_px - y / self.m_per_px) * s)

        if len(trail) > 1:
            p.setPen(QPen(QColor("#4da3ff"), 2))
            pts = [to_px(x, y) for x, y in trail]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        if pose:
            px, py = to_px(pose[0], pose[1])
            stale = time.monotonic() - pose[3] > 0.5
            p.setPen(QPen(QColor("#ffce54" if stale else "#ff5544"), 3))
            p.setBrush(QColor("#ffce54" if stale else "#ff5544"))
            p.drawEllipse(int(px) - 6, int(py) - 6, 12, 12)
            hx = px + 18 * math.cos(-pose[2])
            hy = py + 18 * math.sin(-pose[2])
            p.drawLine(int(px), int(py), int(hx), int(hy))
        if geo:
            p.setPen(QPen(QColor("#ff5544"), 6))
            p.drawRect(3, 3, pm.width() - 6, pm.height() - 6)
        p.end()
        self.map_label.setPixmap(pm)


def main():
    rclpy.init()
    node = DashNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    app = QApplication([])
    w = Dashboard(node)
    w.show()
    app.exec()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
