"""
driving_dashboard_node.py — nav-vla Operator Dashboard (PySide6)

VLA_AD `operator_gui_pkg/dashboard_node.py`(선배 스택)의 충실 이식판.
디자인 토큰·레이아웃·위젯(카메라 3패널, 롤링 바, 채팅, 퀵버튼, Node Health)을
그대로 유지하고, 요청 사양대로 **좌측 하단을 트랙맵(라이다 측위 + 차량
위치)**으로 교체했다. 데이터 소스만 nav-vla 토픽으로 매핑:

    VlaIR 패널        → VLA Policy (instruction + /vla/status 지연/큐/underrun)
    TTL Gate 바       → Pose Age Gate (pose 나이 vs 0.5 s 유실 임계 — 동일 시각화)
    E2E Latency 바    → Policy Latency (/vla/status latency_ms, 기준선 300 ms)
    BehaviorState 배지 → DRIVING / ESTOP / POSE LOST
    좌하단 모션+채팅   → ★트랙맵 + 모션 라벨 | 채팅(→ /vla/instruction 직발행)

아키텍처 (원본 그대로):
  Qt main thread : QTimer 15Hz → deque 읽기 → UI 갱신
  daemon thread  : rclpy.spin — 콜백은 deque appendleft만 수행

실행:
  ros2 run thor_vehicle_pkg driving_dashboard
"""

import json
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# opencv가 Qt5 플러그인 경로를 export해 PySide6(Qt6)를 깨뜨린다.
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
    if "cv2" in os.environ.get(_var, ""):
        os.environ.pop(_var)

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage as RosCompressedImage, Image as RosImage
from std_msgs.msg import String, Bool
from interfaces_pkg.msg import MotionCommand, LaneInfo

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from track_localizer_pkg.bev_detector import DetectorConfig
from track_localizer_pkg.track_pose_node import TemplateFrame, _resolve_config_path

# ── 디자인 토큰 (원본 그대로) ───────────────────────────────────────────────
C_BG     = "#0d1117"
C_BG2    = "#161b22"
C_BG3    = "#21262d"
C_BORDER = "#30363d"
C_TEXT   = "#e6edf3"
C_DIM    = "#8b949e"
C_GREEN  = "#3fb950"
C_YELLOW = "#d29922"
C_RED    = "#f85149"
C_BLUE   = "#58a6ff"

MODE_COLORS = {
    "DRIVING":   C_GREEN,
    "IDLE":      C_DIM,
    "POSE LOST": C_YELLOW,
    "ESTOP":     C_RED,
}

GLOBAL_QSS = f"""
QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: "Noto Sans KR", "Segoe UI", "Ubuntu", monospace;
}}
QGroupBox {{
    background-color: {C_BG2};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px 8px 6px 8px;
    font-weight: bold;
    font-size: 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {C_DIM};
}}
QLineEdit {{
    background-color: {C_BG3};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 6px 10px;
    color: {C_TEXT};
    font-size: 13px;
}}
QPushButton {{
    background-color: {C_BG3};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 5px 12px;
    color: {C_TEXT};
    font-size: 12px;
    min-height: 28px;
}}
QPushButton:hover {{
    background-color: {C_BORDER};
}}
QPushButton:pressed {{
    background-color: {C_BLUE};
    color: black;
}}
"""

POSE_STALE_S = 0.5      # 마스터플랜 §4.4 pose 유실 정지 규칙과 동일 임계
POLICY_REF_MS = 300.0   # Thor 실측 233~299 ms — 이 위면 경고색


# ═══════════════════════════════════════════════════════════════════════════════
# ROS2 백엔드 노드 (daemon thread) — 원본 구조, 토픽만 nav-vla
# ═══════════════════════════════════════════════════════════════════════════════
class DashboardROSNode(Node):
    def __init__(self):
        super().__init__("driving_dashboard")

        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        qos_best = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # deque 버퍼 (항상 최신 1개만)
        self.buf_scene      = deque(maxlen=1)   # raw 카메라
        self.buf_bev_color  = deque(maxlen=1)   # BEV 컬러 (선생 스택 병행 시)
        self.buf_debug_img  = deque(maxlen=1)   # debug overlay
        self.buf_status     = deque(maxlen=1)   # /vla/status JSON
        self.buf_instr      = deque(maxlen=1)
        self.buf_cmd_vel    = deque(maxlen=1)
        self.buf_cmd        = deque(maxlen=1)
        self.buf_lane_info  = deque(maxlen=1)
        self.buf_pose       = deque(maxlen=1)   # /track/vehicle_pose_map
        self.buf_geofence   = deque(maxlen=1)
        self.pose_trail     = deque(maxlen=600)
        self.last_pose_mono = None

        # 구독
        self.create_subscription(RosCompressedImage, "image_raw/compressed",
                                 lambda m: self.buf_scene.appendleft(m.data), qos_best)
        self.create_subscription(RosCompressedImage, "bev/color/compressed",
                                 lambda m: self.buf_bev_color.appendleft(m.data), qos_best)
        # ★좌하단: 노트북의 aligned_bev_publisher가 보내는 정합 BEV
        self.buf_bev_aligned = deque(maxlen=1)
        self.last_bev_aligned_mono = None
        self.create_subscription(RosCompressedImage, "/track/aligned_bev/compressed",
                                 self._cb_bev_aligned, qos_best)
        self.create_subscription(RosImage, "debug/pipeline_overlay",
                                 self._cb_debug_img, qos_reliable)
        self.create_subscription(String, "/vla/status",
                                 self._cb_status, qos_reliable)
        self.create_subscription(String, "/vla/instruction",
                                 lambda m: self.buf_instr.appendleft(m.data), qos_latched)
        self.create_subscription(Twist, "/cmd_vel",
                                 lambda m: self.buf_cmd_vel.appendleft(m), qos_reliable)
        self.create_subscription(MotionCommand, "topic_control_signal",
                                 lambda m: self.buf_cmd.appendleft(m), qos_reliable)
        self.create_subscription(LaneInfo, "yolov8_lane_info",
                                 lambda m: self.buf_lane_info.appendleft(m), qos_reliable)
        self.create_subscription(Odometry, "/track/vehicle_pose_map",
                                 self._cb_pose, qos_best)
        self.create_subscription(Bool, "/track/geofence_estop",
                                 lambda m: self.buf_geofence.appendleft(bool(m.data)), 10)

        # 발행
        self._instr_pub = self.create_publisher(String, "/vla/instruction", qos_latched)
        self._estop_pub = self.create_publisher(Bool, "operator/estop", qos_reliable)
        self._lane_override_pub = self.create_publisher(
            String, "lane/manual_override", qos_reliable)
        self._estop_active = False

        # 노드 alive 추적
        self.node_alive = {}

        self.get_logger().info("Dashboard ROS node ready")

    def _cb_bev_aligned(self, msg: RosCompressedImage):
        self.buf_bev_aligned.appendleft(msg.data)
        self.last_bev_aligned_mono = time.monotonic()

    def _cb_debug_img(self, msg: RosImage):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            _, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.buf_debug_img.appendleft(jpg.tobytes())
        except Exception:
            pass

    def _cb_status(self, msg: String):
        try:
            self.buf_status.appendleft(json.loads(msg.data))
        except Exception:
            pass

    def _cb_pose(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z), 1 - 2 * q.z * q.z)
        p = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.buf_pose.appendleft(p)
        self.pose_trail.append((p[0], p[1]))
        self.last_pose_mono = time.monotonic()

    def send_instruction(self, text: str):
        self._instr_pub.publish(String(data=text))

    def send_estop(self, activate: bool):
        self._estop_pub.publish(Bool(data=activate))
        self._estop_active = activate

    def send_lane_override(self, lane: str):
        self._lane_override_pub.publish(String(data=lane))


# ═══════════════════════════════════════════════════════════════════════════════
# 카메라 위젯 (원본 그대로)
# ═══════════════════════════════════════════════════════════════════════════════
class CameraWidget(QLabel):
    def __init__(self, title="Camera", width=420, height=315, parent=None):
        super().__init__(parent)
        self._title = title
        self._w = width
        self._h = height
        self.setFixedSize(width, height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background: {C_BG3}; border: 1px solid {C_BORDER}; border-radius: 4px;")
        self._set_placeholder()

    def _set_placeholder(self):
        self.setText(f"{self._title}\nWaiting...")

    def update_frame(self, jpeg_bytes, border_color=None):
        try:
            arr = np.frombuffer(jpeg_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            img = cv2.resize(img, (self._w, self._h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img.data, img.shape[1], img.shape[0],
                          img.strides[0], QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            if border_color:
                painter = QPainter(pix)
                pen = QPen(QColor(border_color), 3)
                painter.setPen(pen)
                painter.drawRect(1, 1, pix.width() - 2, pix.height() - 2)
                painter.end()

            self.setPixmap(pix)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# ★트랙맵 위젯 — 좌측 하단. 도안 템플릿 + 라이다 측위 차량점/궤적
# ═══════════════════════════════════════════════════════════════════════════════
class TrackMapWidget(QLabel):
    def __init__(self, width=420, height=300, parent=None):
        super().__init__(parent)
        self._w, self._h = width, height
        self.setFixedSize(width, height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background: {C_BG3}; border: 1px solid {C_BORDER}; border-radius: 4px;")
        self.setText("Track Map\nWaiting...")

        self._bg = None
        self._m_per_px = None
        self._height_px = None
        self._scale = 1.0
        try:
            homo = _resolve_config_path("alignment/track_map_aligned_homography.json")
            tf = TemplateFrame(homo, DetectorConfig())
            img = cv2.imread(_resolve_config_path("alignment/track2.png"))
            if tf.ok and img is not None:
                self._m_per_px = tf.m_per_px
                self._height_px = tf.height_px
                # 위젯 안에 맞는 스케일 (여백 포함)
                self._scale = min(width / img.shape[1], height / img.shape[0])
                img = cv2.resize(img, (int(img.shape[1] * self._scale),
                                       int(img.shape[0] * self._scale)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._bg = QImage(img.data, img.shape[1], img.shape[0],
                                  img.strides[0], QImage.Format.Format_RGB888).copy()
        except Exception as e:  # noqa: BLE001
            print(f"track map load failed: {e}")

    def _to_px(self, x, y):
        px = x / self._m_per_px * self._scale
        py = (self._height_px - y / self._m_per_px) * self._scale
        return px, py

    def update_map(self, pose, trail, stale, geofence):
        if self._bg is None:
            self.setText("Track Map\n(정합 파일 없음)"
                         + (f"\npose: {pose[0]:.2f}, {pose[1]:.2f}" if pose else ""))
            return
        pix = QPixmap.fromImage(self._bg)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if len(trail) > 1:
            painter.setPen(QPen(QColor(C_BLUE), 2))
            pts = [self._to_px(x, y) for x, y in trail]
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))

        if pose:
            px, py = self._to_px(pose[0], pose[1])
            color = QColor(C_YELLOW if stale else C_RED)
            painter.setPen(QPen(color, 3))
            painter.setBrush(color)
            painter.drawEllipse(int(px) - 6, int(py) - 6, 12, 12)
            hx = px + 18 * math.cos(-pose[2])
            hy = py + 18 * math.sin(-pose[2])
            painter.drawLine(int(px), int(py), int(hx), int(hy))

        if geofence:
            painter.setPen(QPen(QColor(C_RED), 5))
            painter.drawRect(2, 2, pix.width() - 4, pix.height() - 4)
        painter.end()
        self.setPixmap(pix)


# ═══════════════════════════════════════════════════════════════════════════════
# 레이턴시 바 위젯 (원본 — 기준선만 인자화: 카메라 33 ms / 정책 300 ms)
# ═══════════════════════════════════════════════════════════════════════════════
class LatencyBarWidget(QWidget):
    def __init__(self, ref_ms=1000.0 / 30.0, ref_label="1 frame @ 30 Hz", parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self._data = deque(maxlen=100)
        self._frame_interval = ref_ms
        self._ref_label = ref_label

    def add_sample(self, e2e_ms):
        self._data.append(e2e_ms)
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        max_val = max(max(self._data), self._frame_interval * 4, 1)
        bar_w = max(2, w / 100)

        for i, val in enumerate(self._data):
            x = int(i * bar_w)
            bh = int(val / max_val * (h - 20))
            if val <= self._frame_interval:
                color = QColor(C_GREEN)
            elif val <= self._frame_interval * 3:
                color = QColor(C_YELLOW)
            else:
                color = QColor(C_RED)
            painter.fillRect(int(x), h - 15 - bh, max(int(bar_w) - 1, 1), bh, color)

        y_ref = h - 15 - int(self._frame_interval / max_val * (h - 20))
        painter.setPen(QPen(QColor(C_DIM), 1, Qt.PenStyle.DashLine))
        painter.drawLine(0, y_ref, w, y_ref)

        arr = list(self._data)
        now = arr[-1]
        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))
        painter.setPen(QColor(C_DIM))
        painter.setFont(QFont("monospace", 9))
        painter.drawText(
            5, h - 2,
            f"now={now:.0f}ms  P50={p50:.0f}ms  P99={p99:.0f}ms  "
            f"(ref: {self._ref_label} = {self._frame_interval:.0f}ms)"
        )

        painter.end()


# ═══════════════════════════════════════════════════════════════════════════════
# Pose Age Gate 위젯 — 원본 TTLBarWidget 재사용, 기준 = pose 유실 임계 0.5 s
# ═══════════════════════════════════════════════════════════════════════════════
class TTLBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self._data = deque(maxlen=60)
        self._ttl = POSE_STALE_S * 1000.0
        self._n_discard = 0

    def add_sample(self, age_ms, ttl_ms, n_discard=None):
        self._data.append(float(age_ms))
        self._ttl = float(ttl_ms)
        if n_discard is not None:
            self._n_discard = int(n_discard)
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        max_val = max(max(self._data), self._ttl * 1.3, 1.0)
        bar_w = max(2, w / max(1, len(self._data)))

        for i, val in enumerate(self._data):
            x = int(i * bar_w)
            bh = int(val / max_val * (h - 20))
            color = QColor(C_GREEN) if val <= self._ttl else QColor(C_RED)
            painter.fillRect(int(x), h - 15 - bh, max(int(bar_w) - 1, 1), bh, color)

        y_ttl = h - 15 - int(self._ttl / max_val * (h - 20))
        painter.setPen(QPen(QColor(C_RED), 2, Qt.PenStyle.SolidLine))
        painter.drawLine(0, y_ttl, w, y_ttl)

        arr = list(self._data)
        now = arr[-1]
        p99 = float(np.percentile(arr, 99))
        status = "VALID" if now <= self._ttl else "STALE"
        painter.setPen(QColor(C_DIM))
        painter.setFont(QFont("monospace", 9))
        painter.drawText(
            5, h - 2,
            f"age now={now:.0f}ms  P99={p99:.0f}ms  limit={self._ttl:.0f}ms  [{status}]"
        )

        painter.end()


# ═══════════════════════════════════════════════════════════════════════════════
# 메인 윈도우 — 원본 레이아웃, 좌하단만 트랙맵
# ═══════════════════════════════════════════════════════════════════════════════
class DashboardWindow(QMainWindow):
    def __init__(self, ros_node: DashboardROSNode):
        super().__init__()
        self._ros = ros_node

        self.setWindowTitle("nav-vla Operator Dashboard")
        self.setFixedSize(1500, 960)
        self.setStyleSheet(GLOBAL_QSS)

        self._mode = "IDLE"
        self._instruction = ""
        self._status = {}
        self._pose = None
        self._geofence = False
        self._estop = False
        self._motion = (0, 0, 0)   # steering, left, right

        self._build_ui()
        self._start_timer()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── 상단: 타이틀 바 ──
        title_bar = QHBoxLayout()
        title_lbl = QLabel("nav-vla Operator Dashboard")
        title_lbl.setFont(QFont("monospace", 16, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {C_BLUE};")
        title_bar.addWidget(title_lbl)
        title_bar.addStretch()

        self._behavior_badge = QLabel("IDLE")
        self._behavior_badge.setFont(QFont("monospace", 12, QFont.Weight.Bold))
        self._behavior_badge.setStyleSheet(
            f"background: {C_DIM}; color: black; padding: 4px 12px; border-radius: 4px;")
        title_bar.addWidget(self._behavior_badge)

        self._time_label = QLabel()
        self._time_label.setFont(QFont("monospace", 12))
        self._time_label.setStyleSheet(f"color: {C_DIM};")
        title_bar.addWidget(self._time_label)
        main_layout.addLayout(title_bar)

        # ── 중단: 좌(카메라·맵) / 우(상태) 스플리터 ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # 카메라 3패널 (원본 그대로 — 선생 스택 병행 시 BEV/Lane도 살아남)
        cam_row = QHBoxLayout()
        self._cam_scene = CameraWidget("Raw Camera", 420, 315)
        cam_row.addWidget(self._cam_scene)
        self._cam_bev = CameraWidget("BEV Color", 420, 315)
        cam_row.addWidget(self._cam_bev)
        self._cam_debug = CameraWidget("Lane + Path", 420, 315)
        cam_row.addWidget(self._cam_debug)
        left_layout.addLayout(cam_row)

        # ★좌하단: 정합 BEV (라이브 점군 + 트랙 라인 — 스튜디오와 같은 화면).
        # BEV 스트림이 없을 때(노트북 미연결)는 도안+pose 트랙맵으로 폴백.
        from PySide6.QtWidgets import QStackedWidget
        bottom_split = QHBoxLayout()

        self._bev_aligned = CameraWidget("Aligned BEV (LiDAR)", 420, 340)
        self._track_map = TrackMapWidget(420, 340)
        self._map_stack = QStackedWidget()
        self._map_stack.setFixedSize(420, 340)
        self._map_stack.addWidget(self._bev_aligned)   # index 0: 정합 BEV
        self._map_stack.addWidget(self._track_map)     # index 1: 폴백 트랙맵
        self._map_stack.setCurrentIndex(1)
        bottom_split.addWidget(self._map_stack)

        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(4, 0, 0, 0)
        chat_layout.setSpacing(2)

        self._chat_log = QListWidget()
        self._chat_log.setFont(QFont("monospace", 10))
        self._chat_log.setStyleSheet(
            f"background: {C_BG2}; color: {C_DIM}; border: 1px solid {C_BORDER}; border-radius: 4px;")
        chat_layout.addWidget(self._chat_log, stretch=1)

        chat_input_row = QHBoxLayout()
        self._cmd_input = QLineEdit()
        self._cmd_input.setPlaceholderText(
            "학습 문장 입력 → /vla/instruction  (예: Start driving in the inner lane, at a slow speed.)")
        self._cmd_input.returnPressed.connect(self._send_command)
        chat_input_row.addWidget(self._cmd_input, stretch=1)

        send_btn = QPushButton("SEND")
        send_btn.setStyleSheet(f"background: {C_BLUE}; color: black; font-weight: bold;")
        send_btn.clicked.connect(self._send_command)
        chat_input_row.addWidget(send_btn)
        chat_layout.addLayout(chat_input_row)

        bottom_split.addWidget(chat_panel, stretch=1)
        left_layout.addLayout(bottom_split, stretch=1)

        splitter.addWidget(left_panel)

        # ── 우측: 상태 패널 (원본 구조) ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        # 차선 패널 (원본 그대로 — 시뮬 선생 스택에서 동작)
        lane_group = QGroupBox("Lane Detection")
        lane_lay = QVBoxLayout(lane_group)

        curr_row = QHBoxLayout()
        curr_label = QLabel("Current:")
        curr_label.setFont(QFont("monospace", 11))
        curr_label.setStyleSheet(f"color: {C_DIM};")
        curr_label.setFixedWidth(70)
        curr_row.addWidget(curr_label)
        self._lane_badge = QLabel("unknown")
        self._lane_badge.setFont(QFont("monospace", 14, QFont.Weight.Bold))
        self._lane_badge.setAlignment(Qt.AlignCenter)
        self._lane_badge.setFixedHeight(32)
        self._lane_badge.setStyleSheet(
            f"background: {C_BG3}; border-radius: 6px; padding: 4px;")
        curr_row.addWidget(self._lane_badge)
        lane_lay.addLayout(curr_row)

        force_row = QHBoxLayout()
        force_label = QLabel("Force:")
        force_label.setFont(QFont("monospace", 11))
        force_label.setStyleSheet(f"color: {C_DIM};")
        force_label.setFixedWidth(70)
        force_row.addWidget(force_label)
        for text, arg in (("Lane 1", "lane1"), ("Lane 2", "lane2"), ("Auto", "auto")):
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked=False, a=arg: self._force_lane(a))
            force_row.addWidget(btn)
        lane_lay.addLayout(force_row)
        right_layout.addWidget(lane_group)

        # VLA Policy 패널 (원본 VlaIR 패널 자리)
        vla_group = QGroupBox("VLA Policy (SmolVLA)")
        vla_lay = QVBoxLayout(vla_group)

        self._vla_mode_badge = QLabel("—")
        self._vla_mode_badge.setFont(QFont("monospace", 18, QFont.Weight.Bold))
        self._vla_mode_badge.setAlignment(Qt.AlignCenter)
        self._vla_mode_badge.setFixedHeight(40)
        self._vla_mode_badge.setStyleSheet(
            f"background: {C_BG3}; border-radius: 6px; padding: 4px;")
        vla_lay.addWidget(self._vla_mode_badge)

        self._vla_details = QLabel("No policy data")
        self._vla_details.setFont(QFont("monospace", 11))
        self._vla_details.setStyleSheet(f"color: {C_DIM};")
        self._vla_details.setWordWrap(True)
        vla_lay.addWidget(self._vla_details)

        self._vla_reasoning = QLabel("")
        self._vla_reasoning.setFont(QFont("monospace", 10))
        self._vla_reasoning.setStyleSheet(f"color: {C_DIM}; font-style: italic;")
        self._vla_reasoning.setWordWrap(True)
        vla_lay.addWidget(self._vla_reasoning)
        right_layout.addWidget(vla_group)

        # Pose Age Gate (원본 TTL Gate 자리 — 같은 시각화, 기준 = 0.5 s)
        ttl_group = QGroupBox("Pose Age Gate (age vs 0.5 s loss rule)")
        ttl_lay = QVBoxLayout(ttl_group)
        self._ttl_bar = TTLBarWidget()
        ttl_lay.addWidget(self._ttl_bar)
        right_layout.addWidget(ttl_group)

        # Policy Latency (원본 E2E Latency 자리)
        lat_group = QGroupBox("Policy Latency (/vla/status)")
        lat_lay = QVBoxLayout(lat_group)
        self._latency_bar = LatencyBarWidget(
            ref_ms=POLICY_REF_MS, ref_label="Thor idle P50 233 ms + margin")
        lat_lay.addWidget(self._latency_bar)
        right_layout.addWidget(lat_group)

        # Node Health (원본 그대로, 항목만 nav-vla)
        health_group = QGroupBox("Node Health")
        health_lay = QVBoxLayout(health_group)
        self._health_label = QLabel("Waiting for data...")
        self._health_label.setFont(QFont("monospace", 10))
        self._health_label.setStyleSheet(f"color: {C_DIM};")
        health_lay.addWidget(self._health_label)
        right_layout.addWidget(health_group)

        right_layout.addStretch()
        splitter.addWidget(right_panel)
        splitter.setSizes([1300, 400])
        main_layout.addWidget(splitter, stretch=1)

        # ── 하단: 퀵 버튼 (원본 구조, 명령은 nav-vla 학습 문장) ──
        quick_row = QHBoxLayout()
        quick_cmds = [
            ("STOP", f"background: {C_RED}; color: white; font-weight: bold;"),
            ("Resume", ""),
            ("Slow", ""),
            ("Normal", ""),
            ("Fast", ""),
        ]
        for text, style in quick_cmds:
            btn = QPushButton(text)
            if style:
                btn.setStyleSheet(style)
            btn.clicked.connect(lambda checked=False, t=text: self._quick_command(t))
            quick_row.addWidget(btn)
        main_layout.addLayout(quick_row)

    def _start_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(67)  # ~15Hz (원본 그대로)

    # ── 15Hz UI 갱신 ──────────────────────────────────────────────────────────
    def _tick(self):
        now = time.time()
        now_mono = time.monotonic()

        from datetime import datetime
        self._time_label.setText(datetime.now().strftime("%H:%M:%S"))

        # Scene 카메라 (배지 색 테두리 — 원본과 동일한 모드 연동)
        if self._ros.buf_scene:
            data = self._ros.buf_scene.popleft()
            self._cam_scene.update_frame(data, MODE_COLORS.get(self._mode))
            self._ros.node_alive["scene"] = now
        if self._ros.buf_bev_color:
            self._cam_bev.update_frame(self._ros.buf_bev_color.popleft())
            self._ros.node_alive["bev"] = now
        if self._ros.buf_debug_img:
            self._cam_debug.update_frame(self._ros.buf_debug_img.popleft())
            self._ros.node_alive["debug"] = now

        # Lane info (필드셋이 달라도 죽지 않게 getattr)
        if self._ros.buf_lane_info:
            lane_msg = self._ros.buf_lane_info.popleft()
            lane = getattr(lane_msg, "current_lane", "unknown")
            lane_color = C_BLUE if lane != "unknown" else C_DIM
            self._lane_badge.setText(str(lane))
            self._lane_badge.setStyleSheet(
                f"background: {lane_color}; color: black; border-radius: 6px; "
                f"font-size: 14px; font-weight: bold;")
            self._ros.node_alive["lane"] = now

        # /vla/status → VLA Policy 패널 + Policy Latency 바
        if self._ros.buf_status:
            st = self._ros.buf_status.popleft()
            self._status.update(st)
            if "latency_ms" in st:
                self._latency_bar.add_sample(float(st["latency_ms"]))
            self._ros.node_alive["status"] = now
        if self._ros.buf_instr:
            self._instruction = self._ros.buf_instr.popleft()
            self._append_chat("<", f"[INSTR] {self._instruction or '(정지)'}", C_GREEN)

        # 모드 배지: estop > pose lost > driving > idle
        # cmd_vel은 10 Hz 발행 vs 15 Hz 틱이라 빈 틱이 생긴다 — 최근 1 s
        # 캐시로 표시가 0으로 깜빡이는 것을 막는다.
        if self._ros.buf_cmd_vel:
            self._last_cmd_vel = self._ros.buf_cmd_vel.popleft()
            self._last_cmd_vel_t = now
            self._ros.node_alive["cmd_vel"] = now
        cmd_vel = (self._last_cmd_vel
                   if getattr(self, "_last_cmd_vel_t", 0) > now - 1.0 else None)
        pose_age_s = (time.monotonic() - self._ros.last_pose_mono
                      if self._ros.last_pose_mono else None)
        if self._ros.buf_geofence:
            self._geofence = self._ros.buf_geofence.popleft()
        if self._estop or self._geofence:
            self._mode = "ESTOP"
        elif pose_age_s is not None and pose_age_s > POSE_STALE_S:
            self._mode = "POSE LOST"
        elif cmd_vel is not None and abs(cmd_vel.linear.x) > 0.05:
            self._mode = "DRIVING"
        elif self._mode not in ("ESTOP",):
            self._mode = "IDLE"
        color = MODE_COLORS.get(self._mode, C_DIM)
        self._behavior_badge.setText(self._mode)
        self._behavior_badge.setStyleSheet(
            f"background: {color}; color: black; padding: 4px 12px; border-radius: 4px;")

        self._vla_mode_badge.setText(self._mode)
        self._vla_mode_badge.setStyleSheet(
            f"background: {color}; color: black; border-radius: 6px; "
            f"font-size: 18px; font-weight: bold;")
        v = cmd_vel.linear.x if cmd_vel else 0.0
        w = cmd_vel.angular.z if cmd_vel else 0.0
        st, ls, rs = self._motion
        self._vla_details.setText(
            f"latency: {self._status.get('latency_ms', 0):.0f} ms    "
            f"queue: {self._status.get('queue', '—')}/30\n"
            f"underrun: {self._status.get('underrun_pct', 0):.2f} %    "
            f"cmd_vel: v {v:+.2f}  w {w:+.2f}\n"
            f"Steer: {st:+d} ({st * 5:+d}°)    L: {ls}    R: {rs}")
        self._vla_reasoning.setText(self._instruction)

        # ★좌하단: 정합 BEV 신선하면 그것, 아니면 도안+pose 트랙맵 폴백
        bev_fresh = (self._ros.last_bev_aligned_mono is not None
                     and now_mono - self._ros.last_bev_aligned_mono < 2.0)
        if self._ros.buf_bev_aligned:
            self._bev_aligned.update_frame(
                self._ros.buf_bev_aligned.popleft(),
                C_RED if self._geofence else None)
            self._ros.node_alive["bev_map"] = now
        self._map_stack.setCurrentIndex(0 if bev_fresh else 1)

        # Pose → (폴백) 트랙맵 + Age Gate
        pose = self._ros.buf_pose[0] if self._ros.buf_pose else None
        if pose:
            self._pose = pose
            self._ros.node_alive["pose"] = now
        stale = pose_age_s is None or pose_age_s > POSE_STALE_S
        if not bev_fresh:
            self._track_map.update_map(
                self._pose, list(self._ros.pose_trail), stale, self._geofence)
        if pose_age_s is not None:
            self._ttl_bar.add_sample(pose_age_s * 1000.0, POSE_STALE_S * 1000.0)

        # Motion command (라벨 제거 — VLA Policy 패널에 표기)
        if self._ros.buf_cmd:
            cmd = self._ros.buf_cmd.popleft()
            self._motion = (cmd.steering, cmd.left_speed, cmd.right_speed)
            self._ros.node_alive["cmd"] = now

        # Node health
        health_parts = []
        for name in ["scene", "bev", "debug", "lane", "status",
                     "cmd_vel", "pose", "cmd", "bev_map"]:
            last = self._ros.node_alive.get(name, 0)
            alive = (now - last) < 3.0
            dot = (f"<span style='color:{C_GREEN}'>●</span>" if alive
                   else f"<span style='color:{C_RED}'>○</span>")
            health_parts.append(f"{dot} {name}")
        self._health_label.setText("  ".join(health_parts))

    # ── 명령 전송 ─────────────────────────────────────────────────────────────
    def _append_chat(self, prefix: str, text: str, color: str = None):
        c = color or C_DIM
        item = QListWidgetItem(f"{prefix} {text}")
        item.setForeground(QColor(c))
        self._chat_log.addItem(item)
        self._chat_log.scrollToBottom()

    def _send_command(self):
        text = self._cmd_input.text().strip()
        if not text:
            return
        self._ros.send_instruction(text)
        self._append_chat(">", text, C_BLUE)
        self._cmd_input.clear()

    def _force_lane(self, lane: str):
        self._ros.send_lane_override(lane)
        color = C_BLUE if lane in ("lane1", "lane2") else C_YELLOW
        self._append_chat(">", f"[LANE] force → {lane}", color)

    def _quick_command(self, cmd_text: str):
        # STOP = 운영자 estop(어댑터 게이트) + 빈 문장(브리지 정지) 이중 정지.
        # Resume 은 estop 만 해제한다 — 자동 재주행 금지, 문장은 사람이 입력.
        if cmd_text == "STOP":
            self._ros.send_estop(True)
            self._ros.send_instruction("")
            self._estop = True
            self._append_chat(">", "[STOP] estop + 정지 문장", C_RED)
            return
        if cmd_text == "Resume":
            self._ros.send_estop(False)
            self._estop = False
            self._append_chat(">", "[Resume] estop 해제 — 주행 문장을 입력하세요", C_YELLOW)
            return
        sentence = {
            "Slow":   "Start driving in the inner lane, at a slow speed.",
            "Normal": "Start driving in the inner lane, at a normal speed.",
            "Fast":   "Start driving in the inner lane, at a fast speed.",
        }.get(cmd_text)
        if sentence:
            self._ros.send_instruction(sentence)
            self._append_chat(">", f"[{cmd_text}] {sentence}", C_BLUE)


# ═══════════════════════════════════════════════════════════════════════════════
# main (원본 그대로)
# ═══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    ros_node = DashboardROSNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DashboardWindow(ros_node)
    window.show()

    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
