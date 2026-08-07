#!/usr/bin/env python3
"""LiDAR BEV Studio — PySide6 live viewer + track-map 4-point alignment.

라이브 LiDAR BEV 뷰와 트랙맵 4점 정합을 하나로 합친 GUI.
정합은 점을 다 찍어도 자동 반영되지 않고, ‘최종 확인’을 눌러야 저장·적용된다.

실행:
    python3 lidar_bev_studio.py            # 기본: /lidar_points
    python3 lidar_bev_studio.py --topic /lidar_points
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

# opencv-python exports its bundled Qt5 plugin dir on import, which makes
# PySide6 (Qt6) try to load an incompatible platform plugin and crash.
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
    if "cv2" in os.environ.get(_var, ""):
        os.environ.pop(_var)

from PySide6.QtCore import Qt, QRect, QRectF, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QRadioButton,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRubberBand,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from bev_pipeline import (
    BevGeometry,
    DetectionParams,
    OverlayError,
    VehicleTracker,
    _world_to_pixel_float,
    build_exclusion_mask,
    load_track_overlay,
    render_alignment_preview,
    save_alignment,
)
from layout_rendering import blend_layout_lines

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="PySide6 LiDAR BEV studio")
    parser.add_argument("--topic", default="/lidar_points")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--forward-axis", choices=["-y", "+y", "+x", "-x"], default="+y")
    parser.add_argument(
        "--full-range",
        type=float,
        default=25.0,
        help="전체 보기에서 각 축으로 ±이 값(m)까지 표시.",
    )
    parser.add_argument("--forward-min", type=float, default=None)
    parser.add_argument("--forward-max", type=float, default=None)
    parser.add_argument("--lateral-min", type=float, default=None)
    parser.add_argument("--lateral-max", type=float, default=None)
    parser.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="BEV yaw 회전(도). GUI의 회전 스핀박스로도 조절 가능.",
    )
    parser.add_argument("--z-min", type=float, default=-2.2)
    parser.add_argument("--z-max", type=float, default=0.6)
    parser.add_argument(
        "--accumulate",
        type=int,
        default=2,
        help="표시용 프레임 누적 수. 1이면 잔상 없음, 크면 밝지만 움직이는 차가 번짐.",
    )
    parser.add_argument("--intensity-max", type=float, default=120.0)
    parser.add_argument(
        "--render-scale",
        type=float,
        default=3.0,
        help="표시용 렌더 배율 상한. 오버레이 선을 캔버스보다 높은 해상도로 그린다.",
    )
    parser.add_argument(
        "--render-max-dim",
        type=int,
        default=1600,
        help="렌더 이미지 최대 변 길이(px). 전체 뷰처럼 큰 캔버스에서 배율을 자동 제한.",
    )
    parser.add_argument("--snapshot-frames", type=int, default=40)
    parser.add_argument("--snapshot-intensity-max", type=float, default=50.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.65)
    parser.add_argument("--track-map", default=str(SCRIPT_DIR / "assets" / "track_map.png"))
    parser.add_argument(
        "--homography-json",
        default=str(SCRIPT_DIR / "output" / "config" / "track_map_aligned_homography.json"),
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "output"))
    parser.add_argument(
        "--zones-json",
        default=str(SCRIPT_DIR / "output" / "config" / "track_zones.json"),
        help="구역(차선·기준선·랩타이머·신호등) 저장 파일.",
    )
    parser.add_argument(
        "--exclusions-json",
        default=str(SCRIPT_DIR / "output" / "config" / "exclusions.json"),
        help="우클릭으로 등록한 후보 제외 영역 저장 파일.",
    )
    parser.add_argument(
        "--detect-vehicle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vehicle-height-min", type=float, default=0.03)
    parser.add_argument("--vehicle-height-max", type=float, default=0.35)
    parser.add_argument("--min-cluster-pixels", type=int, default=3)
    parser.add_argument("--max-cluster-pixels", type=int, default=500)
    parser.add_argument(
        "--bg-subtraction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="(비권장) 적응형 배경 차분 — 정지 차량 지정을 방해하므로 기본 꺼짐.",
    )
    parser.add_argument(
        "--bg-alpha",
        type=float,
        default=0.015,
        help="배경 학습 속도. 클수록 정지 물체가 더 빨리 배경으로 흡수됨.",
    )
    parser.add_argument("--confirm-detections", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# ROS worker thread
# ---------------------------------------------------------------------------


class RosWorker(QThread):
    frame_ready = Signal(QImage, object)
    snapshot_progress = Signal(int, int)
    snapshot_ready = Signal(object)
    log = Signal(str)

    def __init__(self, geometry, det_params, args, parent=None):
        super().__init__(parent)
        self.geom = geometry
        self.args = args
        self.tracker = VehicleTracker(geometry, det_params, log=self.log.emit)
        self.overlay_alpha = float(args.overlay_alpha)
        self.overlay_enabled = True
        self.detect_enabled = bool(det_params.detect_vehicle)
        self._overlay = None
        self._overlay_lock = threading.Lock()
        self._frames = deque(maxlen=max(1, args.accumulate))
        self._stop = False
        self._snap_buf = None
        self._snap_count = 0
        self._snap_target = 0
        self._last_msg_time_ns = None
        self._fps = 0.0
        self._last_frame_monotonic = None
        self._render_scale = self._compute_render_scale()
        self.zones = []
        self.zones_enabled = True
        self.debug_candidates = True
        self._valid_ema = None
        self._degenerate_frames = 0

    def _compute_render_scale(self):
        limit = float(self.args.render_max_dim)
        scale = min(float(self.args.render_scale), limit / max(self.geom.width, self.geom.height))
        return max(1.0, scale)

    def current_render_scale(self):
        return self._render_scale

    # -- called from GUI thread ------------------------------------------------

    def set_overlay(self, overlay):
        if overlay is not None:
            # 선이 있는 픽셀만 합성하도록 좌표·가중치를 미리 계산 (전체 float
            # 블렌딩 대비 프레임당 수십 ms 절약 → 지연 감소).
            ys, xs = np.nonzero(overlay["line_alpha"])
            overlay["_ys"] = ys
            overlay["_xs"] = xs
            overlay["_weight"] = (
                overlay["line_alpha"][ys, xs].astype(np.float32) / 255.0
            )
        with self._overlay_lock:
            self._overlay = overlay
        self.tracker.reset()

    def set_geometry(self, geometry):
        with self._overlay_lock:
            self.geom = geometry
            self._overlay = None
        self._render_scale = self._compute_render_scale()
        self._snap_target = 0
        self._frames.clear()
        self.tracker.geom = geometry
        self.tracker.reset()

    def request_snapshot(self, frames):
        self._snap_buf = np.zeros((self.geom.height, self.geom.width), dtype=np.float32)
        self._snap_count = 0
        self._snap_target = max(1, int(frames))

    def set_zones(self, zones):
        self.zones = list(zones or [])

    def set_accumulate(self, frames):
        frames = max(1, int(frames))
        self._frames = deque(list(self._frames)[-frames:], maxlen=frames)

    def stop(self):
        self._stop = True
        self.wait(3000)

    # -- thread body -----------------------------------------------------------

    def run(self):
        import rclpy
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2

        rclpy.init()
        node = rclpy.create_node("lidar_bev_studio")
        # depth=1 + best-effort: 처리 지연 시 밀린 프레임을 버리고 항상 최신
        # 프레임만 받는다. RELIABLE depth=10이면 큐에 쌓인 과거 프레임을
        # 순서대로 처리해 화면이 현실보다 늦게 따라간다.
        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        node.create_subscription(PointCloud2, self.args.topic, self._on_cloud, qos)
        self.log.emit(
            f"{self.args.topic} 구독 시작, BEV {self.geom.width}x{self.geom.height} "
            f"({self.geom.resolution:.3f} m/px)"
        )
        try:
            while not self._stop and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    ZONE_ROLE_COLORS = {
        "랩타이머": (0, 140, 255),
        "신호등": (255, 90, 210),
        "출발": (90, 220, 90),
        "정차": (60, 60, 230),
    }

    def _draw_zones(self, image, geom, scale):
        for zone in self.zones:
            world = zone.get("world") or []
            if len(world) < 2:
                continue
            pts = []
            for wx, wy in world:
                col, row = _world_to_pixel_float(geom, wx, wy)
                pts.append(
                    (int(round((col + 0.5) * scale)), int(round((row + 0.5) * scale)))
                )
            color = None
            for role in zone.get("role") or []:
                if role in self.ZONE_ROLE_COLORS:
                    color = self.ZONE_ROLE_COLORS[role]
                    break
            if color is None:
                color = (255, 220, 0) if zone.get("geom") == "line" else (0, 213, 255)
            array = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                image,
                [array],
                zone.get("geom") != "line",
                color,
                max(3, int(round(scale * 0.5)) + 3),
                cv2.LINE_AA,
            )
            cx = int(round(sum(p[0] for p in pts) / len(pts)))
            cy = int(round(sum(p[1] for p in pts) / len(pts)))
            cv2.putText(
                image,
                str(zone.get("name", "")),
                (cx + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.7, 0.35 * scale),
                color,
                max(2, int(round(scale * 0.6))),
                cv2.LINE_AA,
            )

    def _check_time_jump(self, msg):
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if (
            self._last_msg_time_ns is not None
            and stamp_ns + 1_000_000 < self._last_msg_time_ns
        ):
            self.log.emit("타임스탬프가 뒤로 이동해 추적 상태를 초기화합니다 (rosbag loop).")
            self.tracker.reset(reset_floor=True)
            self._frames.clear()
        if self._last_msg_time_ns is not None:
            # 실제 프레임 간격 / 공칭 100ms — 끊김(프레임 드랍) 뒤 도착한
            # 프레임에서는 매칭 이동 상한이 이에 비례해 커진다.
            gap_s = max(0.0, (stamp_ns - self._last_msg_time_ns) / 1e9)
            self.tracker.frame_gap = float(np.clip(gap_s / 0.1, 1.0, 4.0))
        self._last_msg_time_ns = stamp_ns

    def _on_cloud(self, msg):
        from sensor_msgs_py import point_cloud2

        self._check_time_jump(msg)
        points = point_cloud2.read_points(
            msg, field_names=["x", "y", "z", "intensity"], skip_nans=False
        )
        x, y, z, intensity = points["x"], points["y"], points["z"], points["intensity"]

        geom = self.geom
        forward, lateral = geom.split_axes(x, y)
        base_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        base_valid &= ~((x == 0) & (y == 0) & (z == 0))
        base_valid &= (forward >= geom.forward_min) & (forward < geom.forward_max)
        base_valid &= (lateral >= geom.lateral_min) & (lateral < geom.lateral_max)
        valid = base_valid & (z >= geom.z_min) & (z <= geom.z_max)

        # 패킷 손실로 점이 거의 없는 부분 스캔은 통째로 건너뛴다 — 화면이
        # 깜빡 검어지거나 추적기가 헛되이 ‘소실’을 세지 않도록.
        valid_count = int(valid.sum())
        if self._valid_ema is not None and self._valid_ema > 2000:
            if valid_count < 0.2 * self._valid_ema:
                self._degenerate_frames += 1
                if self._degenerate_frames % 30 == 1:
                    self.log.emit(
                        f"불완전 스캔 건너뜀 (points={valid_count}, "
                        f"평소~{int(self._valid_ema)}) — 패킷 손실 가능성"
                    )
                return
        self._valid_ema = (
            valid_count
            if self._valid_ema is None
            else self._valid_ema * 0.95 + valid_count * 0.05
        )

        if (
            self.detect_enabled
            and self.tracker.params.height_mode == "floor"
            and self.tracker.floor_normal is None
        ):
            self.tracker.estimate_floor_plane(x, y, z, base_valid)

        bev = geom.rasterize_max(forward, lateral, intensity, valid)

        if self._snap_target > 0 and self._snap_buf.shape == bev.shape:
            self._snap_buf = np.maximum(self._snap_buf, bev)
            self._snap_count += 1
            self.snapshot_progress.emit(self._snap_count, self._snap_target)
            if self._snap_count >= self._snap_target:
                gain = max(self.args.snapshot_intensity_max, 1e-6)
                gray = (np.clip(self._snap_buf / gain, 0.0, 1.0) * 255.0).astype(np.uint8)
                self._snap_target = 0
                self.snapshot_ready.emit(gray)

        if self._frames and self._frames[0].shape != bev.shape:
            self._frames.clear()
        self._frames.append(bev)
        accum = np.maximum.reduce(list(self._frames))
        image = np.clip(accum / max(self.args.intensity_max, 1e-6), 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)

        render_scale = self._render_scale
        if render_scale > 1.001:
            image = cv2.resize(
                image,
                (int(round(geom.width * render_scale)), int(round(geom.height * render_scale))),
                interpolation=cv2.INTER_LINEAR,
            )

        with self._overlay_lock:
            overlay = self._overlay
        if overlay is not None and overlay["line_alpha"].shape != image.shape[:2]:
            overlay = None
        if overlay is not None and self.overlay_enabled:
            ys, xs = overlay["_ys"], overlay["_xs"]
            weight = (overlay["_weight"] * self.overlay_alpha)[:, None]
            base = image[ys, xs].astype(np.float32)
            line_color = np.array([0.0, 255.0, 255.0], dtype=np.float32)
            image[ys, xs] = np.clip(
                base * (1.0 - weight) + line_color * weight, 0.0, 255.0
            ).astype(np.uint8)

        summary = None
        vehicle_points = 0
        if self.detect_enabled:
            if self.tracker.params.height_mode == "floor":
                height = self.tracker.floor_height(x, y, z)
            else:
                height = z
            road_mask = overlay["road_mask"] if overlay is not None else None
            search_mask = overlay["search_mask"] if overlay is not None else None
            self.tracker.arena_mask = (
                overlay.get("arena_mask") if overlay is not None else None
            )
            candidates, vehicle_points = self.tracker.detect(
                forward, lateral, height, base_valid, road_mask, search_mask
            )
            summary = self.tracker.step(
                image, candidates, road_mask, scale=render_scale
            )
            if self.debug_candidates:
                # 후보 표시: 회색, 사람(tall) 판정은 자홍 'T'. 클릭하면 지정.
                minimum_dim = max(1, self.tracker.params.min_candidate_display_dim)
                for candidate in self.tracker.last_candidates:
                    bx, by, bw, bh = candidate["bbox"]
                    if max(bw, bh) < minimum_dim:
                        continue  # 자잘한 노이즈 박스는 표시하지 않는다.
                    tall = candidate.get("tall")
                    color = (200, 80, 220) if tall else (130, 130, 130)
                    top_left = (
                        int(round(bx * render_scale)),
                        int(round(by * render_scale)),
                    )
                    bottom_right = (
                        int(round((bx + bw) * render_scale)),
                        int(round((by + bh) * render_scale)),
                    )
                    cv2.rectangle(image, top_left, bottom_right, color, 1)
                    if tall:
                        cv2.putText(
                            image,
                            "T",
                            (top_left[0], top_left[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.35, 0.25 * render_scale),
                            color,
                            1,
                            cv2.LINE_AA,
                        )

        if self.zones_enabled and self.zones:
            self._draw_zones(image, geom, render_scale)

        now = time.monotonic()
        if self._last_frame_monotonic is not None:
            dt = now - self._last_frame_monotonic
            if dt > 1e-6:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0.0 else self._fps * 0.85 + inst * 0.15
        self._last_frame_monotonic = now

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        ).copy()
        stats = {
            "valid": int(valid.sum()),
            "vehicle_points": int(vehicle_points),
            "summary": summary,
            "fps": self._fps,
        }
        self.frame_ready.emit(qimg, stats)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class PointPickView(QGraphicsView):
    """Zoomable/pannable image view with numbered, draggable pick points."""

    pointsChanged = Signal()
    regionSelected = Signal(float, float, float, float)  # x0, y0, x1, y1 (image px)
    imageClicked = Signal(float, float)  # x, y (image px) — left click without drag
    imageRightClicked = Signal(float, float)  # x, y (image px) — right click, no drag
    cursorMoved = Signal(float, float)  # x, y (image px); (-1, -1) = outside image

    MARKER_HIT_RADIUS = 14.0

    def __init__(self, max_points=4, parent=None):
        super().__init__(parent)
        self.max_points = max_points
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = None
        self._image_w = 0
        self._image_h = 0
        self.points = []
        self._markers = []
        self._drag_index = None
        self._pan_last = None
        self._user_zoomed = False
        self._select_mode = False
        self._rubber = None
        self._rubber_origin = None
        self._pan_press_pos = None
        self._placeholder = ""
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#101318"))
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    # -- image / points API ----------------------------------------------------

    def set_image_bgr(self, image_bgr):
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        ).copy()
        self._scene.clear()
        self._markers = []
        self.points = []
        pixmap = QPixmap.fromImage(qimg)
        self._pix_item = self._scene.addPixmap(pixmap)
        self._image_w = pixmap.width()
        self._image_h = pixmap.height()
        margin = max(self._image_w, self._image_h) * 0.05 + 10
        self._scene.setSceneRect(
            QRectF(-margin, -margin, self._image_w + 2 * margin, self._image_h + 2 * margin)
        )
        self._user_zoomed = False
        self.fit()
        self.pointsChanged.emit()

    def has_image(self):
        return self._pix_item is not None

    def set_placeholder(self, text):
        self._placeholder = text
        if self._pix_item is None:
            self.viewport().update()

    def clear_image(self):
        self._scene.clear()
        self._markers = []
        self.points = []
        self._pix_item = None
        self.viewport().update()

    def update_qimage(self, qimage):
        """Fast path for live frames: swap the pixmap, keep zoom/pan."""
        pixmap = QPixmap.fromImage(qimage)
        if (
            self._pix_item is None
            or pixmap.width() != self._image_w
            or pixmap.height() != self._image_h
        ):
            self._scene.clear()
            self._markers = []
            self.points = []
            self._pix_item = self._scene.addPixmap(pixmap)
            self._image_w = pixmap.width()
            self._image_h = pixmap.height()
            margin = max(self._image_w, self._image_h) * 0.05 + 10
            self._scene.setSceneRect(
                QRectF(
                    -margin, -margin,
                    self._image_w + 2 * margin, self._image_h + 2 * margin,
                )
            )
            self._user_zoomed = False
            self.fit()
        else:
            self._pix_item.setPixmap(pixmap)

    def fit(self):
        if self._pix_item is not None:
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)
            self._user_zoomed = False

    def set_select_mode(self, enabled):
        self._select_mode = bool(enabled)
        if not enabled and self._rubber is not None:
            self._rubber.hide()
            self._rubber_origin = None

    def undo(self):
        if self.points:
            self.points.pop()
            self._rebuild_markers()
            self.pointsChanged.emit()

    def reset_points(self):
        if self.points:
            self.points.clear()
            self._rebuild_markers()
            self.pointsChanged.emit()

    # -- markers ---------------------------------------------------------------

    def _rebuild_markers(self):
        for marker in self._markers:
            self._scene.removeItem(marker)
        self._markers = []
        for index, (px, py) in enumerate(self.points, start=1):
            self._markers.append(self._make_marker(index, px, py))

    def _make_marker(self, index, x, y):
        group = QGraphicsItemGroup()
        group.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        radius = 6.0
        dot = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
        dot.setBrush(QColor("#ff5252"))
        dot.setPen(QPen(QColor("#ffffff"), 1.4))
        dot.setParentItem(group)
        label = QGraphicsSimpleTextItem(str(index))
        label.setBrush(QColor("#ffb4b4"))
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        label.setFont(font)
        label.setPos(radius + 3, -radius - 16)
        label.setParentItem(group)
        group.setPos(x, y)
        group.setZValue(10)
        self._scene.addItem(group)
        return group

    # -- interaction -----------------------------------------------------------

    def wheelEvent(self, event):
        if self._pix_item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        current = self.transform().m11()
        fit_scale = min(
            self.viewport().width() / max(1, self._image_w),
            self.viewport().height() / max(1, self._image_h),
        )
        target = float(np.clip(current * factor, fit_scale * 0.3, 150.0))
        self.scale(target / current, target / current)
        self._user_zoomed = True

    def _nearest_point_index(self, view_pos):
        best_index, best_distance = None, self.MARKER_HIT_RADIUS
        for index, (px, py) in enumerate(self.points):
            marker_view = self.mapFromScene(px, py)
            distance = np.hypot(
                marker_view.x() - view_pos.x(), marker_view.y() - view_pos.y()
            )
            if distance <= best_distance:
                best_index, best_distance = index, distance
        return best_index

    def mousePressEvent(self, event):
        if self._pix_item is None:
            return
        pos = event.position().toPoint()
        if self._select_mode and event.button() == Qt.LeftButton:
            self._rubber_origin = pos
            if self._rubber is None:
                self._rubber = QRubberBand(QRubberBand.Rectangle, self.viewport())
            self._rubber.setGeometry(QRect(pos, pos))
            self._rubber.show()
            return
        if event.button() == Qt.LeftButton and self.max_points == 0:
            self._pan_last = pos
            self._pan_press_pos = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            nearest = self._nearest_point_index(pos)
            if nearest is not None:
                self._drag_index = nearest
                return
            scene_pos = self.mapToScene(pos)
            if (
                0 <= scene_pos.x() < self._image_w
                and 0 <= scene_pos.y() < self._image_h
                and len(self.points) < self.max_points
            ):
                self.points.append([float(scene_pos.x()), float(scene_pos.y())])
                self._rebuild_markers()
                self.pointsChanged.emit()
        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_last = pos
            self._pan_press_pos = pos
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._pix_item is not None:
            scene_pos = self.mapToScene(pos)
            if 0 <= scene_pos.x() < self._image_w and 0 <= scene_pos.y() < self._image_h:
                self.cursorMoved.emit(float(scene_pos.x()), float(scene_pos.y()))
            else:
                self.cursorMoved.emit(-1.0, -1.0)
        if self._rubber_origin is not None:
            self._rubber.setGeometry(QRect(self._rubber_origin, pos).normalized())
            return
        if self._drag_index is not None:
            scene_pos = self.mapToScene(pos)
            px = float(np.clip(scene_pos.x(), 0, self._image_w - 1))
            py = float(np.clip(scene_pos.y(), 0, self._image_h - 1))
            self.points[self._drag_index] = [px, py]
            self._markers[self._drag_index].setPos(px, py)
            self.pointsChanged.emit()
        elif self._pan_last is not None:
            delta = pos - self._pan_last
            self._pan_last = pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rubber_origin is not None and event.button() == Qt.LeftButton:
            origin = self._rubber_origin
            self._rubber_origin = None
            self._rubber.hide()
            start = self.mapToScene(origin)
            end = self.mapToScene(event.position().toPoint())
            x0, x1 = sorted(
                (float(np.clip(start.x(), 0, self._image_w)),
                 float(np.clip(end.x(), 0, self._image_w)))
            )
            y0, y1 = sorted(
                (float(np.clip(start.y(), 0, self._image_h)),
                 float(np.clip(end.y(), 0, self._image_h)))
            )
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                self.regionSelected.emit(x0, y0, x1, y1)
            return
        self._drag_index = None
        if self._pan_last is not None:
            self._pan_last = None
            self.setCursor(Qt.CrossCursor)
            if (
                event.button() in (Qt.LeftButton, Qt.RightButton)
                and self._pan_press_pos is not None
                and self._pix_item is not None
            ):
                delta = event.position().toPoint() - self._pan_press_pos
                if abs(delta.x()) <= 4 and abs(delta.y()) <= 4:
                    scene_pos = self.mapToScene(event.position().toPoint())
                    if (
                        0 <= scene_pos.x() < self._image_w
                        and 0 <= scene_pos.y() < self._image_h
                    ):
                        if event.button() == Qt.LeftButton:
                            self.imageClicked.emit(
                                float(scene_pos.x()), float(scene_pos.y())
                            )
                        else:
                            self.imageRightClicked.emit(
                                float(scene_pos.x()), float(scene_pos.y())
                            )
            self._pan_press_pos = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Backspace, Qt.Key_Z):
            self.undo()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pix_item is None and self._placeholder:
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#5d6675"))
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(self.viewport().rect(), Qt.AlignCenter, self._placeholder)


class PickPanel(QFrame):
    """Titled point-pick panel with counter and undo/reset controls."""

    def __init__(self, title, max_points=4, with_crop=False, parent=None, min_points=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.view = PointPickView(max_points=max_points)
        # min_points < max_points allows extra correspondences beyond the
        # minimum — least-squares over 5+ points is what makes the stored
        # residuals nonzero and the alignment able to report its own error.
        self.min_points = min_points if min_points is not None else max_points
        self.max_points = max_points
        self.counter = QLabel(self._counter_text(0))
        self.counter.setObjectName("chip")
        self.crop_btn = None
        self.uncrop_btn = None

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        undo_btn = QPushButton("점 취소")
        reset_btn = QPushButton("초기화")
        fit_btn = QPushButton("화면 맞춤")
        undo_btn.clicked.connect(self.view.undo)
        reset_btn.clicked.connect(self.view.reset_points)
        fit_btn.clicked.connect(self.view.fit)

        header = QHBoxLayout()
        header.setContentsMargins(12, 10, 12, 6)
        header.addWidget(title_label)
        header.addWidget(self.counter)
        header.addStretch(1)
        if with_crop:
            self.crop_btn = QPushButton("영역 크롭")
            self.crop_btn.setCheckable(True)
            self.uncrop_btn = QPushButton("크롭 해제")
            self.uncrop_btn.setEnabled(False)
            self.crop_btn.toggled.connect(self.view.set_select_mode)
            header.addWidget(self.crop_btn)
            header.addWidget(self.uncrop_btn)
            header.addSpacing(10)
        header.addWidget(undo_btn)
        header.addWidget(reset_btn)
        header.addWidget(fit_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 6)
        layout.addLayout(header)
        layout.addWidget(self.view, 1)

        self.view.pointsChanged.connect(self._update_counter)

    def _counter_text(self, count):
        if self.min_points < self.max_points:
            return f"{count} (≥{self.min_points})"
        return f"{count}/{self.max_points}"

    def _update_counter(self):
        count = len(self.view.points)
        self.counter.setText(self._counter_text(count))
        state = "ok" if count >= self.min_points else ""
        self.counter.setProperty("state", state)
        self.counter.style().unpolish(self.counter)
        self.counter.style().polish(self.counter)


class PreviewDialog(QDialog):
    """Zoomable overlay preview. confirm=True adds apply/cancel buttons."""

    def __init__(self, preview_bgr, confirm=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("정합 미리보기" if not confirm else "최종 확인 — 정합 저장·적용")
        self.resize(980, 900)
        view = PointPickView(max_points=0)
        view.set_image_bgr(preview_bgr)

        message = (
            "노란 트랙 선이 LiDAR BEV와 잘 겹치는지 확인하세요. (휠: 확대 · 우클릭 드래그: 이동)"
            if not confirm
            else "이 정합을 저장하고 라이브 화면에 적용할까요? 적용 전까지는 아무것도 바뀌지 않습니다."
        )
        info = QLabel(message)
        info.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        if confirm:
            cancel_btn = QPushButton("돌아가기")
            apply_btn = QPushButton("적용하고 저장")
            apply_btn.setObjectName("accent")
            cancel_btn.clicked.connect(self.reject)
            apply_btn.clicked.connect(self.accept)
            buttons.addWidget(cancel_btn)
            buttons.addWidget(apply_btn)
        else:
            close_btn = QPushButton("닫기")
            close_btn.clicked.connect(self.accept)
            buttons.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(view, 1)
        layout.addLayout(buttons)


class AlignmentPage(QWidget):
    """BEV snapshot + track map side-by-side 4-point picking page."""

    cancelled = Signal()
    applied = Signal(object)

    def __init__(self, worker, track_bgr, args, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.track_bgr = track_bgr
        self.args = args
        self._bev_gray = None
        self._capture_geom = None
        self._base_gray = None
        self._base_geom = None

        instructions = QLabel(
            "① 양쪽 이미지에서 <b>같은 물리 지점 4곳</b>을 같은 순서로 클릭  "
            "② ‘정합 미리보기’로 확인  ③ <b>‘최종 확인’을 눌러야 저장·적용</b>됩니다 — "
            "4번째 점을 찍어도 자동 반영되지 않습니다.  "
            "<span style='color:#8a93a5'>(휠: 확대 · 우클릭 드래그: 이동 · 점 드래그: 미세 조정 · Z/⌫: 점 취소)</span>"
        )
        instructions.setWordWrap(True)
        instructions.setObjectName("infoBar")

        # 최소 4점, 최대 12점. 4점이면 잔차가 구조적으로 0이라 정합 오차를
        # 스스로 잴 수 없다 — 6~8점을 권장 (compute_homography 주석 참조).
        self.bev_panel = PickPanel("① LiDAR BEV", 12, with_crop=True, min_points=4)
        self.track_panel = PickPanel("② 트랙 이미지", 12, min_points=4)
        self.track_panel.view.set_image_bgr(track_bgr)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.bev_panel)
        splitter.addWidget(self.track_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([1000, 1000])

        self.recapture_btn = QPushButton("BEV 다시 캡처")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(190)
        self.progress.setVisible(False)
        self.status_label = QLabel("")
        self.preview_btn = QPushButton("정합 미리보기")
        self.confirm_btn = QPushButton("최종 확인 (저장·적용)")
        self.confirm_btn.setObjectName("accent")
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setObjectName("danger")
        self.preview_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)

        footer = QHBoxLayout()
        footer.addWidget(self.recapture_btn)
        footer.addWidget(self.progress)
        footer.addStretch(1)
        footer.addWidget(self.status_label)
        footer.addStretch(1)
        footer.addWidget(self.preview_btn)
        footer.addWidget(self.confirm_btn)
        footer.addWidget(self.cancel_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(instructions)
        layout.addWidget(splitter, 1)
        layout.addLayout(footer)

        self.recapture_btn.clicked.connect(self.begin_capture)
        self.preview_btn.clicked.connect(self._show_preview)
        self.confirm_btn.clicked.connect(self._confirm)
        self.cancel_btn.clicked.connect(self._cancel)
        self.bev_panel.view.pointsChanged.connect(self._update_ready)
        self.track_panel.view.pointsChanged.connect(self._update_ready)
        self.bev_panel.view.regionSelected.connect(self._on_bev_crop)
        self.bev_panel.crop_btn.toggled.connect(self._on_crop_mode)
        self.bev_panel.uncrop_btn.clicked.connect(self._reset_crop)
        self.worker.snapshot_progress.connect(self._on_snapshot_progress)
        self.worker.snapshot_ready.connect(self._on_snapshot_ready)

    # -- capture ---------------------------------------------------------------

    def begin_capture(self):
        self.progress.setVisible(True)
        self.progress.setRange(0, self.args.snapshot_frames)
        self.progress.setValue(0)
        self.recapture_btn.setEnabled(False)
        self.status_label.setText("LiDAR BEV 캡처 중…")
        self._capture_geom = self.worker.geom
        self.worker.request_snapshot(self.args.snapshot_frames)

    def _on_snapshot_progress(self, count, target):
        self.progress.setMaximum(target)
        self.progress.setValue(count)

    def _on_snapshot_ready(self, gray):
        self._base_gray = gray
        self._base_geom = self._capture_geom or self.worker.geom
        self._bev_gray = gray
        self._capture_geom = self._base_geom
        self.bev_panel.view.set_image_bgr(gray)
        self.bev_panel.crop_btn.setChecked(False)
        self.bev_panel.uncrop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.recapture_btn.setEnabled(True)
        self.status_label.setText("캡처 완료 — 필요하면 ‘영역 크롭’ 후 4점을 선택하세요.")
        self._update_ready()

    # -- BEV crop --------------------------------------------------------------

    def _on_crop_mode(self, active):
        if active:
            self.status_label.setText("잘라낼 영역을 BEV 위에서 드래그하세요.")

    def _on_bev_crop(self, x0, y0, x1, y1):
        self.bev_panel.crop_btn.setChecked(False)
        if self._bev_gray is None or self._capture_geom is None:
            return
        geom = self._capture_geom
        res = geom.resolution
        col0, col1 = int(round(x0)), int(round(x1))
        row0, row1 = int(round(y0)), int(round(y1))
        col0 = max(0, min(col0, geom.width - 1))
        col1 = max(col0 + 1, min(col1, geom.width))
        row0 = max(0, min(row0, geom.height - 1))
        row1 = max(row0 + 1, min(row1, geom.height))
        if (col1 - col0) * res < 2.0 or (row1 - row0) * res < 2.0:
            QMessageBox.information(
                self, "영역이 너무 작음", "각 변이 2 m 이상이 되도록 드래그하세요."
            )
            return
        new_geom = replace(
            geom,
            forward_min=geom.forward_min + col0 * res,
            forward_max=geom.forward_min + col1 * res,
            lateral_min=geom.lateral_max - row1 * res,
            lateral_max=geom.lateral_max - row0 * res,
        )
        self._capture_geom = new_geom
        self._bev_gray = self._bev_gray[row0:row1, col0:col1].copy()
        self.bev_panel.view.set_image_bgr(self._bev_gray)
        self.bev_panel.uncrop_btn.setEnabled(True)
        self.status_label.setText(
            f"크롭 적용 ({new_geom.width}x{new_geom.height}px) — 4점을 선택하세요."
        )
        self._update_ready()

    def _reset_crop(self):
        if self._base_gray is None:
            return
        self._bev_gray = self._base_gray
        self._capture_geom = self._base_geom
        self.bev_panel.view.set_image_bgr(self._base_gray)
        self.bev_panel.crop_btn.setChecked(False)
        self.bev_panel.uncrop_btn.setEnabled(False)
        self.status_label.setText("크롭 해제 — 캡처 전체 영역으로 복원했습니다.")
        self._update_ready()

    # -- state -----------------------------------------------------------------

    def _points(self):
        return self.bev_panel.view.points, self.track_panel.view.points

    def _update_ready(self):
        bev_points, track_points = self._points()
        n_bev, n_track = len(bev_points), len(track_points)
        ready = (
            self._bev_gray is not None
            and n_bev >= 4
            and n_bev == n_track
        )
        self.preview_btn.setEnabled(ready)
        self.confirm_btn.setEnabled(ready)
        if ready:
            hint = (
                " 4점은 잔차가 0으로 고정됩니다 — 6~8점을 권장."
                if n_bev == 4 else ""
            )
            self.status_label.setText(
                f"{n_bev}점 완료 — 미리보기로 확인한 뒤 ‘최종 확인’을 누르세요.{hint}"
            )
        elif self._bev_gray is not None:
            self.status_label.setText(
                f"BEV {n_bev} · 트랙 {n_track} — 같은 물리 지점을 같은 순서로, "
                f"각각 4점 이상(동수) 필요"
            )

    def _render_preview(self):
        bev_points, track_points = self._points()
        bev_bgr = cv2.cvtColor(self._bev_gray, cv2.COLOR_GRAY2BGR)
        return render_alignment_preview(bev_bgr, self.track_bgr, track_points, bev_points)

    # -- actions ---------------------------------------------------------------

    def _show_preview(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            preview = self._render_preview()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "정합 오류", str(exc))
            return
        QApplication.restoreOverrideCursor()
        PreviewDialog(preview, confirm=False, parent=self).exec()

    def _confirm(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            preview = self._render_preview()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "정합 오류", str(exc))
            return
        QApplication.restoreOverrideCursor()

        dialog = PreviewDialog(preview, confirm=True, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return

        bev_points, track_points = self._points()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            paths = save_alignment(
                self.track_bgr,
                self._bev_gray,
                track_points,
                bev_points,
                self.args.track_map,
                self.args.output_dir,
                overlay_alpha=self.args.overlay_alpha,
                geometry=self._capture_geom,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "저장 실패", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.applied.emit(paths)

    def _cancel(self):
        bev_points, track_points = self._points()
        if bev_points or track_points:
            answer = QMessageBox.question(
                self,
                "정합 취소",
                "선택한 점을 저장하지 않고 라이브 화면으로 돌아갈까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.cancelled.emit()


# ---------------------------------------------------------------------------
# Zone editor (차선·기준선·랩타이머·신호등 구역)
# ---------------------------------------------------------------------------


ZONE_ROLES = ["출발", "재위치", "정차", "종료", "랩타이머", "신호등"]


class ZoneEditorPage(QWidget):
    """Draw named line/rect zones on a BEV snapshot; save sensor-frame coords.

    Port of nav-vla track_roi_editor_node minus calibration — the BEV grid
    already maps 1:1 to sensor-frame meters, so world coords come for free.
    """

    cancelled = Signal()
    saved = Signal(object)  # list of zone dicts

    def __init__(self, worker, args, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.args = args
        self._geom = None
        self._base_bgr = None
        self._es = 1.0
        self.zones = []
        self._dirty = False

        instructions = QLabel(
            "구역을 그립니다: <b>선=2점, 사각형=4점</b> 클릭 — 이름을 먼저 입력하세요. "
            "저장 시 라이다 기준(m) 좌표로 기록되고 라이브 화면에 표시됩니다. "
            "<span style='color:#8a93a5'>(휠: 확대 · 우클릭 드래그: 이동 · Z/⌫: 점 취소)</span>"
        )
        instructions.setWordWrap(True)
        instructions.setObjectName("infoBar")

        self.view = PointPickView(max_points=8)

        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(270)
        panel_title = QLabel("구역 정의")
        panel_title.setObjectName("panelTitle")
        self.line_radio = QRadioButton("선 (2점)")
        self.rect_radio = QRadioButton("사각형 (4점)")
        self.line_radio.setChecked(True)
        geom_row = QHBoxLayout()
        geom_row.addWidget(self.line_radio)
        geom_row.addWidget(self.rect_radio)
        geom_row.addStretch(1)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("이름"))
        self.name_edit = QLineEdit()
        name_row.addWidget(self.name_edit, 1)
        self.role_checks = {}
        roles_grid = QVBoxLayout()
        row_layout = None
        for index, role in enumerate(ZONE_ROLES):
            if index % 3 == 0:
                row_layout = QHBoxLayout()
                roles_grid.addLayout(row_layout)
            check = QCheckBox(role)
            self.role_checks[role] = check
            row_layout.addWidget(check)
        self.pending_label = QLabel("점 0/2")
        cancel_shape_btn = QPushButton("현재 도형 취소")
        cancel_shape_btn.clicked.connect(self._reset_pending)
        self.zone_list = QListWidget()
        delete_btn = QPushButton("선택 삭제")
        delete_btn.clicked.connect(self._delete_selected)

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 10, 12, 10)
        panel_layout.setSpacing(7)
        panel_layout.addWidget(panel_title)
        panel_layout.addLayout(geom_row)
        panel_layout.addLayout(name_row)
        panel_layout.addLayout(roles_grid)
        panel_layout.addWidget(self.pending_label)
        panel_layout.addWidget(cancel_shape_btn)
        panel_layout.addWidget(QLabel("저장 목록"))
        panel_layout.addWidget(self.zone_list, 1)
        panel_layout.addWidget(delete_btn)

        center = QHBoxLayout()
        center.setSpacing(10)
        center.addWidget(self.view, 1)
        center.addWidget(panel)

        self.recapture_btn = QPushButton("BEV 다시 캡처")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(190)
        self.progress.setVisible(False)
        self.status_label = QLabel("")
        save_btn = QPushButton("저장")
        save_btn.setObjectName("accent")
        close_btn = QPushButton("닫기")
        footer = QHBoxLayout()
        footer.addWidget(self.recapture_btn)
        footer.addWidget(self.progress)
        footer.addStretch(1)
        footer.addWidget(self.status_label)
        footer.addStretch(1)
        footer.addWidget(save_btn)
        footer.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(instructions)
        layout.addLayout(center, 1)
        layout.addLayout(footer)

        self.recapture_btn.clicked.connect(self.begin_capture)
        save_btn.clicked.connect(self._save)
        close_btn.clicked.connect(self._close)
        self.view.pointsChanged.connect(self._on_points_changed)
        self.line_radio.toggled.connect(lambda _checked: self._reset_pending())
        self.worker.snapshot_progress.connect(self._on_snapshot_progress)
        self.worker.snapshot_ready.connect(self._on_snapshot_ready)

        self._load_existing()

    # -- capture ---------------------------------------------------------------

    def begin_capture(self):
        self.progress.setVisible(True)
        self.progress.setRange(0, self.args.snapshot_frames)
        self.progress.setValue(0)
        self.recapture_btn.setEnabled(False)
        self.status_label.setText("LiDAR BEV 캡처 중…")
        self._geom = self.worker.geom
        self.worker.request_snapshot(self.args.snapshot_frames)

    def _on_snapshot_progress(self, count, target):
        self.progress.setMaximum(target)
        self.progress.setValue(count)

    def _on_snapshot_ready(self, gray):
        geom = self._geom or self.worker.geom
        self._es = max(
            1.0, min(4.0, 1600.0 / max(gray.shape[0], gray.shape[1]))
        )
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if self._es > 1.001:
            bgr = cv2.resize(
                bgr,
                (int(round(gray.shape[1] * self._es)), int(round(gray.shape[0] * self._es))),
                interpolation=cv2.INTER_CUBIC,
            )
        # LiDAR 인텐시티만으로는 트랙 형상이 흐릿하므로 노란 트랙 오버레이를 합성.
        try:
            overlay = load_track_overlay(
                self.args.track_map,
                self.args.homography_json,
                geom,
                render_scale=self._es,
            )
            if overlay["line_alpha"].shape == bgr.shape[:2]:
                bgr = blend_layout_lines(
                    bgr, overlay["line_alpha"], opacity=self.args.overlay_alpha
                )
        except OverlayError:
            pass
        self._base_bgr = bgr
        self.view.set_image_bgr(self._render_image())
        self.progress.setVisible(False)
        self.recapture_btn.setEnabled(True)
        self.status_label.setText("이름 입력 후 점을 찍어 구역을 만드세요.")

    # -- shapes ----------------------------------------------------------------

    def _need(self):
        return 2 if self.line_radio.isChecked() else 4

    def _zone_color(self, zone):
        for role in zone.get("role") or []:
            if role in RosWorker.ZONE_ROLE_COLORS:
                return RosWorker.ZONE_ROLE_COLORS[role]
        return (255, 220, 0) if zone["geom"] == "line" else (0, 213, 255)

    def _render_image(self):
        image = self._base_bgr.copy()
        geom = self._geom or self.worker.geom
        for zone in self.zones:
            pts = []
            for wx, wy in zone["world"]:
                col, row = _world_to_pixel_float(geom, wx, wy)
                pts.append(
                    (int(round((col + 0.5) * self._es)), int(round((row + 0.5) * self._es)))
                )
            if len(pts) < 2:
                continue
            color = self._zone_color(zone)
            array = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                image, [array], zone["geom"] != "line", color, 5, cv2.LINE_AA
            )
            cx = int(round(sum(p[0] for p in pts) / len(pts)))
            cy = int(round(sum(p[1] for p in pts) / len(pts)))
            cv2.putText(
                image, str(zone["name"]), (cx + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
            )
        return image

    def _rerender(self):
        image = self._render_image()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        ).copy()
        self.view.update_qimage(qimg)

    def _reset_pending(self):
        self.view.reset_points()
        self.pending_label.setText(f"점 0/{self._need()}")

    def _on_points_changed(self):
        count = len(self.view.points)
        need = self._need()
        self.pending_label.setText(f"점 {count}/{need}")
        if count >= need:
            self._commit_shape()

    def _commit_shape(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "이름 필요", "구역 이름을 먼저 입력하세요.")
            self._reset_pending()
            return
        geom = self._geom or self.worker.geom
        world = []
        for u, v in self.view.points[: self._need()]:
            cu, cv_ = u / self._es, v / self._es
            wx, wy = geom.pixel_to_sensor(cu - 0.5, cv_ - 0.5)
            world.append([round(float(wx), 3), round(float(wy), 3)])
        roles = [role for role, check in self.role_checks.items() if check.isChecked()]
        self.zones.append(
            {
                "name": name,
                "role": roles,
                "geom": "line" if self.line_radio.isChecked() else "rect",
                "world": world,
            }
        )
        self._dirty = True
        self.name_edit.clear()
        self._refresh_list()
        self._reset_pending()
        self._rerender()
        self.status_label.setText(f"구역 추가: {name}")

    def _refresh_list(self):
        self.zone_list.clear()
        for zone in self.zones:
            roles = "/".join(zone["role"]) or "-"
            geom_name = "선" if zone["geom"] == "line" else "사각형"
            self.zone_list.addItem(f"{zone['name']} [{roles}·{geom_name}]")

    def _delete_selected(self):
        row = self.zone_list.currentRow()
        if row < 0:
            return
        removed = self.zones.pop(row)
        self._dirty = True
        self._refresh_list()
        self._rerender()
        self.status_label.setText(f"삭제: {removed['name']}")

    # -- persistence -----------------------------------------------------------

    def _load_existing(self):
        path = Path(self.args.zones_json)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.zones = [
                {
                    "name": roi["name"],
                    "role": roi.get("role", []),
                    "geom": roi.get("geom", "line"),
                    "world": roi["world"],
                }
                for roi in data.get("rois", [])
                if roi.get("world")
            ]
        except Exception:
            self.zones = []
        self._refresh_list()

    def _save(self):
        geom = self._geom or self.worker.geom
        path = Path(self.args.zones_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        rois = []
        for zone in self.zones:
            pixels = []
            for wx, wy in zone["world"]:
                col, row = _world_to_pixel_float(geom, wx, wy)
                pixels.append([round(col + 0.5, 1), round(row + 0.5, 1)])
            rois.append({**zone, "pixels": pixels})
        path.write_text(
            json.dumps(
                {
                    "frame": "lidar_sensor",
                    "bev_geometry": geom.to_dict(),
                    "rois": rois,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._dirty = False
        self.status_label.setText(f"저장됨: {path.name}")
        self.saved.emit(list(self.zones))

    def _close(self):
        if self._dirty:
            answer = QMessageBox.question(
                self,
                "저장 안 됨",
                "저장하지 않은 구역 변경이 있습니다. 저장하지 않고 닫을까요?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.cancelled.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


def make_chip(text, state=""):
    chip = QLabel(text)
    chip.setObjectName("chip")
    chip.setProperty("state", state)
    return chip


def set_chip(chip, text, state=""):
    chip.setText(text)
    if chip.property("state") != state:
        chip.setProperty("state", state)
        chip.style().unpolish(chip)
        chip.style().polish(chip)


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("LiDAR BEV Studio")
        self.view_state_path = Path(args.output_dir) / "config" / "studio_view.json"
        self.geom = self._initial_geometry()
        det_params = DetectionParams(
            detect_vehicle=args.detect_vehicle,
            vehicle_height_min=args.vehicle_height_min,
            vehicle_height_max=args.vehicle_height_max,
            min_cluster_pixels=args.min_cluster_pixels,
            max_cluster_pixels=args.max_cluster_pixels,
            use_background_subtraction=args.bg_subtraction,
            bg_alpha=args.bg_alpha,
            confirm_detections=args.confirm_detections,
        )
        self.worker = RosWorker(self.geom, det_params, args)
        self._last_frame_time = None
        self._align_page = None
        self._zone_page = None
        self._track_bgr = None

        # -- header ------------------------------------------------------------
        title = QLabel("LiDAR BEV Studio")
        title.setObjectName("appTitle")
        self.topic_chip = make_chip(args.topic)
        self.fps_chip = make_chip("— fps")
        self.points_chip = make_chip("포인트 —")
        self.vehicle_chip = make_chip("대기", "")
        self.pose_chip = make_chip("f — · l —")
        self.overlay_chip = make_chip("오버레이 —")

        header = QHBoxLayout()
        header.setContentsMargins(16, 10, 16, 10)
        header.addWidget(title)
        header.addSpacing(14)
        header.addWidget(self.topic_chip)
        header.addWidget(self.fps_chip)
        header.addWidget(self.points_chip)
        header.addStretch(1)
        header.addWidget(self.overlay_chip)
        header.addWidget(self.vehicle_chip)
        header.addWidget(self.pose_chip)
        header_widget = QFrame()
        header_widget.setObjectName("header")
        header_widget.setLayout(header)

        # -- live page ---------------------------------------------------------
        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)

        self.live_view = PointPickView(max_points=0)
        self.live_view.setMinimumSize(320, 400)
        self.live_view.set_placeholder(f"포인트클라우드 수신 대기 중…  ({args.topic})")

        self.align_btn = QPushButton("트랙 정합 시작")
        self.align_btn.setObjectName("accent")
        self.zone_btn = QPushButton("구역 설정")
        self.display_btn = QPushButton("디스플레이")
        self.reset_btn = QPushButton("초기화")
        self.reset_btn.setObjectName("danger")
        self._select_purpose = None

        # -- display dialog widgets (크롭·보기·회전·오버레이·검출) -------------
        self.crop_btn = QPushButton("크롭")
        self.full_btn = QPushButton("전체 보기")
        self.live_fit_btn = QPushButton("화면 맞춤")
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setSingleStep(5)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setWrapping(True)
        self.rotation_spin.setValue(int(round(self.geom.rotation_deg)))
        self._rotation_timer = QTimer(self)
        self._rotation_timer.setSingleShot(True)
        self._rotation_timer.setInterval(400)
        self.overlay_check = QCheckBox("트랙 오버레이")
        self.overlay_check.setChecked(True)
        self.zones_check = QCheckBox("구역 표시")
        self.zones_check.setChecked(True)
        self.traj_check = QCheckBox("궤적 표시")
        self.traj_check.setChecked(True)
        self.cand_check = QCheckBox("후보 표시")
        self.cand_check.setChecked(True)
        self.cand_check.setToolTip(
            "검출된 모든 후보 클러스터를 회색으로, 사람(tall) 판정은 자홍 T로 표시합니다."
        )
        self.detect_check = QCheckBox("차량 검출")
        self.detect_check.setChecked(args.detect_vehicle)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(round(args.overlay_alpha * 100)))
        self.alpha_slider.setFixedWidth(140)
        self.display_dialog = self._build_display_dialog()

        controls = QHBoxLayout()
        controls.setContentsMargins(2, 4, 2, 0)
        controls.addWidget(self.align_btn)
        controls.addSpacing(8)
        controls.addWidget(self.zone_btn)
        controls.addWidget(self.display_btn)
        controls.addStretch(1)
        self.cursor_label = QLabel("커서: —")
        self.cursor_label.setStyleSheet("color: #9aa3b4;")
        controls.addWidget(self.cursor_label)
        controls.addSpacing(12)
        zoom_hint = QLabel("휠: 확대 · 드래그: 이동")
        zoom_hint.setStyleSheet("color: #6b7484;")
        controls.addWidget(zoom_hint)
        controls.addSpacing(10)
        controls.addWidget(self.reset_btn)

        side_panel = QFrame()
        side_panel.setObjectName("panel")
        side_panel.setFixedWidth(252)
        side_title = QLabel("검출 차량")
        side_title.setObjectName("panelTitle")
        self.vehicle_table = QTableWidget(0, 4)
        self.vehicle_table.setHorizontalHeaderLabels(["ID", "x (m)", "y (m)", "상태"])
        self.vehicle_table.verticalHeader().setVisible(False)
        self.vehicle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vehicle_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.vehicle_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.vehicle_table.setSelectionMode(QTableWidget.SingleSelection)
        self.vehicle_table.setFocusPolicy(Qt.NoFocus)
        self.clear_focus_btn = QPushButton("선택 해제")
        side_hint = QLabel(
            "회색 후보 박스 클릭: 차량으로 지정\n"
            "지정 박스/행 클릭: 해당 차량 강조 추적\n"
            "지정 박스 우클릭: 지정 해제\n"
            "후보 박스 우클릭: 그 영역 영구 제외\n"
            "좌표는 라이다 기준(m)입니다."
        )
        side_hint.setStyleSheet("color: #6b7484;")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 10, 12, 10)
        side_layout.setSpacing(8)
        side_layout.addWidget(side_title)
        side_layout.addWidget(self.vehicle_table, 1)
        side_layout.addWidget(side_hint)
        side_layout.addWidget(self.clear_focus_btn)

        live_center = QHBoxLayout()
        live_center.setSpacing(10)
        live_center.addWidget(self.live_view, 1)
        live_center.addWidget(side_panel)

        live_layout = QVBoxLayout()
        live_layout.setContentsMargins(14, 10, 14, 12)
        live_layout.setSpacing(8)
        live_layout.addWidget(self.banner)
        live_layout.addLayout(live_center, 1)
        live_layout.addLayout(controls)
        self.live_page = QWidget()
        self.live_page.setLayout(live_layout)

        # -- stack -------------------------------------------------------------
        self.stack = QStackedWidget()
        self.stack.addWidget(self.live_page)

        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(header_widget)
        central_layout.addWidget(self.stack, 1)
        central = QWidget()
        central.setLayout(central_layout)
        self.setCentralWidget(central)

        # -- wiring ------------------------------------------------------------
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.log.connect(lambda message: self.statusBar().showMessage(message, 6000))
        self.align_btn.clicked.connect(self._start_alignment)
        self.zone_btn.clicked.connect(self._start_zone_editor)
        self.display_btn.clicked.connect(self.display_dialog.show)
        self.crop_btn.clicked.connect(self._start_crop)
        self.full_btn.clicked.connect(self._reset_to_full_view)
        self.live_fit_btn.clicked.connect(self.live_view.fit)
        self.zones_check.toggled.connect(
            lambda checked: setattr(self.worker, "zones_enabled", bool(checked))
        )
        self.traj_check.toggled.connect(
            lambda checked: setattr(
                self.worker.tracker.params, "show_trajectory", bool(checked)
            )
        )
        self.cand_check.toggled.connect(
            lambda checked: setattr(self.worker, "debug_candidates", bool(checked))
        )
        self.live_view.regionSelected.connect(self._on_region_selected)
        self.rotation_spin.valueChanged.connect(
            lambda _value: self._rotation_timer.start()
        )
        self._rotation_timer.timeout.connect(self._apply_rotation)
        self.reset_btn.clicked.connect(self._reset_all)
        self.live_view.imageClicked.connect(self._on_live_clicked)
        self.live_view.imageRightClicked.connect(self._on_live_right_clicked)
        self.live_view.cursorMoved.connect(self._on_cursor_moved)
        self.vehicle_table.cellClicked.connect(self._on_table_clicked)
        self.clear_focus_btn.clicked.connect(self._clear_focus)
        self.overlay_check.toggled.connect(self._set_overlay_enabled)
        self.detect_check.toggled.connect(self._set_detect_enabled)
        self.alpha_slider.valueChanged.connect(
            lambda value: setattr(self.worker, "overlay_alpha", value / 100.0)
        )

        self._stale_timer = QTimer(self)
        self._stale_timer.timeout.connect(self._check_stale)
        self._stale_timer.start(1000)

        self._load_overlay(startup=True)
        self._load_zones(startup=True)
        self._exclusions = self._load_exclusions()
        self._rebuild_exclusions()
        self.worker.start()

    # -- candidate exclusions --------------------------------------------------

    def _load_exclusions(self):
        path = Path(self.args.exclusions_json)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [rect for rect in data.get("rects", []) if len(rect) >= 3]
        except Exception:
            return []

    def _save_exclusions(self):
        path = Path(self.args.exclusions_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"rects": self._exclusions}, indent=2), encoding="utf-8"
        )

    def _rebuild_exclusions(self):
        self.worker.tracker.exclusion_mask = build_exclusion_mask(
            self._exclusions, self.geom
        )

    def _clear_exclusions(self):
        if not self._exclusions:
            self.statusBar().showMessage("등록된 제외 영역이 없습니다.", 4000)
            return
        count = len(self._exclusions)
        self._exclusions = []
        path = Path(self.args.exclusions_json)
        if path.exists():
            path.unlink()
        self._rebuild_exclusions()
        self.statusBar().showMessage(f"제외 영역 {count}개를 모두 지웠습니다.", 5000)

    # -- display dialog --------------------------------------------------------

    def _build_display_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("디스플레이 설정")
        dialog.setModal(False)

        view_row = QHBoxLayout()
        view_row.addWidget(self.crop_btn)
        view_row.addWidget(self.full_btn)
        view_row.addWidget(self.live_fit_btn)
        view_row.addStretch(1)

        rotation_row = QHBoxLayout()
        rotation_row.addWidget(QLabel("회전"))
        rotation_row.addWidget(self.rotation_spin)
        rotation_row.addSpacing(14)
        rotation_row.addWidget(QLabel("잔상 누적"))
        self.accumulate_spin = QSpinBox()
        self.accumulate_spin.setRange(1, 10)
        self.accumulate_spin.setValue(max(1, int(self.args.accumulate)))
        self.accumulate_spin.setToolTip(
            "표시용 프레임 누적 수 — 1이면 잔상 없음, 클수록 화면이 밝아지지만 움직이는 차가 번짐."
        )
        self.accumulate_spin.valueChanged.connect(self.worker.set_accumulate)
        rotation_row.addWidget(self.accumulate_spin)
        rotation_row.addStretch(1)

        overlay_row = QHBoxLayout()
        overlay_row.addWidget(self.overlay_check)
        overlay_row.addSpacing(8)
        overlay_row.addWidget(QLabel("투명도"))
        overlay_row.addWidget(self.alpha_slider)
        overlay_row.addStretch(1)

        toggles_row = QHBoxLayout()
        toggles_row.addWidget(self.zones_check)
        toggles_row.addSpacing(10)
        toggles_row.addWidget(self.traj_check)
        toggles_row.addSpacing(10)
        toggles_row.addWidget(self.cand_check)
        toggles_row.addSpacing(10)
        toggles_row.addWidget(self.detect_check)
        toggles_row.addStretch(1)

        clear_excl_btn = QPushButton("제외 영역 지우기")
        clear_excl_btn.clicked.connect(self._clear_exclusions)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(dialog.close)
        close_row = QHBoxLayout()
        close_row.addWidget(clear_excl_btn)
        close_row.addStretch(1)
        close_row.addWidget(close_btn)

        layout = QVBoxLayout(dialog)
        layout.addLayout(view_row)
        layout.addLayout(rotation_row)
        layout.addLayout(overlay_row)
        layout.addLayout(toggles_row)
        layout.addLayout(close_row)
        return dialog

    # -- selection modes (crop / vehicle designation) --------------------------

    def _start_crop(self):
        self._select_purpose = "crop"
        self.live_view.set_select_mode(True)
        self.display_dialog.close()
        self.statusBar().showMessage(
            "표시할 영역을 드래그하세요. 놓으면 적용 여부를 묻습니다.", 15000
        )

    def _reset_all(self):
        box = QMessageBox(self)
        box.setWindowTitle("초기화")
        box.setText(
            "트랙을 다시 깔았거나 라이다 위치를 바꾼 경우 저장된 상태를 초기화합니다.\n\n"
            "• 보기 초기화: 크롭·회전을 전체 보기(0°)로 되돌립니다.\n"
            "• 전체 초기화: 보기 + 지정 차량 해제 + 트랙 정합 무효화(백업됨).\n"
            "  (구역 파일은 유지됩니다 — 필요하면 ‘구역 설정’에서 수정하세요.)"
        )
        view_btn = box.addButton("보기만 초기화", QMessageBox.ActionRole)
        all_btn = box.addButton("전체 초기화", QMessageBox.DestructiveRole)
        box.addButton("취소", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is view_btn:
            self.rotation_spin.blockSignals(True)
            self.rotation_spin.setValue(0)
            self.rotation_spin.blockSignals(False)
            self._apply_geometry(replace(self._full_geometry(), rotation_deg=0.0))
        elif clicked is all_btn:
            homography_path = Path(self.args.homography_json)
            if homography_path.exists():
                backup = homography_path.with_suffix(homography_path.suffix + ".bak")
                if backup.exists():
                    backup.unlink()
                homography_path.rename(backup)
            self.worker.tracker.designated = []
            self.worker.tracker.reset(reset_floor=True)
            self._exclusions = []
            exclusions_path = Path(self.args.exclusions_json)
            if exclusions_path.exists():
                exclusions_path.unlink()
            self._rebuild_exclusions()
            self.rotation_spin.blockSignals(True)
            self.rotation_spin.setValue(0)
            self.rotation_spin.blockSignals(False)
            self._apply_geometry(replace(self._full_geometry(), rotation_deg=0.0))
            self.statusBar().showMessage(
                "전체 초기화 완료 — 회전을 맞추고 트랙 정합을 다시 실행하세요.", 10000
            )

    # -- geometry / crop -------------------------------------------------------

    def _full_geometry(self):
        span = float(self.args.full_range)
        rotation = (
            self.geom.rotation_deg if hasattr(self, "geom") else self.args.rotation
        )
        return BevGeometry(
            resolution=self.args.resolution,
            forward_axis=self.args.forward_axis,
            forward_min=-span,
            forward_max=span,
            lateral_min=-span,
            lateral_max=span,
            z_min=self.args.z_min,
            z_max=self.args.z_max,
            rotation_deg=rotation,
        )

    def _initial_geometry(self):
        args = self.args
        explicit = [args.forward_min, args.forward_max, args.lateral_min, args.lateral_max]
        if all(value is not None for value in explicit):
            return BevGeometry(
                resolution=args.resolution,
                forward_axis=args.forward_axis,
                forward_min=args.forward_min,
                forward_max=args.forward_max,
                lateral_min=args.lateral_min,
                lateral_max=args.lateral_max,
                z_min=args.z_min,
                z_max=args.z_max,
                rotation_deg=args.rotation,
            )
        if self.view_state_path.exists():
            try:
                saved = BevGeometry.from_dict(
                    json.loads(self.view_state_path.read_text(encoding="utf-8"))
                )
                if (
                    abs(saved.resolution - args.resolution) < 1e-9
                    and saved.forward_axis == args.forward_axis
                ):
                    return saved
            except Exception:
                pass
        return self._full_geometry()

    def _save_view_state(self):
        self.view_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.view_state_path.write_text(
            json.dumps(self.geom.to_dict(), indent=2), encoding="utf-8"
        )

    def _on_region_selected(self, x0, y0, x1, y1):
        self._select_purpose = None
        self.live_view.set_select_mode(False)
        render_scale = self.worker.current_render_scale()
        x0, y0, x1, y1 = (value / render_scale for value in (x0, y0, x1, y1))
        geom = self.geom
        f0 = geom.forward_min + x0 * geom.resolution
        f1 = geom.forward_min + x1 * geom.resolution
        l_top = geom.lateral_max - y0 * geom.resolution
        l_bottom = geom.lateral_max - y1 * geom.resolution
        res = geom.resolution

        def snap(value):
            return round(value / res) * res

        forward_min, forward_max = snap(f0), snap(f1)
        lateral_min, lateral_max = snap(l_bottom), snap(l_top)
        if forward_max - forward_min < 2.0 or lateral_max - lateral_min < 2.0:
            QMessageBox.information(
                self, "영역이 너무 작음", "각 변이 2 m 이상이 되도록 드래그하세요."
            )
            return
        answer = QMessageBox.question(
            self,
            "영역 적용",
            f"선택한 영역으로 BEV를 잘라낼까요?\n\n"
            f"forward {forward_min:.2f} ~ {forward_max:.2f} m\n"
            f"lateral {lateral_min:.2f} ~ {lateral_max:.2f} m\n"
            f"캔버스 {int(round((forward_max - forward_min) / res))}"
            f"x{int(round((lateral_max - lateral_min) / res))} px",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        new_geom = replace(
            geom,
            forward_min=forward_min,
            forward_max=forward_max,
            lateral_min=lateral_min,
            lateral_max=lateral_max,
        )
        self._apply_geometry(new_geom)

    def _apply_rotation(self):
        rotation = float(self.rotation_spin.value())
        if abs(rotation - self.geom.rotation_deg) < 1e-9:
            return
        self._apply_geometry(replace(self.geom, rotation_deg=rotation))

    def _reset_to_full_view(self):
        self._apply_geometry(self._full_geometry())

    def _apply_geometry(self, geometry):
        self.geom = geometry
        self.worker.set_geometry(geometry)
        self._save_view_state()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._load_overlay(startup=True)
            self._rebuild_exclusions()
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"보기 영역 적용: forward {geometry.forward_min:.1f}~{geometry.forward_max:.1f} m · "
            f"lateral {geometry.lateral_min:.1f}~{geometry.lateral_max:.1f} m · "
            f"회전 {geometry.rotation_deg:.0f}° ({geometry.width}x{geometry.height}px) — "
            f"{self.view_state_path.name}에 저장됨",
            8000,
        )

    # -- overlay ---------------------------------------------------------------

    def _load_overlay(self, startup=False):
        try:
            overlay = load_track_overlay(
                self.args.track_map,
                self.args.homography_json,
                self.geom,
                render_scale=self.worker.current_render_scale(),
            )
        except OverlayError as exc:
            self.worker.set_overlay(None)
            set_chip(self.overlay_chip, "오버레이 없음", "warn")
            self.banner.setText(
                f"⚠ 트랙 오버레이를 표시할 수 없습니다 — {exc}  "
                "아래 ‘트랙 정합 시작’으로 4점 정합을 다시 만드세요."
            )
            self.banner.setVisible(True)
            return False
        self.worker.set_overlay(overlay)
        set_chip(self.overlay_chip, "오버레이 ON", "ok")
        self.banner.setVisible(False)
        if not startup:
            msg = "정합이 저장되고 라이브 화면에 적용되었습니다 ✓"
            info = getattr(self, "_last_alignment_info", None)
            if info:
                n = info.get("n_points")
                rms_m = info.get("residual_rms_m")
                if n == 4:
                    msg += "  (4점 — 잔차 측정 불가, 다음엔 6~8점 권장)"
                elif rms_m is not None:
                    msg += f"  ({n}점, 잔차 RMS {rms_m:.3f} m — D2 게이트 <0.05 m)"
            self.statusBar().showMessage(msg, 12000)
        return True

    def _set_overlay_enabled(self, enabled):
        self.worker.overlay_enabled = bool(enabled)

    def _set_detect_enabled(self, enabled):
        self.worker.detect_enabled = bool(enabled)
        if enabled:
            # Fresh start: drop stale tracks, re-estimate the floor plane.
            # The learned frozen background is kept (re-learn via '배경 학습').
            self.worker.tracker.reset(reset_floor=True)
            self.statusBar().showMessage("차량 검출 재시작 — 바닥 평면을 다시 추정합니다.", 6000)
        else:
            set_chip(self.vehicle_chip, "검출 꺼짐", "")

    # -- live frame ------------------------------------------------------------

    def _on_frame(self, qimage, stats):
        self._last_frame_time = time.monotonic()
        if self.stack.currentWidget() is self.live_page:
            self.live_view.update_qimage(qimage)
        set_chip(self.fps_chip, f"{stats['fps']:.1f} fps")
        set_chip(self.points_chip, f"포인트 {stats['valid']:,}")
        summary = stats.get("summary")
        self._update_vehicle_table(summary)
        if not self.worker.detect_enabled or summary is None:
            set_chip(self.vehicle_chip, "검출 꺼짐", "")
            set_chip(self.pose_chip, "—")
            return
        if self.worker.tracker.capturing_background():
            set_chip(self.vehicle_chip, "배경 학습 중…", "warn")
            set_chip(self.pose_chip, "—")
            return
        count = summary["count"]
        if count == 0:
            set_chip(self.vehicle_chip, "차량 없음", "")
            set_chip(self.pose_chip, "—")
            return
        lost = summary.get("lost", 0)
        if summary["off_road"] > 0:
            state = "err"
        elif lost > 0:
            state = "warn"
        else:
            state = "ok"
        set_chip(self.vehicle_chip, f"차량 {count}", state)
        pose_text = f"ON {summary['on_road']} · OFF {summary['off_road']}"
        if lost:
            pose_text += f" · 소실 {lost}"
        set_chip(self.pose_chip, pose_text)

    def _update_vehicle_table(self, summary):
        vehicles = [] if summary is None else sorted(
            summary["vehicles"], key=lambda v: v["id"]
        )
        self.vehicle_table.setRowCount(len(vehicles))
        focused_row = -1
        for row, vehicle in enumerate(vehicles):
            status = vehicle.get("status", "ON" if vehicle["on_road"] else "OFF")
            status_text = {"ON": "ON", "OFF": "OFF", "LOST": "소실"}.get(status, status)
            status_color = {
                "ON": QColor("#6fe3a1"),
                "OFF": QColor("#ff8484"),
                "LOST": QColor("#ffd479"),
            }.get(status, QColor("#aeb6c4"))
            values = (
                str(vehicle["id"]),
                f"{vehicle['x']:.2f}",
                f"{vehicle['y']:.2f}",
                status_text,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                if column == 3:
                    item.setForeground(QBrush(status_color))
                self.vehicle_table.setItem(row, column, item)
            if vehicle["focused"]:
                focused_row = row
        if focused_row >= 0:
            self.vehicle_table.selectRow(focused_row)
        else:
            self.vehicle_table.clearSelection()

    def _on_live_clicked(self, x, y):
        if not self.worker.detect_enabled:
            return
        scale = self.worker.current_render_scale()
        col, row = x / scale, y / scale
        tracker = self.worker.tracker

        # 1) 지정 차량 박스 클릭 → 강조 추적
        for vehicle in tracker.designated:
            bx, by, bw, bh = vehicle["bbox"]
            if bx - 2 <= col <= bx + bw + 2 and by - 2 <= row <= by + bh + 2:
                tracker.focused_id = vehicle["id"]
                self.statusBar().showMessage(
                    f"차량 {vehicle['id']} 추적 중 — 빈 곳을 클릭하면 해제됩니다.", 6000
                )
                return

        # 2) 후보 클러스터 클릭 → 그 클러스터를 씨앗으로 차량 지정
        candidate = tracker.candidate_at(col, row)
        if candidate is not None:
            existing = {d["id"] for d in tracker.designated}
            number = 1
            while str(number) in existing:
                number += 1
            name, ok = QInputDialog.getText(
                self,
                "차량 지정",
                "이 클러스터를 차량으로 지정합니다 — 이름/번호:",
                text=str(number),
            )
            if not ok or not name.strip():
                return
            tracker.designate_from_candidate(name.strip(), candidate)
            self.statusBar().showMessage(
                f"차량 '{name.strip()}' 지정 — 추적을 시작합니다.", 6000
            )
            return

        # 3) 빈 곳 → 강조 해제
        tracker.focused_id = None
        self.statusBar().showMessage("차량 선택 해제", 4000)

    def _on_live_right_clicked(self, x, y):
        if not self.worker.detect_enabled:
            return
        tracker = self.worker.tracker
        scale = self.worker.current_render_scale()
        col, row = x / scale, y / scale

        # 1) 지정 차량 우클릭 → 지정 해제
        for vehicle in tracker.designated:
            bx, by, bw, bh = vehicle["bbox"]
            if bx - 2 <= col <= bx + bw + 2 and by - 2 <= row <= by + bh + 2:
                answer = QMessageBox.question(
                    self,
                    "차량 지정 해제",
                    f"차량 '{vehicle['id']}' 지정을 해제할까요?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer == QMessageBox.Yes:
                    tracker.remove_designated(vehicle["id"])
                    self.statusBar().showMessage(
                        f"차량 '{vehicle['id']}' 지정 해제됨", 5000
                    )
                return

        # 2) 후보 박스 우클릭 → 그 영역을 후보에서 영구 제외
        candidate = tracker.candidate_at(col, row)
        if candidate is None:
            return
        answer = QMessageBox.question(
            self,
            "영역 제외",
            "이 박스 영역을 후보 검출에서 영구 제외할까요?\n\n"
            "고정 반사체가 만드는 불필요한 박스를 지울 때 사용하세요.\n"
            "센서 좌표로 저장되어 재시작·크롭·회전 후에도 유지되며,\n"
            "‘디스플레이 → 제외 영역 지우기’로 전부 되돌릴 수 있습니다.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        bx, by, bw, bh = candidate["bbox"]
        corners_px = [
            (bx - 2, by - 2),
            (bx + bw + 2, by - 2),
            (bx + bw + 2, by + bh + 2),
            (bx - 2, by + bh + 2),
        ]
        corners_sensor = [
            [round(v, 3) for v in self.geom.pixel_to_sensor(cx0 - 0.5, cy0 - 0.5)]
            for cx0, cy0 in corners_px
        ]
        self._exclusions.append(corners_sensor)
        self._save_exclusions()
        self._rebuild_exclusions()
        self.statusBar().showMessage(
            f"영역을 후보에서 제외했습니다 (총 {len(self._exclusions)}개).", 5000
        )

    def _on_table_clicked(self, row, _column):
        item = self.vehicle_table.item(row, 0)
        if item is None:
            return
        vehicle_id = int(item.text())
        self.worker.tracker.focused_id = vehicle_id
        self.statusBar().showMessage(f"차량 {vehicle_id} 추적 중", 5000)

    def _clear_focus(self):
        self.worker.tracker.focused_id = None
        self.vehicle_table.clearSelection()
        self.statusBar().showMessage("차량 선택 해제", 4000)

    def _on_cursor_moved(self, x, y):
        if x < 0:
            self.cursor_label.setText("커서: —")
            return
        scale = self.worker.current_render_scale()
        sensor_x, sensor_y = self.geom.pixel_to_sensor(
            x / scale - 0.5, y / scale - 0.5
        )
        self.cursor_label.setText(f"커서: x {sensor_x:+.2f} · y {sensor_y:+.2f} m")

    def _check_stale(self):
        stale = (
            self._last_frame_time is None
            or time.monotonic() - self._last_frame_time > 3.0
        )
        if stale:
            set_chip(self.fps_chip, "수신 없음", "err")
            if self._last_frame_time is None:
                self.live_view.clear_image()

    # -- alignment flow --------------------------------------------------------

    def _start_alignment(self):
        if self._track_bgr is None:
            track = cv2.imread(str(self.args.track_map), cv2.IMREAD_COLOR)
            if track is None:
                QMessageBox.critical(
                    self, "트랙맵 없음", f"트랙맵 이미지를 읽지 못했습니다:\n{self.args.track_map}"
                )
                return
            self._track_bgr = track
        page = AlignmentPage(self.worker, self._track_bgr, self.args)
        page.cancelled.connect(self._leave_alignment)
        page.applied.connect(self._on_alignment_applied)
        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)
        self._align_page = page
        page.begin_capture()

    def _leave_alignment(self):
        if self._align_page is not None:
            self.stack.removeWidget(self._align_page)
            self._align_page.deleteLater()
            self._align_page = None
        self.stack.setCurrentWidget(self.live_page)

    # -- zone editor flow ------------------------------------------------------

    def _load_zones(self, startup=False):
        path = Path(self.args.zones_json)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            zones = [roi for roi in data.get("rois", []) if roi.get("world")]
        except Exception:
            return
        self.worker.set_zones(zones)
        if startup and zones:
            self.statusBar().showMessage(f"구역 {len(zones)}개를 불러왔습니다.", 5000)

    def _start_zone_editor(self):
        page = ZoneEditorPage(self.worker, self.args)
        page.cancelled.connect(self._leave_zone_editor)
        page.saved.connect(self._on_zones_saved)
        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)
        self._zone_page = page
        page.begin_capture()

    def _leave_zone_editor(self):
        if self._zone_page is not None:
            self.stack.removeWidget(self._zone_page)
            self._zone_page.deleteLater()
            self._zone_page = None
        self.stack.setCurrentWidget(self.live_page)

    def _on_zones_saved(self, zones):
        self.worker.set_zones(zones)
        self.statusBar().showMessage(
            f"구역 {len(zones)}개 저장·적용됨 — 라이브 화면에 표시됩니다.", 6000
        )

    def _on_alignment_applied(self, paths):
        self.args.homography_json = str(paths["homography_json"])
        self._last_alignment_info = {
            "n_points": paths.get("n_points"),
            "residual_rms_m": paths.get("residual_rms_m"),
        }
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._load_overlay()
        finally:
            QApplication.restoreOverrideCursor()
        self._leave_alignment()

    # -- shutdown --------------------------------------------------------------

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)


QSS = """
* { font-family: 'Noto Sans KR','Noto Sans CJK KR','NanumGothic','Malgun Gothic',sans-serif; }
QMainWindow, QDialog, QWidget { background: #14171d; color: #e8ebf2; font-size: 13px; }
QFrame#header { background: #191d25; border-bottom: 1px solid #262c38; }
QLabel { background: transparent; }
QLabel#appTitle { font-size: 16px; font-weight: 700; letter-spacing: 0.3px; }
QLabel#panelTitle { font-size: 14px; font-weight: 600; }
QLabel#infoBar { background: #191d25; border: 1px solid #262c38; border-radius: 8px; padding: 8px 12px; }
QLabel#chip {
    background: #20262f; border: 1px solid #2c3442; border-radius: 11px;
    padding: 3px 10px; color: #aeb6c4;
}
QLabel#chip[state="ok"]   { color: #6fe3a1; border-color: #2f5c44; background: #16241d; }
QLabel#chip[state="warn"] { color: #ffd479; border-color: #6a5426; background: #262013; }
QLabel#chip[state="err"]  { color: #ff8484; border-color: #6a2f33; background: #261618; }
QLabel#banner {
    background: #2c2213; border: 1px solid #7a5a2a; color: #ffd479;
    border-radius: 8px; padding: 9px 12px;
}
QFrame#panel { background: #191d25; border: 1px solid #262c38; border-radius: 10px; }
QPushButton {
    background: #232a35; border: 1px solid #303a49; border-radius: 8px;
    padding: 7px 16px; color: #dde3ec;
}
QPushButton:hover { background: #2b3341; }
QPushButton:pressed { background: #1d232d; }
QPushButton:disabled { color: #5b6270; background: #1b2029; border-color: #262c38; }
QPushButton#accent { background: #2f6bff; border-color: #2f6bff; color: #ffffff; font-weight: 600; }
QPushButton#accent:hover { background: #4a80ff; }
QPushButton#accent:pressed { background: #2657d6; }
QPushButton#accent:disabled { background: #223252; border-color: #223252; color: #5f6f92; }
QPushButton#danger { border-color: #5a3238; color: #ff9d9d; }
QPushButton#danger:hover { background: #34242a; }
QSpinBox, QLineEdit {
    background: #1b2029; border: 1px solid #303a49; border-radius: 6px;
    padding: 4px 8px; color: #dde3ec;
}
QListWidget {
    background: #14171d; border: 1px solid #262c38; border-radius: 6px;
}
QListWidget::item:selected { background: #24344f; }
QRadioButton { spacing: 6px; }
QSpinBox::up-button, QSpinBox::down-button { width: 16px; background: #232a35; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid #3a4456; background: #1b2029;
}
QCheckBox::indicator:checked { background: #2f6bff; border-color: #2f6bff; }
QSlider::groove:horizontal { height: 4px; background: #2a3140; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; margin: -6px 0; border-radius: 7px; background: #4a80ff;
}
QProgressBar {
    background: #1b2029; border: 1px solid #2c3442; border-radius: 7px;
    text-align: center; color: #aeb6c4;
}
QProgressBar::chunk { background: #2f6bff; border-radius: 6px; }
QSplitter::handle { background: #14171d; }
QTableWidget {
    background: #14171d; gridline-color: #262c38; border: none;
    selection-background-color: #24344f; selection-color: #e8ebf2;
}
QHeaderView::section {
    background: #1b2029; color: #aeb6c4; border: none; padding: 5px 4px;
    font-weight: 600;
}
QTableWidget::item { padding: 3px; }
QStatusBar { background: #191d25; color: #8a93a5; }
QGraphicsView { border: none; background: #101318; border-radius: 8px; }
"""


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    window = MainWindow(args)
    window.resize(1500, 950)
    window.show()
    if args.smoke_test:
        QTimer.singleShot(3000, app.quit)
    exit_code = app.exec()
    window.worker.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
