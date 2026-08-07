#!/usr/bin/env python3
"""정합 BEV 발행기 — 스튜디오의 '정합된 BEV 화면'을 토픽으로.

라이다가 물린 머신(트랙사이드 노트북)에서 돌며, 라이브 점군을 BEV 강도
이미지로 래스터화하고 그 위에 4점 정합 homography로 워프한 트랙 도안
라인을 얹는다 — lidar_bev_studio 가 화면에 그리는 것과 같은 그림. 결과를
JPEG(CompressedImage)로 발행해 Thor 대시보드 좌하단이 구독한다.

    /lidar_points (PointCloud2)
        → BEV 강도 래스터 (track_pose_node 와 동일 캔버스·픽셀 규약)
        → + 트랙 라인 오버레이 (homography 워프, 시작 시 1회 계산)
        → /track/aligned_bev/compressed (JPEG, 기본 ~8 Hz)

원시 점군(수십 Mbps)은 여전히 네트워크를 건너지 않는다 — 건너는 것은
240x320 JPEG(~20 KB) 뿐이다 (마스터플랜 §4.1 원칙 유지).

    ros2 run track_localizer_pkg aligned_bev_publisher
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs_py import point_cloud2

from track_localizer_pkg.bev_detector import DetectorConfig
from track_localizer_pkg.track_pose_node import _resolve_config_path

import json
from pathlib import Path


class AlignedBevPublisher(Node):
    def __init__(self):
        super().__init__("aligned_bev_publisher")
        self.cloud_topic = self.declare_parameter(
            "cloud_topic", "/lidar_points").value
        self.out_topic = self.declare_parameter(
            "out_topic", "/track/aligned_bev/compressed").value
        self.rate_hz = float(self.declare_parameter("rate_hz", 8.0).value)
        self.jpeg_quality = int(self.declare_parameter("jpeg_quality", 80).value)
        self.overlay_alpha = float(
            self.declare_parameter("overlay_alpha", 0.85).value)
        homography_json = _resolve_config_path(
            self.declare_parameter(
                "homography_json",
                "alignment/track_map_aligned_homography.json").value)
        track_map = _resolve_config_path(
            self.declare_parameter("track_map", "alignment/track2.png").value)

        # 캔버스 기하 = track_pose_node/homography와 동일해야 오버레이가 맞는다
        cfg = DetectorConfig()
        self.res = cfg.resolution
        self.f_min, self.f_max = cfg.forward_min, cfg.forward_max
        self.l_min, self.l_max = cfg.lateral_min, cfg.lateral_max
        self.w = int(round((self.f_max - self.f_min) / self.res))   # 240 (col=전방)
        self.h = int(round((self.l_max - self.l_min) / self.res))   # 320 (row=횡)

        # 트랙 라인 오버레이 — 스튜디오와 동일한 추출·워프 로직을 그대로
        # 임포트한다 (도안은 초록 배경+회색 노면 풀컬러라 단순 임계값은
        # 배경까지 잡는다 — HSV 기반 extract_layout_line_mask가 정답).
        self.line_alpha = None
        self._blend = None
        try:
            import sys
            tools_dir = (Path(__file__).resolve().parents[3]
                         / "tools" / "lidar_alignment_gui")
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
            from layout_rendering import (
                blend_layout_lines,
                extract_layout_line_mask,
                warp_layout_line_mask,
            )
            data = json.loads(Path(homography_json).read_text(encoding="utf-8"))
            H = np.asarray(data["homography_track_to_bev"], dtype=np.float64)
            tpl = cv2.imread(track_map)
            if tpl is None:
                raise FileNotFoundError(track_map)
            line_mask = extract_layout_line_mask(tpl)
            self.line_alpha = warp_layout_line_mask(
                line_mask, H, (self.w, self.h), supersample=4)
            self._blend = blend_layout_lines
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"오버레이 없음 (BEV만 발행): {e}")

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.pub = self.create_publisher(CompressedImage, self.out_topic, sensor_qos)
        self.create_subscription(
            PointCloud2, self.cloud_topic, self._on_cloud, sensor_qos)
        self._last_pub = 0.0
        self.get_logger().info(
            f"aligned bev — in={self.cloud_topic} out={self.out_topic} "
            f"{self.w}x{self.h} @{self.rate_hz} Hz, "
            f"overlay={'on' if self.line_alpha is not None else 'off'}")

    def _on_cloud(self, msg):
        now = time.monotonic()
        if now - self._last_pub < 1.0 / self.rate_hz:
            return
        self._last_pub = now

        fields = [f.name for f in msg.fields]
        want = ["x", "y", "intensity"] if "intensity" in fields else ["x", "y"]
        pts = point_cloud2.read_points(msg, field_names=want, skip_nans=True)
        x = np.asarray(pts["x"], dtype=np.float32)
        y = np.asarray(pts["y"], dtype=np.float32)
        val = (np.asarray(pts["intensity"], dtype=np.float32)
               if "intensity" in want else np.full(x.shape, 255.0, np.float32))

        # track_pose_node.pixel_to_world 의 역: col=전방, row=횡(위가 +lateral)
        col = ((x - self.f_min) / self.res - 0.5).astype(np.int32)
        row = ((self.l_max - y) / self.res - 0.5).astype(np.int32)
        keep = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        col, row, val = col[keep], row[keep], val[keep]

        img = np.zeros((self.h, self.w), np.float32)
        if val.size:
            np.maximum.at(img, (row, col), val)
            p99 = max(float(np.percentile(val, 99)), 1.0)
            img = np.clip(img / p99, 0.0, 1.0) * 255.0
        gray = img.astype(np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if self.line_alpha is not None and self._blend is not None:
            # 스튜디오와 동일한 블렌드
            bgr = self._blend(bgr, self.line_alpha, opacity=self.overlay_alpha)

        ok, buf = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        out = CompressedImage()
        out.header = msg.header
        out.format = "jpeg"
        out.data = buf.tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = AlignedBevPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
