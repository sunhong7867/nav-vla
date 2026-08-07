#!/usr/bin/env python3
"""Hesai OT128 ingest — PointCloud2를 이 패키지의 기존 /scan 계약으로.

이 패키지는 원래 차량 부착 2D LiDAR(RPLidar, serial)용이었다:
publisher -> /scan(LaserScan) -> processor / obstacle_detector.
Hesai OT128은 트랙사이드 고정 3D 라이다라 드라이버(hesai_ros_driver)가
PointCloud2를 발행한다. 이 노드가 그 사이를 잇는다:

    /lidar_points (PointCloud2, 드라이버)
        -> z-밴드 크롭 + 각도 빈별 최소거리       -> /scan (LaserScan)
        -> 수신율·포인트수 감시                    -> /lidar/health (String JSON)

/scan 이 살아나면 lidar_processor_node / lidar_obstacle_detector_node 는
수정 없이 Hesai 를 소비한다. LaserScan 은 ~수 KB/프레임이라 원시 점군과
달리 WiFi로 Thor에 보내도 된다 (원시 PointCloud2 전송은 계속 금지 —
real_car_master_plan.md §4.1).

실행 (라이다가 물린 머신 = 트랙사이드 노트북):
    ros2 run lidar_perception_pkg hesai_ingest_node
    ros2 run lidar_perception_pkg hesai_ingest_node --ros-args \
        -p input_topic:=/lidar_points -p z_min:=0.05 -p z_max:=0.5
"""

import json
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String


class HesaiIngest(Node):
    def __init__(self):
        super().__init__("hesai_ingest_node")
        self.input_topic = self.declare_parameter(
            "input_topic", "/lidar_points").value
        self.scan_topic = self.declare_parameter("scan_topic", "/scan").value
        self.health_topic = self.declare_parameter(
            "health_topic", "/lidar/health").value
        # z-밴드: 이 높이 구간의 점만 2D 스캔으로 본다. 트랙사이드 설치에선
        # 바닥 위 장애물 높이, 차량 탑재로 바꾸면 센서 기준 높이로 재설정.
        self.z_min = float(self.declare_parameter("z_min", 0.05).value)
        self.z_max = float(self.declare_parameter("z_max", 0.5).value)
        self.range_min = float(self.declare_parameter("range_min", 0.3).value)
        self.range_max = float(self.declare_parameter("range_max", 30.0).value)
        # 1° 빈 = LaserScan 360포인트. 기존 2D 체인이 기대하는 해상도 수준.
        self.angle_inc_deg = float(
            self.declare_parameter("angle_increment_deg", 1.0).value)
        self.stale_warn_s = float(
            self.declare_parameter("stale_warn_seconds", 1.0).value)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, sensor_qos)
        self.health_pub = self.create_publisher(String, self.health_topic, 10)
        self.create_subscription(
            PointCloud2, self.input_topic, self._on_cloud, sensor_qos)

        self.n_bins = max(8, int(round(360.0 / self.angle_inc_deg)))
        self.frames = 0
        self.last_cloud_wall = None
        self.last_points = 0
        self.last_band_points = 0
        self._win_t0 = time.monotonic()
        self._win_frames = 0
        self.create_timer(1.0, self._health_tick)
        self.get_logger().info(
            f"hesai ingest — in={self.input_topic} out={self.scan_topic} "
            f"z=[{self.z_min}, {self.z_max}] m, {self.n_bins} bins")

    def _on_cloud(self, msg):
        pts = point_cloud2.read_points(
            msg, field_names=["x", "y", "z"], skip_nans=True)
        x = np.asarray(pts["x"], dtype=np.float32)
        y = np.asarray(pts["y"], dtype=np.float32)
        z = np.asarray(pts["z"], dtype=np.float32)
        self.frames += 1
        self._win_frames += 1
        self.last_cloud_wall = time.monotonic()
        self.last_points = int(x.size)

        band = (z >= self.z_min) & (z <= self.z_max)
        x, y = x[band], y[band]
        self.last_band_points = int(x.size)

        rng = np.hypot(x, y)
        keep = (rng >= self.range_min) & (rng <= self.range_max)
        rng, ang = rng[keep], np.arctan2(y[keep], x[keep])

        ranges = np.full(self.n_bins, float("inf"), dtype=np.float32)
        if rng.size:
            bins = ((ang + math.pi) / (2.0 * math.pi) * self.n_bins).astype(int)
            bins = np.clip(bins, 0, self.n_bins - 1)
            # 빈별 최소거리 — 같은 빈에 여러 점이면 가장 가까운 것
            np.minimum.at(ranges, bins, rng)

        scan = LaserScan()
        scan.header = msg.header
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = 2.0 * math.pi / self.n_bins
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.scan_time = 0.1
        scan.ranges = [float(r) for r in ranges]
        self.scan_pub.publish(scan)

    def _health_tick(self):
        now = time.monotonic()
        rate = self._win_frames / max(now - self._win_t0, 1e-6)
        self._win_t0, self._win_frames = now, 0
        stale = (self.last_cloud_wall is None
                 or now - self.last_cloud_wall > self.stale_warn_s)
        if stale:
            self.get_logger().warn(
                f"no clouds on {self.input_topic} — 드라이버/링크 확인")
        self.health_pub.publish(String(data=json.dumps({
            "ok": not stale,
            "rate_hz": round(rate, 2),
            "frames": self.frames,
            "points_last": self.last_points,
            "points_in_band": self.last_band_points,
        })))


def main():
    rclpy.init()
    node = HesaiIngest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
