#!/usr/bin/env python3
"""USB 카메라 -> CompressedImage(JPEG). VLA_AD image_publisher_node 이식판.

원본: VLA_AD/src/camera_perception_pkg/image_publisher_node.py (선배 스택).
검증된 결정들을 그대로 유지한다:
  - MJPG 640x480 @30fps, buffersize 1
  - CompressedImage(JPEG q80): raw 900KB -> ~50KB, 역직렬화 18x 감소
  - 캡처 스레드가 직접 퍼블리시 (타이머 없음 -> GIL 경합 없음)
  - v4l2 수동 노출 고정: auto_exposure=1 + exposure_time_absolute
    (30=3ms 낮 / 156=15.6ms 밤) — FPS 스파이크 방지

이식하며 뺀 것: image 디렉토리/CARLA 모드 (원본에 남아 있음).
camera / video(재생 벤치) 두 모드만 유지.
"""

import os
import subprocess
import sys
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage

JPEG_QUALITY = 80


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__("camera_publisher_node")
        self.data_source = self.declare_parameter("data_source", "camera").value
        self.cam_num = int(self.declare_parameter("cam_num", 0).value)
        self.video_path = self.declare_parameter("video_path", "").value
        self.pub_topic = self.declare_parameter(
            "pub_topic", "image_raw/compressed").value
        self.exposure_us = int(self.declare_parameter("exposure_us", 30).value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.publisher = self.create_publisher(CompressedImage, self.pub_topic, qos)
        self.cap = None

        if self.data_source == "camera":
            self._init_camera()
        elif self.data_source == "video":
            if not os.path.isfile(self.video_path):
                self.get_logger().error(f"video 없음: {self.video_path}")
                rclpy.shutdown()
                sys.exit(1)
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.get_logger().error(f"지원하지 않는 data_source: {self.data_source}")
            rclpy.shutdown()
            sys.exit(1)

        threading.Thread(target=self._publish_loop, daemon=True).start()
        self.get_logger().info(
            f"camera publisher — source={self.data_source} topic={self.pub_topic}")

    def _init_camera(self):
        dev = f"/dev/video{self.cam_num}"
        # 오픈 먼저(OpenCV가 V4L2 컨트롤을 초기화하므로 순서 중요), 그 다음
        # v4l2-ctl 로 수동 노출을 덮어쓴다 — 원본의 실측 결론 그대로.
        self.cap = cv2.VideoCapture(self.cam_num)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev,
             "--set-ctrl=auto_exposure=1,"
             "exposure_dynamic_framerate=0,"
             f"exposure_time_absolute={self.exposure_us}"],
            capture_output=True)
        if result.returncode != 0:
            self.get_logger().warn(f"v4l2-ctl 실패: {result.stderr.decode()}")
        else:
            self.get_logger().info(
                f"수동 노출 {self.exposure_us} (={self.exposure_us * 0.1:.1f} ms)")

    def _publish_loop(self):
        period = 1.0 / 30.0
        last = time.monotonic()
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                if self.data_source == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (640, 480))
            if self.data_source == "video":
                now = time.monotonic()
                if now - last < period:
                    time.sleep(period - (now - last))
                last = time.monotonic()
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "image_frame"
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            self.publisher.publish(msg)


def main():
    rclpy.init()
    node = CameraPublisherNode()
    try:
        while rclpy.ok():
            time.sleep(0.05)  # 캡처 스레드가 발행 — executor 불필요 (원본 방식)
    except KeyboardInterrupt:
        pass
    finally:
        if node.cap is not None and node.cap.isOpened():
            node.cap.release()


if __name__ == "__main__":
    main()
