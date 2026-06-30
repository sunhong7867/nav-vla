"""Obstacle-detection data collector for nav-vla.

Records (camera frame, obstacle label) pairs while the car drives, using the
lidar's /lidar_obstacle_info (Bool) as the FREE auto-label. This data trains a
camera->"obstacle ahead" classifier (the learned reactive VLA), with lidar as
the teacher. Unlike localization, the obstacle IS visible in the camera, so the
camera model can learn it.

Run alongside the working Phase-1 stack (so labels are correct):
    mission_sim.launch.py use_obstacles:=true
    obstacle_data_collector_node

Output: <out_dir>/<session>/frames/NNNNNN.jpg + labels.jsonl {image,label,t}
"""

import json
import os
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
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

import cv2

DEFAULT_OUT = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/data_obstacle")


class ObstacleDataCollector(Node):
    def __init__(self):
        super().__init__("obstacle_data_collector_node")
        self.image_topic = self.declare_parameter(
            "image_topic", "/camera/image_raw").value
        self.label_topic = self.declare_parameter(
            "label_topic", "/lidar_obstacle_info").value
        self.dist_topic = self.declare_parameter(
            "dist_topic", "/lidar_obstacle_distance").value
        self.out_dir = self.declare_parameter("out_dir", DEFAULT_OUT).value
        self.fps = float(self.declare_parameter("fps", 5.0).value)

        self.session = os.path.join(self.out_dir, time.strftime("session_%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(self.session, "frames"), exist_ok=True)
        self.labels_f = open(os.path.join(self.session, "labels.jsonl"), "w",
                             encoding="utf-8")

        self.label = None
        self.distance = None
        self.idx = 0
        self.n_pos = 0
        self.n_neg = 0
        self._last_t = 0.0

        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(Image, self.image_topic, self._img_cb, img_qos)
        self.create_subscription(Bool, self.label_topic, self._label_cb, 10)
        self.create_subscription(Float32, self.dist_topic, self._dist_cb, 10)
        self.create_timer(5.0, self._report)
        self.get_logger().info(f"collecting -> {self.session}")

    def _label_cb(self, msg):
        self.label = bool(msg.data)

    def _dist_cb(self, msg):
        self.distance = float(msg.data)

    def _img_cb(self, msg):
        if self.label is None:  # wait until we have a label
            return
        now = time.monotonic()
        if now - self._last_t < 1.0 / self.fps:
            return
        self._last_t = now
        ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        try:
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, ch)
        except ValueError:
            return
        bgr = arr[:, :, 2::-1] if msg.encoding in ("rgb8", "rgba8") else arr[:, :, :3]
        name = f"frames/{self.idx:06d}.jpg"
        cv2.imwrite(os.path.join(self.session, name), np.ascontiguousarray(bgr))
        lbl = 1 if self.label else 0
        rec = {"image": name, "label": lbl}
        if self.distance is not None:
            rec["distance"] = round(self.distance, 3)
        self.labels_f.write(json.dumps(rec) + "\n")
        self.labels_f.flush()
        self.idx += 1
        if lbl:
            self.n_pos += 1
        else:
            self.n_neg += 1

    def _report(self):
        self.get_logger().info(
            f"frames={self.idx}  obstacle={self.n_pos}  clear={self.n_neg}")

    def destroy_node(self):
        try:
            self.labels_f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ObstacleDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
