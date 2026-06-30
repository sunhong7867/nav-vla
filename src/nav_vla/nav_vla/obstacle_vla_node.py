"""Camera obstacle VLA node for nav-vla (learned reactive perception).

Loads the trained camera->"obstacle ahead" classifier (train/checkpoints/obstacle.pt)
and publishes /obstacle_ahead (Bool) from the CAMERA only — the learned replacement
for the lidar detector. Feed it to navigator_node for lane-change avoidance:

    navigator_node -p use_obstacle_avoidance:=true -p obstacle_topic:=/obstacle_ahead
    obstacle_vla_node

Lidar is no longer needed at run time (it was only the training teacher).
"""

import os

import numpy as np
import rclpy
import torch
import torch.nn as nn
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from torchvision import models, transforms

DEFAULT_CKPT = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/train/checkpoints/obstacle.pt")


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


class ObstacleVLA(Node):
    def __init__(self):
        super().__init__("obstacle_vla_node")
        ckpt_path = self.declare_parameter("ckpt", DEFAULT_CKPT).value
        self.image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        self.threshold = float(self.declare_parameter("threshold", 0.5).value)
        self.consec = int(self.declare_parameter("consec_count", 2).value)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(ckpt_path, map_location=self.dev)
        img = ckpt["img"]
        self.model = build_model().to(self.dev)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((img, img)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self._hits = 0
        self._latched = False
        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(Image, self.image_topic, self._img_cb, img_qos)
        self.pub = self.create_publisher(Bool, "/obstacle_ahead", 10)
        self.get_logger().info(f"obstacle_vla ready ({self.dev}) — camera->/obstacle_ahead")

    def _img_cb(self, msg):
        ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        try:
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, ch)
        except ValueError:
            return
        rgb = arr[:, :, :3] if msg.encoding in ("rgb8", "rgba8") else arr[:, :, 2::-1]
        x = self.tf(PILImage.fromarray(np.ascontiguousarray(rgb))).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            p = torch.sigmoid(self.model(x)).item()
        hit = p > self.threshold
        self._hits = min(self._hits + 1, self.consec) if hit else 0
        detected = self._hits >= self.consec
        if detected != self._latched:
            self._latched = detected
            self.get_logger().info(
                f"obstacle {'DETECTED' if detected else 'clear'} (p={p:.2f})")
        self.pub.publish(Bool(data=detected))


def main():
    rclpy.init()
    node = ObstacleVLA()
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
