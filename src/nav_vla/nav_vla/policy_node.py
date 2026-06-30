"""Stage-A VLA policy node for nav-vla (closed-loop).

Loads the behavior-cloned policy (train/checkpoints/stage_a.pt) and drives the
car from CAMERA + GOAL ZONE only: pi(image, zone) -> cmd_vel. This is the
learned "vision + control" half of the hierarchical VLA; chat_gui maps
language -> zone upstream and publishes /nav_goal.

The policy uses ONLY the camera image and the goal zone id (no pose). Ground
truth pose is used solely to declare arrival and stop (eval convenience).

Run instead of navigator_node:
    sim (use_perception_pipeline:=false use_driver:=false use_camera:=true)
    + this node + chat_gui_node
    ros2 run nav_vla policy_node
"""

import math
import os

import numpy as np
import rclpy
import torch
import torch.nn as nn
from geometry_msgs.msg import Twist
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from torchvision import models, transforms

from nav_vla.gz_pose import WorldPoseStream, query_world_pose, resolve_gz_bin

DEFAULT_CKPT = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/train/checkpoints/stage_a.pt"
)
ZONE_MAP = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml")


class VisionGoalPolicy(nn.Module):
    """Must match the architecture in train/train_stage_a.py."""

    def __init__(self, n_zones, emb=32):
        super().__init__()
        bb = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.zone_emb = nn.Embedding(n_zones, emb)
        self.head = nn.Sequential(
            nn.Linear(512 + emb, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, img, zidx):
        f = self.backbone(img).flatten(1)
        z = self.zone_emb(zidx)
        return self.head(torch.cat([f, z], dim=1))


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        ckpt_path = self.declare_parameter("ckpt", DEFAULT_CKPT).value
        self.image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.tol_pos = float(self.declare_parameter("tol_pos", 0.8).value)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(ckpt_path, map_location=self.dev)
        self.vocab = ckpt["vocab"]
        self.lin_scale = ckpt["lin_scale"]
        self.ang_scale = ckpt["ang_scale"]
        img_size = ckpt["img"]
        self.model = VisionGoalPolicy(len(self.vocab)).to(self.dev)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.goal_pose = self._load_goal_poses()

        self.latest_img = None
        self.goal_zone = None

        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(Image, self.image_topic, self._img_cb, img_qos)
        self.create_subscription(String, "/nav_goal", self._goal_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/nav_status", 10)
        self.stream = WorldPoseStream(resolve_gz_bin(""), self.model_name).start()
        seed = query_world_pose(resolve_gz_bin(""), self.model_name)
        if seed:
            self.stream.latest = seed
        self.create_timer(0.1, self._control)
        self.get_logger().info(
            f"policy_node ready ({self.dev}, {len(self.vocab)} zones) — "
            f"drives from camera+goal")

    def _load_goal_poses(self):
        import yaml
        zones = (yaml.safe_load(open(ZONE_MAP)) or {}).get("zones", {})
        return {n: z.get("pose", {}) for n, z in zones.items()}

    def _img_cb(self, msg):
        ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        try:
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, ch)
        except ValueError:
            return
        rgb = arr[:, :, :3] if msg.encoding in ("rgb8", "rgba8") else arr[:, :, 2::-1]
        self.latest_img = PILImage.fromarray(np.ascontiguousarray(rgb))

    def _goal_cb(self, msg):
        name = (msg.data or "").strip()
        if name.lower() in ("stop", "cancel", ""):
            self.goal_zone = None
            self.cmd_pub.publish(Twist())
            self._status("idle: cancelled")
            return
        if name not in self.vocab:
            self._status(f"error: unknown zone '{name}'")
            return
        self.goal_zone = name
        self._status(f"moving: {name} (policy)")

    def _status(self, t):
        self.status_pub.publish(String(data=t))
        self.get_logger().info(t)

    def _control(self):
        if self.goal_zone is None or self.latest_img is None:
            return
        # arrival check (ground-truth pose, eval-only)
        pose = self.stream.latest
        g = self.goal_pose.get(self.goal_zone, {})
        if pose and g:
            if math.hypot(g.get("x", 0) - pose[0], g.get("y", 0) - pose[1]) <= self.tol_pos:
                self.cmd_pub.publish(Twist())
                self._status(f"arrived: {self.goal_zone}")
                self.goal_zone = None
                return
        # policy inference: camera + goal -> action
        x = self.tf(self.latest_img).unsqueeze(0).to(self.dev)
        z = torch.tensor([self.vocab[self.goal_zone]], device=self.dev)
        with torch.no_grad():
            out = self.model(x, z)[0].cpu()
        msg = Twist()
        msg.linear.x = float(out[0]) * self.lin_scale
        msg.angular.z = float(out[1]) * self.ang_scale
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = PolicyNode()
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
