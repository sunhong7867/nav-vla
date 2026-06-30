"""Data engine for nav-vla (oracle rollouts -> VLA training data).

Drives the ego to each zone via the navigator (publishes /nav_goal, navigator
executes) and records, per episode, synchronized tuples of:
    (camera frame, language instruction, action=cmd_vel, pose, dist-to-goal).

This doubles as the multi-zone TOUR demo: the car visibly drives zone to zone.

Output layout (one session dir, one folder per episode):
    <out_dir>/<session>/
        ep_0000/
            meta.json          # instruction, goal_zone, role, success, n_frames, ...
            steps.jsonl        # per-frame: image, cmd{linear,angular}, pose, dist, t
            frames/0000.jpg ...

Run with:  sim + navigator_node + this node.

Usage:
    ros2 run nav_vla data_engine_node
    ros2 run nav_vla data_engine_node --ros-args -p rounds:=5 -p save:=true
"""

import json
import math
import os
import random
import threading
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2

from nav_vla.gz_pose import WorldPoseStream, query_world_pose, resolve_gz_bin

DEFAULT_MAP_PATH = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml"
)
DEFAULT_OUT = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/data")

GENERIC = ["go to {p}", "drive to {p}", "head to {p}", "navigate to {p}",
           "take me to {p}", "move to {p}"]
ROLE_TEMPLATES = {
    "출발": ["go to the start line", "head to the start", "drive to the starting point"],
    "정차": ["stop at {p}", "pull up to {p}", "go and stop at {p}"],
    "종료": ["go to the finish line", "drive to the end at {p}"],
}


def zone_phrase(name):
    low = name.lower()
    if low.startswith("slot"):
        return f"parking slot {name[4:]}"
    if low == "start":
        return "the start line"
    if low == "in":
        return "the entrance"
    if low.startswith("out"):
        return "the exit"
    if "crosswalk" in low:
        return "the crosswalk stop"
    return name  # T2, M2, T1/M1, ...


def make_instruction(name, roles):
    pool = list(GENERIC)
    for r in roles or []:
        pool += ROLE_TEMPLATES.get(r, [])
    tmpl = random.choice(pool)
    return tmpl.format(p=zone_phrase(name))


class DataEngine(Node):
    def __init__(self):
        super().__init__("data_engine_node")
        self.map_path = self.declare_parameter("map_path", DEFAULT_MAP_PATH).value
        self.out_dir = self.declare_parameter("out_dir", DEFAULT_OUT).value
        self.image_topic = self.declare_parameter(
            "image_topic", "/camera/image_raw").value
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        zones_param = self.declare_parameter("zones", "all").value
        self.rounds = int(self.declare_parameter("rounds", 3).value)
        self.shuffle = bool(self.declare_parameter("shuffle", True).value)
        self.save = bool(self.declare_parameter("save", True).value)
        self.fps = float(self.declare_parameter("record_fps", 5.0).value)
        self.timeout_sec = float(self.declare_parameter("timeout_sec", 60.0).value)
        self.settle_sec = float(self.declare_parameter("settle_sec", 1.0).value)

        self.zones = self._load_zones()
        names = list(self.zones)
        if zones_param and zones_param != "all":
            wanted = [z.strip() for z in zones_param.split(",") if z.strip()]
            names = [n for n in wanted if n in self.zones]
        self.zone_names = names

        self.latest_cmd = (0.0, 0.0)
        self.latest_img = None        # (cv image, stamp)
        self.arrived_zone = None

        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(Image, self.image_topic, self._img_cb, img_qos)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)
        self.create_subscription(String, "/nav_status", self._status_cb, 10)
        self.goal_pub = self.create_publisher(String, "/nav_goal", 10)

        self.stream = WorldPoseStream(self.gz_bin(), self.model_name).start()
        seed = query_world_pose(self.gz_bin(), self.model_name)
        if seed:
            self.stream.latest = seed

        # recording state (written by image cb, read/reset by orchestrator)
        self._rec = False
        self._records = []
        self._frame_idx = 0
        self._last_frame_t = 0.0
        self._ep_dir = None

        threading.Thread(target=self._run, daemon=True).start()

    def gz_bin(self):
        if not hasattr(self, "_gz"):
            self._gz = resolve_gz_bin(self.declare_parameter("gz_bin", "").value)
        return self._gz

    def _load_zones(self):
        with open(self.map_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("zones", {})

    # ---- callbacks ----------------------------------------------------
    def _cmd_cb(self, msg):
        self.latest_cmd = (msg.linear.x, msg.angular.z)

    def _status_cb(self, msg):
        text = msg.data or ""
        if text.startswith("arrived:"):
            self.arrived_zone = text.split(":", 1)[1].strip()

    def _img_to_bgr(self, msg):
        """Convert sensor_msgs/Image to a BGR uint8 array without cv_bridge."""
        ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ch)
        if msg.encoding in ("rgb8", "rgba8"):
            arr = arr[:, :, :3][:, :, ::-1]  # RGB(A) -> BGR
        else:  # bgr8 / bgra8
            arr = arr[:, :, :3]
        return np.ascontiguousarray(arr)

    def _img_cb(self, msg):
        try:
            img = self._img_to_bgr(msg)
        except Exception:
            return
        self.latest_img = img
        if not self._rec:
            return
        now = time.monotonic()
        if now - self._last_frame_t < 1.0 / self.fps:
            return
        self._last_frame_t = now
        pose = self.stream.latest or (0.0, 0.0, 0.0)
        i = self._frame_idx
        if self.save and self._ep_dir:
            cv2.imwrite(os.path.join(self._ep_dir, "frames", f"{i:04d}.jpg"), img)
        self._records.append({
            "i": i,
            "image": f"frames/{i:04d}.jpg",
            "cmd": {"linear": round(self.latest_cmd[0], 4),
                    "angular": round(self.latest_cmd[1], 4)},
            "pose": {"x": round(pose[0], 4), "y": round(pose[1], 4),
                     "yaw": round(pose[2], 4)},
        })
        self._frame_idx += 1

    # ---- orchestration ------------------------------------------------
    def _run(self):
        time.sleep(2.0)  # let topics connect
        session = time.strftime("session_%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.out_dir, session)
        self.get_logger().info(
            f"data engine: {len(self.zone_names)} zones x {self.rounds} rounds "
            f"-> {session_dir if self.save else '(no save)'}")
        ep = 0
        for r in range(self.rounds):
            order = list(self.zone_names)
            if self.shuffle:
                random.shuffle(order)
            for name in order:
                self._episode(ep, name, session_dir)
                ep += 1
        self.get_logger().info(f"data engine DONE: {ep} episodes")

    def _episode(self, ep, name, session_dir):
        roles = self.zones[name].get("role", [])
        instruction = make_instruction(name, roles)
        ep_dir = os.path.join(session_dir, f"ep_{ep:04d}")
        if self.save:
            os.makedirs(os.path.join(ep_dir, "frames"), exist_ok=True)
        start_pose = self.stream.latest

        # start recording
        self._records = []
        self._frame_idx = 0
        self._ep_dir = ep_dir
        self.arrived_zone = None
        self._rec = True
        self.get_logger().info(f"ep {ep}: '{instruction}' -> {name}")
        self.goal_pub.publish(String(data=name))

        deadline = time.monotonic() + self.timeout_sec
        success = False
        while time.monotonic() < deadline:
            if self.arrived_zone == name:
                success = True
                break
            time.sleep(0.1)
        self._rec = False
        time.sleep(self.settle_sec)

        end_pose = self.stream.latest
        if not success:
            self.goal_pub.publish(String(data="stop"))
            self.get_logger().warn(f"ep {ep}: TIMEOUT ({name})")

        if self.save:
            with open(os.path.join(ep_dir, "steps.jsonl"), "w", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            meta = {
                "instruction": instruction,
                "goal_zone": name,
                "role": roles,
                "success": success,
                "n_frames": len(self._records),
                "start_pose": start_pose,
                "end_pose": end_pose,
                "goal_pose": self.zones[name].get("pose"),
            }
            with open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    rclpy.init()
    node = DataEngine()
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
