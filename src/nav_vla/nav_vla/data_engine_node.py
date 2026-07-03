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
    ros2 run nav_vla data_engine_node --ros-args -p nav_mode:=lane -p rounds:=5
    ros2 run nav_vla data_engine_node --ros-args -p nav_mode:=lane -p goal_lane:=lane1
    ros2 run nav_vla data_engine_node --ros-args -p nav_mode:=direct -p rounds:=5
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
DIRECT_ONLY_ZONES = {
    "IN",
    "OUT(통과직전)",
    "OUT(통과직후)",
    "Slot1",
    "Slot2",
    "Slot3",
    "Slot4",
}

GENERIC = ["go to {p}", "drive to {p}", "head to {p}", "navigate to {p}",
           "take me to {p}", "move to {p}"]
LANE_GENERIC = [
    "follow the lane to {p}",
    "drive along the lane to {p}",
    "go to {p} through the lane",
]
DIRECT_GENERIC = [
    "go directly to {p}",
    "drive straight to {p} ignoring lanes",
    "take the shortest path to {p}",
]
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


def make_instruction(name, roles, nav_mode="lane", goal_lane="default"):
    pool = list(GENERIC)
    if nav_mode == "direct":
        pool += DIRECT_GENERIC
    else:
        pool += LANE_GENERIC
    for r in roles or []:
        pool += ROLE_TEMPLATES.get(r, [])
    tmpl = random.choice(pool)
    text = tmpl.format(p=zone_phrase(name))
    if nav_mode == "lane" and goal_lane in {"lane1", "lane2"}:
        return f"{text} through {goal_lane}"
    return text


class DataEngine(Node):
    def __init__(self):
        super().__init__("data_engine_node")
        self.map_path = self.declare_parameter("map_path", DEFAULT_MAP_PATH).value
        self.out_dir = self.declare_parameter("out_dir", DEFAULT_OUT).value
        self.image_topic = self.declare_parameter(
            "image_topic", "/camera/image_raw").value
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        zones_param = self.declare_parameter("zones", "all").value
        self.nav_mode = str(
            self.declare_parameter("nav_mode", "lane").value
        ).strip().lower()
        if self.nav_mode not in {"lane", "direct"}:
            self.get_logger().warn(
                f"unknown nav_mode '{self.nav_mode}', using lane"
            )
            self.nav_mode = "lane"
        self.goal_lane = str(
            self.declare_parameter("goal_lane", "default").value
        ).strip().lower()
        if self.goal_lane not in {"default", "lane1", "lane2"}:
            self.goal_lane = "default"
        self.rounds = int(self.declare_parameter("rounds", 3).value)
        self.shuffle = bool(self.declare_parameter("shuffle", True).value)
        self.save = bool(self.declare_parameter("save", True).value)
        self.fps = float(self.declare_parameter("record_fps", 5.0).value)
        self.timeout_sec = float(self.declare_parameter("timeout_sec", 60.0).value)
        self.settle_sec = float(self.declare_parameter("settle_sec", 1.0).value)
        self.navigator_wait_sec = float(
            self.declare_parameter("navigator_wait_sec", 10.0).value)

        self.zones = self._load_zones()
        names = list(self.zones)
        if zones_param and zones_param != "all":
            wanted = [z.strip() for z in zones_param.split(",") if z.strip()]
            names = [n for n in wanted if n in self.zones]
        self.zone_names = names

        self.latest_cmd = (0.0, 0.0)
        self.latest_lane = "unknown"
        self.latest_img = None        # (cv image, stamp)
        self.arrived_zone = None
        self.current_goal = None
        self.current_goal_lane = self.goal_lane
        self.current_goal_pose = None

        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(Image, self.image_topic, self._img_cb, img_qos)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)
        self.create_subscription(String, "/nav_status", self._status_cb, 10)
        self.create_subscription(String, "/lane_mode_state", self._lane_cb, 10)
        self.goal_pub = self.create_publisher(String, "/nav_goal", 10)
        self.direct_goal_pub = self.create_publisher(String, "/direct_nav_goal", 10)
        self.motion_pub = self.create_publisher(
            String,
            "/motion_control_command",
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )

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

    def _lane_cb(self, msg):
        lane = (msg.data or "").strip()
        if lane:
            self.latest_lane = lane

    def _status_cb(self, msg):
        text = msg.data or ""
        if text.startswith("arrived:"):
            self.arrived_zone = text.split(":", 1)[1].strip().split()[0]
        elif text.startswith("direct arrived:"):
            self.arrived_zone = text.split(":", 1)[1].strip().split()[0]

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
        dist_to_goal = None
        rel_goal = None
        if self.current_goal_pose:
            gx = float(self.current_goal_pose.get("x", 0.0))
            gy = float(self.current_goal_pose.get("y", 0.0))
            dx = gx - pose[0]
            dy = gy - pose[1]
            dist_to_goal = round(math.hypot(dx, dy), 4)
            rel_goal = {"dx": round(dx, 4), "dy": round(dy, 4)}
        i = self._frame_idx
        if self.save and self._ep_dir:
            cv2.imwrite(os.path.join(self._ep_dir, "frames", f"{i:04d}.jpg"), img)
        record = {
            "i": i,
            "image": f"frames/{i:04d}.jpg",
            "goal_zone": self.current_goal,
            "goal_lane": self.current_goal_lane,
            "lane": self.latest_lane,
            "cmd": {"linear": round(self.latest_cmd[0], 4),
                    "angular": round(self.latest_cmd[1], 4)},
            "pose": {"x": round(pose[0], 4), "y": round(pose[1], 4),
                     "yaw": round(pose[2], 4)},
        }
        if dist_to_goal is not None:
            record["dist_to_goal"] = dist_to_goal
            record["rel_goal"] = rel_goal
        self._records.append(record)
        self._frame_idx += 1

    # ---- orchestration ------------------------------------------------
    def _run(self):
        time.sleep(2.0)  # let topics connect
        if not self._wait_for_navigator():
            return
        session = time.strftime("session_%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.out_dir, session)
        self.get_logger().info(
            f"data engine: mode={self.nav_mode}, lanes={self._lane_plan()}, "
            f"{len(self.zone_names)} zones x {self.rounds} rounds "
            f"-> {session_dir if self.save else '(no save)'}")
        ep = 0
        for r in range(self.rounds):
            order = self._episode_plan()
            if self.shuffle:
                random.shuffle(order)
            for name, goal_lane in order:
                self._episode(ep, name, goal_lane, session_dir)
                ep += 1
        self.get_logger().info(f"data engine DONE: {ep} episodes")

    def _episode(self, ep, name, goal_lane, session_dir):
        roles = self.zones[name].get("role", [])
        if self.nav_mode == "lane" and self._is_direct_only_zone(name):
            self.get_logger().info(
                f"ep {ep}: skip {name} in lane mode; this zone requires direct mode"
            )
            return False
        instruction = make_instruction(name, roles, self.nav_mode, goal_lane)
        ep_dir = os.path.join(session_dir, f"ep_{ep:04d}")
        if self.save:
            os.makedirs(os.path.join(ep_dir, "frames"), exist_ok=True)
        start_pose = self.stream.latest

        # start recording
        self._records = []
        self._frame_idx = 0
        self._ep_dir = ep_dir
        self.arrived_zone = None
        self.current_goal = name
        self.current_goal_lane = goal_lane
        self.current_goal_pose = self.zones[name].get("pose")
        self._rec = True
        self.get_logger().info(
            f"ep {ep}: '{instruction}' -> {name} ({self.nav_mode}, {goal_lane})"
        )
        self._publish_goal(name, goal_lane)

        deadline = time.monotonic() + self.timeout_sec
        success = False
        while time.monotonic() < deadline:
            if self.arrived_zone == name:
                success = True
                break
            time.sleep(0.1)
        self._rec = False
        self.current_goal = None
        self.current_goal_lane = self.goal_lane
        self.current_goal_pose = None
        time.sleep(self.settle_sec)

        end_pose = self.stream.latest
        if not success:
            self.goal_pub.publish(String(data="stop"))
            self.direct_goal_pub.publish(String(data="stop"))
            self.get_logger().warn(f"ep {ep}: TIMEOUT ({name})")

        if self.save:
            with open(os.path.join(ep_dir, "steps.jsonl"), "w", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            meta = {
                "instruction": instruction,
                "goal_zone": name,
                "nav_mode": self.nav_mode,
                "goal_lane": goal_lane,
                "role": roles,
                "success": success,
                "n_frames": len(self._records),
                "start_pose": start_pose,
                "end_pose": end_pose,
                "goal_pose": self.zones[name].get("pose"),
            }
            with open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    def _wait_for_navigator(self):
        deadline = time.monotonic() + self.navigator_wait_sec
        while time.monotonic() < deadline:
            if self.goal_pub.get_subscription_count() > 0:
                return True
            time.sleep(0.2)
        self.get_logger().error(
            "navigator_node is not connected to /nav_goal. "
            "Run `navvla` then `ros2 run nav_vla navigator_node` in another terminal "
            "before starting data_engine_node."
        )
        return False

    def _publish_goal(self, name, goal_lane):
        self.motion_pub.publish(String(data="start"))
        if self.nav_mode == "direct":
            self.goal_pub.publish(String(data="stop"))
            self.direct_goal_pub.publish(String(data=name))
            return

        payload = {"zone": name}
        if goal_lane in {"lane1", "lane2"}:
            payload["lane"] = goal_lane
        self.goal_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _lane_plan(self):
        if self.nav_mode != "lane":
            return ["default"]
        if self.goal_lane in {"lane1", "lane2"}:
            return [self.goal_lane]
        return ["lane1", "lane2"]

    def _episode_plan(self):
        if self.nav_mode != "lane":
            return [(name, "default") for name in self.zone_names]
        lanes = self._lane_plan()
        return [(name, lane) for lane in lanes for name in self.zone_names]

    @staticmethod
    def _is_direct_only_zone(name):
        return str(name or "") in DIRECT_ONLY_ZONES


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
