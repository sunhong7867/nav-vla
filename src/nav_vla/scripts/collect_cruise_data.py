#!/usr/bin/env python3
"""One-off cruise data collector.

Collects continuous lane-following data with the existing navigator_node:
lane2 laps first, then lane1 laps. Data is saved under src/nav_vla/data so it
can be inspected together with the existing Stage-A recordings.

Run:
    python3 src/nav_vla/scripts/collect_cruise_data.py --laps-per-lane 10
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
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

from nav_vla.gz_pose import WorldPoseStream, query_world_pose, resolve_gz_bin


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_PATH = str(ROOT / "config" / "zone_map.yaml")
DEFAULT_OUT = str(ROOT / "data")
DEFAULT_ZONES = "M2,T2,M3,T3,T4,Start"


class CruiseCollector(Node):
    def __init__(self, args):
        super().__init__("cruise_data_script")
        self.map_path = args.map_path
        self.out_dir = args.out_dir
        self.image_topic = args.image_topic
        self.model_name = args.model_name
        self.gz_bin = resolve_gz_bin(args.gz_bin)
        self.route = [z.strip() for z in args.zones.split(",") if z.strip()]
        self.lanes = [l.strip().lower() for l in args.lanes.split(",") if l.strip()]
        self.laps_per_lane = int(args.laps_per_lane)
        self.fps = float(args.record_fps)
        self.timeout_per_zone = float(args.timeout_per_zone)
        self.navigator_wait_sec = float(args.navigator_wait_sec)
        self.save = not args.no_save

        self.zones = self._load_zones()
        self.route = [z for z in self.route if z in self.zones]
        self.lanes = [l for l in self.lanes if l in {"lane1", "lane2"}]
        if not self.route:
            raise RuntimeError("cruise route is empty")
        if not self.lanes:
            raise RuntimeError("cruise lane list is empty")

        self.latest_cmd = (0.0, 0.0)
        self.latest_lane_state = {}
        self.arrived_zone = None
        self.current_target = None
        self.current_lane = None
        self.current_target_pose = None

        self._rec = False
        self._records = []
        self._frame_idx = 0
        self._last_frame_t = 0.0
        self._ep_dir = None
        self.done = False

        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        command_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(Image, self.image_topic, self._img_cb, img_qos)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)
        self.create_subscription(String, "/nav_status", self._status_cb, 10)
        self.create_subscription(String, "/lane_mode_state", self._lane_state_cb, command_qos)
        self.goal_pub = self.create_publisher(String, "/nav_goal", 10)
        self.lane_pub = self.create_publisher(String, "/lane_mode_command", command_qos)
        self.motion_pub = self.create_publisher(
            String, "/motion_control_command", command_qos
        )

        self.stream = WorldPoseStream(self.gz_bin, self.model_name).start()
        seed = query_world_pose(self.gz_bin, self.model_name)
        if seed is not None:
            self.stream.latest = seed

        threading.Thread(target=self._run, daemon=True).start()

    def _load_zones(self):
        with open(self.map_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("zones", {})

    def _cmd_cb(self, msg):
        self.latest_cmd = (float(msg.linear.x), float(msg.angular.z))

    def _status_cb(self, msg):
        text = msg.data or ""
        if text.startswith("arrived:"):
            self.arrived_zone = text.split(":", 1)[1].strip().split()[0]

    def _lane_state_cb(self, msg):
        try:
            self.latest_lane_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.latest_lane_state = {"raw": msg.data}

    def _img_to_bgr(self, msg):
        ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ch)
        if msg.encoding in ("rgb8", "rgba8"):
            arr = arr[:, :, :3][:, :, ::-1]
        else:
            arr = arr[:, :, :3]
        return np.ascontiguousarray(arr)

    def _img_cb(self, msg):
        if not self._rec:
            return
        now = time.monotonic()
        if now - self._last_frame_t < 1.0 / self.fps:
            return
        try:
            img = self._img_to_bgr(msg)
        except Exception:
            return
        self._last_frame_t = now
        i = self._frame_idx
        if self.save and self._ep_dir:
            cv2.imwrite(os.path.join(self._ep_dir, "frames", f"{i:04d}.jpg"), img)

        pose = self.stream.latest or (0.0, 0.0, 0.0)
        record = {
            "i": i,
            "image": f"frames/{i:04d}.jpg",
            "task_type": "cruise",
            "goal_zone": self.current_target,
            "goal_lane": self.current_lane,
            "lane": json.dumps(self.latest_lane_state, ensure_ascii=False),
            "cmd": {
                "linear": round(self.latest_cmd[0], 4),
                "angular": round(self.latest_cmd[1], 4),
            },
            "pose": {
                "x": round(pose[0], 4),
                "y": round(pose[1], 4),
                "yaw": round(pose[2], 4),
            },
        }
        if self.current_target_pose:
            gx = float(self.current_target_pose.get("x", 0.0))
            gy = float(self.current_target_pose.get("y", 0.0))
            dx = gx - pose[0]
            dy = gy - pose[1]
            record["dist_to_goal"] = round(math.hypot(dx, dy), 4)
            record["rel_goal"] = {"dx": round(dx, 4), "dy": round(dy, 4)}
        self._records.append(record)
        self._frame_idx += 1

    def _run(self):
        time.sleep(2.0)
        if not self._wait_for_navigator():
            self.done = True
            return

        session = time.strftime("session_%Y%m%d_%H%M%S_cruise")
        session_dir = os.path.join(self.out_dir, session)
        if self.save:
            os.makedirs(session_dir, exist_ok=True)
        self.get_logger().info(
            f"collect cruise: lanes={self.lanes}, laps_per_lane={self.laps_per_lane}, "
            f"route={self.route} -> {session_dir if self.save else '(no save)'}"
        )

        summaries = []
        for ep, lane in enumerate(self.lanes):
            summaries.append(self._collect_lane_episode(ep, lane, session_dir))

        if self.save:
            with open(os.path.join(session_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump({"episodes": summaries}, f, ensure_ascii=False, indent=2)
        self.motion_pub.publish(String(data="stop"))
        self.goal_pub.publish(String(data="stop"))
        self.get_logger().info("collect cruise DONE")
        self.done = True

    def _collect_lane_episode(self, ep, lane, session_dir):
        ep_dir = os.path.join(session_dir, f"ep_{ep:04d}_{lane}")
        if self.save:
            os.makedirs(os.path.join(ep_dir, "frames"), exist_ok=True)

        self._records = []
        self._frame_idx = 0
        self._ep_dir = ep_dir
        self.current_lane = lane
        self._rec = True
        self.lane_pub.publish(String(data=lane))
        self.motion_pub.publish(String(data="start"))
        self.get_logger().info(f"ep {ep}: cruise {lane}, {self.laps_per_lane} laps")

        start_pose = self.stream.latest
        success_segments = 0
        total_segments = self.laps_per_lane * len(self.route)
        for lap in range(self.laps_per_lane):
            for target in self.route:
                success_segments += int(self._drive_to(target, lane, lap))

        self._rec = False
        self.current_target = None
        self.current_target_pose = None
        end_pose = self.stream.latest
        meta = {
            "instruction": f"cruise in {lane} for {self.laps_per_lane} laps",
            "task_type": "cruise",
            "nav_mode": "cruise",
            "goal_lane": lane,
            "route": self.route,
            "laps": self.laps_per_lane,
            "success": success_segments == total_segments,
            "success_segments": success_segments,
            "total_segments": total_segments,
            "n_frames": len(self._records),
            "start_pose": start_pose,
            "end_pose": end_pose,
        }

        if self.save:
            with open(os.path.join(ep_dir, "steps.jsonl"), "w", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        self.get_logger().info(
            f"ep {ep}: {lane} saved {len(self._records)} frames, "
            f"{success_segments}/{total_segments} segments"
        )
        return meta

    def _drive_to(self, target, lane, lap):
        self.current_target = target
        self.current_target_pose = self.zones[target].get("pose")
        self.arrived_zone = None
        self.goal_pub.publish(String(data=json.dumps({"zone": target, "lane": lane})))
        deadline = time.monotonic() + self.timeout_per_zone
        while time.monotonic() < deadline:
            if self.arrived_zone == target:
                self.get_logger().info(f"lap {lap + 1}: arrived {target} ({lane})")
                return True
            time.sleep(0.05)
        self.get_logger().warn(f"lap {lap + 1}: TIMEOUT {target} ({lane})")
        return False

    def _wait_for_navigator(self):
        deadline = time.monotonic() + self.navigator_wait_sec
        while time.monotonic() < deadline:
            if self.goal_pub.get_subscription_count() > 0:
                return True
            time.sleep(0.2)
        self.get_logger().error(
            "navigator_node is not connected to /nav_goal. "
            "Run `ros2 run nav_vla navigator_node` first."
        )
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", default=DEFAULT_MAP_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--image-topic", default="/camera/image_raw")
    parser.add_argument("--model-name", default="ego_vehicle")
    parser.add_argument("--gz-bin", default="")
    parser.add_argument("--zones", default=DEFAULT_ZONES)
    parser.add_argument("--lanes", default="lane2,lane1")
    parser.add_argument("--laps-per-lane", type=int, default=10)
    parser.add_argument("--record-fps", type=float, default=5.0)
    parser.add_argument("--timeout-per-zone", type=float, default=45.0)
    parser.add_argument("--navigator-wait-sec", type=float, default=10.0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CruiseCollector(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.motion_pub.publish(String(data="stop"))
        node.goal_pub.publish(String(data="stop"))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
