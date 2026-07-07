"""Stage-A VLA policy node for nav-vla (closed-loop).

Loads the behavior-cloned policy (train/checkpoints/stage_a.pt) and drives the
car from CAMERA + GOAL ZONE + RELATIVE GOAL POSE: pi(image, zone, lane, rel_goal)
-> cmd_vel. This is the learned "vision + control" half of the hierarchical
VLA; chat_gui maps language -> zone upstream and publishes goals.

Ground-truth pose is used to provide the relative goal vector and to declare
arrival/stop in this simulation policy.

Run instead of navigator_node:
    sim (use_perception_pipeline:=false use_driver:=false use_camera:=true)
    + this node + a publisher on /policy_nav_goal
    ros2 run nav_vla policy_node
"""

import math
import os
import json

import numpy as np
import rclpy
import torch
import torch.nn as nn
import yaml
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
DEFAULT_LANE_VOCAB = {"default": 0, "lane1": 1, "lane2": 2, "direct": 3}
DEFAULT_TASK_VOCAB = {"drive_to_zone": 0, "direct": 1, "cruise": 2}
DEFAULT_CRUISE_ROUTE = ("M2", "T2", "M3", "T3", "T4", "Start")
AREA_STOP_ZONES = {"crosswalk_stop"}
AREA_FALLBACK_ZONES = {"T1/M1"}
TARGET_YAW_STOP_LINE_ZONES = {"T3"}


class VisionGoalPolicy(nn.Module):
    """Must match the architecture in train/train_stage_a.py."""

    def __init__(self, n_zones, n_lanes=len(DEFAULT_LANE_VOCAB),
                 n_tasks=len(DEFAULT_TASK_VOCAB), emb=32):
        super().__init__()
        bb = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.zone_emb = nn.Embedding(n_zones, emb)
        self.lane_emb = nn.Embedding(n_lanes, 8)
        self.task_emb = nn.Embedding(n_tasks, 8)
        self.head = nn.Sequential(
            nn.Linear(512 + emb + 8 + 8 + 5, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, img, zidx, lidx, tidx, state):
        f = self.backbone(img).flatten(1)
        z = self.zone_emb(zidx)
        lane = self.lane_emb(lidx)
        task = self.task_emb(tidx)
        return self.head(torch.cat([f, z, lane, task, state], dim=1))


class LegacyVisionGoalPolicy(nn.Module):
    """Stage-A checkpoints before task_type was added."""

    def __init__(self, n_zones, n_lanes=len(DEFAULT_LANE_VOCAB), emb=32):
        super().__init__()
        bb = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.zone_emb = nn.Embedding(n_zones, emb)
        self.lane_emb = nn.Embedding(n_lanes, 8)
        self.head = nn.Sequential(
            nn.Linear(512 + emb + 8 + 5, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, img, zidx, lidx, _tidx, state):
        f = self.backbone(img).flatten(1)
        z = self.zone_emb(zidx)
        lane = self.lane_emb(lidx)
        return self.head(torch.cat([f, z, lane, state], dim=1))


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        ckpt_path = self.declare_parameter("ckpt", DEFAULT_CKPT).value
        self.image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        self.goal_topic = self.declare_parameter("goal_topic", "/policy_nav_goal").value
        self.direct_goal_topic = self.declare_parameter(
            "direct_goal_topic", "/policy_direct_nav_goal"
        ).value
        self.cmd_vel_topic = self.declare_parameter("cmd_vel_topic", "/cmd_vel").value
        self.motion_control_topic = self.declare_parameter(
            "motion_control_topic", "/motion_control_command"
        ).value
        self.status_topic = self.declare_parameter("status_topic", "/policy_nav_status").value
        self.lane_state_topic = self.declare_parameter(
            "lane_state_topic", "/lane_mode_state"
        ).value
        self.lane_command_topic = self.declare_parameter(
            "lane_command_topic", "/lane_mode_command"
        ).value
        cruise_route = self.declare_parameter(
            "cruise_route", ",".join(DEFAULT_CRUISE_ROUTE)
        ).value
        self.cruise_route = [z.strip() for z in str(cruise_route).split(",") if z.strip()]
        self.current_lane = self._normalize_lane(
            self.declare_parameter("initial_lane", "lane2").value,
            "lane2",
        )
        self.target_lane = self.current_lane
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.tol_pos = float(self.declare_parameter("tol_pos", 0.8).value)
        self.direct_tol_pos = float(self.declare_parameter("direct_tol_pos", 0.8).value)
        self.stop_offset = float(self.declare_parameter("stop_offset", 0.8).value)
        self.stop_line_arm_radius = float(
            self.declare_parameter("stop_line_arm_radius", 8.0).value
        )
        self.stop_line_lateral_radius = float(
            self.declare_parameter("stop_line_lateral_radius", 8.0).value
        )
        self.stop_line_pass_margin = float(
            self.declare_parameter("stop_line_pass_margin", 0.0).value
        )
        self.area_arrival_radius = float(
            self.declare_parameter("area_arrival_radius", 2.0).value
        )
        self.area_fallback_radius = float(
            self.declare_parameter("area_fallback_radius", 3.0).value
        )
        self.stop_external_motion = bool(
            self.declare_parameter("stop_external_motion", True).value
        )
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(ckpt_path, map_location=self.dev)
        self.vocab = ckpt["vocab"]
        self.lane_vocab = ckpt.get("lane_vocab", DEFAULT_LANE_VOCAB)
        self.task_vocab = ckpt.get("task_vocab", {"drive_to_zone": 0})
        self.has_task_input = "task_vocab" in ckpt and "task_emb.weight" in ckpt["model"]
        self.lin_scale = ckpt["lin_scale"]
        self.ang_scale = ckpt["ang_scale"]
        self.pos_scale = float(ckpt.get("pos_scale", 50.0))
        img_size = ckpt["img"]
        if self.has_task_input:
            self.model = VisionGoalPolicy(
                len(self.vocab),
                len(self.lane_vocab),
                len(self.task_vocab),
            ).to(self.dev)
        else:
            self.model = LegacyVisionGoalPolicy(
                len(self.vocab),
                len(self.lane_vocab),
            ).to(self.dev)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.zones = self._load_zones()
        self.cruise_route = [z for z in self.cruise_route if z in self.zones and z in self.vocab]
        if not self.cruise_route:
            self.cruise_route = [z for z in DEFAULT_CRUISE_ROUTE if z in self.zones and z in self.vocab]
        self.goal_pose = {n: z.get("pose", {}) for n, z in self.zones.items()}

        self.latest_img = None
        self.goal_zone = None
        self.goal_lane = "default"
        self.task_type = "drive_to_zone"
        self.cruise_enabled = False
        self.goal = None

        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(Image, self.image_topic, self._img_cb, img_qos)
        self.create_subscription(String, self.goal_topic, self._goal_cb, 10)
        self.create_subscription(String, self.direct_goal_topic, self._direct_goal_cb, 10)
        self.create_subscription(String, self.lane_command_topic, self._lane_command_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        command_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.motion_pub = self.create_publisher(
            String,
            self.motion_control_topic,
            command_qos,
        )
        self.lane_state_pub = self.create_publisher(
            String, self.lane_state_topic, command_qos
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.stream = WorldPoseStream(resolve_gz_bin(""), self.model_name).start()
        seed = query_world_pose(resolve_gz_bin(""), self.model_name)
        if seed:
            self.stream.latest = seed
        self.create_timer(0.1, self._control)
        self.stop_until = 0.0
        self.get_logger().info(
            f"policy_node ready ({self.dev}, {len(self.vocab)} zones) — "
            f"goals={self.goal_topic}, direct={self.direct_goal_topic}, "
            f"status={self.status_topic}")
        self._publish_lane_state("idle")

    def _load_zones(self):
        with open(ZONE_MAP, "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("zones", {})

    def _img_cb(self, msg):
        ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        try:
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, ch)
        except ValueError:
            return
        rgb = arr[:, :, :3] if msg.encoding in ("rgb8", "rgba8") else arr[:, :, 2::-1]
        self.latest_img = PILImage.fromarray(np.ascontiguousarray(rgb))

    def _goal_cb(self, msg):
        name, lane, mode = self._parse_goal(msg.data)
        if name.lower() in ("stop", "cancel", ""):
            self.goal_zone = None
            self.goal = None
            self.cruise_enabled = False
            self.task_type = "drive_to_zone"
            self._hold_stop()
            self._status("idle: cancelled")
            self._publish_lane_state("idle")
            return
        if name not in self.vocab:
            self._status(f"error: unknown zone '{name}'")
            return
        lane = self._effective_lane(lane)
        self.cruise_enabled = mode == "cruise"
        self.task_type = "cruise" if self.cruise_enabled else "drive_to_zone"
        self.goal_zone = name
        self.goal_lane = lane
        if lane in {"lane1", "lane2"}:
            self.target_lane = lane
        self.goal = self._make_lane_goal(name, lane)
        self.stop_until = 0.0
        self._stop_external_motion()
        self._publish_lane_state("cruise" if self.cruise_enabled else "moving")
        label = "cruising" if self.cruise_enabled else "moving"
        self._status(f"{label}: {name} via {lane} (policy)")

    def _direct_goal_cb(self, msg):
        name = (msg.data or "").strip()
        if name.lower() in ("stop", "cancel", ""):
            self.goal_zone = None
            self.goal = None
            self.cruise_enabled = False
            self.task_type = "drive_to_zone"
            self._hold_stop()
            self._status("idle: direct cancelled")
            self._publish_lane_state("idle")
            return
        if name not in self.vocab:
            self._status(f"error: unknown direct zone '{name}'")
            return
        self.goal_zone = name
        self.goal_lane = "direct"
        self.cruise_enabled = False
        self.task_type = "direct"
        self.goal = self._make_direct_goal(name)
        self.stop_until = 0.0
        self._stop_external_motion()
        self._publish_lane_state("direct")
        self._status(f"direct moving: {name} (policy)")

    def _lane_command_cb(self, msg):
        lane = self._normalize_lane(msg.data, None)
        if lane is None:
            return
        self.target_lane = lane
        if self.goal_zone is None:
            # No separate lane follower exists in policy-only mode. An idle lane
            # command is only an intent; current_lane changes after a policy goal
            # to that lane arrives.
            pass
        elif self.goal_lane in {"lane1", "lane2"}:
            # If a lane command arrives during an active goal, retarget the learned
            # policy to the requested lane for the remainder of that drive.
            self.goal_lane = lane
            self.goal = self._make_lane_goal(self.goal_zone, lane)
        self._publish_lane_state("lane_command")

    def _parse_goal(self, raw):
        text = (raw or "").strip()
        lane = "default"
        mode = "drive_to_zone"
        if text.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return text, lane, mode
            text = str(obj.get("zone") or obj.get("name") or "").strip()
            maybe_lane = str(obj.get("lane") or "default").strip().lower()
            if maybe_lane in self.lane_vocab:
                lane = maybe_lane
            maybe_mode = str(obj.get("mode") or obj.get("task_type") or mode).strip().lower()
            if maybe_mode in {"cruise", "drive_to_zone"}:
                mode = maybe_mode
        return text, lane, mode

    def _status(self, t):
        self.status_pub.publish(String(data=t))
        self.get_logger().info(t)

    @staticmethod
    def _normalize_lane(raw, default):
        lane = str(raw or "").strip().lower()
        return lane if lane in {"lane1", "lane2"} else default

    def _effective_lane(self, lane):
        if lane in {"lane1", "lane2", "direct"}:
            return lane
        return self.current_lane

    def _publish_lane_state(self, mode):
        payload = {
            "current_lane": self.current_lane,
            "target_lane": self.target_lane,
            "is_lane_changing": self.current_lane != self.target_lane,
            "mode": f"policy_{mode}",
        }
        self.lane_state_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _control(self):
        if self.goal_zone is None:
            if self.get_clock().now().nanoseconds * 1e-9 < self.stop_until:
                self._publish_zero()
            return
        if self.latest_img is None:
            return
        # arrival check (ground-truth pose, eval-only)
        pose = self.stream.latest
        g = self.goal_pose.get(self.goal_zone, {})
        if pose and self.goal:
            arrived, reason = self._arrived(pose, self.goal)
            if arrived:
                if self.goal_lane in {"lane1", "lane2"}:
                    self.current_lane = self.goal_lane
                    self.target_lane = self.goal_lane
                arrived_zone = self.goal_zone
                if self.cruise_enabled:
                    self._status(f"cruise passed: {arrived_zone} reason={reason}")
                    self._advance_cruise_goal()
                    self._publish_lane_state("cruise")
                    return
                else:
                    self._hold_stop()
                    self._publish_lane_state("arrived")
                    self._status(f"arrived: {arrived_zone} reason={reason}")
                    self.goal_zone = None
                    self.goal = None
                    self.task_type = "drive_to_zone"
                    return
        # policy inference: camera + goal -> action
        x = self.tf(self.latest_img).unsqueeze(0).to(self.dev)
        z = torch.tensor([self.vocab[self.goal_zone]], device=self.dev)
        lane_name = self.goal_lane if self.goal_lane in self.lane_vocab else "default"
        lane = torch.tensor([self.lane_vocab[lane_name]], device=self.dev)
        task_name = self.task_type if self.task_type in self.task_vocab else "drive_to_zone"
        task = torch.tensor([self.task_vocab[task_name]], device=self.dev)
        state = torch.tensor([self._state_features(pose, g)], device=self.dev)
        with torch.no_grad():
            out = self.model(x, z, lane, task, state)[0].cpu()
        msg = Twist()
        msg.linear.x = float(out[0]) * self.lin_scale
        msg.angular.z = float(out[1]) * self.ang_scale
        self.cmd_pub.publish(msg)

    def _advance_cruise_goal(self):
        lane = self.goal_lane if self.goal_lane in {"lane1", "lane2"} else self.current_lane
        next_zone = self._next_cruise_zone(self.goal_zone)
        self.goal_zone = next_zone
        self.goal_lane = lane
        self.task_type = "cruise"
        self.goal = self._make_lane_goal(next_zone, lane)
        self._status(f"cruising: {next_zone} via {lane} (policy)")

    def _next_cruise_zone(self, current):
        route = self.cruise_route or [z for z in DEFAULT_CRUISE_ROUTE if z in self.vocab]
        if not route:
            return current
        if current in route:
            return route[(route.index(current) + 1) % len(route)]
        return route[0]

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    def _stop_external_motion(self):
        if not self.stop_external_motion:
            return
        # Optional safety for launch setups where the old motion planner is
        # running but not publishing /cmd_vel zero continuously.
        self.motion_pub.publish(String(data="stop"))

    def _hold_stop(self, seconds=None):
        self._stop_external_motion()
        if seconds is None:
            self.stop_until = math.inf
        else:
            self.stop_until = self.get_clock().now().nanoseconds * 1e-9 + seconds
        self._publish_zero()

    def _make_direct_goal(self, zone_name):
        zone = self.zones.get(zone_name, {})
        pose = zone.get("pose", {})
        tol = zone.get("tol", {})
        return {
            "mode": "direct",
            "name": zone_name,
            "target_x": float(pose.get("x", 0.0)),
            "target_y": float(pose.get("y", 0.0)),
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "tol_pos": max(float(tol.get("pos", self.direct_tol_pos)), self.direct_tol_pos),
            "closest_dist": None,
        }

    def _make_lane_goal(self, zone_name, lane_name):
        zone = self.zones.get(zone_name, {})
        pose = zone.get("pose", {})
        tol = zone.get("tol", {})
        lane_config = self._lane_config_for_zone(zone, lane_name)
        target_x = float(pose.get("x", 0.0))
        target_y = float(pose.get("y", 0.0))
        target_yaw = float(pose.get("yaw", 0.0))
        stop_offset = float(
            lane_config.get("stop_offset", zone.get("stop_offset", self.stop_offset))
        )
        travel_yaw = self._travel_yaw_for_zone(
            zone_name, target_x, target_y, target_yaw
        )
        line_yaw = self._line_yaw_for_zone(zone_name, zone, travel_yaw, target_yaw)
        arrival_mode = str(
            zone.get("arrival_mode", self._arrival_mode_for_zone(zone_name))
        ).strip()
        if arrival_mode == "area":
            stop_x = target_x
            stop_y = target_y
        else:
            stop_x = target_x - math.cos(travel_yaw) * stop_offset
            stop_y = target_y - math.sin(travel_yaw) * stop_offset
        return {
            "mode": "lane",
            "name": zone_name,
            "lane": lane_name,
            "target_x": target_x,
            "target_y": target_y,
            "x": stop_x,
            "y": stop_y,
            "yaw": travel_yaw,
            "line_yaw": line_yaw,
            "arrival_mode": arrival_mode,
            "area_radius": float(
                lane_config.get("area_radius", zone.get("area_radius", self.area_arrival_radius))
            ),
            "area_fallback_radius": float(
                lane_config.get(
                    "area_fallback_radius",
                    zone.get("area_fallback_radius", self.area_fallback_radius),
                )
            ),
            "area_fallback": bool(
                lane_config.get("area_fallback", zone.get("area_fallback", False))
            ) or zone_name in AREA_FALLBACK_ZONES,
            "stop_line_arm_radius": float(
                lane_config.get(
                    "stop_line_arm_radius",
                    zone.get("stop_line_arm_radius", self.stop_line_arm_radius),
                )
            ),
            "stop_line_lateral_radius": float(
                lane_config.get(
                    "stop_line_lateral_radius", 
                    zone.get("stop_line_lateral_radius", self.stop_line_lateral_radius),
                )
            ),
            "stop_line_pass_margin": float(
                lane_config.get(
                    "stop_line_pass_margin",
                    zone.get("stop_line_pass_margin", self.stop_line_pass_margin),
                )
            ),
            "tol_pos": max(float(tol.get("pos", self.tol_pos)), 0.25),
            "closest_dist": None,
        }

    def _arrived(self, pose, goal):
        x, y, _yaw = pose
        dist = math.hypot(goal["x"] - x, goal["y"] - y)
        closest = goal.get("closest_dist")
        if closest is None or dist < closest:
            closest = dist
            goal["closest_dist"] = dist
        if goal.get("mode") == "direct":
            if dist <= goal["tol_pos"]:
                return True, "direct"
            if closest <= goal["tol_pos"] and dist > closest + 0.5:
                return True, "direct_passed"
            return False, None
        if goal.get("arrival_mode") == "area":
            inside_area = (
                dist <= goal["area_radius"]
                or (
                    closest is not None
                    and closest <= goal["area_radius"] + 0.3
                    and dist > closest + 0.5
                )
            )
            return (True, "area") if inside_area else (False, None)
        crossed_stop_line, _signed, _lateral = self._stop_line_state(x, y, goal)
        if crossed_stop_line:
            return True, "stop_line"
        if goal.get("area_fallback"):
            inside_area = (
                dist <= goal["area_fallback_radius"]
                or (
                    closest is not None
                    and closest <= goal["area_fallback_radius"]
                    and dist > closest + 0.5
                )
            )
            if inside_area:
                return True, "area_fallback"
        return False, None

    def _state_features(self, pose, goal):
        if not pose or not goal:
            return [0.0, 0.0, 0.0, 0.0, 1.0]
        dx = float(goal.get("x", 0.0)) - float(pose[0])
        dy = float(goal.get("y", 0.0)) - float(pose[1])
        dist = math.hypot(dx, dy)
        yaw = float(pose[2])
        return [
            dx / self.pos_scale,
            dy / self.pos_scale,
            dist / self.pos_scale,
            math.sin(yaw),
            math.cos(yaw),
        ]

    def _travel_yaw_for_zone(self, zone_name, target_x, target_y, fallback_yaw):
        zone_names = list(self.zones)
        try:
            index = zone_names.index(zone_name)
        except ValueError:
            return fallback_yaw
        if index <= 0:
            return fallback_yaw
        for previous_name in reversed(zone_names[:index]):
            previous_pose = self.zones.get(previous_name, {}).get("pose", {})
            if not previous_pose:
                continue
            prev_x = float(previous_pose.get("x", target_x))
            prev_y = float(previous_pose.get("y", target_y))
            dx = target_x - prev_x
            dy = target_y - prev_y
            if math.hypot(dx, dy) > 0.5:
                return math.atan2(dy, dx)
        return fallback_yaw

    def _stop_line_state(self, x, y, goal):
        line_yaw = goal.get("line_yaw", goal["yaw"])
        hx = math.cos(line_yaw)
        hy = math.sin(line_yaw)
        dx = x - goal["x"]
        dy = y - goal["y"]
        signed_to_stop = dx * hx + dy * hy
        lateral_to_stop = abs(dx * -hy + dy * hx)
        dist_to_stop = math.hypot(dx, dy)
        armed = (
            dist_to_stop <= goal["stop_line_arm_radius"]
            and lateral_to_stop <= goal["stop_line_lateral_radius"]
        )
        crossed = armed and signed_to_stop >= goal["stop_line_pass_margin"]
        return crossed, signed_to_stop, lateral_to_stop

    def _line_yaw_for_zone(self, zone_name, zone, travel_yaw, target_yaw):
        line_source = str(zone.get("line_yaw_source", "")).strip().lower()
        if line_source == "target":
            return target_yaw
        if line_source == "travel":
            return travel_yaw
        if zone_name in TARGET_YAW_STOP_LINE_ZONES:
            return target_yaw
        return travel_yaw

    @staticmethod
    def _arrival_mode_for_zone(zone_name):
        if zone_name in AREA_STOP_ZONES:
            return "area"
        return "line"

    @staticmethod
    def _lane_config_for_zone(zone, lane_name):
        configs = zone.get("lane_overrides", {})
        if not isinstance(configs, dict):
            return {}
        config = configs.get(lane_name, {})
        return config if isinstance(config, dict) else {}


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
