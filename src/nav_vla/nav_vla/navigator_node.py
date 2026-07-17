"""Unified zone navigator for nav-vla.

The node supports two navigation modes:
- lane-follow: command the lane follower and stop it near the target zone.
- direct: override motion_planner with direct MotionCommand and drive to the zone.

Accepted /nav_goal String forms for lane-follow mode:
  - "T2"
  - "lane1:T2"
  - "T2 lane2"
  - {"zone": "T2", "lane": "lane1"}
  - "stop" / "cancel"

Accepted /direct_nav_goal String forms for direct mode:
  - "T2"
  - {"zone": "T2"}
  - "stop" / "cancel"

The stop point is shifted before the mapped zone by stop_offset meters using the
zone yaw. Set stop_offset negative if a specific map's yaw points the opposite
way.
"""

import json
import math
import os
import re
import time

import rclpy
import yaml
from interfaces_pkg.msg import MotionCommand
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from std_msgs.msg import Bool, String

from nav_vla.gz_pose import WorldPoseStream, query_world_pose, resolve_gz_bin


DEFAULT_MAP_PATH = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml"
)
DEFAULT_ZONE_LOG_DIR = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/data/zone_logs"
)
VALID_LANES = {"lane1", "lane2"}
ZONE_ALIASES = {
    "t1": "T1/M1",
    "m1": "T1/M1",
    "t1m1": "T1/M1",
    "t1/m1": "T1/M1",
    "m1t1": "T1/M1",
    "m1/t1": "T1/M1",
    "crosswalk": "crosswalk_stop",
    "crosswalkstop": "crosswalk_stop",
    "crosswalk_stop": "crosswalk_stop",
    "횡단보도": "crosswalk_stop",
}
MAX_STEERING_COMMAND = 7
MAX_MAPPED_YAW_RATE = 0.6458
AREA_STOP_ZONES = {"crosswalk_stop"}
AREA_FALLBACK_ZONES = {"T1/M1"}
TARGET_YAW_STOP_LINE_ZONES = {"T3"}


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, limit):
    return max(-limit, min(limit, value))


class NavigatorNode(Node):
    def __init__(self):
        super().__init__("navigator_node")
        self.map_path = self.declare_parameter("map_path", DEFAULT_MAP_PATH).value
        self.default_lane = self._normalize_lane(
            self.declare_parameter("default_lane", "lane2").value,
            "lane2",
        )
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.gz_bin = resolve_gz_bin(self.declare_parameter("gz_bin", "").value)
        self.zone_log_dir = self.declare_parameter(
            "zone_log_dir", DEFAULT_ZONE_LOG_DIR
        ).value
        self.zone_log_enabled = bool(
            self.declare_parameter("zone_log_enabled", True).value
        )
        self.zone_log_session = time.strftime("%Y%m%d_%H%M%S")
        if self.zone_log_enabled:
            os.makedirs(self.zone_log_dir, exist_ok=True)

        goal_topic = self.declare_parameter("goal_topic", "/nav_goal").value
        direct_goal_topic = self.declare_parameter(
            "direct_goal_topic", "/direct_nav_goal"
        ).value
        status_topic = self.declare_parameter("status_topic", "/nav_status").value
        direct_motion_topic = self.declare_parameter(
            "direct_motion_topic", "/direct_motion_command"
        ).value
        lane_command_topic = self.declare_parameter(
            "lane_command_topic", "/lane_mode_command"
        ).value
        lane_state_topic = self.declare_parameter(
            "lane_state_topic", "/lane_mode_state"
        ).value
        motion_control_topic = self.declare_parameter(
            "motion_control_topic", "/motion_control_command"
        ).value
        obstacle_topic = self.declare_parameter("obstacle_topic", "/obstacle_ahead").value
        self.stop_on_start = bool(self.declare_parameter("stop_on_start", True).value)
        self.use_obstacle_avoidance = bool(
            self.declare_parameter("use_obstacle_avoidance", False).value
        )
        self.arrival_hold_sec = float(
            self.declare_parameter("arrival_hold_sec", 0.0).value
        )
        self.default_tol_pos = float(self.declare_parameter("tol_pos", 0.45).value)
        self.min_arrival_radius = float(
            self.declare_parameter("min_arrival_radius", 0.25).value
        )
        self.pass_arrival_radius = float(
            self.declare_parameter("pass_arrival_radius", 0.8).value
        )
        self.pass_arrival_margin = float(
            self.declare_parameter("pass_arrival_margin", 0.25).value
        )
        self.stop_line_arm_radius = float(
            self.declare_parameter("stop_line_arm_radius", 8.0).value
        )
        self.stop_line_lateral_radius = float(
            self.declare_parameter("stop_line_lateral_radius", 8.0).value
        )
        self.area_arrival_radius = float(
            self.declare_parameter("area_arrival_radius", 2.0).value
        )
        self.area_fallback_radius = float(
            self.declare_parameter("area_fallback_radius", 3.0).value
        )
        self.stop_line_early_margin = float(
            self.declare_parameter("stop_line_early_margin", 0.0).value
        )
        self.stop_line_pass_margin = float(
            self.declare_parameter("stop_line_pass_margin", 0.0).value
        )
        self.default_stop_offset = float(
            self.declare_parameter("stop_offset", 0.8).value
        )
        self.yaw_offset = float(self.declare_parameter("yaw_offset", -math.pi / 2).value)
        self.max_linear = float(self.declare_parameter("direct_max_linear", 140.0).value)
        self.min_linear = float(self.declare_parameter("direct_min_linear", 35.0).value)
        self.max_angular = float(self.declare_parameter("direct_max_angular", 0.6458).value)
        self.k_ang = float(self.declare_parameter("direct_k_ang", 2.0).value)
        self.slow_radius = float(self.declare_parameter("direct_slow_radius", 4.0).value)
        self.direct_tol_pos = float(self.declare_parameter("direct_tol_pos", 0.6).value)
        self.direct_steering_sign = float(
            self.declare_parameter("direct_steering_sign", -1.0).value
        )
        self.direct_turn_in_place_angle = float(
            self.declare_parameter("direct_turn_in_place_angle", 0.35).value
        )
        self.direct_turn_speed = float(
            self.declare_parameter("direct_turn_speed", self.min_linear).value
        )
        self.direct_turn_speed = max(
            self.min_linear, min(self.max_linear, self.direct_turn_speed)
        )
        self.direct_heading_deadband = float(
            self.declare_parameter("direct_heading_deadband", 0.06).value
        )
        self.avoidance_preferred = self._normalize_lane(
            self.declare_parameter("avoidance_preferred_lane", self.default_lane).value,
            self.default_lane,
        )
        self.avoidance_other = self._normalize_lane(
            self.declare_parameter(
                "avoidance_other_lane", self._opposite_lane(self.avoidance_preferred)
            ).value,
            self._opposite_lane(self.avoidance_preferred),
        )
        self.avoidance_return_delay = float(
            self.declare_parameter("avoidance_return_delay", 3.0).value
        )
        self.avoidance_cooldown = float(
            self.declare_parameter("avoidance_cooldown", 2.5).value
        )
        self.avoidance_commit_delay = float(
            self.declare_parameter("avoidance_commit_delay", 0.25).value
        )

        self.zones = self._load_zones()
        self.mode = "idle"
        self.goal = None
        self.direct_goal = None
        self.arrival_started_at = None
        self.timer_count = 0
        self.obstacle_ahead = False
        self.in_obstacle_avoidance = False
        self.obstacle_clear_since = None
        self.obstacle_seen_since = None
        self.last_avoidance_cmd_t = 0.0
        self.current_lane = self.default_lane

        self.stream = WorldPoseStream(self.gz_bin, self.model_name).start()
        seed = query_world_pose(self.gz_bin, self.model_name)
        if seed is not None:
            self.stream.latest = seed

        command_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.lane_pub = self.create_publisher(String, lane_command_topic, command_qos)
        self.motion_pub = self.create_publisher(String, motion_control_topic, command_qos)
        self.direct_motion_pub = self.create_publisher(
            MotionCommand, direct_motion_topic, command_qos
        )
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.create_subscription(String, goal_topic, self._goal_cb, 10)
        self.create_subscription(String, direct_goal_topic, self._direct_goal_cb, 10)
        self.create_subscription(
            String, lane_state_topic, self._lane_state_cb, command_qos
        )
        self.create_subscription(Bool, obstacle_topic, self._obstacle_cb, 10)
        self.create_timer(0.1, self._tick)

        self._publish_lane(self.default_lane)
        if self.stop_on_start:
            self._publish_motion("stop")
        self._status(
            f"navigator ready: default={self.default_lane}, zones={len(self.zones)}, "
            f"lane_topic={goal_topic}, direct_topic={direct_goal_topic}, "
            f"obstacle_avoidance={self.use_obstacle_avoidance}"
        )

    def _load_zones(self):
        with open(self.map_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("zones", {})

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

    def _goal_cb(self, msg):
        zone_name, lane_name, is_cancel = self._parse_goal(msg.data)
        if is_cancel:
            self._cancel("cancelled")
            return

        if zone_name not in self.zones:
            self._status(f"error: unknown zone '{zone_name}'")
            return

        zone = self.zones[zone_name]
        pose = zone.get("pose", {})
        tol = zone.get("tol", {})
        self.default_stop_offset = float(self.get_parameter("stop_offset").value)
        self.stop_line_pass_margin = float(
            self.get_parameter("stop_line_pass_margin").value
        )
        lane_name = self._resolve_goal_lane(lane_name)
        lane_config = self._lane_config_for_zone(zone, lane_name)
        stop_offset = float(
            lane_config.get("stop_offset", zone.get("stop_offset", self.default_stop_offset))
        )
        target_x = float(pose.get("x", 0.0))
        target_y = float(pose.get("y", 0.0))
        target_yaw = float(pose.get("yaw", 0.0))
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
        self.avoidance_preferred = lane_name
        self.avoidance_other = self._opposite_lane(lane_name)
        self._reset_obstacle_avoidance()
        self.mode = "lane_follow"
        self.direct_goal = None
        self.goal = {
            "name": zone_name,
            "lane": lane_name,
            "x": stop_x,
            "y": stop_y,
            "target_x": target_x,
            "target_y": target_y,
            "yaw": travel_yaw,
            "line_yaw": line_yaw,
            "target_yaw": target_yaw,
            "stop_offset": stop_offset,
            "travel_yaw": travel_yaw,
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
            "closest_dist": None,
            "entered_pass_radius": False,
            "tol_pos": max(
                float(tol.get("pos", self.default_tol_pos)),
                self.min_arrival_radius,
            ),
        }
        self.arrival_started_at = None
        self._publish_lane(lane_name)
        self._publish_motion("start")
        self._log_zone_event(
            zone_name,
            "start",
            {
                "mode": "lane",
                "lane": lane_name,
                "arrival_mode": arrival_mode,
                "target": {"x": target_x, "y": target_y, "yaw": target_yaw},
                "stop": {"x": stop_x, "y": stop_y},
                "stop_offset": stop_offset,
                "travel_yaw": travel_yaw,
                "line_yaw": line_yaw,
                "area_radius": self.goal["area_radius"],
                "area_fallback_radius": self.goal["area_fallback_radius"],
                "stop_line_arm_radius": self.goal["stop_line_arm_radius"],
                "stop_line_lateral_radius": self.goal["stop_line_lateral_radius"],
                "stop_line_pass_margin": self.goal["stop_line_pass_margin"],
            },
        )
        self._status(
            f"moving: {zone_name} via {lane_name} "
            f"mode={arrival_mode} "
            f"stop_offset={stop_offset:.2f} travel_yaw={travel_yaw:.2f} "
            f"line_yaw={line_yaw:.2f} "
            f"stop=({stop_x:.2f},{stop_y:.2f}) "
            f"target=({target_x:.2f},{target_y:.2f})"
        )

    def _direct_goal_cb(self, msg):
        zone_name, is_cancel = self._parse_direct_goal(msg.data)
        if is_cancel:
            self._cancel("direct cancelled")
            return

        if zone_name not in self.zones:
            self._status(f"error: unknown direct zone '{zone_name}'")
            return

        zone = self.zones[zone_name]
        pose = zone.get("pose", {})
        tol = zone.get("tol", {})
        self.mode = "direct"
        self.goal = None
        self.arrival_started_at = None
        self._reset_obstacle_avoidance()
        self.direct_goal = {
            "name": zone_name,
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "tol_pos": max(float(tol.get("pos", self.direct_tol_pos)), self.direct_tol_pos),
        }
        self._publish_motion("stop")
        self._publish_direct_motion(0, 0)
        self._log_zone_event(
            zone_name,
            "direct_start",
            {
                "mode": "direct",
                "target": {"x": self.direct_goal["x"], "y": self.direct_goal["y"]},
                "tol_pos": self.direct_goal["tol_pos"],
            },
        )
        self._status(f"direct moving: {zone_name}")

    def _tick(self):
        self.timer_count += 1
        if self.use_obstacle_avoidance and self.mode != "direct":
            self._tick_obstacle_avoidance()
        if self.mode == "direct":
            self._tick_direct()
        elif self.mode == "lane_follow":
            self._tick_lane_follow()

    def _obstacle_cb(self, msg):
        self.obstacle_ahead = bool(msg.data)

    def _lane_state_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        lane = self._normalize_lane(payload.get("current_lane"), None)
        if lane in VALID_LANES:
            self.current_lane = lane

    def _tick_obstacle_avoidance(self):
        now = self._now()
        if self.obstacle_ahead:
            self.obstacle_clear_since = None
            if self.obstacle_seen_since is None:
                self.obstacle_seen_since = now
            sustained = now - self.obstacle_seen_since >= self.avoidance_commit_delay
            can_command = now - self.last_avoidance_cmd_t > self.avoidance_cooldown
            if not self.in_obstacle_avoidance and sustained and can_command:
                self._publish_lane(self.avoidance_other)
                self.last_avoidance_cmd_t = now
                self.in_obstacle_avoidance = True
                self._status(f"avoidance: lane change -> {self.avoidance_other}")
            return

        self.obstacle_seen_since = None
        if not self.in_obstacle_avoidance:
            return
        if self.obstacle_clear_since is None:
            self.obstacle_clear_since = now
            return
        if now - self.obstacle_clear_since > self.avoidance_return_delay:
            self._publish_lane(self.avoidance_preferred)
            self.last_avoidance_cmd_t = now
            self.in_obstacle_avoidance = False
            self.obstacle_clear_since = None
            self._status(f"avoidance: return -> {self.avoidance_preferred}")

    def _tick_lane_follow(self):
        if self.goal is None:
            return

        pose = self.stream.latest
        if pose is None:
            self._status("warning: waiting for gz world pose")
            return

        x, y, _yaw = pose
        dist = math.hypot(self.goal["x"] - x, self.goal["y"] - y)
        closest = self.goal.get("closest_dist")
        if closest is None or dist < closest:
            closest = dist
            self.goal["closest_dist"] = dist
        signed_to_stop = 0.0
        lateral_to_stop = dist
        crossed_stop_line = False
        inside_area = False
        if self.goal.get("arrival_mode") == "area":
            inside_area = (
                dist <= self.goal["area_radius"]
                or (
                    closest is not None
                    and closest <= self.goal["area_radius"] + 0.3
                    and dist > closest + 0.5
                )
            )
        else:
            crossed_stop_line, signed_to_stop, lateral_to_stop = self._stop_line_state(
                x, y, self.goal
            )
            if self.goal.get("area_fallback"):
                inside_area = (
                    dist <= self.goal["area_fallback_radius"]
                    or (
                        closest is not None
                        and closest <= self.goal["area_fallback_radius"]
                        and dist > closest + 0.5
                    )
                )
        if self.timer_count % 20 == 1:
            self._log_zone_event(
                self.goal["name"],
                "progress",
                {
                    "mode": "lane",
                    "lane": self.goal["lane"],
                    "arrival_mode": self.goal["arrival_mode"],
                    "pose": {"x": x, "y": y},
                    "target": {
                        "x": self.goal["target_x"],
                        "y": self.goal["target_y"],
                    },
                    "stop": {"x": self.goal["x"], "y": self.goal["y"]},
                    "dist": dist,
                    "signed_to_stop": signed_to_stop,
                    "lateral_to_stop": lateral_to_stop,
                    "closest_dist": closest,
                    "inside_area": inside_area,
                    "crossed_stop_line": crossed_stop_line,
                    "stop_offset": self.goal["stop_offset"],
                    "area_radius": self.goal["area_radius"],
                    "area_fallback_radius": self.goal["area_fallback_radius"],
                    "stop_line_arm_radius": self.goal["stop_line_arm_radius"],
                    "stop_line_lateral_radius": self.goal["stop_line_lateral_radius"],
                    "stop_line_pass_margin": self.goal["stop_line_pass_margin"],
                },
            )
            self._status(
                f"moving: {self.goal['name']} via {self.goal['lane']} "
                f"mode={self.goal['arrival_mode']} "
                f"dist={dist:.2f} tol={self.goal['tol_pos']:.2f} "
                f"line={signed_to_stop:.2f}/{lateral_to_stop:.2f} "
                f"line_stop>={self.goal['stop_line_pass_margin']:.2f} "
                f"arm_max={self.goal['stop_line_arm_radius']:.2f} "
                f"lat_max={self.goal['stop_line_lateral_radius']:.2f} "
                f"area={self.goal['area_radius']:.2f} "
                f"fallback={self.goal['area_fallback_radius']:.2f} "
                f"closest={closest:.2f} offset={self.goal['stop_offset']:.2f}"
            )
        now = self.get_clock().now().nanoseconds * 1e-9
        arrival_reason = None
        if inside_area:
            arrival_reason = "area"
        elif crossed_stop_line:
            arrival_reason = "stop_line"

        if arrival_reason is not None:
            if self.arrival_started_at is None:
                self.arrival_started_at = now
            if now - self.arrival_started_at >= self.arrival_hold_sec:
                name = self.goal["name"]
                target_x = self.goal["target_x"]
                target_y = self.goal["target_y"]
                self.goal = None
                self.mode = "idle"
                self.arrival_started_at = None
                self._publish_motion("stop")
                self._log_zone_event(
                    name,
                    "arrived",
                    {
                        "mode": "lane",
                        "reason": arrival_reason,
                        "pose": {"x": x, "y": y},
                        "target": {"x": target_x, "y": target_y},
                        "dist": dist,
                        "signed_to_stop": signed_to_stop,
                        "lateral_to_stop": lateral_to_stop,
                        "closest_dist": closest,
                    },
                )
                self._status(
                    f"arrived: {name} reason={arrival_reason} "
                    f"pose=({x:.2f},{y:.2f}) "
                    f"target=({target_x:.2f},{target_y:.2f})"
                )
        else:
            self.arrival_started_at = None

    def _tick_direct(self):
        if self.direct_goal is None:
            return

        pose = self.stream.latest
        if pose is None:
            self._status("warning: waiting for gz world pose")
            self._publish_direct_motion(0, 0)
            return

        x, y, yaw = pose
        goal = self.direct_goal
        dx = goal["x"] - x
        dy = goal["y"] - y
        dist = math.hypot(dx, dy)
        if self.timer_count % 20 == 1:
            self._log_zone_event(
                goal["name"],
                "direct_progress",
                {
                    "mode": "direct",
                    "pose": {"x": x, "y": y, "yaw": yaw},
                    "target": {"x": goal["x"], "y": goal["y"]},
                    "dist": dist,
                    "tol_pos": goal["tol_pos"],
                },
            )
            self._status(
                f"direct moving: {goal['name']} dist={dist:.2f} "
                f"tol={goal['tol_pos']:.2f}"
            )

        if dist <= goal["tol_pos"]:
            name = goal["name"]
            self.direct_goal = None
            self.mode = "idle"
            self._publish_direct_motion(0, 0)
            self._log_zone_event(
                name,
                "direct_arrived",
                {
                    "mode": "direct",
                    "pose": {"x": x, "y": y, "yaw": yaw},
                    "target": {"x": goal["x"], "y": goal["y"]},
                    "dist": dist,
                    "tol_pos": goal["tol_pos"],
                },
            )
            self._status(f"direct arrived: {name}")
            return

        heading = yaw + self.yaw_offset
        heading_err = wrap(math.atan2(dy, dx) - heading)
        angular = clamp(self.k_ang * heading_err, self.max_angular)
        # An Ackermann car cannot rotate in place.  Keep moving at a low speed
        # under large heading error so the steered front wheels can change yaw.
        if abs(heading_err) > self.direct_turn_in_place_angle:
            speed = self.direct_turn_speed
        else:
            speed = self.max_linear * min(1.0, dist / self.slow_radius)
            speed *= max(0.25, math.cos(heading_err))
        raw_speed = int(round(max(self.min_linear, min(self.max_linear, speed))))
        steering = self._yaw_rate_to_steering_command(
            angular * self.direct_steering_sign
        )
        if abs(heading_err) < self.direct_heading_deadband:
            steering = 0
        if self.timer_count % 20 == 1:
            self._status(
                f"direct cmd: heading_err={heading_err:.2f} "
                f"steering={steering} speed={raw_speed}"
            )
        self._publish_direct_motion(steering, raw_speed)

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

    def _arrival_mode_for_zone(self, zone_name):
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

    def _parse_goal(self, raw):
        text = str(raw or "").strip()
        if not text or text.lower() in {"stop", "cancel", "pause"}:
            return None, None, True

        if text.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return self._normalize_zone(text), None, False
            zone = self._normalize_zone(str(obj.get("zone") or obj.get("name") or "").strip())
            lane = self._lane_from_text(obj.get("lane"))
            if not zone or zone.lower() in {"none", "stop", "cancel"}:
                return None, None, True
            return zone, lane, False

        lane = self._lane_from_text(text)
        match = re.match(r"^(lane[12])\s*[:/]\s*(.+)$", text, flags=re.IGNORECASE)
        if match:
            lane = match.group(1).lower()
            text = match.group(2).strip()
        else:
            match = re.search(r"\b(lane[12])\b", text, flags=re.IGNORECASE)
            if match:
                lane = match.group(1).lower()
                text = (text[: match.start()] + text[match.end() :]).strip()
        if lane is not None:
            text = self._strip_lane_text(text)

        return self._normalize_zone(text), lane, False

    def _parse_direct_goal(self, raw):
        text = str(raw or "").strip()
        if not text or text.lower() in {"stop", "cancel", "pause"}:
            return None, True

        if text.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return self._normalize_zone(text), False
            zone = self._normalize_zone(str(obj.get("zone") or obj.get("name") or "").strip())
            if not zone or zone.lower() in {"none", "stop", "cancel"}:
                return None, True
            return zone, False

        return self._normalize_zone(text), False

    def _cancel(self, reason):
        if self.goal is not None:
            self._log_zone_event(
                self.goal["name"],
                "cancel",
                {"mode": "lane", "reason": reason, "lane": self.goal["lane"]},
            )
        if self.direct_goal is not None:
            self._log_zone_event(
                self.direct_goal["name"],
                "direct_cancel",
                {"mode": "direct", "reason": reason},
            )
        self.mode = "idle"
        self.goal = None
        self.direct_goal = None
        self.arrival_started_at = None
        self._reset_obstacle_avoidance()
        self._publish_motion("stop")
        self._publish_direct_motion(0, 0)
        self._status(f"idle: {reason}")

    def _reset_obstacle_avoidance(self):
        self.in_obstacle_avoidance = False
        self.obstacle_clear_since = None
        self.obstacle_seen_since = None

    def _resolve_goal_lane(self, lane):
        explicit_lane = self._normalize_lane(lane, None)
        if explicit_lane in VALID_LANES:
            return explicit_lane
        return self._normalize_lane(self.current_lane, self.default_lane)

    def _publish_lane(self, lane):
        lane = self._normalize_lane(lane, self.default_lane)
        self.current_lane = lane
        self.lane_pub.publish(String(data=lane))

    def _publish_motion(self, command):
        self.motion_pub.publish(String(data=command))

    def _publish_direct_motion(self, steering, speed):
        msg = MotionCommand()
        msg.steering = int(max(-MAX_STEERING_COMMAND, min(MAX_STEERING_COMMAND, steering)))
        msg.left_speed = int(max(0, min(255, speed)))
        msg.right_speed = int(max(0, min(255, speed)))
        self.direct_motion_pub.publish(msg)

    @staticmethod
    def _yaw_rate_to_steering_command(yaw_rate):
        yaw_rate = max(-MAX_MAPPED_YAW_RATE, min(MAX_MAPPED_YAW_RATE, yaw_rate))
        command = round((yaw_rate / MAX_MAPPED_YAW_RATE) * MAX_STEERING_COMMAND)
        return int(max(-MAX_STEERING_COMMAND, min(MAX_STEERING_COMMAND, command)))

    def _status(self, text):
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def _log_zone_event(self, zone_name, event, fields):
        if not self.zone_log_enabled or not zone_name:
            return
        record = {
            "t": round(time.time(), 3),
            "session": self.zone_log_session,
            "zone": zone_name,
            "event": event,
        }
        record.update(self._round_for_log(fields))
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(zone_name)).strip("_")
        path = os.path.join(self.zone_log_dir, f"{safe_name or 'zone'}.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.get_logger().warn(f"failed to write zone log {path}: {exc}")

    def _round_for_log(self, value):
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {k: self._round_for_log(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._round_for_log(v) for v in value]
        return value

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _normalize_lane(lane, fallback):
        lane = NavigatorNode._lane_from_text(lane)
        return lane if lane in VALID_LANES else fallback

    @staticmethod
    def _opposite_lane(lane):
        return "lane1" if lane == "lane2" else "lane2"

    @staticmethod
    def _normalize_zone(zone):
        text = str(zone or "").strip()
        compact = re.sub(r"[\s_-]+", "", text.lower())
        return ZONE_ALIASES.get(compact, text)

    @staticmethod
    def _lane_from_text(text):
        text = str(text or "").strip().lower()
        compact = re.sub(r"\s+", "", text)
        if (
            "1차선" in compact
            or "lane1" in compact
            or "laneone" in compact
            or "firstlane" in compact
            or "leftlane" in compact
            or "innerlane" in compact
        ):
            return "lane1"
        if (
            "2차선" in compact
            or "lane2" in compact
            or "lanetwo" in compact
            or "secondlane" in compact
            or "rightlane" in compact
            or "outerlane" in compact
        ):
            return "lane2"
        return text if text in VALID_LANES else None

    @staticmethod
    def _strip_lane_text(text):
        text = str(text or "")
        patterns = (
            r"1\s*차선",
            r"2\s*차선",
            r"lane\s*1",
            r"lane\s*2",
            r"lane1",
            r"lane2",
            r"first\s*lane",
            r"second\s*lane",
            r"left\s*lane",
            r"right\s*lane",
            r"inner\s*lane",
            r"outer\s*lane",
            r"따라서?",
            r"으로",
            r"로",
            r"까지",
            r"가",
            r"go\s*to",
            r"drive\s*to",
        )
        for pattern in patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
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
