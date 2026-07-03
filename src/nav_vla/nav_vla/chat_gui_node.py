"""Simplified qwen3:4b chat GUI for lane-aware nav-vla driving.

The GUI accepts natural-language commands and publishes one of:
  - /nav_goal JSON for lane-following zone navigation
  - /direct_nav_goal plain zone names for direct shortest-path navigation
  - /lane_mode_command for lane-only changes
  - /motion_control_command for start/stop

Run with navigator_node:
    ros2 run nav_vla navigator_node
    ros2 run nav_vla chat_gui_node
"""

import json
import math
import os
import queue
import re
import base64
import csv
import datetime as dt
import io
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import scrolledtext
from tkinter import ttk

import numpy as np
import rclpy
import yaml
from interfaces_pkg.msg import DetectionArray
from interfaces_pkg.msg import LaneInfo
from interfaces_pkg.msg import PathPlanningResult
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

try:
    from nav_vla.action_policy_model import ActionPolicyPredictor
except ImportError:
    ActionPolicyPredictor = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


MODEL = "qwen3:4b"
VOICE_SAMPLE_RATE = 16000
WHISPER_MODEL = os.environ.get("NAV_VLA_WHISPER_MODEL", "base")
DEFAULT_MAP_PATH = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml"
)
DEFAULT_ACTION_POLICY_CKPT = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/train/checkpoints/action_policy.pt"
)
DEFAULT_ALPAMAYO_LOG_DIR = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/logs/alpamayo"
)
ALPAMAYO_MODEL_ID = "nvidia/Alpamayo-1.5-10B"
ALPAMAYO_REPO_URL = "https://github.com/NVlabs/alpamayo1.5"
ALPAMAYO_HF_URL = "https://huggingface.co/nvidia/Alpamayo-1.5-10B"
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
DIRECT_ONLY_ZONES = {
    "IN",
    "OUT(통과직전)",
    "OUT(통과직후)",
    "Slot1",
    "Slot2",
    "Slot3",
    "Slot4",
}

SYSTEM_TEMPLATE = """You are a ROS 2 driving-command interpreter for a small track car.
The user may write Korean or English.

Return one compact JSON object with an ordered "steps" list. Each step is one
action. A single-intent command has one step; a multi-intent command has one
step per intent, in the order the user stated them. Do not include explanations
outside JSON.

Available actions:
- drive_to_zone: drive along a lane until one listed zone is reached.
- drive_direct: ignore lanes and drive directly to one listed zone.
- change_lane: change to lane1 or lane2, without selecting a zone.
- keep_lane: keep/follow lane1 or lane2, without selecting a zone.
- stop: stop/pause/cancel driving.
- start: start/resume driving.
- none: unrelated, unsafe, or impossible request.

Available lanes:
- lane1: 1차선, lane 1, first lane, inner lane, left lane
- lane2: 2차선, lane 2, second lane, outer lane, right lane
- default: use this when the user did not explicitly specify lane1 or lane2

Zones (name : roles):
{zones}

Guidelines:
- If the user is asking a question about whether something is possible, such as
  "가능한가?", "can I", "is it possible", or "할 수 있어?", action=none.
- Position-triggered actions, where the trigger is REACHING a listed zone (words
  like "at <zone>", "<zone>에서", "<zone> 지나서", "after <zone>", "<zone>까지 가서"),
  ARE supported by sequencing through that zone as a waypoint. See the waypoint
  rule below. Time-based or sensor-based triggers ("after 5 seconds", "5초 뒤",
  "when you see an obstacle", "장애물 보이면") are NOT supported: action=none.
- If the user says change lane without specifying lane1/lane2, use the opposite
  of the current lane from the context. If current_lane=lane2, return lane=lane1.
  If current_lane=lane1, return lane=lane2.
- Do NOT change lanes just because the user says "lane", "through lane",
  "차선따라", or "차선으로". If the user asks to go to a zone by lane but does
  not explicitly say lane1/lane2/1차선/2차선, use lane=default so the current
  lane is kept.
- Treat "start line", "start 선", and "출발선" as the Start zone when the user says
  stop at, go to, or drive to that line.
- For commands like "1차선 따라 T2까지 가", one step: drive_to_zone, zone=T2, lane=lane1.
- For commands like "stop at M3", "M3에서 멈춰", "M3까지 가서 정지",
  "go to M3 and stop", or "stop at crosswalk", one step: drive_to_zone with that zone.
- Do not infer a lane from a zone name, target name, or the word "line".
- Do not infer a lane from the word "lane" alone.
- "line", "기준선", "재위치선", and "stop line" mean a target zone/line, not lane1.
- For commands like "go M2 line", one step: drive_to_zone, zone=M2, lane=default.
- For commands like "차선 무시하고 M3로 가", "최단거리로 M3", or "direct to M3",
  one step: drive_direct, zone=M3, lane=default.
- IN, OUT(통과직전), OUT(통과직후), and Slot1~Slot4 are inside the track/parking
  area, not lane-follow targets. For these zones, use drive_direct unless the
  user is only asking a question.
- For commands like "2차선으로 변경", one step: change_lane, lane=lane2, zone=null.
- Treat T1, M1, and T1/M1 as the same zone. Return zone=T1/M1 for all three.
- Treat crosswalk and 횡단보도 as crosswalk_stop. Return zone=crosswalk_stop.
- For standalone "정지", "stop", or "cancel" without a target zone, one step: stop.
- For standalone "출발", "start", "resume", or "continue" without a target zone,
  one step: start.
- Use only exact zone names from the list.
- If no zone matches for a drive_to_zone request, use action=none for that step.

Merging vs. sequencing:
- MERGE a lane choice with a single destination that is driven in that lane into
  ONE drive_to_zone step, but ONLY when the lane change is not tied to a location.
  "change lane and go to T4" or "change lane and stop at T4" -> one step:
  drive_to_zone, zone=T4, lane=the opposite of current_lane. "start drive and stop
  at T4 on lane1" -> one step: drive_to_zone, zone=T4, lane=lane1.
- WAYPOINT: when the lane change (or other action) is tied to a listed zone AND a
  further destination is given, sequence through the zone. Drive to the waypoint
  zone in the CURRENT lane (lane=default), then drive to the destination in the
  new lane. "M2에서 차선 변경하고 T4까지 가" or "change lane at M2 then go to T4" ->
  two steps: [drive_to_zone zone=M2 lane=default, drive_to_zone zone=T4
  lane=opposite]. The lane switch happens when the car reaches M2, not at the
  start, and M2 is not skipped.
- SEQUENCE when the user states actions to perform in order. Trigger words:
  "first", "then", "after", "next", "and then", "먼저", "그다음", "그리고", "-고
  나서", or two different destinations. Emit one step per intent, in order.
  "Go to start line first, then change lane" -> two steps:
  [drive_to_zone zone=Start lane=default, change_lane lane=opposite].
  "Drive to M2 then to T3" -> [drive_to_zone M2, drive_to_zone T3].
- change_lane already starts motion. So combine start/stop with change-lane into
  ONE step: "start driving and change lane" or "change lane and start driving"
  -> change_lane, lane=opposite. "change lane and stop" or "stop and change lane"
  -> stop (stopping overrides the lane change).

Schema (return exactly this shape):
{{
  "steps": [
    {{
      "action": "drive_to_zone" | "drive_direct" | "change_lane" | "keep_lane" | "stop" | "start" | "none",
      "zone": <one listed zone name or null>,
      "lane": "lane1" | "lane2" | "default"
    }}
  ],
  "reason": <short Korean or English reason>
}}"""


class ChatGuiNode(Node):
    def __init__(self):
        super().__init__("nav_vla_chat_gui_node")
        self.map_path = self.declare_parameter("map_path", DEFAULT_MAP_PATH).value
        self.host = self.declare_parameter(
            "ollama_host", "http://localhost:11434"
        ).value.rstrip("/")
        self.timeout = float(self.declare_parameter("timeout", 30.0).value)
        self.parser_backend = str(
            self.declare_parameter("parser_backend", "llm").value
        ).strip().lower()
        self.action_policy_ckpt = self.declare_parameter(
            "action_policy_ckpt", DEFAULT_ACTION_POLICY_CKPT
        ).value
        nav_goal_topic = self.declare_parameter("nav_goal_topic", "/nav_goal").value
        direct_nav_goal_topic = self.declare_parameter(
            "direct_nav_goal_topic", "/direct_nav_goal"
        ).value
        lane_command_topic = self.declare_parameter(
            "lane_command_topic", "/lane_mode_command"
        ).value
        motion_control_topic = self.declare_parameter(
            "motion_control_topic", "/motion_control_command"
        ).value
        status_topic = self.declare_parameter("status_topic", "/nav_status").value
        lane_state_topic = self.declare_parameter(
            "lane_state_topic", "/lane_mode_state"
        ).value
        detection_topic = self.declare_parameter("detection_topic", "/detections").value
        lane_info_topic = self.declare_parameter(
            "lane_info_topic", "/yolov8_lane_info"
        ).value
        path_topic = self.declare_parameter(
            "path_topic", "/path_planning_result"
        ).value
        odom_topic = self.declare_parameter("odom_topic", "/odom").value
        alpamayo_image_topic = self.declare_parameter(
            "alpamayo_image_topic", "/camera/image_raw"
        ).value
        self.alpamayo_image_max_width = int(
            self.declare_parameter("alpamayo_image_max_width", 384).value
        )
        self.alpamayo_image_quality = int(
            self.declare_parameter("alpamayo_image_quality", 75).value
        )
        self.vla_judgment_backend = str(
            self.declare_parameter("vla_judgment_backend", "local").value
        ).strip().lower()
        self.alpamayo_endpoint = str(
            self.declare_parameter("alpamayo_endpoint", "").value
        ).strip()
        self.alpamayo_model_id = str(
            self.declare_parameter("alpamayo_model_id", ALPAMAYO_MODEL_ID).value
        ).strip()
        self.alpamayo_period = float(
            self.declare_parameter("alpamayo_period", 2.0).value
        )
        self.alpamayo_timeout = float(
            self.declare_parameter("alpamayo_timeout", 3.0).value
        )
        self.alpamayo_log_dir = str(
            self.declare_parameter("alpamayo_log_dir", DEFAULT_ALPAMAYO_LOG_DIR).value
        ).strip()

        self.zones = self._load_zones()
        self.zone_names = list(self.zones)
        self.system_prompt = SYSTEM_TEMPLATE.format(zones=self._zone_lines())
        self.current_lane = "lane2"
        self.last_user_text = "-"
        self.last_action_text = "-"
        self.last_nav_status = "-"
        self.latest_detections = []
        self.latest_detection_time = None
        self.latest_lane_info = None
        self.latest_lane_info_time = None
        self.latest_path_info = None
        self.latest_path_time = None
        self.latest_pose = None
        self.latest_pose_time = None
        self.latest_alpamayo_image = None
        self.latest_alpamayo_image_time = None
        self.alpamayo_busy = False
        self.alpamayo_last_call = 0.0
        self.alpamayo_last_text = ""
        self.alpamayo_last_error = ""
        self.alpamayo_last_stamp = None
        self.alpamayo_log_jsonl = ""
        self.alpamayo_log_csv = ""
        self._init_alpamayo_logs()
        self.action_policy = None
        if self.parser_backend == "action_policy":
            if ActionPolicyPredictor is None:
                raise RuntimeError("ActionPolicyPredictor import failed")
            self.action_policy = ActionPolicyPredictor(self.action_policy_ckpt)

        transient_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.nav_goal_pub = self.create_publisher(String, nav_goal_topic, 10)
        self.direct_goal_pub = self.create_publisher(String, direct_nav_goal_topic, 10)
        self.lane_pub = self.create_publisher(String, lane_command_topic, transient_qos)
        self.motion_pub = self.create_publisher(String, motion_control_topic, transient_qos)
        self.status_q = queue.Queue()
        self.event_q = queue.Queue()
        self.last_parsed = None
        self.last_dispatch = "-"
        self._plan_lock = threading.Lock()
        self.pending_steps = []
        self._waiting = None
        self.create_subscription(String, status_topic, lambda msg: self._handle_status(msg.data), 10)
        self.create_subscription(String, lane_state_topic, self._lane_state_cb, transient_qos)
        self.create_subscription(DetectionArray, detection_topic, self._detections_cb, 10)
        self.create_subscription(LaneInfo, lane_info_topic, self._lane_info_cb, 10)
        self.create_subscription(PathPlanningResult, path_topic, self._path_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(Image, alpamayo_image_topic, self._alpamayo_image_cb, 10)

        parser_desc = (
            f"action_policy={self.action_policy_ckpt}"
            if self.parser_backend == "action_policy"
            else f"model={MODEL}"
        )
        self.get_logger().info(
            f"chat gui ready: backend={self.parser_backend}, {parser_desc}, "
            f"vla_judgment={self.vla_judgment_backend}, zones={len(self.zone_names)}"
        )

    def _init_alpamayo_logs(self):
        if not self.alpamayo_log_dir:
            return
        os.makedirs(self.alpamayo_log_dir, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.alpamayo_log_jsonl = os.path.join(
            self.alpamayo_log_dir,
            f"alpamayo_judgments_{stamp}.jsonl",
        )
        self.alpamayo_log_csv = os.path.join(
            self.alpamayo_log_dir,
            f"alpamayo_judgments_{stamp}.csv",
        )
        with open(self.alpamayo_log_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self._alpamayo_log_fields())
            writer.writeheader()

    @staticmethod
    def _alpamayo_log_fields():
        return [
            "time",
            "model",
            "source",
            "command",
            "steps",
            "current_lane",
            "nav_status",
            "dispatch",
            "pose",
            "image_count",
            "reasoning",
            "endpoint",
        ]

    def _lane_state_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        lane = str(payload.get("current_lane") or "").strip().lower()
        if lane in {"lane1", "lane2"}:
            self.current_lane = lane

    def _detections_cb(self, msg):
        detections = []
        for det in msg.detections[:12]:
            bbox = det.bbox
            detections.append({
                "class": str(det.class_name or f"class_{det.class_id}"),
                "score": float(det.score),
                "cx": float(bbox.center.position.x),
                "cy": float(bbox.center.position.y),
                "w": float(bbox.size.x),
                "h": float(bbox.size.y),
            })
        detections.sort(key=lambda item: item["score"], reverse=True)
        self.latest_detections = detections
        self.latest_detection_time = time.monotonic()

    def _lane_info_cb(self, msg):
        points = [
            (int(point.target_x), int(point.target_y))
            for point in msg.target_points[:5]
        ]
        self.latest_lane_info = {
            "slope": float(msg.slope),
            "points": points,
            "point_count": len(msg.target_points),
            "is_lane_changing": bool(msg.is_lane_changing),
        }
        self.latest_lane_info_time = time.monotonic()

    def _path_cb(self, msg):
        first = None
        last = None
        if msg.x_points and msg.y_points:
            first = (float(msg.x_points[0]), float(msg.y_points[0]))
            last = (float(msg.x_points[-1]), float(msg.y_points[-1]))
        self.latest_path_info = {
            "point_count": min(len(msg.x_points), len(msg.y_points)),
            "first": first,
            "last": last,
            "is_lane_changing": bool(msg.is_lane_changing),
        }
        self.latest_path_time = time.monotonic()

    def _odom_cb(self, msg):
        pose = msg.pose.pose
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.latest_pose = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "yaw": yaw,
            "speed": float(msg.twist.twist.linear.x),
        }
        self.latest_pose_time = time.monotonic()

    def _alpamayo_image_cb(self, msg):
        if PILImage is None:
            return
        try:
            payload = self._image_msg_to_jpeg_payload(msg)
        except Exception:
            return
        self.latest_alpamayo_image = payload
        self.latest_alpamayo_image_time = time.monotonic()

    def _image_msg_to_jpeg_payload(self, msg):
        ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ch)
        if msg.encoding in ("bgr8", "bgra8"):
            rgb = arr[:, :, :3][:, :, ::-1]
        else:
            rgb = arr[:, :, :3]
        image = PILImage.fromarray(np.ascontiguousarray(rgb))
        if self.alpamayo_image_max_width > 0 and image.width > self.alpamayo_image_max_width:
            scale = self.alpamayo_image_max_width / float(image.width)
            image = image.resize(
                (self.alpamayo_image_max_width, max(1, int(image.height * scale)))
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.alpamayo_image_quality)
        return {
            "encoding": "jpeg_base64",
            "topic_encoding": str(msg.encoding),
            "width": image.width,
            "height": image.height,
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    def _load_zones(self):
        with open(self.map_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        return data.get("zones", {})

    def _zone_lines(self):
        lines = []
        for name, zone in self.zones.items():
            roles = ", ".join(zone.get("role", []) or ["-"])
            lines.append(f"- {name} : {roles}")
        return "\n".join(lines)

    def parse_command(self, text):
        self.last_user_text = text
        if self.parser_backend == "action_policy":
            started = time.monotonic()
            plan = self.action_policy.predict(text, self.current_lane)
            plan["reason"] = (
                f"action_policy confidence={plan.get('confidence', 0.0):.2f}"
            )
            return self._normalize_plan(plan), time.monotonic() - started, None

        zone_enum = self.zone_names + ["T1", "M1", None]
        step_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "drive_to_zone",
                        "drive_direct",
                        "change_lane",
                        "keep_lane",
                        "stop",
                        "start",
                        "none",
                    ],
                },
                "zone": {"type": ["string", "null"], "enum": zone_enum},
                "lane": {"type": "string", "enum": ["lane1", "lane2", "default"]},
            },
            "required": ["action", "lane"],
        }
        schema = {
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": step_schema, "minItems": 1},
                "reason": {"type": "string"},
            },
            "required": ["steps"],
        }
        payload = {
            "model": MODEL,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "system",
                    "content": (
                        f"Runtime context: current_lane={self.current_lane}. "
                        "Use this only to resolve unspecified lane-change direction."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "format": schema,
            "options": {"temperature": 0, "num_predict": 256},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return None, time.monotonic() - started, str(exc)

        content = body.get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None, time.monotonic() - started, f"bad json: {content[:160]}"
        return self._normalize_plan(parsed), time.monotonic() - started, None

    def vla_judgment_text(self):
        if self.vla_judgment_backend == "alpamayo":
            return self._alpamayo_judgment_text()
        return self._local_vla_judgment_text()

    def _local_vla_judgment_text(self):
        parsed = self.last_parsed or {}
        steps = parsed.get("steps") or []
        detections = self._fresh(self.latest_detections, self.latest_detection_time, [])
        lane_info = self._fresh(self.latest_lane_info, self.latest_lane_info_time)
        path_info = self._fresh(self.latest_path_info, self.latest_path_time)
        pose = self._fresh(self.latest_pose, self.latest_pose_time)

        detection_text = self._detection_summary(detections)
        lane_text = self._lane_info_summary(lane_info)
        path_text = self._path_summary(path_info)
        pose_text = self._pose_summary(pose)
        step_text = self._steps_summary(steps)
        evidence = self._evidence_summary(detections, lane_info, path_info)

        lines = [
            "VLA Judgment (CoC-lite)",
            "",
            "Observation",
            f"- camera/YOLO: {detection_text}",
            f"- lane info: {lane_text}",
            f"- path planner: {path_text}",
            f"- pose: {pose_text}",
            f"- current lane: {self.current_lane}",
            f"- navigator: {self.last_nav_status}",
            "",
            "Language Intent",
            f"- command: {self.last_user_text}",
            f"- interpreted steps: {step_text}",
            "",
            "Causal Check",
            f"- perception evidence: {evidence}",
            f"- selected dispatch: {self.last_dispatch}",
            f"- action summary: {self.last_action_text}",
        ]
        return "\n".join(lines)

    def _alpamayo_judgment_text(self):
        self._maybe_request_alpamayo()
        if self.alpamayo_last_text.strip():
            return self.alpamayo_last_text.strip()
        if self.alpamayo_last_error:
            return f"Alpamayo 응답을 기다리는 중입니다. 현재 연결 상태: {self.alpamayo_last_error}"
        if not self.alpamayo_endpoint:
            return "Alpamayo endpoint가 설정되지 않았습니다."
        return "Alpamayo가 현재 장면을 분석하는 중입니다."

    def alpamayo_debug_text(self):
        header = [
            "Alpamayo Teacher",
            f"- model: {self.alpamayo_model_id}",
            f"- repo: {ALPAMAYO_REPO_URL}",
            f"- weights: {ALPAMAYO_HF_URL}",
        ]
        if not self.alpamayo_endpoint:
            header.extend([
                "- status: endpoint not configured",
                "",
                "Run chat_gui_node with:",
                "  -p vla_judgment_backend:=alpamayo",
                "  -p alpamayo_endpoint:=http://127.0.0.1:8765/judge",
                "",
                "The endpoint should accept POST JSON:",
                "  {model, prompt, snapshot}",
                "and return JSON/text with a reasoning or judgment field.",
            ])
        else:
            header.append(f"- endpoint: {self.alpamayo_endpoint}")
            if self.alpamayo_last_stamp is None:
                header.append("- status: waiting for first teacher response")
            else:
                age = time.monotonic() - self.alpamayo_last_stamp
                header.append(f"- status: last response {age:.1f}s ago")
            if self.alpamayo_last_error:
                header.append(f"- error: {self.alpamayo_last_error}")

        teacher_text = self.alpamayo_last_text.strip()
        if not teacher_text:
            teacher_text = "No Alpamayo teacher output yet."

        return "\n".join([
            *header,
            "",
            "Teacher Output",
            teacher_text,
            "",
            "Local ROS Snapshot",
            self._local_vla_judgment_text(),
        ])

    def _maybe_request_alpamayo(self):
        if not self.alpamayo_endpoint or self.alpamayo_busy:
            return
        now = time.monotonic()
        if now - self.alpamayo_last_call < max(0.5, self.alpamayo_period):
            return
        self.alpamayo_last_call = now
        self.alpamayo_busy = True
        snapshot = self._vla_snapshot()
        prompt = self._alpamayo_prompt(snapshot)
        threading.Thread(
            target=self._request_alpamayo_worker,
            args=(prompt, snapshot),
            daemon=True,
        ).start()

    def _request_alpamayo_worker(self, prompt, snapshot):
        payload = {
            "model": self.alpamayo_model_id,
            "prompt": prompt,
            "snapshot": snapshot,
        }
        request = urllib.request.Request(
            self.alpamayo_endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.alpamayo_timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            teacher_text, teacher_payload = self._extract_alpamayo_payload(raw)
            self.alpamayo_last_text = teacher_text
            self.alpamayo_last_error = ""
            self.alpamayo_last_stamp = time.monotonic()
            self._log_alpamayo_response(snapshot, teacher_text, teacher_payload)
        except Exception as exc:  # Keep GUI alive even if the external teacher is down.
            self.alpamayo_last_error = str(exc)
        finally:
            self.alpamayo_busy = False

    @staticmethod
    def _extract_alpamayo_text(raw):
        return ChatGuiNode._extract_alpamayo_payload(raw)[0]

    @staticmethod
    def _extract_alpamayo_payload(raw):
        text = raw.strip()
        if not text:
            return "Empty Alpamayo teacher response.", {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text, {"raw": text}
        for key in (
            "reasoning",
            "judgment",
            "coc",
            "chain_of_causation",
            "text",
            "output",
            "message",
        ):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), data
        return json.dumps(data, ensure_ascii=False, indent=2), data

    def _log_alpamayo_response(self, snapshot, teacher_text, teacher_payload):
        if not self.alpamayo_log_jsonl or not self.alpamayo_log_csv:
            return
        now = dt.datetime.now().isoformat(timespec="seconds")
        clean_snapshot = self._snapshot_for_log(snapshot)
        row = {
            "time": now,
            "model": str(teacher_payload.get("model") or self.alpamayo_model_id),
            "source": str(teacher_payload.get("source") or ""),
            "command": str(clean_snapshot.get("command") or ""),
            "steps": json.dumps(clean_snapshot.get("parsed_steps") or [], ensure_ascii=False),
            "current_lane": str(clean_snapshot.get("current_lane") or ""),
            "nav_status": str(clean_snapshot.get("nav_status") or ""),
            "dispatch": str(clean_snapshot.get("last_dispatch") or ""),
            "pose": json.dumps(clean_snapshot.get("pose") or {}, ensure_ascii=False),
            "image_count": str(len(clean_snapshot.get("images") or [])),
            "reasoning": teacher_text,
            "endpoint": self.alpamayo_endpoint,
        }
        record = {
            **row,
            "snapshot": clean_snapshot,
            "teacher_payload": teacher_payload,
        }
        try:
            with open(self.alpamayo_log_jsonl, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open(self.alpamayo_log_csv, "a", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self._alpamayo_log_fields())
                writer.writerow(row)
        except OSError as exc:
            self.alpamayo_last_error = f"log write failed: {exc}"

    @staticmethod
    def _snapshot_for_log(snapshot):
        clean = dict(snapshot)
        images = []
        for image in clean.get("images") or []:
            images.append({
                "encoding": image.get("encoding"),
                "topic_encoding": image.get("topic_encoding"),
                "width": image.get("width"),
                "height": image.get("height"),
                "data_bytes_base64": len(image.get("data") or ""),
            })
        clean["images"] = images
        return clean

    def _vla_snapshot(self):
        parsed = self.last_parsed or {}
        return {
            "command": self.last_user_text,
            "action_summary": self.last_action_text,
            "parsed_steps": parsed.get("steps") or [],
            "reason": parsed.get("reason", ""),
            "current_lane": self.current_lane,
            "nav_status": self.last_nav_status,
            "last_dispatch": self.last_dispatch,
            "detections": self._fresh(self.latest_detections, self.latest_detection_time, []),
            "lane_info": self._fresh(self.latest_lane_info, self.latest_lane_info_time),
            "path": self._fresh(self.latest_path_info, self.latest_path_time),
            "pose": self._fresh(self.latest_pose, self.latest_pose_time),
            "images": self._fresh(
                [self.latest_alpamayo_image] if self.latest_alpamayo_image else [],
                self.latest_alpamayo_image_time,
                [],
            ),
            "zones": self.zone_names,
        }

    def _alpamayo_prompt(self, snapshot):
        return (
            "You are Alpamayo 1.5 used as a non-controlling teacher for a small "
            "ROS2 track vehicle. Review the latest camera/perception/planner "
            "snapshot and produce one concise natural-language paragraph. "
            "Do not command the vehicle directly. Focus on whether the parsed "
            "intent, lane choice, target zone, and current motion are consistent. "
            "Do not use bullets, headings, JSON, numbered lists, or labels. "
            "Write 2 to 4 complete sentences as if explaining the current driving "
            "situation to an operator. If visual evidence is insufficient, say "
            "what is missing in the same paragraph.\n\n"
            f"Snapshot JSON:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _fresh(value, stamp, default=None, max_age=2.5):
        if stamp is None:
            return default
        if time.monotonic() - stamp > max_age:
            return default
        return value

    @staticmethod
    def _detection_summary(detections):
        if not detections:
            return "no fresh detections"
        counts = {}
        best = {}
        for det in detections:
            name = det["class"]
            counts[name] = counts.get(name, 0) + 1
            best[name] = max(best.get(name, 0.0), det["score"])
        parts = [
            f"{name} x{counts[name]} best={best[name]:.2f}"
            for name in sorted(counts)
        ]
        return ", ".join(parts)

    @staticmethod
    def _lane_info_summary(lane_info):
        if not lane_info:
            return "no fresh lane info"
        points = lane_info["points"]
        point_text = ", ".join(f"({x},{y})" for x, y in points[:3]) or "-"
        return (
            f"slope={lane_info['slope']:.2f}, "
            f"points={lane_info['point_count']} [{point_text}], "
            f"changing={lane_info['is_lane_changing']}"
        )

    @staticmethod
    def _path_summary(path_info):
        if not path_info:
            return "no fresh path"
        first = path_info["first"]
        last = path_info["last"]
        if first is None or last is None:
            span = "-"
        else:
            span = f"({first[0]:.1f},{first[1]:.1f}) -> ({last[0]:.1f},{last[1]:.1f})"
        return (
            f"points={path_info['point_count']}, "
            f"changing={path_info['is_lane_changing']}, span={span}"
        )

    @staticmethod
    def _pose_summary(pose):
        if not pose:
            return "no fresh odom"
        return (
            f"x={pose['x']:.2f}, y={pose['y']:.2f}, "
            f"yaw={pose['yaw']:.2f}, v={pose['speed']:.2f}"
        )

    @staticmethod
    def _steps_summary(steps):
        if not steps:
            return "-"
        parts = []
        for step in steps:
            action = step.get("action", "?")
            zone = step.get("zone")
            lane = step.get("lane")
            text = action
            if zone:
                text += f"({zone})"
            if lane and lane != "default":
                text += f"[{lane}]"
            parts.append(text)
        return " -> ".join(parts)

    @staticmethod
    def _evidence_summary(detections, lane_info, path_info):
        lane_seen = any(
            str(det.get("class", "")).lower() in {"lane1", "lane2"}
            for det in detections or []
        )
        if lane_seen and lane_info and path_info:
            return "lane markings, lane target points, and planned path are all fresh"
        if lane_seen and lane_info:
            return "lane markings and lane target points are fresh"
        if lane_info or path_info:
            return "planner/lane geometry is fresh, visual detections may be stale"
        if detections:
            return "visual detections are fresh, planner/lane geometry may be stale"
        return "waiting for fresh camera/perception/planner data"

    def dispatch_plan(self, plan):
        """Queue an ordered plan and dispatch steps up to the first blocking drive.

        A new plan replaces any plan still in flight. Returns a summary string of
        the steps dispatched in this synchronous burst; later steps (unblocked by
        the navigator reaching a zone) are announced through event_q.
        """
        steps = plan.get("steps", [])
        with self._plan_lock:
            self.last_parsed = {"steps": [dict(s) for s in steps], "reason": plan.get("reason", "")}
            self.pending_steps = [dict(s) for s in steps]
            self._waiting = None
            messages = self._run_pending()
        summary = " / ".join(m for m in messages if m)
        return summary or "처리할 수 있는 주행 명령을 찾지 못했습니다."

    def _run_pending(self):
        """Dispatch queued steps until a drive step starts (and we must wait for
        arrival) or the queue empties. Caller must hold self._plan_lock.
        Returns the list of message strings produced in this burst.
        """
        messages = []
        while self.pending_steps:
            step = self.pending_steps.pop(0)
            message, wait = self._dispatch_step(step)
            messages.append(message)
            if wait is not None:
                self._waiting = wait
                return messages
        self._waiting = None
        return messages

    def _dispatch_step(self, step):
        """Execute one step. Returns (message, wait) where wait is None for an
        instant step, or ("zone"|"direct", zone_name) if the step started an
        asynchronous drive that must complete before the next step runs.
        """
        action = step["action"]
        lane = step["lane"]
        zone = step.get("zone")

        if action == "drive_to_zone":
            if zone not in self.zones:
                self.last_dispatch = f"invalid zone: {zone}"
                return f"zone을 찾지 못했습니다: {zone}", None
            if self._is_direct_only_zone(zone):
                return self._dispatch_step({
                    "action": "drive_direct",
                    "zone": zone,
                    "lane": "default",
                })
            payload = {"zone": zone}
            if lane in {"lane1", "lane2"}:
                payload["lane"] = lane
                self.lane_pub.publish(String(data=lane))
            self.nav_goal_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            lane_text = f" / {lane}" if lane in {"lane1", "lane2"} else ""
            self.last_dispatch = f"/nav_goal {payload}"
            return f"{zone}{lane_text} 목표로 이동합니다.", ("zone", zone)

        if action == "drive_direct":
            if zone not in self.zones:
                self.last_dispatch = f"invalid direct zone: {zone}"
                return f"zone을 찾지 못했습니다: {zone}", None
            self.nav_goal_pub.publish(String(data="stop"))
            self.motion_pub.publish(String(data="stop"))
            self.direct_goal_pub.publish(String(data=zone))
            self.last_dispatch = f"/direct_nav_goal {zone}"
            return f"차선을 무시하고 {zone}까지 직접 이동합니다.", ("direct", zone)

        if action in {"change_lane", "keep_lane"}:
            if lane not in {"lane1", "lane2"}:
                self.last_dispatch = "missing lane"
                return "차선이 명확하지 않습니다.", None
            self.lane_pub.publish(String(data=lane))
            self.motion_pub.publish(String(data="start"))
            self.last_dispatch = f"/lane_mode_command {lane}"
            return f"{lane}으로 주행합니다.", None

        if action == "stop":
            self.motion_pub.publish(String(data="stop"))
            self.nav_goal_pub.publish(String(data="stop"))
            self.direct_goal_pub.publish(String(data="stop"))
            self.last_dispatch = "/motion_control_command stop"
            return "정지합니다.", None

        if action == "start":
            self.motion_pub.publish(String(data="start"))
            self.last_dispatch = "/motion_control_command start"
            return "주행을 시작합니다.", None

        self.last_dispatch = "none"
        return None, None

    def _handle_status(self, text):
        """Forward navigator status to the GUI and advance the plan queue when the
        drive we were waiting on completes (or is cancelled)."""
        self.last_nav_status = text
        self.status_q.put(text)
        with self._plan_lock:
            waiting = self._waiting
            if waiting is None:
                return
            kind, name = waiting
            if kind == "zone" and text.startswith("arrived:"):
                parts = text.split(None, 2)
                if len(parts) > 1 and self._zone_match(parts[1], name):
                    self._advance_locked()
            elif kind == "direct" and text.startswith("direct arrived:"):
                parts = text.split(None, 2)
                if len(parts) > 2 and self._zone_match(parts[2], name):
                    self._advance_locked()
            elif kind == "zone" and (text.startswith("idle:") or text.startswith("error:")):
                # A lane-follow drive never self-cancels, so idle/error here means the
                # drive was cancelled or failed externally; abandon the rest of the plan.
                # (A drive_direct step self-cancels /nav_goal, so its idle is ignored.)
                self.pending_steps = []
                self._waiting = None
                self.event_q.put(("system", "남은 계획 단계를 취소했습니다."))

    def _advance_locked(self):
        """Continue the plan after an arrival. Caller must hold self._plan_lock."""
        self._waiting = None
        for message in self._run_pending():
            if message:
                self.event_q.put(("assistant", message))

    def _zone_match(self, got, name):
        if got == name:
            return True
        return self._normalize_zone(got) == self._normalize_zone(name)

    @staticmethod
    def _is_direct_only_zone(zone):
        return str(zone or "") in DIRECT_ONLY_ZONES

    def _normalize_plan(self, parsed):
        raw_steps = parsed.get("steps")
        if not isinstance(raw_steps, list):
            # Tolerate a legacy single-object response.
            raw_steps = [parsed] if parsed.get("action") is not None else []
        steps = [self._normalize_step(step) for step in raw_steps if isinstance(step, dict)]
        steps = [step for step in steps if step["action"] != "none"]
        if not steps:
            steps = [{"action": "none", "zone": None, "lane": "default"}]
        return {"steps": steps, "reason": str(parsed.get("reason") or "")}

    def _normalize_step(self, step):
        action = str(step.get("action") or "none").strip()
        if action not in {
            "drive_to_zone",
            "drive_direct",
            "change_lane",
            "keep_lane",
            "stop",
            "start",
            "none",
        }:
            action = "none"
        lane = str(step.get("lane") or "default").strip().lower()
        if lane not in {"lane1", "lane2"}:
            lane = "default"
        zone = step.get("zone")
        if zone is not None:
            zone = self._normalize_zone(str(zone).strip())
        return {"action": action, "zone": zone, "lane": lane}

    def _normalize_zone(self, zone):
        if zone in self.zones:
            return zone
        compact = re.sub(r"[\s_-]+", "", str(zone or "").lower())
        return ZONE_ALIASES.get(compact, zone)


class ChatGuiWindow:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("nav-vla Lane Chat Console")
        self.root.geometry("1120x620")
        self.root.minsize(900, 500)

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("Title.TLabel", font=("TkDefaultFont", 14, "bold"))
        self.style.configure("Status.TLabel", foreground="#344054")
        self.style.configure("Primary.TButton", padding=(12, 6))
        self.style.configure("Record.TButton", padding=(12, 6))

        self.recording = False
        self.record_stream = None
        self.record_frames = []
        self.voice_busy = False
        self.whisper_model = None
        self.voice_available = sd is not None and WhisperModel is not None
        self.debug_window = None
        self.debug_log = None
        self.vla_debug_window = None
        self.vla_debug_log = None
        self.debug_lines = []
        self.status_text = tk.StringVar(value=self._debug_text())

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            header,
            text="nav-vla Lane Chat Console",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        parser_label = (
            "Backend: learned action_policy"
            if node.parser_backend == "action_policy"
            else f"Model fixed: {MODEL}"
        )
        ttk.Label(header, text=parser_label, style="Status.TLabel").pack(anchor=tk.W)

        entry_row = ttk.Frame(outer)
        entry_row.pack(fill=tk.X, pady=(0, 10))
        self.entry = ttk.Entry(entry_row)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda _event: self._send())
        self.send_button = ttk.Button(
            entry_row,
            text="Send",
            command=self._send,
            style="Primary.TButton",
        )
        self.send_button.pack(side=tk.LEFT, padx=(8, 0))
        self.voice_auto_send = tk.BooleanVar(value=True)
        self.voice_button = ttk.Button(
            entry_row,
            text="Voice",
            command=self._toggle_voice,
            style="Record.TButton",
        )
        self.voice_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(
            entry_row,
            text="Auto send",
            variable=self.voice_auto_send,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            entry_row,
            text="Debug",
            command=self._open_debug_window,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            entry_row,
            text="VLA Debug",
            command=self._open_vla_debug_window,
        ).pack(side=tk.LEFT, padx=(8, 0))
        if not self.voice_available:
            self.voice_button.config(state=tk.DISABLED)

        content = ttk.Frame(outer)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=1, uniform="main")
        content.columnconfigure(1, weight=1, uniform="main")
        content.rowconfigure(0, weight=1)

        chat_frame = ttk.LabelFrame(content, text="Conversation", padding=8)
        vla_frame = ttk.LabelFrame(content, text="VLA Judgment", padding=8)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        vla_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.log = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_config("user", foreground="#1565c0", font=("TkDefaultFont", 10, "bold"))
        self.log.tag_config("assistant", foreground="#2e7d32", font=("TkDefaultFont", 10, "bold"))
        self.log.tag_config("system", foreground="#666666")
        self.log.tag_config("error", foreground="#b00020")

        self.vla_text = scrolledtext.ScrolledText(
            vla_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=12,
        )
        self.vla_text.pack(fill=tk.BOTH, expand=True)

        self._append_debug(f"zones: {', '.join(node.zone_names)}")
        self._append_debug("예: 'M3로 가', '2차선 따라서 crosswalk_stop까지 가', '1차선으로 변경', '정지'")
        if self.voice_available:
            self._append_debug(f"voice ready: whisper={WHISPER_MODEL}")
            self._append_debug("voice auto-send: on")
        else:
            self._append_debug("voice disabled: install sounddevice and faster-whisper")
        self.entry.focus_set()
        self.root.after(150, self._drain_status)
        self.root.after(250, self._refresh_vla_panel)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._send_text(text)

    def _send_text(self, text):
        self._append("user", f"User: {text}")
        self._append_debug(f"User: {text}")
        self.send_button.config(state=tk.DISABLED)
        self.voice_button.config(state=tk.DISABLED)
        self.status_text.set(self._debug_text(extra="thinking..."))

        def worker():
            parsed, latency, error = self.node.parse_command(text)
            self.root.after(0, lambda: self._handle_result(parsed, latency, error))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_result(self, parsed, latency, error):
        self.send_button.config(state=tk.NORMAL)
        if self.voice_available and not self.recording and not self.voice_busy:
            self.voice_button.config(state=tk.NORMAL)
        if error:
            self.status_text.set(self._debug_text(extra=f"error: {error}"))
            self._append("assistant", f"Action: 오류: {error}")
            self._append_debug(f"Error: {error}")
            return
        response = self.node.dispatch_plan(parsed)
        self.node.last_action_text = response
        compact = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        self.status_text.set(self._debug_text(latency=latency))
        self._append("assistant", f"Action: {response}")
        self._append_debug(f"Action: {response}")
        self._append_debug(compact)
        self._append_debug(self._debug_text(latency=latency))

    def _toggle_voice(self):
        if not self.voice_available or self.voice_busy:
            return
        if self.recording:
            self._stop_voice_recording()
        else:
            self._start_voice_recording()

    def _start_voice_recording(self):
        self.record_frames = []

        def audio_cb(indata, _frames, _time_info, status):
            if status:
                self.root.after(0, lambda: self._append_debug(f"Voice status: {status}"))
            self.record_frames.append(indata.copy())

        try:
            self.record_stream = sd.InputStream(
                samplerate=VOICE_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=audio_cb,
            )
            self.record_stream.start()
        except Exception as exc:
            self.record_stream = None
            self.record_frames = []
            self._append_debug(f"Voice error: {exc}")
            return

        self.recording = True
        self.voice_button.config(text="Stop voice")
        self.send_button.config(state=tk.DISABLED)
        self.status_text.set(self._debug_text(extra="recording voice..."))
        self._append_debug("Voice: recording started")

    def _stop_voice_recording(self):
        stream = self.record_stream
        self.record_stream = None
        self.recording = False
        self.voice_busy = True
        self.voice_button.config(text="Transcribing...", state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                self._append_debug(f"Voice stop error: {exc}")

        frames = list(self.record_frames)
        self.record_frames = []
        if not frames:
            self._finish_voice("")
            return

        audio = np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)

        def worker():
            try:
                text = self._transcribe_voice(audio)
                error = None
            except Exception as exc:
                text = ""
                error = str(exc)
            self.root.after(0, lambda: self._finish_voice(text, error))

        threading.Thread(target=worker, daemon=True).start()

    def _transcribe_voice(self, audio):
        if self.whisper_model is None:
            self.whisper_model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
            )
        segments, _info = self.whisper_model.transcribe(
            audio,
            beam_size=3,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return self._normalize_voice_text(text)

    @staticmethod
    def _normalize_voice_text(text):
        replacements = {
            "엠 원": "M1",
            "엠원": "M1",
            "엠 투": "M2",
            "엠투": "M2",
            "엠 쓰리": "M3",
            "엠쓰리": "M3",
            "티 원": "T1",
            "티원": "T1",
            "티 투": "T2",
            "티투": "T2",
            "티 쓰리": "T3",
            "티쓰리": "T3",
            "티 포": "T4",
            "티포": "T4",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text.strip()

    def _finish_voice(self, text, error=None):
        self.voice_busy = False
        self.voice_button.config(text="Voice")
        if self.voice_available:
            self.voice_button.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        if error:
            self.status_text.set(self._debug_text(extra=f"voice error: {error}"))
            self._append_debug(f"Voice error: {error}")
            return
        if not text:
            self.status_text.set(self._debug_text(extra="voice: no speech"))
            self._append_debug("Voice: no speech")
            return
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.status_text.set(self._debug_text(extra=f"voice: {text}"))
        self._append_debug(f"Voice: {text}")
        if self.voice_auto_send.get():
            self._send_text(text)

    def _drain_status(self):
        try:
            while True:
                tag, message = self.node.event_q.get_nowait()
                if tag == "assistant":
                    self._append("assistant", f"Action: {message}")
                    self.node.last_action_text = message
                self._append_debug(f"Event[{tag}]: {message}")
        except queue.Empty:
            pass
        try:
            while True:
                status = self.node.status_q.get_nowait()
                self.status_text.set(self._debug_text(status=status))
                self._append_debug("Status: " + status)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_status)

    def _refresh_vla_panel(self):
        text = self.node.vla_judgment_text()
        self.vla_text.configure(state=tk.NORMAL)
        self.vla_text.delete("1.0", tk.END)
        self.vla_text.insert(tk.END, text)
        self.vla_text.configure(state=tk.DISABLED)
        self._refresh_vla_debug_window()
        self.root.after(500, self._refresh_vla_panel)

    def _debug_text(self, latency=None, status=None, extra=None):
        parsed = self.node.last_parsed or {}
        steps = parsed.get("steps") or []
        if steps:
            steps_text = " -> ".join(self._step_label(step) for step in steps)
        else:
            steps_text = "-"
        parser_label = (
            "learned action_policy"
            if self.node.parser_backend == "action_policy"
            else MODEL
        )
        lines = [
            f"Parser: {parser_label}",
            f"Steps: {steps_text}",
            f"Reason: {parsed.get('reason', '-')}",
            f"Dispatch: {self.node.last_dispatch}",
        ]
        if latency is not None:
            lines.append(f"Parse latency: {latency * 1000.0:.0f} ms")
        if status:
            lines.append(f"Navigator status: {status}")
        if extra:
            lines.append(str(extra))
        return "\n".join(lines)

    def _open_debug_window(self):
        if self.debug_window is not None and self.debug_window.winfo_exists():
            self.debug_window.lift()
            return

        self.debug_window = tk.Toplevel(self.root)
        self.debug_window.title("nav-vla Debug")
        self.debug_window.geometry("820x520")
        self.debug_window.minsize(560, 360)
        self.debug_window.protocol("WM_DELETE_WINDOW", self._close_debug_window)

        outer = ttk.Frame(self.debug_window, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            outer,
            textvariable=self.status_text,
            style="Status.TLabel",
            justify=tk.LEFT,
            anchor=tk.NW,
        ).pack(fill=tk.X, anchor=tk.NW, pady=(0, 8))

        self.debug_log = scrolledtext.ScrolledText(
            outer,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.debug_log.pack(fill=tk.BOTH, expand=True)
        for line in self.debug_lines:
            self._write_debug_line(line)

    def _close_debug_window(self):
        if self.debug_window is not None:
            self.debug_window.destroy()
        self.debug_window = None
        self.debug_log = None

    def _open_vla_debug_window(self):
        if self.vla_debug_window is not None and self.vla_debug_window.winfo_exists():
            self.vla_debug_window.lift()
            return

        self.vla_debug_window = tk.Toplevel(self.root)
        self.vla_debug_window.title("nav-vla VLA Debug")
        self.vla_debug_window.geometry("920x640")
        self.vla_debug_window.minsize(620, 420)
        self.vla_debug_window.protocol("WM_DELETE_WINDOW", self._close_vla_debug_window)

        outer = ttk.Frame(self.vla_debug_window, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        self.vla_debug_log = scrolledtext.ScrolledText(
            outer,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.vla_debug_log.pack(fill=tk.BOTH, expand=True)
        self._refresh_vla_debug_window()

    def _close_vla_debug_window(self):
        if self.vla_debug_window is not None:
            self.vla_debug_window.destroy()
        self.vla_debug_window = None
        self.vla_debug_log = None

    def _refresh_vla_debug_window(self):
        if self.vla_debug_log is None:
            return
        self.vla_debug_log.configure(state=tk.NORMAL)
        self.vla_debug_log.delete("1.0", tk.END)
        self.vla_debug_log.insert(tk.END, self.node.alpamayo_debug_text())
        self.vla_debug_log.configure(state=tk.DISABLED)

    def _append_debug(self, text):
        line = str(text)
        self.debug_lines.append(line)
        if len(self.debug_lines) > 1000:
            self.debug_lines = self.debug_lines[-1000:]
        self._write_debug_line(line)

    def _write_debug_line(self, line):
        if self.debug_log is None:
            return
        self.debug_log.configure(state=tk.NORMAL)
        self.debug_log.insert(tk.END, str(line) + "\n")
        self.debug_log.see(tk.END)
        self.debug_log.configure(state=tk.DISABLED)

    @staticmethod
    def _step_label(step):
        action = step.get("action", "?")
        zone = step.get("zone")
        lane = step.get("lane")
        parts = [action]
        if zone:
            parts.append(str(zone))
        if lane and lane != "default":
            parts.append(str(lane))
        return ":".join(parts)

    def _append(self, tag, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n", tag)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def close(self):
        if self.debug_window is not None and self.debug_window.winfo_exists():
            self.debug_window.destroy()
            self.debug_window = None
            self.debug_log = None
        if self.vla_debug_window is not None and self.vla_debug_window.winfo_exists():
            self.vla_debug_window.destroy()
            self.vla_debug_window = None
            self.vla_debug_log = None
        if self.record_stream is not None:
            try:
                self.record_stream.stop()
                self.record_stream.close()
            except Exception:
                pass
            self.record_stream = None
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = ChatGuiNode()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    window = ChatGuiWindow(node)
    try:
        window.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
