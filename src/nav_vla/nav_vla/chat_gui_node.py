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
import os
import queue
import re
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
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from std_msgs.msg import String

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

SYSTEM_TEMPLATE = """You are a ROS 2 driving-command interpreter for a small track car.
The user may write Korean or English.

Return exactly one compact JSON object. Do not include explanations outside JSON.

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
- Scheduled future actions like "change lane at T2" are not supported. If the
  user asks about them, action=none. If the user explicitly commands one, choose
  the closest immediate supported action only when the target and lane are clear.
- If a command contains both start/go/drive/change-lane words and a target zone,
  prefer one drive_to_zone action. Include the requested lane if lane1/lane2 is present.
- If the user says change lane without specifying lane1/lane2, use the opposite
  of the current lane from the context. If current_lane=lane2, return lane=lane1.
  If current_lane=lane1, return lane=lane2.
- Treat "start line", "start 선", and "출발선" as the Start zone when the user says
  stop at, go to, or drive to that line.
- For commands like "change lane and stop at T4", action=drive_to_zone, zone=T4,
  lane=the opposite of current_lane.
- For commands like "start drive and stop at T4. change lane1",
  action=drive_to_zone, zone=T4, lane=lane1.
- For commands like "1차선 따라 T2까지 가", action=drive_to_zone, zone=T2, lane=lane1.
- For commands like "stop at M3", "M3에서 멈춰", "M3까지 가서 정지",
  "go to M3 and stop", or "stop at crosswalk", action=drive_to_zone with that zone.
- Do not infer a lane from a zone name, target name, or the word "line".
- "line", "기준선", "재위치선", and "stop line" mean a target zone/line, not lane1.
- For commands like "go M2 line", action=drive_to_zone, zone=M2, lane=default.
- For commands like "차선 무시하고 M3로 가", "최단거리로 M3", or "direct to M3",
  action=drive_direct, zone=M3, lane=default.
- For commands like "2차선으로 변경", action=change_lane, lane=lane2, zone=null.
- Treat T1, M1, and T1/M1 as the same zone. Return zone=T1/M1 for all three.
- Treat crosswalk and 횡단보도 as crosswalk_stop. Return zone=crosswalk_stop.
- For standalone "정지", "stop", or "cancel" without a target zone, action=stop.
- For standalone "출발", "start", "resume", or "continue" without a target zone,
  action=start.
- Use only exact zone names from the list.
- If no zone matches for a drive_to_zone request, action=none.

Schema:
{{
  "action": "drive_to_zone" | "drive_direct" | "change_lane" | "keep_lane" | "stop" | "start" | "none",
  "zone": <one listed zone name or null>,
  "lane": "lane1" | "lane2" | "default",
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

        self.zones = self._load_zones()
        self.zone_names = list(self.zones)
        self.system_prompt = SYSTEM_TEMPLATE.format(zones=self._zone_lines())
        self.current_lane = "lane2"

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
        self.last_parsed = None
        self.last_dispatch = "-"
        self.create_subscription(String, status_topic, lambda msg: self.status_q.put(msg.data), 10)
        self.create_subscription(String, lane_state_topic, self._lane_state_cb, transient_qos)

        self.get_logger().info(
            f"chat gui ready: model={MODEL}, zones={len(self.zone_names)}"
        )

    def _lane_state_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        lane = str(payload.get("current_lane") or "").strip().lower()
        if lane in {"lane1", "lane2"}:
            self.current_lane = lane

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
        zone_enum = self.zone_names + ["T1", "M1", None]
        schema = {
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
                "reason": {"type": "string"},
            },
            "required": ["action", "lane"],
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
            "options": {"temperature": 0, "num_predict": 180},
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
        return self._normalize_parsed(parsed), time.monotonic() - started, None

    def dispatch(self, parsed):
        action = parsed["action"]
        lane = parsed["lane"]
        zone = parsed.get("zone")
        self.last_parsed = dict(parsed)

        if action == "drive_to_zone":
            if zone not in self.zones:
                self.last_dispatch = f"invalid zone: {zone}"
                return f"zone을 찾지 못했습니다: {zone}"
            if self._is_parking_slot(zone):
                self.last_dispatch = f"unsupported lane goal: {zone}"
                return f"{zone}은 주차 공간이라 차선 추종만으로는 도착할 수 없습니다. 주차 controller 단계에서 처리해야 합니다."
            payload = {"zone": zone}
            if lane in {"lane1", "lane2"}:
                payload["lane"] = lane
                self.lane_pub.publish(String(data=lane))
            self.motion_pub.publish(String(data="start"))
            self.nav_goal_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            lane_text = f" / {lane}" if lane in {"lane1", "lane2"} else ""
            self.last_dispatch = f"/nav_goal {payload}"
            return f"{zone}{lane_text} 목표로 이동합니다."

        if action == "drive_direct":
            if zone not in self.zones:
                self.last_dispatch = f"invalid direct zone: {zone}"
                return f"zone을 찾지 못했습니다: {zone}"
            self.nav_goal_pub.publish(String(data="stop"))
            self.motion_pub.publish(String(data="stop"))
            self.direct_goal_pub.publish(String(data=zone))
            self.last_dispatch = f"/direct_nav_goal {zone}"
            return f"차선을 무시하고 {zone}까지 직접 이동합니다."

        if action in {"change_lane", "keep_lane"}:
            if lane not in {"lane1", "lane2"}:
                self.last_dispatch = "missing lane"
                return "차선이 명확하지 않습니다."
            self.lane_pub.publish(String(data=lane))
            self.motion_pub.publish(String(data="start"))
            self.last_dispatch = f"/lane_mode_command {lane}"
            return f"{lane}으로 주행합니다."

        if action == "stop":
            self.motion_pub.publish(String(data="stop"))
            self.nav_goal_pub.publish(String(data="stop"))
            self.direct_goal_pub.publish(String(data="stop"))
            self.last_dispatch = "/motion_control_command stop"
            return "정지합니다."

        if action == "start":
            self.motion_pub.publish(String(data="start"))
            self.last_dispatch = "/motion_control_command start"
            return "주행을 시작합니다."

        self.last_dispatch = "none"
        return "처리할 수 있는 주행 명령을 찾지 못했습니다."

    @staticmethod
    def _is_parking_slot(zone):
        return str(zone or "").lower().startswith("slot")

    def _normalize_parsed(self, parsed):
        action = str(parsed.get("action") or "none").strip()
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
        lane = str(parsed.get("lane") or "default").strip().lower()
        if lane not in {"lane1", "lane2"}:
            lane = "default"
        zone = parsed.get("zone")
        if zone is not None:
            zone = self._normalize_zone(str(zone).strip())
        return {
            "action": action,
            "zone": zone,
            "lane": lane,
            "reason": str(parsed.get("reason") or ""),
        }

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

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            header,
            text="nav-vla Lane Chat Console",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(header, text=f"Model fixed: {MODEL}", style="Status.TLabel").pack(anchor=tk.W)

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
        if not self.voice_available:
            self.voice_button.config(state=tk.DISABLED)

        content = ttk.Frame(outer)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=1, uniform="panes")
        content.columnconfigure(1, weight=1, uniform="panes")
        content.rowconfigure(0, weight=1)

        chat_frame = ttk.LabelFrame(content, text="Conversation", padding=8)
        debug_frame = ttk.LabelFrame(content, text="LLM Debug", padding=8)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        debug_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.log = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_config("user", foreground="#1565c0", font=("TkDefaultFont", 10, "bold"))
        self.log.tag_config("assistant", foreground="#2e7d32", font=("TkDefaultFont", 10, "bold"))
        self.log.tag_config("system", foreground="#666666")
        self.log.tag_config("error", foreground="#b00020")

        self.status_text = tk.StringVar(value=self._debug_text())
        ttk.Label(
            debug_frame,
            textvariable=self.status_text,
            style="Status.TLabel",
            justify=tk.LEFT,
            anchor=tk.NW,
        ).pack(fill=tk.BOTH, expand=True, anchor=tk.NW)

        shortcuts = ttk.Frame(outer)
        shortcuts.pack(fill=tk.X, pady=(8, 0))
        for label, text in [
            ("Stop", "정지"),
            ("Start", "출발"),
            ("Lane 1", "1차선으로 변경"),
            ("Lane 2", "2차선으로 변경"),
            ("M3", "M3로 가"),
            ("Lane2 M3", "2차선 따라서 M3로 가"),
        ]:
            ttk.Button(shortcuts, text=label, command=lambda value=text: self._send_text(value)).pack(
                side=tk.LEFT,
                padx=(0, 6),
            )

        self._append("system", f"zones: {', '.join(node.zone_names)}")
        self._append("system", "예: 'M3로 가', '2차선 따라서 crosswalk_stop까지 가', '1차선으로 변경', '정지'")
        if self.voice_available:
            self._append("system", f"voice ready: whisper={WHISPER_MODEL}")
            self._append("system", "voice auto-send: on")
        else:
            self._append("system", "voice disabled: install sounddevice and faster-whisper")
        self.entry.focus_set()
        self.root.after(150, self._drain_status)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._send_text(text)

    def _send_text(self, text):
        self._append("user", f"User: {text}")
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
            self._append("error", f"Error: {error}")
            return
        response = self.node.dispatch(parsed)
        compact = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        self.status_text.set(self._debug_text(latency=latency))
        self._append("assistant", f"Action: {response}")
        self._append("system", compact)

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
                self.root.after(0, lambda: self._append("system", f"Voice status: {status}"))
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
            self._append("error", f"Voice error: {exc}")
            return

        self.recording = True
        self.voice_button.config(text="Stop voice")
        self.send_button.config(state=tk.DISABLED)
        self.status_text.set(self._debug_text(extra="recording voice..."))
        self._append("system", "Voice: recording started")

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
                self._append("error", f"Voice stop error: {exc}")

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
            self._append("error", f"Voice error: {error}")
            return
        if not text:
            self.status_text.set(self._debug_text(extra="voice: no speech"))
            self._append("system", "Voice: no speech")
            return
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.status_text.set(self._debug_text(extra=f"voice: {text}"))
        self._append("user", f"Voice: {text}")
        if self.voice_auto_send.get():
            self._send_text(text)

    def _drain_status(self):
        try:
            while True:
                status = self.node.status_q.get_nowait()
                self._append("system", "Status: " + status)
                self.status_text.set(self._debug_text(status=status))
        except queue.Empty:
            pass
        self.root.after(150, self._drain_status)

    def _debug_text(self, latency=None, status=None, extra=None):
        parsed = self.node.last_parsed or {}
        lines = [
            f"Model: {MODEL}",
            f"Action: {parsed.get('action', '-')}",
            f"Zone: {parsed.get('zone', '-')}",
            f"Lane: {parsed.get('lane', '-')}",
            f"Reason: {parsed.get('reason', '-')}",
            f"Dispatch: {self.node.last_dispatch}",
        ]
        if latency is not None:
            lines.append(f"LLM latency: {latency * 1000.0:.0f} ms")
        if status:
            lines.append(f"Navigator status: {status}")
        if extra:
            lines.append(str(extra))
        return "\n".join(lines)

    def _append(self, tag, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n", tag)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def close(self):
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
