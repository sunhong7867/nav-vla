"""Small HTTP adapter for the chat GUI's Alpamayo teacher panel.

This server is intentionally lightweight. It does not load the 10B Alpamayo
model; it validates the endpoint contract and returns a concise teacher-style
judgment from the ROS snapshot. A real Alpamayo inference process can replace
this server later as long as it keeps the same /judge JSON interface.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer


def _step_text(steps):
    if not steps:
        return "no parsed steps"
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


def _detection_text(detections):
    if not detections:
        return "no fresh visual detections"
    counts = {}
    best = {}
    for det in detections:
        name = str(det.get("class") or "unknown")
        counts[name] = counts.get(name, 0) + 1
        best[name] = max(best.get(name, 0.0), float(det.get("score") or 0.0))
    return ", ".join(
        f"{name} x{counts[name]} best={best[name]:.2f}" for name in sorted(counts)
    )


def _lane_text(lane_info):
    if not lane_info:
        return "no fresh lane geometry"
    return (
        f"slope={float(lane_info.get('slope') or 0.0):.2f}, "
        f"points={lane_info.get('point_count', 0)}, "
        f"changing={bool(lane_info.get('is_lane_changing'))}"
    )


def _path_text(path):
    if not path:
        return "no fresh path"
    first = path.get("first")
    last = path.get("last")
    if first is not None and last is not None:
        span = f"{first} -> {last}"
    else:
        span = "-"
    return (
        f"points={path.get('point_count', 0)}, "
        f"changing={bool(path.get('is_lane_changing'))}, span={span}"
    )


def _pose_text(pose):
    if not pose:
        return "no fresh odom"
    return (
        f"x={float(pose.get('x') or 0.0):.2f}, "
        f"y={float(pose.get('y') or 0.0):.2f}, "
        f"yaw={float(pose.get('yaw') or 0.0):.2f}, "
        f"v={float(pose.get('speed') or 0.0):.2f}"
    )


def _target_from_steps(steps):
    for step in steps:
        action = step.get("action")
        zone = step.get("zone")
        if action in {"drive_to_zone", "drive_direct"} and zone:
            return str(zone)
    return None


def build_reasoning(payload):
    snapshot = payload.get("snapshot") or {}
    command = snapshot.get("command") or "-"
    steps = snapshot.get("parsed_steps") or []
    current_lane = snapshot.get("current_lane") or "unknown"
    nav_status = snapshot.get("nav_status") or "-"
    dispatch = snapshot.get("last_dispatch") or "-"
    detections = snapshot.get("detections") or []
    lane_info = snapshot.get("lane_info")
    path = snapshot.get("path")
    pose = snapshot.get("pose")
    target = _target_from_steps(steps)

    checks = []
    if not steps:
        checks.append("No action plan is available yet, so the teacher cannot audit intent.")
    elif target and f"arrived: {target}" in nav_status:
        checks.append(
            f"Navigator reports arrival at {target}; verify the vehicle is physically stopped at the marked target."
        )
    elif target:
        checks.append(
            f"Target {target} is active; compare odom and navigator status until arrival is reported."
        )
    else:
        checks.append("The command is a motion/lane command without a target zone.")

    if lane_info and path:
        checks.append("Lane geometry and path planner outputs are fresh enough for lane-follow auditing.")
    elif lane_info or path:
        checks.append("Only part of the lane/path evidence is fresh, so judgment confidence is limited.")
    else:
        checks.append("No fresh lane/path evidence is available.")

    if detections:
        checks.append("Camera detections are present, but the teacher should not infer zone identity from lane-only visuals.")
    else:
        checks.append("No visual detections are available; rely on map/odom/navigation state.")

    return (
        "This local Alpamayo-compatible test adapter is reviewing the current "
        f"command, '{command}', which was parsed as {_step_text(steps)} while the "
        f"vehicle is in {current_lane}. The latest state reports navigator status "
        f"'{nav_status}', dispatch '{dispatch}', pose {_pose_text(pose)}, lane "
        f"evidence {_lane_text(lane_info)}, path evidence {_path_text(path)}, and "
        f"camera evidence {_detection_text(detections)}. "
        + " ".join(checks)
    )


class AlpamayoTeacherHandler(BaseHTTPRequestHandler):
    server_version = "NavVLAAlpamayoTeacher/0.1"

    def log_message(self, fmt, *args):
        print(f"[alpamayo_teacher_server] {self.address_string()} - {fmt % args}")

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in {"/", "/health"}:
            self._send_json(200, {"ok": True, "service": "alpamayo_teacher_server"})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/judge":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
            reasoning = build_reasoning(payload)
            self._send_json(
                200,
                {
                    "ok": True,
                    "source": "local_alpamayo_compatible_adapter",
                    "reasoning": reasoning,
                },
            )
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AlpamayoTeacherHandler)
    print(
        f"alpamayo teacher endpoint ready: http://{args.host}:{args.port}/judge"
    )
    print("mode: local adapter; replace this with real Alpamayo inference later")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
