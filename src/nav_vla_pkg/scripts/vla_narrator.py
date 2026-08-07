#!/usr/bin/env python3
"""Real-time Korean action narrator for the VLA demo.

Pure observer: subscribes to the bridge's telemetry (`/vla/plan`, the raw
chunk mirror), the active instruction, and the sim's ground-truth pose stream,
and publishes one Korean line per *event* on `/vla/narration`. It never
touches `/cmd_vel` or any command topic — nothing here can alter the eval.

Run directly (not installed as an entry point)::

    python3 src/nav_vla_pkg/scripts/vla_narrator.py

`NarratorCore` at the top is deliberately ROS-free so an offline harness can
feed it synthetic plans/poses; the rclpy wiring lives below the import guard.
"""

import json
import math
import os
import sys
import time

import yaml

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

CFG = os.path.join(PKG_PARENT, "config")
PATHS_FILE = os.path.join(CFG, "track_paths.json")
ZONE_FILE = os.path.join(CFG, "zone_map.yaml")

RATE_HZ = 10.0                 # bridge control rate; dx per step -> m/s is *10
SPEED_TREND_EPS = 0.25         # m/s difference that counts as accel/decel
TURN_YAW_RAD = 0.45            # |sum dyaw| over a chunk that counts as a turn.
                               # 0.20 fired constantly: the ring's gentle CCW
                               # curvature alone sums to ~0.2-0.4 per 3 s chunk,
                               # so only sharper-than-ring turns should speak.
CURV_TIGHT = 0.095             # 1/m: above -> 곡선 구간. The ring's own
                               # curvature tops out at ~0.12, so the old 0.12
                               # bar classified NOTHING as a curve and the
                               # tight corners all read "완만한 곡선".
                               # Measured over ring_center (±3 idx window):
                               # p75=0.090, p90=0.114 — 0.095 puts the top
                               # ~quarter (the actual corners) in 곡선 구간.
CURV_STRAIGHT = 0.05           # 1/m: below -> 직선 구간 (~37% of the ring)
ZONE_RADIUS_M = 8.0            # announce a zone approach inside this radius
MIN_GAP_S = 2.0                # minimum spacing between published lines

# The instruction sentences carry the lane as "the inner lane"/"the outer
# lane" (chat_gui_node.VLA_LANE_WORDS).
LANE_KR = {"inner": "안쪽", "outer": "바깥쪽"}

# crosswalk_stop and the plain "T3" zone are co-located in zone_map.yaml
# (same stop line, ~2 cm apart); keeping both would make the announced name
# depend on float noise. The crosswalk label is the demo-relevant one.
CROSSWALK_ZONE = "crosswalk_stop"
CROSSWALK_LABEL = "T3 횡단보도"
DROP_ZONES = {"T3"}


def load_ring(path=PATHS_FILE):
    with open(path) as f:
        return json.load(f)["ring_center"]


def load_zones(path=ZONE_FILE):
    with open(path) as f:
        doc = yaml.safe_load(f)
    zones = {}
    for name, z in (doc.get("zones") or {}).items():
        if name in DROP_ZONES:
            continue
        pose = z.get("pose") or {}
        if "x" in pose and "y" in pose:
            zones[name] = (float(pose["x"]), float(pose["y"]))
    return zones


class NarratorCore:
    """Compose logic only — no ROS, so the offline harness can drive it."""

    def __init__(self, ring, zones, turn_sign=1.0,
                 min_gap_s=MIN_GAP_S, zone_radius_m=ZONE_RADIUS_M):
        self.ring = [tuple(p) for p in ring]
        self.n = len(self.ring)
        self.zones = dict(zones)
        self.turn_sign = float(turn_sign)
        self.min_gap_s = min_gap_s
        self.zone_radius = zone_radius_m
        self._last_pose = None     # (t_mono, x, y) for measured-speed baseline
        self.instruction = None    # None = never seen, "" = explicit stop
        self.lane_kr = None
        self._cur_zone = None      # zone we are currently inside (announced)
        self._last_state = None
        self._last_line = None
        self._last_pub_t = -1e9
        self._force = False        # publish next line regardless of gap/state

    # ------------------------------------------------------------- geometry

    def curvature_at(self, x, y):
        """Local curvature of ring_center at the nearest polyline index.

        3-point circumradius over +/-3 indices: k = 2*cross(ab, ac) /
        (|ab| |bc| |ac|). Same formula the y4 speed-vs-curvature analysis
        binned the corpus with, so the 0.05/0.12 thresholds carry over.
        """
        i = min(range(self.n),
                key=lambda j: (self.ring[j][0] - x) ** 2
                + (self.ring[j][1] - y) ** 2)
        a = self.ring[(i - 3) % self.n]
        b = self.ring[i]
        c = self.ring[(i + 3) % self.n]
        ab = math.hypot(b[0] - a[0], b[1] - a[1])
        bc = math.hypot(c[0] - b[0], c[1] - b[1])
        ac = math.hypot(c[0] - a[0], c[1] - a[1])
        if ab * bc * ac == 0:
            return 0.0
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return abs(2.0 * cross / (ab * bc * ac))

    def nearest_zone(self, x, y):
        best, best_d = None, self.zone_radius
        for name, (zx, zy) in self.zones.items():
            d = math.hypot(zx - x, zy - y)
            if d < best_d:
                best, best_d = name, d
        return best

    # --------------------------------------------------------------- events

    def on_instruction(self, text):
        """Returns a line to publish, or None."""
        text = (text or "").strip()
        if text == self.instruction:
            return None
        self.instruction = text
        self._last_state = None
        self._last_line = None
        self._cur_zone = None
        if not text:
            self.lane_kr = None
            # One line, then silence: the bridge stops requesting chunks on an
            # empty instruction, so no /vla/plan will arrive to narrate.
            return "정지 명령 — 대기 중"
        low = text.lower()
        self.lane_kr = ("안쪽" if "inner" in low
                        else "바깥쪽" if "outer" in low else None)
        self._force = True         # first line after an instruction change
        return None

    def on_plan(self, actions, pose, now):
        """Evaluate one /vla/plan chunk. Returns a line to publish, or None.

        `pose` is (x, y) in the world frame or None while the ego is not yet
        identified; `now` is a monotonic timestamp.
        """
        if not self.instruction or len(actions) < 10:
            return None

        v_now = sum(abs(a[0]) for a in actions[:5]) / 5.0 * RATE_HZ
        v_end = sum(abs(a[0]) for a in actions[-5:]) / 5.0 * RATE_HZ
        # Trend against the MEASURED speed, not the chunk front: every chunk
        # ramps internally from the current state toward its target, so
        # (v_end - v_now) reads as "가속" almost always. Measured pose speed
        # between consecutive plans is the honest baseline.
        v_meas = None
        if pose is not None and self._last_pose is not None:
            dt = now - self._last_pose[0]
            if 0.2 < dt < 4.0:
                v_meas = math.hypot(pose[0] - self._last_pose[1],
                                    pose[1] - self._last_pose[2]) / dt
        if pose is not None:
            self._last_pose = (now, pose[0], pose[1])
        base = v_meas if v_meas is not None else v_now
        dv = v_end - base
        trend = ("가속" if dv > SPEED_TREND_EPS
                 else "감속" if dv < -SPEED_TREND_EPS else "유지")
        v_now = base

        yaw = self.turn_sign * sum(a[2] for a in actions)
        turn = ("좌회전 예정" if yaw > TURN_YAW_RAD
                else "우회전 예정" if yaw < -TURN_YAW_RAD else None)

        context = None
        zone = None
        if pose is not None:
            k = self.curvature_at(pose[0], pose[1])
            context = ("곡선 구간" if k > CURV_TIGHT
                       else "직선 구간" if k < CURV_STRAIGHT else "완만한 곡선")
            zone = self.nearest_zone(pose[0], pose[1])

        # Approach is announced once per visit: only on the transition into a
        # zone's radius. Leaving resets, so a later lap announces it again.
        # The tracker is committed only when a line is actually published —
        # otherwise an entry landing inside the 2 s gap would consume the
        # visit without ever being mentioned.
        zone_event = zone if zone != self._cur_zone and zone is not None else None

        state = (trend, context, zone, turn, self.lane_kr)
        line = self._compose(trend, context, turn, zone_event, v_now, v_end)

        if self._force:
            self._force = False
        elif state == self._last_state or line == self._last_line:
            return None
        elif now - self._last_pub_t < self.min_gap_s:
            return None
        self._cur_zone = zone
        self._last_state = state
        self._last_line = line
        self._last_pub_t = now
        return line

    def _compose(self, trend, context, turn, zone_event, v_now, v_end):
        prefix = f"{context} — " if context else ""
        if zone_event:
            label = (CROSSWALK_LABEL if zone_event == CROSSWALK_ZONE
                     else zone_event)
            line = f"{label} 접근"
            if trend != "유지":
                line += f" — {trend} 중"
        elif trend == "가속":
            line = f"{prefix}가속 ({v_end:.1f} m/s로)"
        elif trend == "감속":
            line = f"{prefix}감속 ({v_now:.1f} → {v_end:.1f} m/s)"
        elif self.lane_kr:
            line = f"{self.lane_kr} 차선 유지"
        else:
            line = f"{prefix}유지 ({v_now:.1f} m/s)"
        if turn:
            line += f" · {turn}"
        return line


# ---------------------------------------------------------------------- ROS
# Guarded so the offline harness can import NarratorCore from an interpreter
# without a sourced ROS environment.
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    from nav_vla_pkg.gz_pose import query_world_pose, resolve_gz_bin
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

if _ROS_OK:

    class NarratorNode(Node):

        def __init__(self):
            super().__init__("vla_narrator")
            # Sign convention of dyaw vs. left/right is UNVERIFIED in the sim.
            # If the first live test shows inverted 좌/우 labels, flip with:
            #   ros2 param set /vla_narrator turn_sign -- -1.0
            # (or relaunch with -p turn_sign:=-1.0) — no code change needed.
            turn_sign = float(self.declare_parameter("turn_sign", 1.0).value)
            tf_topic = self.declare_parameter(
                "tf_topic", "/world/default/dynamic_pose/info").value

            self.core = NarratorCore(load_ring(), load_zones(),
                                     turn_sign=turn_sign)
            self.gz = resolve_gz_bin("")
            self.ego_idx = None
            self.pose = None

            self.pub = self.create_publisher(String, "/vla/narration", 5)

            # Same dual-QoS pattern as vla_bridge_node: a TRANSIENT_LOCAL
            # subscriber alone will not hear a VOLATILE publisher (plain
            # `ros2 topic pub`), and a VOLATILE subscriber alone misses an
            # instruction published before this node attached. on_instruction
            # dedupes, so hearing the same text twice costs nothing.
            instr_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, depth=1)
            instr_qos_volatile = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE, depth=1)
            self.create_subscription(String, "/vla/instruction",
                                     self._instr_cb, instr_qos)
            self.create_subscription(String, "/vla/instruction",
                                     self._instr_cb, instr_qos_volatile)

            self.create_subscription(String, "/vla/plan", self._plan_cb, 5)

            tf_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST,
                                durability=QoSDurabilityPolicy.VOLATILE,
                                depth=5)
            self.create_subscription(TFMessage, tf_topic, self._tf_cb, tf_qos)

            self.get_logger().info(
                f"vla_narrator ready: zones={len(self.core.zones)}, "
                f"ring={self.core.n} pts, turn_sign={turn_sign:+.0f}")

        # Ego identification pattern from scripts/collect_corpus.py: the pose
        # stream carries every model unnamed, so match once against gz
        # ground truth by distance, then trust that index for the session.
        def _tf_cb(self, msg):
            if self.ego_idx is None:
                truth = query_world_pose(self.gz, "ego_vehicle")
                if truth and msg.transforms:
                    d = sorted((math.hypot(t.transform.translation.x - truth[0],
                                           t.transform.translation.y - truth[1]),
                                i)
                               for i, t in enumerate(msg.transforms))
                    if d[0][0] < 2.0:
                        self.ego_idx = d[0][1]
                        self.get_logger().info(
                            f"ego identified: transform index {self.ego_idx}")
                return
            if self.ego_idx < len(msg.transforms):
                t = msg.transforms[self.ego_idx].transform
                self.pose = (t.translation.x, t.translation.y)

        def _instr_cb(self, msg):
            line = self.core.on_instruction(msg.data)
            if line:
                self._emit(line)

        def _plan_cb(self, msg):
            try:
                actions = json.loads(msg.data).get("actions") or []
            except (json.JSONDecodeError, TypeError):
                return
            line = self.core.on_plan(actions, self.pose, time.monotonic())
            if line:
                self._emit(line)

        def _emit(self, line):
            self.pub.publish(String(data=line))
            self.get_logger().info(line)


def main():
    if not _ROS_OK:
        print("rclpy not importable — source the ROS environment first",
              file=sys.stderr)
        return 1
    rclpy.init()
    node = NarratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
