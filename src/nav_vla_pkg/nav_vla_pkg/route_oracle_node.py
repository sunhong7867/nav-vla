"""Goal-conditioned oracle: the demonstrator whose steering depends on the goal.

This is the piece the whole plan turns on. Today's stack cannot produce VLA
training data, and the reason is not the model — it is that **the recorded
steering does not depend on the instruction**:

    navigator_node.py:362-363   publishes a lane name and "start", nothing else
    lane_info_extractor:176     the lane name only selects a YOLO class mask
    motion_planner_node:246     steering comes from the camera spline slope

So "drive to T2" and "drive to M3" produce identical wheel motion and differ only
in where the car is told to stop. Measured over 5,523 pose-matched pairs, the
trajectories diverge by **0.063 m**. Any policy trained on that learns to ignore
language, which is exactly what the 0.0049 rad/s goal-shuffle ablation shows.

This node replaces that path rather than patching it. It steers **itself**, by
pure pursuit along a lane centreline extracted from the track texture
(``config/track_paths.json``, see ``scripts/extract_track_paths.py``), so the
emitted command is a function of the requested lane, goal and speed.

It does not go through ``motion_planner_node``, and that is deliberate: line 37
there is ``round(7 / max_angle**3 * angle**3)``, which collapses steering onto 15
symbols. The old corpus contains exactly 9 distinct angular values. Publishing
``/cmd_vel`` directly keeps the command continuous.

Topics
------
sub  ``/oracle_goal``    ``std_msgs/String`` JSON::

        {"goal": "T2", "lane": "lane1"|"lane2", "target_speed": 1.2}
        {"goal": "stop"}

pub  ``/cmd_vel``        ``geometry_msgs/Twist``  — sole controller, continuous
pub  ``/oracle_action``  ``std_msgs/String`` JSON — every pre-quantization float,
                         which is what fills layers 1 and 2 of ``control.jsonl``
pub  ``/nav_status``     ``std_msgs/String``      — ``"arrived: <zone>"``, same
                         format ``data_engine_node.py:214`` already parses
"""

import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
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

DEFAULT_PATHS = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla_pkg/config/track_paths.json")

# Rotation from the model's reported yaw to its direction of travel. Measured,
# not assumed: straight-line runs from four headings give -90.00 deg with sd 0.00.
# See docs/ver/20260728_1713_recorder-v2-debug.md section 7.
YAW_OFFSET = -math.pi / 2
SIM_WHEEL_BASE = 2.86      # ackermann_cmd_adapter_node.py:23
SIM_MAX_STEER = 0.6        # policy_node.py:59
# Nothing tighter than this is drivable: R_min = L / tan(steer_max) = 4.18 m.
MAX_CURVATURE = math.tan(SIM_MAX_STEER) / SIM_WHEEL_BASE


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


class LanePath:
    """Closed, uniformly spaced polyline with arc-length lookup."""

    def __init__(self, pts, spacing):
        self.pts = pts
        self.spacing = spacing
        self.n = len(pts)
        self.length = self.n * spacing

    def nearest_index(self, x, y, near=None, window=60):
        """Closest path index. `near` restricts the search to avoid snapping to
        the far side of the loop where it doubles back on itself."""
        if near is None:
            rng = range(self.n)
        else:
            rng = ((near + d) % self.n for d in range(-window, window + 1))
        best, bd = 0, float("inf")
        for i in rng:
            px, py = self.pts[i]
            d = (px - x) ** 2 + (py - y) ** 2
            if d < bd:
                best, bd = i, d
        return best, math.sqrt(bd)

    def point_at(self, i):
        return self.pts[i % self.n]

    def advance(self, i, metres):
        return int(i + round(metres / self.spacing)) % self.n

    def forward_gap(self, i_from, i_to):
        """Arc length from i_from forward to i_to, always non-negative."""
        return ((i_to - i_from) % self.n) * self.spacing


class OpenPath:
    """Non-closing polyline, for a parking approach that ends at a bay.

    Kept separate from LanePath rather than adding a `closed` flag, because every
    wrap in the closed version (nearest-index search, forward_gap, advance) would
    need the opposite behaviour here, and a path that silently wraps at a dead end
    sends the car back to the aisle entrance instead of stopping.
    """

    def __init__(self, pts):
        self.pts = pts
        self.n = len(pts)
        self.cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.length = self.cum[-1]

    def nearest_index(self, x, y, near=None, window=40):
        rng = range(self.n) if near is None else range(
            max(0, near - window), min(self.n, near + window + 1))
        best, bd = 0, float("inf")
        for i in rng:
            px, py = self.pts[i]
            d = (px - x) ** 2 + (py - y) ** 2
            if d < bd:
                best, bd = i, d
        return best, math.sqrt(bd)

    def point_at(self, i):
        return self.pts[max(0, min(i, self.n - 1))]

    def advance_by(self, i, metres):
        target = self.cum[i] + metres
        j = i
        while j + 1 < self.n and self.cum[j] < target:
            j += 1
        return j

    def remaining(self, i):
        return self.length - self.cum[min(i, self.n - 1)]


class RouteOracle(Node):
    def __init__(self):
        super().__init__("route_oracle")

        self.paths_file = self.declare_parameter("paths_file", DEFAULT_PATHS).value
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.rate_hz = float(self.declare_parameter("rate_hz", 20.0).value)
        self.lookahead_base = float(self.declare_parameter("lookahead_base", 2.1).value)
        self.lookahead_gain = float(self.declare_parameter("lookahead_gain", 0.9).value)
        self.lookahead_max = float(self.declare_parameter("lookahead_max", 5.0).value)
        self.accel_limit = float(self.declare_parameter("accel_limit", 0.55).value)
        self.arrive_tol = float(self.declare_parameter("arrive_tol_m", 1.0).value)
        # A 1.0 m arrival tolerance is fine on a 142 m ring and badly wrong in a
        # 5.66 m bay: stopping a metre early leaves the tail 0.5 m outside the
        # mouth, which is what the parked screenshots showed.
        self.park_arrive_tol = float(
            self.declare_parameter("park_arrive_tol_m", 0.15).value)
        self.park_lookahead = float(
            self.declare_parameter("park_lookahead_m", 1.6).value)
        self.park_final_lookahead = float(
            self.declare_parameter("park_final_lookahead_m", 0.8).value)
        self.brake_dist = float(self.declare_parameter("brake_dist_m", 6.0).value)
        self.max_offpath = float(self.declare_parameter("max_offpath_m", 6.0).value)
        self.default_speed = float(self.declare_parameter("default_speed", 1.2).value)
        self.tf_topic = self.declare_parameter(
            "tf_topic", "/world/default/dynamic_pose/info").value

        with open(self.paths_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        sp = float(data["meta"]["spacing_m"])
        self.lanes = {k: LanePath([tuple(p) for p in data[k]], sp)
                      for k in ("lane1", "lane2") if k in data}
        self.zones = data.get("zones", {})
        # Full parking episode = ring connector + aisle + bay, as one open path.
        # Only the lane2 connector is used: lane1 sits 3.82 m from the aisle
        # entrance and the 90 deg turn needs 5.23 m of lateral room, so entering
        # from the inner lane would demand a 3.2 m radius against the vehicle's
        # 4.18 m limit. Moving the aisle 2.6 m deeper would fix that but would cut
        # the shared aisle from 5.60 m to 2.99 m, and the shared aisle is the
        # entire basis of the ordinal axis.
        conns = data.get("ring_to_aisle", {})
        self.connector = conns.get("lane2")
        self.parking = {}
        for k, v in data.get("parking", {}).items():
            bay = [tuple(p) for p in v["path"]]
            full = ([tuple(p) for p in self.connector["path"]] + bay
                    if self.connector else bay)
            self.parking[k] = {
                "path": OpenPath(bay),
                "full": OpenPath(full),
                "connector_points": len(self.connector["path"]) if self.connector else 0,
                "stop": v["stop"],
                "aisle_points": v.get("aisle_points", 0),
            }
        self.get_logger().info(
            f"paths: {', '.join(self.lanes)} — {self.lanes['lane1'].length:.1f} m loop, "
            f"{len(self.zones)} zones, {len(self.parking)} bays")

        self.gz_bin = resolve_gz_bin("")
        self.pose = None
        self._ego_idx = None
        self._probe = None

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=5)
        self.create_subscription(TFMessage, self.tf_topic, self._tf_cb, sensor_qos)
        self.create_subscription(String, "/oracle_goal", self._goal_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.act_pub = self.create_publisher(String, "/oracle_action", 10)
        self.status_pub = self.create_publisher(String, "/nav_status", 10)

        self.goal = None            # dict: zone, lane, target_speed, goal_index
        self.speed = 0.0
        self.clamped = False
        self.path_i = None
        self.create_timer(1.0 / self.rate_hz, self._control)
        self._last_log = 0.0
        self.get_logger().info("route_oracle ready — waiting on /oracle_goal")

    # ------------------------------------------------------------------ pose

    def _tf_cb(self, msg):
        self._probe = msg.transforms
        if self._ego_idx is None or self._ego_idx >= len(msg.transforms):
            return
        tf = msg.transforms[self._ego_idx].transform
        q = tf.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (tf.translation.x, tf.translation.y, yaw)

    def _identify_ego(self):
        truth = query_world_pose(self.gz_bin, self.model_name)
        if truth is None or not self._probe:
            return False
        d = [(math.hypot(t.transform.translation.x - truth[0],
                         t.transform.translation.y - truth[1]), i)
             for i, t in enumerate(self._probe)]
        d.sort()
        if d[0][0] > 2.0:
            self.get_logger().error(f"ego not found in {self.tf_topic}")
            return False
        self._ego_idx = d[0][1]
        self.get_logger().info(f"ego at dynamic_pose index {d[0][1]}")
        return True

    # ------------------------------------------------------------------ goal

    def _goal_cb(self, msg):
        try:
            req = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            req = {"goal": msg.data.strip()}
        zone = str(req.get("goal", "")).strip()
        if zone.lower() in ("stop", "cancel", ""):
            self.goal = None
            self._halt()
            self._status("idle: cancelled")
            return
        lane = str(req.get("lane", "lane2"))
        if lane not in self.lanes:
            self._status(f"error: unknown lane '{lane}'")
            return
        if zone.lower() == "cruise":
            # No endpoint: lap the lane until cancelled. The collector ends
            # these episodes by duration, so the demonstration never brakes.
            self.goal = {
                "zone": "cruise", "lane": lane, "kind": "cruise",
                "target_speed": float(req.get("target_speed",
                                              self.default_speed)),
            }
            self.path_i = None
            self._status(f"cruising: {lane} at "
                         f"{self.goal['target_speed']:.2f} m/s")
            return
        if zone in self.parking:
            # "from": "ring" prepends the connector, so an episode can start on
            # the loop and end in a bay. Default "aisle" starts at the entrance.
            frm = str(req.get("from", "aisle"))
            self.goal = {
                "zone": zone, "lane": "parking", "kind": "parking",
                "from": frm,
                "target_speed": float(req.get("target_speed", self.default_speed)),
            }
            self.path_i = None
            self._status(f"parking: {zone} from {frm} at "
                         f"{self.goal['target_speed']:.2f} m/s")
            return
        z = self.zones.get(zone)
        if z is None:
            self._status(f"error: unknown zone '{zone}'")
            return
        if not z.get("on_ring", False):
            # Parking-area zones are not reachable by following a ring lane.
            # Refuse rather than drive to the nearest ring point and call it done.
            self._status(f"error: '{zone}' is off the ring "
                         f"({z[lane]['dist_m']:.1f} m) — needs a parking path")
            return
        self.goal = {
            "zone": zone,
            "lane": lane,
            "kind": "ring",
            "target_speed": float(req.get("target_speed", self.default_speed)),
            "goal_index": int(z[lane]["index"]),
        }
        self.path_i = None
        self._status(f"moving: {zone} via {lane} at "
                     f"{self.goal['target_speed']:.2f} m/s")

    # --------------------------------------------------------------- control

    def _control(self):
        if self._ego_idx is None:
            self._identify_ego()
            return
        if self.goal is None or self.pose is None:
            return

        x, y, raw_yaw = self.pose
        heading = wrap_pi(raw_yaw + YAW_OFFSET)
        parking = self.goal["kind"] == "parking"
        if parking:
            ent = self.parking[self.goal["zone"]]
            path = ent["full"] if self.goal.get("from") == "ring" else ent["path"]
        else:
            path = self.lanes[self.goal["lane"]]

        i, off = path.nearest_index(x, y, near=self.path_i)
        self.path_i = i
        if off > self.max_offpath:
            self.goal = None
            self._halt()
            self._status(f"abort: {off:.1f} m off path")
            return

        if self.goal["kind"] == "cruise":
            # No goal index exists; nothing to arrive at, no brake ramp.
            remaining, arrived = None, False
        elif parking:
            remaining = path.remaining(i)
            arrived = remaining <= self.park_arrive_tol
        else:
            remaining = path.forward_gap(i, self.goal["goal_index"])
            arrived = (remaining <= self.arrive_tol or
                       remaining >= path.length - self.arrive_tol)
        if arrived:
            zone = self.goal["zone"]
            target = (self.parking[zone]["stop"] if parking
                      else list(path.point_at(self.goal["goal_index"])))
            ex, ey = x - target[0], y - target[1]
            self.goal = None
            self._halt()
            self._status(
                f"arrived: {zone} (err {math.hypot(ex, ey):.2f} m: "
                f"along {ex:+.2f}, lateral {ey:+.2f})")
            return

        # Speed: cruise, then a linear ramp into the goal. A learned policy can
        # only produce a stop if the demonstrations contain the deceleration.
        target = self.goal["target_speed"]
        if remaining is not None and remaining < self.brake_dist:
            target *= max(0.0, remaining / self.brake_dist)
        dt = 1.0 / self.rate_hz
        self.speed += max(-self.accel_limit * dt,
                          min(self.accel_limit * dt, target - self.speed))

        # Pure pursuit. Lookahead grows with speed so the same gains work across
        # the speed axis instead of oscillating at the top of it.
        ld = min(self.lookahead_max,
                 self.lookahead_base + self.lookahead_gain * self.speed)
        if parking:
            # Shorter lookahead in the bay: the 2.65 m turn-in radius is tighter
            # than anything on the ring, and the ring value cuts the corner.
            # Shorter still on the final straight — the curvature clamp makes the
            # turn under-steer and overshoot in +y, and a long lookahead then has
            # no distance left to null that error before the car stops.
            ld = min(ld, self.park_final_lookahead if remaining < 4.0
                     else self.park_lookahead)
            gx, gy = path.point_at(path.advance_by(i, ld))
        else:
            gx, gy = path.point_at(path.advance(i, ld))
        dx, dy = gx - x, gy - y
        c, s = math.cos(heading), math.sin(heading)
        local_x = dx * c + dy * s
        local_y = -dx * s + dy * c
        dist2 = max(local_x ** 2 + local_y ** 2, 1e-6)
        curvature_raw = 2.0 * local_y / dist2
        # Clamp to what the vehicle can actually execute. Unclamped, recovering
        # from a 0.89 m path error during the parking turn asked for 1.20 1/m —
        # a 0.83 m radius against a 4.18 m limit. The command is then physically
        # impossible, the car under-turns, the error grows and the controller
        # asks for more. Clamping keeps the demonstration inside the envelope the
        # policy will later have to respect.
        curvature = max(-MAX_CURVATURE, min(MAX_CURVATURE, curvature_raw))
        self.clamped = abs(curvature_raw) > MAX_CURVATURE
        yaw_rate = curvature * self.speed

        cmd = Twist()
        cmd.linear.x = self.speed
        cmd.angular.z = yaw_rate
        self.cmd_pub.publish(cmd)

        steer = math.atan(SIM_WHEEL_BASE * curvature)
        self.act_pub.publish(String(data=json.dumps({
            "source": "route_oracle",
            # A clamp is a rule acting on the command, so it is declared rather
            # than hidden — criterion (d) is audited from this field.
            "override": "clamp_curvature" if self.clamped else "none",
            "curvature_raw": curvature_raw,
            "curvature": curvature,
            "steer_angle_rad": steer,
            # The float that motion_planner would have rounded to one of 15
            # symbols. Recorded for comparison and for the real-car port; the
            # command actually published is the continuous one above.
            "steer_float": 7.0 * (steer / 0.6) ** 3 if abs(steer) < 0.6 else
                           math.copysign(7.0, steer),
            "steering_cmd_int": max(-7, min(7, int(round(
                7.0 * (steer / 0.6) ** 3)))),
            "target_speed_mps": self.speed,
            "yaw_rate": yaw_rate,
            "lateral_err": local_y,
            "heading_err": math.atan2(local_y, local_x),
            "s_remaining": remaining,
            "lane": self.goal["lane"],
            "kind": self.goal["kind"],
            "goal": self.goal["zone"],
            "lookahead_m": ld,
        })))

        now = time.monotonic()
        if now - self._last_log > 2.0:
            self._last_log = now
            left = "  ∞  " if remaining is None else f"{remaining:5.1f}"
            self.get_logger().info(
                f"{self.goal['zone']}/{self.goal['lane']}: {left} m left, "
                f"v={self.speed:4.2f} off={off:4.2f} m curv={curvature:+.4f}")

    def _halt(self):
        self.speed = 0.0
        self.cmd_pub.publish(Twist())

    def _status(self, text):
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)


def main():
    rclpy.init()
    node = RouteOracle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._halt()
            node.destroy_node()
        except (KeyboardInterrupt, Exception):
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
