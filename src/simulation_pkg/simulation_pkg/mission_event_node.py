import math
import os
import time

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from std_msgs.msg import String

from nav_vla.gz_pose import WorldPoseStream, query_world_pose, resolve_gz_bin


DEFAULT_MAP_PATH = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml"
)
TARGET_YAW_STOP_LINE_ZONES = {"T3"}

DEFAULT_EVENTS = [
    {
        "name": "M2",
        "zone": "M2",
        "approach_lane": "lane2",
        "action": "lane",
        "lane": "lane1",
    },
    {
        "name": "T2",
        "zone": "T2",
        "approach_lane": "lane1",
        "action": "lane",
        "lane": "lane2",
    },
    {
        "name": "crosswalk_stop",
        "zone": "crosswalk_stop",
        "approach_lane": "lane2",
        "action": "stop",
        "hold_sec": 5.0,
    },
]


class MissionEventNode(Node):
    """Inject simple mission events into the normal lane-following stack.

    The driving pipeline still follows lanes exactly as in driving_sim. This node
    watches Gazebo world pose and publishes:
      - /lane_mode_command at M2 and T2
      - /motion_control_command stop/start at the crosswalk/traffic-light stop
    """

    def __init__(self):
        super().__init__("mission_event_node")
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.gz_bin = resolve_gz_bin(self.declare_parameter("gz_bin", "").value)
        self.lane_topic = self.declare_parameter(
            "lane_command_topic", "/lane_mode_command"
        ).value
        self.motion_topic = self.declare_parameter(
            "motion_control_topic", "/motion_control_command"
        ).value
        self.map_path = self.declare_parameter("map_path", DEFAULT_MAP_PATH).value
        self.default_stop_offset = float(
            self.declare_parameter("stop_offset", 0.8).value
        )
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
        self.timer_period = float(self.declare_parameter("timer_period", 0.1).value)

        self.zones = self._load_zones()
        self.events = [self._build_event_goal(event) for event in DEFAULT_EVENTS]
        self.next_event_index = 0
        self.latest_xy = None
        self.stop_until = None
        self.pending_start_sent = False
        self.last_command = None
        self.last_command_until = 0.0
        self.timer_count = 0

        self.stream = WorldPoseStream(self.gz_bin, self.model_name).start()
        seed = query_world_pose(self.gz_bin, self.model_name)
        if seed is not None:
            self.stream.latest = seed

        motion_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        lane_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.lane_pub = self.create_publisher(String, self.lane_topic, lane_qos)
        self.motion_pub = self.create_publisher(String, self.motion_topic, motion_qos)
        self.create_timer(self.timer_period, self._tick)
        self.get_logger().info(
            "mission events ready: M2 lane1, T2 lane2, crosswalk_stop stop 5s "
            "(navigator stop-line based)"
        )

    def _load_zones(self):
        with open(os.path.expanduser(self.map_path), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("zones", {})

    def _build_event_goal(self, event):
        event = dict(event)
        zone_name = event.get("zone") or event["name"]
        zone = self.zones.get(zone_name)
        if not zone:
            raise RuntimeError(f"unknown mission event zone: {zone_name}")

        pose = zone.get("pose", {})
        lane_name = event.get("approach_lane", "lane2")
        lane_config = self._lane_config_for_zone(zone, lane_name)
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
        stop_offset = float(
            lane_config.get("stop_offset", zone.get("stop_offset", self.default_stop_offset))
        )
        if arrival_mode == "area":
            stop_x = target_x
            stop_y = target_y
        else:
            stop_x = target_x - math.cos(travel_yaw) * stop_offset
            stop_y = target_y - math.sin(travel_yaw) * stop_offset
        event.update(
            {
                "zone": zone_name,
                "target_x": target_x,
                "target_y": target_y,
                "stop_x": stop_x,
                "stop_y": stop_y,
                "arrival_mode": arrival_mode,
                "travel_yaw": travel_yaw,
                "line_yaw": line_yaw,
                "stop_offset": stop_offset,
                "area_radius": float(
                    lane_config.get(
                        "area_radius",
                        zone.get("area_radius", self.area_arrival_radius),
                    )
                ),
                "stop_line_arm_radius": float(
                    lane_config.get(
                        "stop_line_arm_radius",
                        zone.get("stop_line_arm_radius", self.stop_line_arm_radius),
                    )
                ),
                "stop_line_lateral_radius": float(
                    lane_config.get(
                        "stop_line_lateral_radius",
                        zone.get(
                            "stop_line_lateral_radius",
                            self.stop_line_lateral_radius,
                        ),
                    )
                ),
                "stop_line_pass_margin": float(
                    lane_config.get(
                        "stop_line_pass_margin",
                        zone.get("stop_line_pass_margin", self.stop_line_pass_margin),
                    )
                ),
            }
        )
        return event

    def _tick(self):
        self.timer_count += 1
        now = time.monotonic()
        self._republish_recent_command(now)

        if self.stop_until is not None:
            if now >= self.stop_until and not self.pending_start_sent:
                self._publish_motion("start")
                self.pending_start_sent = True
                self.stop_until = None
                self.get_logger().info("crosswalk_stop hold complete: start")
            return

        pose = self.stream.latest
        if pose is None:
            if self.timer_count % 30 == 1:
                self.get_logger().warn("waiting for Gazebo world pose")
            return

        self.latest_xy = (float(pose[0]), float(pose[1]))
        if self.next_event_index >= len(self.events):
            return

        event = self.events[self.next_event_index]
        if event.get("arrival_mode") == "area":
            reached, signed_to_stop, lateral_to_stop, dist_to_stop = self._area_state(
                self.latest_xy[0], self.latest_xy[1], event
            )
        else:
            reached, signed_to_stop, lateral_to_stop, dist_to_stop = self._stop_line_state(
                self.latest_xy[0], self.latest_xy[1], event
            )
        if self.timer_count % 20 == 1:
            if event.get("arrival_mode") == "area":
                self.get_logger().info(
                    f"next event {event['name']}: dist={dist_to_stop:.2f} "
                    f"area_radius={event['area_radius']:.2f} "
                    f"pose=({self.latest_xy[0]:.2f},{self.latest_xy[1]:.2f})"
                )
            else:
                self.get_logger().info(
                    f"next event {event['name']}: dist={dist_to_stop:.2f} "
                    f"line={signed_to_stop:.2f}/{lateral_to_stop:.2f} "
                    f"line_stop>={event['stop_line_pass_margin']:.2f} "
                    f"arm_max={event['stop_line_arm_radius']:.2f} "
                    f"lat_max={event['stop_line_lateral_radius']:.2f} "
                    f"pose=({self.latest_xy[0]:.2f},{self.latest_xy[1]:.2f})"
                )
        if not reached:
            return

        self.next_event_index += 1
        action = event["action"]
        if action == "lane":
            lane = event["lane"]
            self._publish_lane(lane)
            self.get_logger().info(
                f"{event['name']} stop-line reached: lane change -> {lane} "
                f"(line={signed_to_stop:.2f}/{lateral_to_stop:.2f}, "
                f"dist={dist_to_stop:.2f})"
            )
        elif action == "stop":
            hold_sec = float(event.get("hold_sec", 5.0))
            self._publish_motion("stop")
            self.stop_until = now + hold_sec
            self.pending_start_sent = False
            self.get_logger().info(
                f"{event['name']} stop-line reached: stop for {hold_sec:.1f}s "
                f"(line={signed_to_stop:.2f}/{lateral_to_stop:.2f}, "
                f"dist={dist_to_stop:.2f})"
            )

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

    @staticmethod
    def _line_yaw_for_zone(zone_name, zone, travel_yaw, target_yaw):
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
        if zone_name == "crosswalk_stop":
            return "area"
        return "line"

    @staticmethod
    def _lane_config_for_zone(zone, lane_name):
        configs = zone.get("lane_overrides", {})
        if not isinstance(configs, dict):
            return {}
        config = configs.get(lane_name, {})
        return config if isinstance(config, dict) else {}

    @staticmethod
    def _stop_line_state(x, y, event):
        line_yaw = event["line_yaw"]
        hx = math.cos(line_yaw)
        hy = math.sin(line_yaw)
        dx = x - event["stop_x"]
        dy = y - event["stop_y"]
        signed_to_stop = dx * hx + dy * hy
        lateral_to_stop = abs(dx * -hy + dy * hx)
        dist_to_stop = math.hypot(dx, dy)
        armed = (
            dist_to_stop <= event["stop_line_arm_radius"]
            and lateral_to_stop <= event["stop_line_lateral_radius"]
        )
        crossed = armed and signed_to_stop >= event["stop_line_pass_margin"]
        return crossed, signed_to_stop, lateral_to_stop, dist_to_stop

    @staticmethod
    def _area_state(x, y, event):
        dx = x - event["stop_x"]
        dy = y - event["stop_y"]
        dist = math.hypot(dx, dy)
        return dist <= event["area_radius"], 0.0, dist, dist

    def _publish_lane(self, lane):
        self.lane_pub.publish(String(data=lane))
        self.last_command = ("lane", lane)
        self.last_command_until = time.monotonic() + 2.0

    def _publish_motion(self, command):
        self.motion_pub.publish(String(data=command))
        self.last_command = ("motion", command)
        self.last_command_until = time.monotonic() + 2.0

    def _republish_recent_command(self, now):
        if self.last_command is None or now > self.last_command_until:
            return
        kind, value = self.last_command
        if kind == "lane":
            self.lane_pub.publish(String(data=value))
        else:
            self.motion_pub.publish(String(data=value))

    def destroy_node(self):
        try:
            self.stream.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionEventNode()
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
