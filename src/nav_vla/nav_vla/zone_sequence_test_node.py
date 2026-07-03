"""Run a fixed lane-follow zone sequence for stop-position testing."""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_SEQUENCE = "M2,T2,M3,T3,T4,Start,crosswalk_stop,T1/M1"
VALID_LANES = {"default", "lane1", "lane2"}


class ZoneSequenceTestNode(Node):
    def __init__(self):
        super().__init__("zone_sequence_test_node")
        self.sequence = self._parse_sequence(
            self.declare_parameter("sequence", DEFAULT_SEQUENCE).value
        )
        self.goal_lane = str(
            self.declare_parameter("goal_lane", "default").value
        ).strip().lower()
        if self.goal_lane not in VALID_LANES:
            self.get_logger().warn(
                f"unknown goal_lane '{self.goal_lane}', using default"
            )
            self.goal_lane = "default"
        self.timeout_sec = float(self.declare_parameter("timeout_sec", 90.0).value)
        self.settle_sec = float(self.declare_parameter("settle_sec", 1.0).value)
        self.navigator_wait_sec = float(
            self.declare_parameter("navigator_wait_sec", 10.0).value
        )
        self.arrived_zone = None
        self.current_goal = None

        self.goal_pub = self.create_publisher(String, "/nav_goal", 10)
        self.create_subscription(String, "/nav_status", self._status_cb, 10)
        self.create_timer(0.1, self._tick)
        self.started = False
        self.get_logger().info(
            f"zone sequence ready: {', '.join(self.sequence)} lane={self.goal_lane}"
        )

    @staticmethod
    def _parse_sequence(raw):
        zones = [z.strip() for z in str(raw or "").split(",") if z.strip()]
        return zones or DEFAULT_SEQUENCE.split(",")

    def _status_cb(self, msg):
        text = msg.data or ""
        if not text.startswith("arrived:"):
            return
        reached = text.split(" reason=", 1)[0].replace("arrived:", "", 1).strip()
        self.arrived_zone = reached
        self.get_logger().info(text)

    def _tick(self):
        if self.started:
            return
        self.started = True
        threading.Thread(target=self._run_sequence, daemon=True).start()

    def _run_sequence(self):
        if not self._wait_for_navigator():
            return
        ok_count = 0
        for index, zone in enumerate(self.sequence, start=1):
            self.current_goal = zone
            self.arrived_zone = None
            payload = {"zone": zone}
            if self.goal_lane in {"lane1", "lane2"}:
                payload["lane"] = self.goal_lane
            self.get_logger().info(
                f"[{index}/{len(self.sequence)}] send {payload}"
            )
            self.goal_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

            deadline = time.monotonic() + self.timeout_sec
            while time.monotonic() < deadline:
                if self.arrived_zone == zone:
                    ok_count += 1
                    self.get_logger().info(f"[{index}] arrived {zone}")
                    break
                time.sleep(0.1)
            else:
                self.get_logger().warn(f"[{index}] TIMEOUT {zone}")
                self.goal_pub.publish(String(data="stop"))
                break
            time.sleep(self.settle_sec)

        self.current_goal = None
        self.get_logger().info(
            f"zone sequence done: {ok_count}/{len(self.sequence)} reached"
        )

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


def main():
    rclpy.init()
    node = ZoneSequenceTestNode()
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
