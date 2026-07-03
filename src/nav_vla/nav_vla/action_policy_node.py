"""ROS node for learned high-level action parsing.

Subscribes:
    /text_command (std_msgs/String)

Publishes:
    /nav_action_plan (std_msgs/String JSON)

This node only predicts the high-level plan. chat_gui_node can also load the
same checkpoint directly with parser_backend:=action_policy.
"""

import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from nav_vla.action_policy_model import ActionPolicyPredictor


DEFAULT_CKPT = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/train/checkpoints/action_policy.pt"
)


class ActionPolicyNode(Node):
    def __init__(self):
        super().__init__("action_policy_node")
        ckpt = self.declare_parameter("ckpt", DEFAULT_CKPT).value
        input_topic = self.declare_parameter("input_topic", "/text_command").value
        output_topic = self.declare_parameter("output_topic", "/nav_action_plan").value
        lane_state_topic = self.declare_parameter("lane_state_topic", "/lane_mode_state").value

        self.current_lane = "lane2"
        self.predictor = ActionPolicyPredictor(ckpt)
        self.plan_pub = self.create_publisher(String, output_topic, 10)
        self.create_subscription(String, input_topic, self._text_cb, 10)
        self.create_subscription(
            String,
            lane_state_topic,
            self._lane_state_cb,
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )
        self.get_logger().info(f"action policy ready: {ckpt}")

    def _lane_state_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        lane = str(payload.get("current_lane") or "").strip().lower()
        if lane in {"lane1", "lane2"}:
            self.current_lane = lane

    def _text_cb(self, msg):
        plan = self.predictor.predict(msg.data, self.current_lane)
        self.plan_pub.publish(String(data=json.dumps(plan, ensure_ascii=False)))
        self.get_logger().info(f"{msg.data!r} -> {plan}")


def main():
    rclpy.init()
    node = ActionPolicyNode()
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
