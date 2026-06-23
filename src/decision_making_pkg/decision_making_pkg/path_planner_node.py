import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from scipy.interpolate import CubicSpline

from interfaces_pkg.msg import LaneInfo
from interfaces_pkg.msg import PathPlanningResult


SUB_LANE_TOPIC_NAME = "yolov8_lane_info"
PUB_TOPIC_NAME = "path_planning_result"
CAR_CENTER_POINT = (320, 179)


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        self.sub_lane_topic = self.declare_parameter("sub_lane_topic", SUB_LANE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value
        self.car_center_point = self.declare_parameter("car_center_point", CAR_CENTER_POINT).value

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.target_points = []
        self.is_lane_changing = False

        self.create_subscription(LaneInfo, self.sub_lane_topic, self.lane_callback, self.qos_profile)
        self.publisher = self.create_publisher(PathPlanningResult, self.pub_topic, self.qos_profile)

    def lane_callback(self, msg):
        self.target_points = list(msg.target_points)
        self.is_lane_changing = bool(getattr(msg, "is_lane_changing", False))

        if len(self.target_points) >= 3:
            self.plan_path()

    def plan_path(self):
        if not self.target_points:
            return

        x_points, y_points = zip(
            *[(float(point.target_x), float(point.target_y)) for point in self.target_points]
        )
        x_values = list(x_points)
        y_values = list(y_points)
        x_values.append(float(self.car_center_point[0]))
        y_values.append(float(self.car_center_point[1]))

        sorted_points = sorted(zip(y_values, x_values), key=lambda point: point[0])
        unique_points = []
        for y_value, x_value in sorted_points:
            if unique_points and abs(float(y_value) - float(unique_points[-1][0])) < 1e-6:
                previous_y, previous_x, count = unique_points[-1]
                merged_x = (previous_x * count + float(x_value)) / (count + 1)
                unique_points[-1] = (previous_y, merged_x, count + 1)
            else:
                unique_points.append((float(y_value), float(x_value), 1))

        if len(unique_points) < 2:
            return

        y_points = tuple(point[0] for point in unique_points)
        x_points = tuple(point[1] for point in unique_points)
        y_new = np.linspace(min(y_points), max(y_points), 100)
        if len(unique_points) >= 3:
            spline = CubicSpline(y_points, x_points, bc_type="natural")
            x_new = spline(y_new)
        else:
            x_new = np.interp(y_new, y_points, x_points)

        path_msg = PathPlanningResult()
        path_msg.x_points = [float(value) for value in x_new]
        path_msg.y_points = [float(value) for value in y_new]
        path_msg.is_lane_changing = self.is_lane_changing
        self.publisher.publish(path_msg)
        self.target_points.clear()


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
