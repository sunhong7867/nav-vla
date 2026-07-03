import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from std_msgs.msg import String


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def world_to_local(point, origin, yaw):
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return (local_x, local_y)


def densify_loop(points, spacing):
    if not points:
        return []

    dense_points = []
    for start_index in range(len(points)):
        start = points[start_index]
        end = points[(start_index + 1) % len(points)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(segment_length / spacing)))

        for step in range(steps):
            ratio = step / steps
            dense_points.append((
                start[0] + dx * ratio,
                start[1] + dy * ratio,
            ))

    return dense_points


class SimpleTrackDriverNode(Node):
    def __init__(self):
        super().__init__("simple_track_driver_node")

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.motion_control_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            self.qos_profile,
        )
        self.motion_control_topic = self.declare_parameter(
            "motion_control_topic", "motion_control_command"
        ).value
        self.motion_control_sub = self.create_subscription(
            String,
            self.motion_control_topic,
            self.motion_control_callback,
            self.motion_control_qos_profile,
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            self.qos_profile,
        )

        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.current_speed = 0.0
        self.have_odom = False
        self.motion_stopped = False
        self.closest_index = 0
        self.loop_count = 0

        # Approximate lane2 centerline in the /odom frame. The current Gazebo
        # bridge publishes odometry in world-like coordinates, so these points
        # should not be transformed into a spawn-relative local frame.
        lane2_world = [
            (3.37, 24.59),
            (3.20, 16.60),
            (3.05, 7.40),
            (3.05, -3.30),
            (3.35, -13.70),
            (6.80, -21.30),
            (14.80, -24.35),
            (24.00, -22.10),
            (31.70, -15.55),
            (34.95, -5.65),
            (34.35, 5.30),
            (30.60, 14.70),
            (23.25, 21.05),
            (13.80, 23.80),
            (6.20, 24.70),
        ]
        self.path_points = densify_loop(lane2_world, spacing=0.35)

        self.base_speed = 1.05
        self.min_speed = 0.40
        self.max_speed = 1.35
        self.base_lookahead = 2.1
        self.max_lookahead = 3.3
        self.max_angular_speed = 0.95
        self.max_linear_accel = 0.55

        self.get_logger().info(
            f"simple track driver initialized with {len(self.path_points)} path points"
        )

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)

        self.position = (position.x, position.y)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_speed = msg.twist.twist.linear.x
        self.have_odom = True

    def motion_control_callback(self, msg):
        command = str(msg.data or "").strip().lower()
        if command in {"stop", "pause"}:
            self.motion_stopped = True
            self.publish_stop()
            self.get_logger().info("motion control: stop")
        elif command in {"start", "resume"}:
            self.motion_stopped = False
            self.get_logger().info("motion control: start")
        else:
            self.get_logger().warn(f"unknown motion control command: {msg.data}")

    def control_loop(self):
        if not self.path_points:
            return

        if self.motion_stopped:
            self.publish_stop()
            return

        lookahead = min(
            self.max_lookahead,
            self.base_lookahead + max(0.0, self.current_speed) * 0.7,
        )

        self.closest_index = self._find_closest_index(self.closest_index)
        target_index = self._find_lookahead_index(self.closest_index, lookahead)
        target_x, target_y = self.path_points[target_index]

        dx = target_x - self.position[0]
        dy = target_y - self.position[1]
        local_x = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
        local_y = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy

        if local_x < 0.2:
            target_index = self._find_lookahead_index(target_index, lookahead * 1.5)
            target_x, target_y = self.path_points[target_index]
            dx = target_x - self.position[0]
            dy = target_y - self.position[1]
            local_x = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
            local_y = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy

        pursuit_distance = max(lookahead, math.hypot(local_x, local_y))
        heading_error = math.atan2(local_y, max(local_x, 1e-3))
        curvature = (2.0 * local_y) / max(pursuit_distance * pursuit_distance, 1e-3)

        speed_scale = 1.0 - min(0.65, abs(heading_error) * 0.45 + abs(curvature) * 0.9)
        target_speed = self.base_speed * max(0.42, speed_scale)
        target_speed = max(self.min_speed, min(self.max_speed, target_speed))

        accel_limit = self.max_linear_accel * self.timer_period
        next_speed = self.current_speed + max(
            -accel_limit,
            min(accel_limit, target_speed - self.current_speed),
        )

        cmd = Twist()
        cmd.linear.x = next_speed
        cmd.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, curvature * next_speed),
        )
        self.cmd_pub.publish(cmd)

        if not self.have_odom:
            self._integrate_fallback_pose(cmd)

        self.loop_count += 1
        if self.loop_count % 20 == 1:
            self.get_logger().info(
                "track drive: "
                f"pos=({self.position[0]:.2f}, {self.position[1]:.2f}) "
                f"target=({target_x:.2f}, {target_y:.2f}) "
                f"v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )

    def _integrate_fallback_pose(self, cmd):
        self.position = (
            self.position[0] + cmd.linear.x * math.cos(self.yaw) * self.timer_period,
            self.position[1] + cmd.linear.x * math.sin(self.yaw) * self.timer_period,
        )
        self.yaw = normalize_angle(self.yaw + cmd.angular.z * self.timer_period)
        self.current_speed = cmd.linear.x

    def _find_closest_index(self, start_index):
        best_index = start_index
        best_distance = float("inf")
        search_radius = min(len(self.path_points), 45)

        for offset in range(-10, search_radius):
            index = (start_index + offset) % len(self.path_points)
            px, py = self.path_points[index]
            distance = math.hypot(px - self.position[0], py - self.position[1])
            if distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index

    def _find_lookahead_index(self, start_index, lookahead):
        accumulated = 0.0
        current_index = start_index

        for _ in range(len(self.path_points)):
            next_index = (current_index + 1) % len(self.path_points)
            current_point = self.path_points[current_index]
            next_point = self.path_points[next_index]
            accumulated += math.hypot(
                next_point[0] - current_point[0],
                next_point[1] - current_point[1],
            )
            current_index = next_index
            if accumulated >= lookahead:
                return current_index

        return start_index

    def publish_stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTrackDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
