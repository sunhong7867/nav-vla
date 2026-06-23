import json
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from interfaces_pkg.msg import DetectionArray
from interfaces_pkg.msg import LaneInfo
from interfaces_pkg.msg import TargetPoint
from .lib import camera_perception_func_lib as CPFL


SUB_TOPIC_NAME = "detections"
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"
EDGE_IMAGE_TOPIC_NAME = "lane_edge_image"
BIRD_IMAGE_TOPIC_NAME = "lane_bird_image"
TARGET_DEBUG_IMAGE_TOPIC_NAME = "lane_target_debug_image"
MODE_TOPIC_NAME = "lane_mode_command"
LEGACY_LANE_SELECT_TOPIC = "selected_lane"
LANE_STATE_TOPIC_NAME = "lane_mode_state"
CONFIG_TOPIC_NAME = "lane_extractor_config"
SHOW_IMAGE = False
VALID_LANES = {"lane1", "lane2"}
LANE_CHANGE_DURATION_SEC = 5.0

SRC_MAT = [[244, 382], [411, 382], [493, 478], [177, 478]]
TARGET_POINT_START = 5
TARGET_POINT_STOP = 155
TARGET_POINT_STEP = 50
LANE_WIDTH_PX = 300
ROI_CUTTING_IDX = 300


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__("lane_info_extractor_node")

        self.sub_topic = self.declare_parameter("sub_detection_topic", SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter("pub_topic", PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter("show_image", SHOW_IMAGE).value
        self.current_target_lane = self._normalize_lane(
            self.declare_parameter("target_lane", "lane2").value,
            "lane2",
        )
        self.mode_topic = self.declare_parameter("mode_topic", MODE_TOPIC_NAME).value
        self.legacy_lane_select_topic = self.declare_parameter(
            "legacy_lane_select_topic", LEGACY_LANE_SELECT_TOPIC
        ).value
        self.lane_state_topic = self.declare_parameter(
            "lane_state_topic", LANE_STATE_TOPIC_NAME
        ).value
        self.config_topic = self.declare_parameter(
            "config_topic", CONFIG_TOPIC_NAME
        ).value
        self.src_mat = [point[:] for point in SRC_MAT]
        self.roi_cutting_idx = ROI_CUTTING_IDX
        self.target_point_start = TARGET_POINT_START
        self.target_point_stop = TARGET_POINT_STOP
        self.target_point_step = TARGET_POINT_STEP
        self.current_lane = self.current_target_lane
        self.transition_active = False
        self.transition_source_lane = self.current_lane
        self.transition_target_lane = self.current_target_lane
        self.transition_start_time = 0.0
        self.last_lane_infos = {"lane1": None, "lane2": None}

        self.cv_bridge = CvBridge()
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        state_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self.create_subscription(
            DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile
        )
        self.create_subscription(
            String, self.mode_topic, self.lane_command_callback, self.qos_profile
        )
        self.create_subscription(
            String, self.legacy_lane_select_topic, self.lane_command_callback, self.qos_profile
        )
        self.create_subscription(
            String, self.config_topic, self.config_callback, self.qos_profile
        )

        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)
        self.edge_image_publisher = self.create_publisher(Image, EDGE_IMAGE_TOPIC_NAME, self.qos_profile)
        self.bird_image_publisher = self.create_publisher(Image, BIRD_IMAGE_TOPIC_NAME, self.qos_profile)
        self.target_debug_image_publisher = self.create_publisher(
            Image, TARGET_DEBUG_IMAGE_TOPIC_NAME, self.qos_profile
        )
        self.state_publisher = self.create_publisher(String, self.lane_state_topic, state_qos_profile)

        self.publish_lane_state()
        self.get_logger().info(
            f"simple driving lane extractor active: target_lane={self.current_target_lane}, "
            f"src_mat={self.src_mat}"
        )

    def lane_command_callback(self, msg):
        data = (msg.data or "").strip()
        # "reset_lane:laneX" — force both current and target to laneX with no
        # transition. Used by Layer-2 trial harness right after teleporting
        # the vehicle back to the spawn pose.
        if data.lower().startswith("reset_lane:"):
            requested_lane = self._lane_from_command(data.split(":", 1)[1])
            if requested_lane is None:
                self.get_logger().warn(f"Unknown lane in reset_lane command: {data}")
                return
            self.current_lane = requested_lane
            self.current_target_lane = requested_lane
            self.transition_active = False
            self.get_logger().info(f"lane force-reset to {requested_lane}")
            self.publish_lane_state()
            return

        requested_lane = self._lane_from_command(data)
        if requested_lane is None:
            self.get_logger().warn(f"Unknown lane command: {data}")
            return

        self.current_target_lane = requested_lane
        if requested_lane != self.current_lane:
            self._start_lane_transition(requested_lane)
        else:
            self.transition_active = False
        self.get_logger().info(f"lane target changed: {self.current_target_lane}")
        self.publish_lane_state()

    def yolov8_detections_callback(self, detection_msg):
        if len(detection_msg.detections) == 0:
            return

        source_lane = self._extract_lane_info(detection_msg, self.current_lane)
        target_lane = self._extract_lane_info(detection_msg, self.current_target_lane)
        if source_lane is not None:
            self.last_lane_infos[self.current_lane] = self._copy_lane_info(source_lane)
        if target_lane is not None:
            self.last_lane_infos[self.current_target_lane] = self._copy_lane_info(target_lane)

        if self.transition_active:
            lane = self._transition_lane(source_lane, target_lane)
        else:
            lane = target_lane or self.last_lane_infos.get(self.current_target_lane)
            if lane is not None:
                lane = self._copy_lane_info(lane)
                lane.is_lane_changing = False

        if lane is None:
            return

        self.publisher.publish(lane)
        self._publish_lane_debug_images(detection_msg, self.current_target_lane, lane.target_points)

    def _extract_lane_info(self, detection_msg, lane_name):
        edge_image = CPFL.draw_edges(detection_msg, cls_name=lane_name, color=255)
        height, width = edge_image.shape[:2]
        dst_mat = [
            [round(width * 0.3), 0],
            [round(width * 0.7), 0],
            [round(width * 0.7), height],
            [round(width * 0.3), height],
        ]
        bird_image = CPFL.bird_convert(edge_image, srcmat=self.src_mat, dstmat=dst_mat)
        roi_image = CPFL.roi_rectangle_below(bird_image, cutting_idx=self.roi_cutting_idx)
        roi_image = cv2.convertScaleAbs(roi_image)

        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
        target_points = []
        for target_point_y in self._target_point_heights():
            target_point_x = CPFL.get_lane_center(
                roi_image,
                detection_height=target_point_y,
                detection_thickness=10,
                road_gradient=grad,
                lane_width=LANE_WIDTH_PX,
            )

            target_point = TargetPoint()
            target_point.target_x = round(target_point_x)
            target_point.target_y = round(target_point_y)
            target_points.append(target_point)

        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points
        lane.is_lane_changing = False
        return lane

    def _publish_lane_debug_images(self, detection_msg, lane_name, target_points):
        edge_image = CPFL.draw_edges(detection_msg, cls_name=lane_name, color=255)
        height, width = edge_image.shape[:2]
        dst_mat = [
            [round(width * 0.3), 0],
            [round(width * 0.7), 0],
            [round(width * 0.7), height],
            [round(width * 0.3), height],
        ]
        bird_image = CPFL.bird_convert(edge_image, srcmat=self.src_mat, dstmat=dst_mat)
        roi_image = CPFL.roi_rectangle_below(bird_image, cutting_idx=self.roi_cutting_idx)
        roi_image = cv2.convertScaleAbs(roi_image)

        self._publish_image(self.edge_image_publisher, edge_image, "mono8")
        self._publish_image(self.bird_image_publisher, bird_image, "mono8")
        self._publish_image(self.roi_image_publisher, roi_image, "mono8")

        if self.show_image:
            cv2.imshow("lane_edge_image", edge_image)
            cv2.imshow("lane_bird_image", bird_image)
            cv2.imshow("roi_image", roi_image)
            cv2.waitKey(1)

        self._publish_target_debug_image(roi_image, target_points)

    def _start_lane_transition(self, target_lane):
        self.transition_active = True
        self.transition_source_lane = self.current_lane
        self.transition_target_lane = target_lane
        self.transition_start_time = time.monotonic()
        self.get_logger().info(
            f"lane transition start: {self.transition_source_lane} -> {target_lane}"
        )

    def _transition_lane(self, source_lane, target_lane):
        source_lane = (
            source_lane
            or self.last_lane_infos.get(self.transition_source_lane)
            or self.last_lane_infos.get(self.current_lane)
        )
        target_lane = (
            target_lane
            or self.last_lane_infos.get(self.transition_target_lane)
            or self._offset_lane_info(source_lane, self.transition_target_lane, self.transition_source_lane)
        )

        if source_lane is None or target_lane is None:
            fallback = target_lane or source_lane
            if fallback is not None:
                fallback = self._copy_lane_info(fallback)
                fallback.is_lane_changing = True
            return fallback

        elapsed = time.monotonic() - self.transition_start_time
        progress = max(0.0, min(1.0, elapsed / max(LANE_CHANGE_DURATION_SEC, 1e-3)))
        ratio = progress * progress * (3.0 - 2.0 * progress)

        lane = self._blend_lane_infos(source_lane, target_lane, ratio)
        lane.is_lane_changing = True

        if progress >= 1.0:
            self.transition_active = False
            self.current_lane = self.transition_target_lane
            self.current_target_lane = self.transition_target_lane
            lane = self._copy_lane_info(target_lane)
            lane.is_lane_changing = False
            self.get_logger().info(f"lane transition complete: {self.current_lane}")
            self.publish_lane_state()

        return lane

    def _blend_lane_infos(self, source_lane, target_lane, ratio):
        lane = LaneInfo()
        lane.slope = float(source_lane.slope) * (1.0 - ratio) + float(target_lane.slope) * ratio
        lane.target_points = []

        source_points = list(source_lane.target_points)
        target_points = list(target_lane.target_points)
        for source_point, target_point in zip(source_points, target_points):
            point = TargetPoint()
            point.target_x = round(
                float(source_point.target_x) * (1.0 - ratio)
                + float(target_point.target_x) * ratio
            )
            point.target_y = round(
                float(source_point.target_y) * (1.0 - ratio)
                + float(target_point.target_y) * ratio
            )
            lane.target_points.append(point)

        return lane

    def _offset_lane_info(self, source_lane, target_lane, source_lane_name):
        if source_lane is None:
            return None
        if target_lane == source_lane_name:
            offset = 0.0
        else:
            offset = -LANE_WIDTH_PX if target_lane == "lane1" else LANE_WIDTH_PX

        lane = LaneInfo()
        lane.slope = source_lane.slope
        lane.target_points = []
        for source_point in source_lane.target_points:
            point = TargetPoint()
            point.target_x = round(float(source_point.target_x) + offset)
            point.target_y = round(float(source_point.target_y))
            lane.target_points.append(point)
        lane.is_lane_changing = True
        return lane

    def _copy_lane_info(self, source_lane):
        if source_lane is None:
            return None
        lane = LaneInfo()
        lane.slope = source_lane.slope
        lane.is_lane_changing = bool(getattr(source_lane, "is_lane_changing", False))
        lane.target_points = []
        for source_point in source_lane.target_points:
            point = TargetPoint()
            point.target_x = source_point.target_x
            point.target_y = source_point.target_y
            lane.target_points.append(point)
        return lane

    def config_callback(self, msg):
        try:
            config = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid lane extractor config JSON: {exc}")
            return

        if "src_mat" in config:
            src_mat = config["src_mat"]
            if self._valid_src_mat(src_mat):
                self.src_mat = [[int(point[0]), int(point[1])] for point in src_mat]
            else:
                self.get_logger().warn(f"Ignoring invalid src_mat: {src_mat}")

        if "roi_cutting_idx" in config:
            self.roi_cutting_idx = self._clamp_int(config["roi_cutting_idx"], 0, 479)

        if "target_point_start" in config:
            self.target_point_start = self._clamp_int(config["target_point_start"], 0, 300)
        if "target_point_stop" in config:
            self.target_point_stop = self._clamp_int(config["target_point_stop"], 1, 350)
        if "target_point_step" in config:
            self.target_point_step = self._clamp_int(config["target_point_step"], 1, 150)

        self.get_logger().info(
            "lane extractor config updated: "
            f"src_mat={self.src_mat}, roi_cutting_idx={self.roi_cutting_idx}, "
            f"target_points={list(self._target_point_heights())}"
        )

    def publish_lane_state(self):
        msg = String()
        msg.data = json.dumps(
            {
                "mode": "keep_lane",
                "current_lane": self.current_lane,
                "target_lane": self.current_target_lane,
                "is_lane_changing": self.transition_active,
            },
            sort_keys=True,
        )
        self.state_publisher.publish(msg)

    def _publish_target_debug_image(self, roi_image, target_points):
        debug_image = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        height = debug_image.shape[0]
        for point in target_points:
            x = int(max(0, min(debug_image.shape[1] - 1, round(float(point.target_x)))))
            y = int(max(0, min(height - 1, height - round(float(point.target_y)))))
            cv2.circle(debug_image, (x, y), 6, (0, 0, 255), -1)

        cv2.putText(
            debug_image,
            self.current_target_lane,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self._publish_image(self.target_debug_image_publisher, debug_image, "bgr8")

    def _target_point_heights(self):
        stop = max(self.target_point_stop, self.target_point_start + 1)
        step = max(1, self.target_point_step)
        return range(self.target_point_start, stop, step)

    def _publish_image(self, publisher, image, encoding):
        try:
            publisher.publish(self.cv_bridge.cv2_to_imgmsg(image, encoding=encoding))
        except Exception as exc:
            self.get_logger().error(f"Failed to publish debug image: {exc}")

    def _lane_from_command(self, command):
        command = str(command or "").strip().lower().replace(" ", "_")
        if command in {"lane1", "keep_lane1", "keep_lane:lane1", "fixed_lane1"}:
            return "lane1"
        if command in {"lane2", "keep_lane2", "keep_lane:lane2", "fixed_lane2", "outer_lane"}:
            return "lane2"
        return None

    @staticmethod
    def _normalize_lane(lane, fallback):
        lane = str(lane or "").strip().lower()
        return lane if lane in VALID_LANES else fallback

    @staticmethod
    def _valid_src_mat(src_mat):
        if not isinstance(src_mat, list) or len(src_mat) != 4:
            return False
        return all(
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(value, (int, float)) for value in point)
            for point in src_mat
        )

    @staticmethod
    def _clamp_int(value, lower, upper):
        try:
            numeric_value = int(round(float(value)))
        except (TypeError, ValueError):
            return lower
        return max(lower, min(upper, numeric_value))


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
