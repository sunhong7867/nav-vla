# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import List, Dict
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints
from torch import cuda

from sensor_msgs.msg import Image
from interfaces_pkg.msg import Point2D
from interfaces_pkg.msg import BoundingBox2D
from interfaces_pkg.msg import Mask
from interfaces_pkg.msg import KeyPoint2D
from interfaces_pkg.msg import KeyPoint2DArray
from interfaces_pkg.msg import Detection
from interfaces_pkg.msg import DetectionArray

from std_srvs.srv import SetBool


class Yolov8Node(LifecycleNode):

    def __init__(self, **kwargs) -> None:
        super().__init__("yolov8_node", **kwargs)
        
        #---------------Variable Setting---------------
        # 딥러닝 모델 pt 파일명 작성
        #self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("model", "best_cap.pt")
        
        # 추론 하드웨어 선택 (cpu / gpu) 
        #self.declare_parameter("device", "cpu")
        self.declare_parameter("device", "cuda:0")
        #----------------------------------------------
        
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("enable", True)
        self.declare_parameter("inference_period", 0.0)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("allowed_class_names", ["lane1", "lane2"])
        self.declare_parameter("ignore_class_names", ["crosswalk"])
        self.declare_parameter("merge_lane_instances", True)
        self.declare_parameter("lane_instance_class_names", ["lane1", "lane2"])
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_topic", "yolov8_seg_debug_image")
        self.declare_parameter("image_reliability",
                               QoSReliabilityPolicy.RELIABLE)

        self.get_logger().info('Yolov8Node created')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Configuring {self.get_name()}')

        self.model = self.get_parameter(
            "model").get_parameter_value().string_value

        self.device = self.get_parameter(
            "device").get_parameter_value().string_value

        self.threshold = self.get_parameter(
            "threshold").get_parameter_value().double_value

        self.enable = self.get_parameter(
            "enable").get_parameter_value().bool_value

        self.inference_period = self.get_parameter(
            "inference_period").get_parameter_value().double_value

        self.imgsz = self.get_parameter(
            "imgsz").get_parameter_value().integer_value
        self.ignore_class_names = {
            self._normalize_class_name(class_name)
            for class_name in self.get_parameter(
                "ignore_class_names").get_parameter_value().string_array_value
            if class_name.strip()
        }
        self.allowed_class_names = {
            self._normalize_class_name(class_name)
            for class_name in self.get_parameter(
                "allowed_class_names").get_parameter_value().string_array_value
            if class_name.strip()
        }
        self.merge_lane_instances = self.get_parameter(
            "merge_lane_instances").get_parameter_value().bool_value
        self.lane_instance_class_names = {
            self._normalize_class_name(class_name)
            for class_name in self.get_parameter(
                "lane_instance_class_names").get_parameter_value().string_array_value
            if class_name.strip()
        }
        self.publish_debug_image = self.get_parameter(
            "publish_debug_image").get_parameter_value().bool_value
        self.debug_image_topic = self.get_parameter(
            "debug_image_topic").get_parameter_value().string_value

        self.last_inference_time = 0.0
        self._class_to_color = {}

        self.reliability = self.get_parameter(
            "image_reliability").get_parameter_value().integer_value

        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self._pub = self.create_lifecycle_publisher(
            DetectionArray, "detections", 10)
        self._debug_pub = self.create_lifecycle_publisher(
            Image, self.debug_image_topic, 10)
        self._srv = self.create_service(
            SetBool, "enable", self.enable_cb
        )
        self.cv_bridge = CvBridge()

        return TransitionCallbackReturn.SUCCESS

    def enable_cb(self, request, response):
        self.enable = request.data
        response.success = True
        return response

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Activating {self.get_name()}')

        try:
            self.yolo = YOLO(self.model)  # 모델 로딩
            self.yolo.fuse()
        except FileNotFoundError:
            self.get_logger().error(f"Error: Model file '{self.model}' not found!")
            return TransitionCallbackReturn.FAILURE
        except Exception as e:
            self.get_logger().error(f"Error while loading model '{self.model}': {str(e)}")
            return TransitionCallbackReturn.FAILURE

        # subs
        self._sub = self.create_subscription(
            Image,
            "camera/image_raw",
            self.image_cb,
            self.image_qos_profile
        )

        super().on_activate(state)

        return TransitionCallbackReturn.SUCCESS


    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Deactivating {self.get_name()}')

        del self.yolo
        if 'cuda' in self.device:
            self.get_logger().info("Clearing CUDA cache")
            cuda.empty_cache()

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f'Cleaning up {self.get_name()}')

        self.destroy_publisher(self._pub)
        self.destroy_publisher(self._debug_pub)

        del self.image_qos_profile

        return TransitionCallbackReturn.SUCCESS

    def parse_hypothesis(self, results: Results) -> List[Dict]:

        hypothesis_list = []

        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": self.yolo.names[int(box_data.cls)],
                "score": float(box_data.conf)
            }
            hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:

        boxes_list = []

        box_data: Boxes
        for box_data in results.boxes:

            msg = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])

            # append msg
            boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [create_point2d(float(ele[0]), float(ele[1]))
                        for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def image_cb(self, msg: Image) -> None:
        if self.enable:
            now = time.monotonic()
            if now - self.last_inference_time < self.inference_period:
                return
            self.last_inference_time = now

            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                device=self.device,
                imgsz=self.imgsz,
            )
            results: Results = results[0].cpu()

            if results.boxes:
                hypothesis = self.parse_hypothesis(results)
                boxes = self.parse_boxes(results)

            if results.masks:
                masks = self.parse_masks(results)

            if results.keypoints:
                keypoints = self.parse_keypoints(results)

            # create detection msgs
            detections_msg = DetectionArray()

            for i in range(len(results)):

                aux_msg = Detection()

                if results.boxes:
                    normalized_class_name = self._normalize_class_name(hypothesis[i]["class_name"])
                    if self.allowed_class_names and normalized_class_name not in self.allowed_class_names:
                        continue
                    if normalized_class_name in self.ignore_class_names:
                        continue

                    aux_msg.class_id = hypothesis[i]["class_id"]
                    aux_msg.class_name = hypothesis[i]["class_name"]
                    aux_msg.score = hypothesis[i]["score"]

                    aux_msg.bbox = boxes[i]

                if results.masks:
                    aux_msg.mask = masks[i]

                if results.keypoints:
                    aux_msg.keypoints = keypoints[i]

                detections_msg.detections.append(aux_msg)

            if self.merge_lane_instances:
                detections_msg.detections = self._merge_instances_by_class(
                    detections_msg.detections,
                    self.lane_instance_class_names,
                    cv_image.shape[:2],
                )

            # publish detections
            detections_msg.header = msg.header
            self._pub.publish(detections_msg)
            self.publish_yolo_debug_image(cv_image, detections_msg, msg.header)

            del results
            del cv_image

    def publish_yolo_debug_image(self, cv_image, detections_msg, header):
        if not self.publish_debug_image:
            return

        debug_image = cv_image.copy()
        for detection in detections_msg.detections:
            color = self._color_for_class(detection.class_name)
            self._draw_detection(debug_image, detection, color)

        debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = header
        self._debug_pub.publish(debug_msg)

    def _draw_detection(self, image, detection, color):
        bbox = detection.bbox
        if bbox.size.x > 0 and bbox.size.y > 0:
            min_pt = (
                round(bbox.center.position.x - bbox.size.x / 2.0),
                round(bbox.center.position.y - bbox.size.y / 2.0),
            )
            max_pt = (
                round(bbox.center.position.x + bbox.size.x / 2.0),
                round(bbox.center.position.y + bbox.size.y / 2.0),
            )
            cv2.rectangle(image, min_pt, max_pt, color, 2)
            label = f"{detection.class_name} {detection.score:.2f}"
            text_org = (min_pt[0] + 4, max(18, min_pt[1] + 18))
            cv2.putText(
                image,
                label,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        if detection.mask.data:
            mask_array = np.array([
                [int(round(point.x)), int(round(point.y))]
                for point in detection.mask.data
            ], dtype=np.int32)
            if len(mask_array) >= 3:
                layer = image.copy()
                cv2.fillPoly(layer, [mask_array], color)
                cv2.addWeighted(image, 0.55, layer, 0.45, 0, image)
                cv2.polylines(image, [mask_array], True, color, 2, cv2.LINE_AA)

    def _merge_instances_by_class(self, detections, class_names, image_shape):
        grouped = {}
        ordered_classes = []
        passthrough = []

        for detection in detections:
            normalized_class_name = self._normalize_class_name(detection.class_name)
            if normalized_class_name not in class_names:
                passthrough.append(detection)
                continue
            if normalized_class_name not in grouped:
                grouped[normalized_class_name] = []
                ordered_classes.append(normalized_class_name)
            grouped[normalized_class_name].append(detection)

        merged = []
        for class_name in ordered_classes:
            class_detections = grouped[class_name]
            if len(class_detections) == 1:
                merged.append(class_detections[0])
            else:
                merged.append(self._merge_detection_group(class_detections, image_shape))

        return passthrough + merged

    def _merge_detection_group(self, detections, image_shape):
        height, width = image_shape
        mask_image = np.zeros((height, width), dtype=np.uint8)

        x_min = width
        y_min = height
        x_max = 0
        y_max = 0
        best_detection = max(detections, key=lambda detection: detection.score)

        for detection in detections:
            bbox = detection.bbox
            x_min = min(x_min, bbox.center.position.x - bbox.size.x / 2.0)
            y_min = min(y_min, bbox.center.position.y - bbox.size.y / 2.0)
            x_max = max(x_max, bbox.center.position.x + bbox.size.x / 2.0)
            y_max = max(y_max, bbox.center.position.y + bbox.size.y / 2.0)

            if detection.mask.data:
                points = np.array(
                    [[round(point.x), round(point.y)] for point in detection.mask.data],
                    dtype=np.int32,
                )
                if len(points) >= 3:
                    cv2.fillPoly(mask_image, [points], 255)

        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            x_min, y_min, x_max, y_max = x, y, x + w, y + h
            epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
            contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        else:
            contour = np.array(
                [
                    [round(x_min), round(y_min)],
                    [round(x_max), round(y_min)],
                    [round(x_max), round(y_max)],
                    [round(x_min), round(y_max)],
                ],
                dtype=np.int32,
            )

        merged_detection = Detection()
        merged_detection.class_id = best_detection.class_id
        merged_detection.class_name = best_detection.class_name
        merged_detection.score = max(detection.score for detection in detections)

        merged_detection.bbox.center.position.x = float((x_min + x_max) / 2.0)
        merged_detection.bbox.center.position.y = float((y_min + y_max) / 2.0)
        merged_detection.bbox.size.x = float(max(0.0, x_max - x_min))
        merged_detection.bbox.size.y = float(max(0.0, y_max - y_min))

        merged_detection.mask.height = height
        merged_detection.mask.width = width
        merged_detection.mask.data = [
            self._create_point2d(float(point[0]), float(point[1]))
            for point in contour
        ]
        return merged_detection

    def _color_for_class(self, class_name):
        if class_name not in self._class_to_color:
            palette = {
                "lane1": (255, 80, 80),
                "lane2": (80, 180, 255),
                "crosswalk": (80, 255, 120),
                "traffic_light": (80, 80, 255),
                "road-objects": (255, 220, 80),
            }
            self._class_to_color[class_name] = palette.get(
                class_name,
                (180, 180, 180),
            )
        return self._class_to_color[class_name]

    @staticmethod
    def _normalize_class_name(class_name):
        return str(class_name or "").strip().lower().replace("_", "").replace(" ", "")

    @staticmethod
    def _create_point2d(x, y):
        point = Point2D()
        point.x = x
        point.y = y
        return point


def main():
    rclpy.init()
    node = Yolov8Node()
    node.trigger_configure()
    node.trigger_activate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
