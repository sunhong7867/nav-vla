import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class YoloDebugImageViewerNode(Node):
    def __init__(self):
        super().__init__("yolo_debug_image_viewer_node")

        self.image_topic = self.declare_parameter(
            "image_topic",
            "/yolov8_seg_debug_image",
        ).value
        self.window_name = self.declare_parameter(
            "window_name",
            "YOLO Segmentation Debug",
        ).value
        self.bridge = CvBridge()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.get_logger().info(f"showing {self.image_topic} at the image's native size")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"failed to convert debug image: {exc}")
            return

        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDebugImageViewerNode()
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
