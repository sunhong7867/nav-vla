import json
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


DEFAULT_SRC_MAT = [[244, 382], [411, 382], [493, 478], [177, 478]]
POINT_LABELS = ("Left top", "Right top", "Right bottom", "Left bottom")


class LaneTuningGuiNode(Node):
    def __init__(self):
        super().__init__("lane_tuning_gui_node")
        self.config_topic = self.declare_parameter(
            "config_topic",
            "lane_extractor_config",
        ).value
        self.image_topic = self.declare_parameter(
            "image_topic",
            "/camera/image_raw",
        ).value
        self.publisher = self.create_publisher(String, self.config_topic, 10)
        self.bridge = CvBridge()
        self.latest_image = None
        self.click_index = 0
        self.click_window_name = "Click SRC_MAT Points"

        self.root = tk.Tk()
        self.root.title("Lane Bird-Eye Tuning")
        self.root.geometry("470x640")
        self.root.minsize(430, 560)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.running = True

        self.point_vars = []
        self.roi_var = tk.IntVar(value=300)
        self.target_start_var = tk.IntVar(value=5)
        self.target_stop_var = tk.IntVar(value=155)
        self.target_step_var = tk.IntVar(value=50)
        self.status_var = tk.StringVar(value="Ready")
        self.payload_var = tk.StringVar(value="")

        self._build_ui()
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        cv2.namedWindow(self.click_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.click_window_name, 960, 720)
        cv2.setMouseCallback(self.click_window_name, self.on_mouse)
        self.publish_config()
        self.root.after(50, self._spin_once)
        self.root.after(50, self._update_click_window)

    def _build_ui(self):
        root = ttk.Frame(self.root, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root, text="SRC_MAT points (camera image pixels)").pack(anchor=tk.W)
        points_frame = ttk.Frame(root)
        points_frame.pack(fill=tk.X, pady=(6, 12))

        for row, (label, default_point) in enumerate(zip(POINT_LABELS, DEFAULT_SRC_MAT)):
            ttk.Label(points_frame, text=label, width=13).grid(row=row, column=0, sticky=tk.W, pady=3)
            x_var = tk.IntVar(value=default_point[0])
            y_var = tk.IntVar(value=default_point[1])
            self.point_vars.append((x_var, y_var))
            ttk.Spinbox(points_frame, from_=0, to=639, textvariable=x_var, width=7, command=self.publish_config).grid(row=row, column=1, padx=4)
            ttk.Spinbox(points_frame, from_=0, to=479, textvariable=y_var, width=7, command=self.publish_config).grid(row=row, column=2, padx=4)
            ttk.Label(points_frame, text="x").grid(row=row, column=3, padx=(6, 2))
            ttk.Label(points_frame, textvariable=x_var, width=4).grid(row=row, column=4, sticky=tk.W)
            ttk.Label(points_frame, text="y").grid(row=row, column=5, padx=(6, 2))
            ttk.Label(points_frame, textvariable=y_var, width=4).grid(row=row, column=6, sticky=tk.W)

        self._slider(root, "ROI cutting idx", self.roi_var, 0, 479)
        self._slider(root, "Target start", self.target_start_var, 0, 250)
        self._slider(root, "Target stop", self.target_stop_var, 10, 350)
        self._slider(root, "Target step", self.target_step_var, 1, 120)

        buttons = ttk.Frame(root)
        buttons.pack(fill=tk.X, pady=(12, 8))
        ttk.Button(buttons, text="Apply", command=self.publish_config).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Reset driving.zip", command=self.reset_defaults).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Copy JSON", command=self.copy_json).pack(side=tk.LEFT)

        click_buttons = ttk.Frame(root)
        click_buttons.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(click_buttons, text="Start 4-point click", command=self.start_clicking).pack(side=tk.LEFT)
        ttk.Button(click_buttons, text="Undo point", command=self.undo_point).pack(side=tk.LEFT, padx=6)

        ttk.Label(root, textvariable=self.status_var).pack(anchor=tk.W, pady=(6, 4))
        payload_entry = ttk.Entry(root, textvariable=self.payload_var)
        payload_entry.pack(fill=tk.X)

        help_text = (
            "Click order in the image window:\n"
            "1 left-top: far left lane edge\n"
            "2 right-top: far right lane edge\n"
            "3 right-bottom: near right lane edge\n"
            "4 left-bottom: near left lane edge\n"
            "Then watch lane_bird_image / lane_target_debug_image."
        )
        ttk.Label(root, text=help_text, foreground="#555").pack(anchor=tk.W, pady=(10, 0))

    def _slider(self, parent, label, variable, lower, upper):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
        scale = ttk.Scale(
            frame,
            from_=lower,
            to=upper,
            variable=variable,
            command=lambda _value: self.publish_config(),
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        spinbox = ttk.Spinbox(
            frame,
            from_=lower,
            to=upper,
            textvariable=variable,
            width=6,
            command=self.publish_config,
        )
        spinbox.pack(side=tk.LEFT)

    def payload(self):
        return {
            "src_mat": [[x_var.get(), y_var.get()] for x_var, y_var in self.point_vars],
            "roi_cutting_idx": self.roi_var.get(),
            "target_point_start": self.target_start_var.get(),
            "target_point_stop": self.target_stop_var.get(),
            "target_point_step": self.target_step_var.get(),
        }

    def publish_config(self):
        payload = self.payload()
        payload_text = json.dumps(payload, sort_keys=True)
        msg = String()
        msg.data = payload_text
        self.publisher.publish(msg)
        self.payload_var.set(payload_text)
        self.status_var.set(
            "Published target points: "
            + str(list(range(payload["target_point_start"], payload["target_point_stop"], max(1, payload["target_point_step"]))))
        )

    def reset_defaults(self):
        for (x_var, y_var), point in zip(self.point_vars, DEFAULT_SRC_MAT):
            x_var.set(point[0])
            y_var.set(point[1])
        self.roi_var.set(300)
        self.target_start_var.set(5)
        self.target_stop_var.set(155)
        self.target_step_var.set(50)
        self.publish_config()

    def copy_json(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.payload_var.get())
        self.status_var.set("Copied JSON to clipboard")

    def start_clicking(self):
        self.click_index = 0
        self.status_var.set("Click left-top point in the image window")

    def undo_point(self):
        self.click_index = max(0, self.click_index - 1)
        self.status_var.set(f"Undo. Next point: {self._next_point_label()}")

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.status_var.set(f"Image convert failed: {exc}")

    def on_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.click_index >= len(self.point_vars):
            self.click_index = 0

        x_var, y_var = self.point_vars[self.click_index]
        x_var.set(x)
        y_var.set(y)
        clicked_label = POINT_LABELS[self.click_index]
        self.click_index += 1
        self.publish_config()

        if self.click_index >= len(self.point_vars):
            self.status_var.set(f"{clicked_label} set. 4 points complete.")
        else:
            self.status_var.set(f"{clicked_label} set. Next: {self._next_point_label()}")

    def _update_click_window(self):
        if not self.running:
            return

        image = self.latest_image
        if image is None:
            image = self._blank_image()
        else:
            image = image.copy()

        self._draw_click_overlay(image)
        cv2.imshow(self.click_window_name, image)
        cv2.waitKey(1)
        self.root.after(50, self._update_click_window)

    def _blank_image(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (35, 35, 35)
        cv2.putText(
            image,
            f"Waiting for {self.image_topic}",
            (40, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return image

    def _draw_click_overlay(self, image):
        points = [[x_var.get(), y_var.get()] for x_var, y_var in self.point_vars]
        colors = [(0, 255, 255), (0, 200, 255), (0, 140, 255), (0, 80, 255)]
        for index, (point, label, color) in enumerate(zip(points, POINT_LABELS, colors)):
            x, y = point
            cv2.circle(image, (x, y), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{index + 1} {label}",
                (min(x + 8, image.shape[1] - 150), max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], True, (255, 255, 255), 2, cv2.LINE_AA)

        instruction = f"Next click: {self._next_point_label()}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(
            image,
            instruction,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _next_point_label(self):
        if self.click_index >= len(POINT_LABELS):
            return "complete; next click restarts at left-top"
        return f"{self.click_index + 1} {POINT_LABELS[self.click_index]}"

    def _spin_once(self):
        if not self.running:
            return
        rclpy.spin_once(self, timeout_sec=0.0)
        self.root.after(50, self._spin_once)

    def on_close(self):
        self.running = False
        cv2.destroyWindow(self.click_window_name)
        self.root.quit()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = LaneTuningGuiNode()
    try:
        node.run()
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
