import json
import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import String
from simulation_pkg import basic
from decision_making_pkg import motion_planner_node as motion_defaults


MODE_TOPIC_NAME = "lane_mode_command"
TUNING_TOPIC_NAME = "motion_tuning_command"


def _motion_limit(key):
    _, min_value, max_value = motion_defaults.TUNABLE_LIMITS[key]
    return float(min_value), float(max_value)


def _tuning_control(key, label, step, default_value):
    min_value, max_value = _motion_limit(key)
    return key, label, min_value, max_value, step, float(default_value)


TUNING_CONTROLS = [
    _tuning_control("target_speed_raw", "Straight speed raw", 1.0, motion_defaults.STRAIGHT_TARGET_SPEED),
    _tuning_control("curve_speed_raw", "Curve speed raw", 1.0, motion_defaults.CURVE_TARGET_SPEED),
    _tuning_control("offset_px", "Vehicle center offset px", 1.0, motion_defaults.OFFSET_PX),
    _tuning_control("steering_gain_scale", "Overall steering gain", 0.01, motion_defaults.STEERING_GAIN_SCALE),
    _tuning_control(
        "lane_change_steering_gain_scale",
        "Lane change steering gain",
        0.01,
        motion_defaults.LANE_CHANGE_STEERING_GAIN_SCALE,
    ),
    _tuning_control("heading_gain", "Slope / heading gain", 0.01, motion_defaults.STANLEY_HEADING_GAIN),
    _tuning_control("lateral_gain", "Center error gain", 0.01, motion_defaults.STANLEY_LATERAL_GAIN),
    _tuning_control("curve_boost", "Curve steering boost", 0.01, motion_defaults.CURVATURE_STEER_BOOST),
    _tuning_control("near_error_weight", "Near-center weight", 0.01, motion_defaults.NEAR_ERROR_WEIGHT),
    _tuning_control("center_ref_x_m", "Lookahead x m", 0.01, motion_defaults.CENTER_REF_X_M),
    _tuning_control("center_near_x_m", "Near check x m", 0.01, motion_defaults.CENTER_NEAR_X_M),
    _tuning_control("stanley_softening", "Steering softening", 0.01, motion_defaults.STANLEY_SOFTENING),
    _tuning_control("smoothing_alpha", "Steering smoothing", 0.01, motion_defaults.SMOOTHING_ALPHA),
    _tuning_control(
        "lane_change_smoothing_alpha",
        "Lane change smoothing",
        0.01,
        motion_defaults.LANE_CHANGE_SMOOTHING_ALPHA,
    ),
    _tuning_control("max_steering_step", "Max steering step", 0.01, motion_defaults.MAX_STEERING_STEP),
    _tuning_control(
        "lane_change_max_steering_step",
        "Lane change max step",
        0.01,
        motion_defaults.LANE_CHANGE_MAX_STEERING_STEP,
    ),
    _tuning_control("lane_change_speed_raw", "Lane change speed raw", 1.0, motion_defaults.LANE_CHANGE_TARGET_SPEED),
]
TUNING_CONTROLS_BY_KEY = {
    key: (key, label, lower, upper, resolution, default)
    for key, label, lower, upper, resolution, default in TUNING_CONTROLS
}
TUNING_GROUPS = [
    (
        "Speed",
        [
            "target_speed_raw",
            "curve_speed_raw",
            "lane_change_speed_raw",
        ],
    ),
    (
        "Lane Center / Lookahead",
        [
            "offset_px",
            "center_near_x_m",
            "center_ref_x_m",
            "near_error_weight",
        ],
    ),
    (
        "Normal Steering",
        [
            "steering_gain_scale",
            "lateral_gain",
            "heading_gain",
            "curve_boost",
            "stanley_softening",
            "smoothing_alpha",
            "max_steering_step",
        ],
    ),
    (
        "Lane Change",
        [
            "lane_change_steering_gain_scale",
            "lane_change_smoothing_alpha",
            "lane_change_max_steering_step",
        ],
    ),
]


class LaneModeGuiNode(Node):
    def __init__(self):
        super().__init__("lane_mode_gui_node")
        self.mode_topic = self.declare_parameter("mode_topic", MODE_TOPIC_NAME).value
        self.tuning_topic = self.declare_parameter("tuning_topic", TUNING_TOPIC_NAME).value
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.publisher = self.create_publisher(String, self.mode_topic, qos_profile)
        self.tuning_publisher = self.create_publisher(String, self.tuning_topic, qos_profile)
        self.current_command = "keep_lane:lane2"

    def publish_mode(self, command, log=True):
        self.current_command = command
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        if log:
            self.get_logger().info(f"lane mode command: {command}")

    def publish_tuning(self, params, reset_filter=False, log=True):
        msg = String()
        msg.data = json.dumps(
            {
                "params": params,
                "reset_filter": bool(reset_filter),
            },
            sort_keys=True,
        )
        self.tuning_publisher.publish(msg)
        if log:
            compact = ", ".join(f"{key}={value}" for key, value in params.items())
            self.get_logger().info(f"motion tuning command: {compact}")

    def reset_vehicle(self):
        self.get_logger().info("reset ego vehicle requested")
        basic.reset_model("ego_vehicle", "prius_hybrid", basic.driving_ego())
        self.get_logger().info("ego vehicle reset complete")


class LaneModeWindow:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Driving Control")
        self.root.geometry("390x640")
        self.root.minsize(360, 420)
        self.root.resizable(True, True)
        self.status_text = tk.StringVar(value="Mode: keep lane2")
        self.reset_status_text = tk.StringVar(value="Vehicle: ready")
        self.selected_command = node.current_command
        self.buttons = {}
        self.tuning_vars = {}
        self._pending_tuning_after = None
        self._reset_thread = None

        outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        frame = tk.Frame(canvas, padx=12, pady=12)
        frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.root.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        tk.Label(frame, text="Driving Mode", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 8))

        self._add_button(frame, "Change / keep lane2", "keep_lane:lane2", "Mode: change / keep lane2")
        self._add_button(frame, "Change / keep lane1", "keep_lane:lane1", "Mode: change / keep lane1")
        self._add_button(frame, "Auto switch before crosswalk", "lane_change", "Mode: auto switch before crosswalk")

        tk.Frame(frame, height=1, bg="#c7c7c7").pack(fill=tk.X, pady=8)
        tk.Label(frame, textvariable=self.status_text).pack(anchor=tk.W)

        tk.Frame(frame, height=1, bg="#c7c7c7").pack(fill=tk.X, pady=10)
        tk.Label(frame, text="Motion Tuning", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self._add_tuning_controls(frame)

        actions = tk.Frame(frame)
        actions.pack(fill=tk.X, pady=(8, 4))
        tk.Button(actions, text="Apply Tuning", command=lambda: self.publish_tuning(reset_filter=True)).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(0, 4),
        )
        tk.Button(actions, text="Reset Defaults", command=self.reset_tuning_defaults).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(4, 0),
        )

        tk.Frame(frame, height=1, bg="#c7c7c7").pack(fill=tk.X, pady=10)
        tk.Button(
            frame,
            text="Reset ego vehicle",
            command=self.reset_vehicle,
            bg="#b71c1c",
            fg="white",
            activebackground="#7f0000",
            activeforeground="white",
        ).pack(fill=tk.X, pady=3)
        tk.Label(frame, textvariable=self.reset_status_text).pack(anchor=tk.W, pady=(4, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh_buttons()
        self.root.after(250, self._publish_initial_mode)
        self.root.after(300, lambda: self.publish_tuning(reset_filter=True, log=False))
        self.root.after(1000, self._republish_current_mode)

    def _add_button(self, parent, label, command, status):
        button = tk.Button(
            parent,
            text=label,
            command=lambda: self.set_mode(command, status),
            relief=tk.RAISED,
            bd=2,
            padx=8,
            pady=4,
        )
        button.pack(fill=tk.X, pady=3)
        self.buttons[command] = button

    def _add_tuning_controls(self, parent):
        for group_title, control_keys in TUNING_GROUPS:
            self._add_tuning_group(parent, group_title, control_keys)

    def _add_tuning_group(self, parent, title, control_keys):
        tk.Frame(parent, height=1, bg="#d0d0d0").pack(fill=tk.X, pady=(8, 5))
        tk.Label(parent, text=title, font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        for key in control_keys:
            control = TUNING_CONTROLS_BY_KEY.get(key)
            if control is None:
                continue

            key, label, lower, upper, resolution, default = control
            row = tk.Frame(parent)
            row.pack(fill=tk.X, pady=(5, 2))
            tk.Label(row, text=label).pack(anchor=tk.W)
            variable = tk.DoubleVar(value=default)
            scale = tk.Scale(
                row,
                from_=lower,
                to=upper,
                resolution=resolution,
                orient=tk.HORIZONTAL,
                variable=variable,
                command=self._schedule_tuning_publish,
                length=300,
            )
            scale.pack(fill=tk.X)
            self.tuning_vars[key] = variable

    def set_mode(self, command, status):
        self.selected_command = command
        self.status_text.set(status)
        self._refresh_buttons()
        self.node.publish_mode(command)

    def _tuning_params(self):
        return {
            key: round(float(variable.get()), 4)
            for key, variable in self.tuning_vars.items()
        }

    def _schedule_tuning_publish(self, _value=None):
        if self._pending_tuning_after is not None:
            self.root.after_cancel(self._pending_tuning_after)
        self._pending_tuning_after = self.root.after(120, lambda: self.publish_tuning(log=False))

    def publish_tuning(self, reset_filter=False, log=True):
        self._pending_tuning_after = None
        self.node.publish_tuning(self._tuning_params(), reset_filter=reset_filter, log=log)

    def reset_tuning_defaults(self):
        for key, _label, _lower, _upper, _resolution, default in TUNING_CONTROLS:
            self.tuning_vars[key].set(default)
        self.publish_tuning(reset_filter=True)

    def reset_vehicle(self):
        if self._reset_thread is not None and self._reset_thread.is_alive():
            return

        self.reset_status_text.set("Vehicle: resetting...")
        self._reset_thread = threading.Thread(target=self._reset_vehicle_worker, daemon=True)
        self._reset_thread.start()

    def _reset_vehicle_worker(self):
        try:
            self.node.reset_vehicle()
        except Exception as exc:
            self.root.after(0, lambda: self.reset_status_text.set(f"Vehicle reset failed: {exc}"))
            return

        self.root.after(0, lambda: self.reset_status_text.set("Vehicle: reset complete"))

    def _refresh_buttons(self):
        for command, button in self.buttons.items():
            if command == self.selected_command:
                button.configure(bg="#2e7d32", fg="white", activebackground="#1b5e20", activeforeground="white")
            else:
                button.configure(bg="#f0f0f0", fg="black", activebackground="#e0e0e0", activeforeground="black")

    def _publish_initial_mode(self):
        self.node.publish_mode(self.node.current_command)

    def _republish_current_mode(self):
        if rclpy.ok():
            self.node.publish_mode(self.node.current_command, log=False)
            self.root.after(1000, self._republish_current_mode)

    def close(self):
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = LaneModeGuiNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        window = LaneModeWindow(node)
        window.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
