#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR / "assets"
OUTPUT_DIR = SCRIPT_DIR / "output"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
CONFIG_OUTPUT_DIR = OUTPUT_DIR / "config"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Click 4 corresponding points on track map and LiDAR BEV images."
    )
    parser.add_argument(
        "--make-bev",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run --bev-script first and use its raw BEV output for alignment.",
    )
    parser.add_argument(
        "--bev-script", default="/home/autolab/rosbag_data/ot128_accum_bev.py"
    )
    parser.add_argument(
        "--live-bev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture a short BEV snapshot from a live PointCloud2 topic. "
            "Works with a replayed rosbag as well, since it just subscribes to --topic."
        ),
    )
    parser.add_argument("--topic", default="/lidar_points")
    parser.add_argument("--capture-frames", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument(
        "--forward-axis",
        choices=["-y", "+y", "+x", "-x"],
        default="-y",
        help="LiDAR axis pointing toward the track; the lateral axis is chosen to keep a right-handed BEV.",
    )
    # Long-side-center LiDAR placement. 240x320 BEV, same canvas as the Open3D
    # black-road candidate mask, so alignment results stay directly comparable.
    parser.add_argument("--forward-min", type=float, default=1.3)
    parser.add_argument("--forward-max", type=float, default=13.3)
    parser.add_argument("--lateral-min", type=float, default=-7.0)
    parser.add_argument("--lateral-max", type=float, default=9.0)
    parser.add_argument("--z-min", type=float, default=-2.2)
    parser.add_argument("--z-max", type=float, default=0.6)
    parser.add_argument("--intensity-max", type=float, default=50.0)
    parser.add_argument("--display-width", type=int, default=1800)
    parser.add_argument("--display-height", type=int, default=950)
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the interactive side-by-side alignment editor.",
    )
    parser.add_argument(
        "--marker-radius",
        type=int,
        default=4,
        help="On-screen radius in pixels for selected-point markers.",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open each point-picking window fullscreen while preserving image aspect ratio.",
    )
    parser.add_argument(
        "--track-map",
        default=str(ASSET_DIR / "track_map.png"),
        help=(
            "Use the same image in live_bev_intensity_viewer.py. The road mask closing "
            "kernel scales with the image width, so track.png also still works."
        ),
    )
    parser.add_argument(
        "--bev-image",
        default=str(IMAGE_OUTPUT_DIR / "alignment_bev_intensity_raw.png"),
    )
    parser.add_argument(
        "--points-json",
        default=str(CONFIG_OUTPUT_DIR / "track_map_alignment_points.json"),
    )
    parser.add_argument(
        "--align-script",
        default=str(SCRIPT_DIR / "track_map_align.py"),
    )
    parser.add_argument(
        "--out-prefix",
        default="track_map_aligned",
        help="Output path prefix passed to track_map_align.py when --run-align is used.",
    )
    parser.add_argument(
        "--image-output-dir",
        default=str(IMAGE_OUTPUT_DIR),
        help="Directory for generated PNG previews and masks.",
    )
    parser.add_argument(
        "--config-output-dir",
        default=str(CONFIG_OUTPUT_DIR),
        help="Directory for the generated homography JSON.",
    )
    parser.add_argument(
        "--bev-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pick points on the LiDAR BEV image first, then on the track map.",
    )
    parser.add_argument(
        "--run-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run track_map_align.py after saving the point JSON.",
    )
    return parser.parse_args()


def make_bev(args):
    cmd = ["python3", str(args.bev_script)]
    print("making_bev:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    bev_path = Path(args.bev_image)
    if not bev_path.exists():
        raise FileNotFoundError(
            f"Expected BEV image was not created: {bev_path}. "
            "Check ot128_accum_bev.py defaults or pass --bev-image explicitly."
        )
    return bev_path


class LiveBevCapture(Node):
    def __init__(self, args):
        super().__init__("alignment_live_bev_capture")
        self.args = args
        self.width = int(round((args.forward_max - args.forward_min) / args.resolution))
        self.height = int(round((args.lateral_max - args.lateral_min) / args.resolution))
        self.bev = np.zeros((self.height, self.width), dtype=np.float32)
        self.frames = 0
        self.subscription = self.create_subscription(
            PointCloud2, args.topic, self.on_cloud, 10
        )
        self.get_logger().info(
            f"Capturing {args.capture_frames} frames from {args.topic}, "
            f"BEV size={self.width}x{self.height}"
        )

    def on_cloud(self, msg):
        points = point_cloud2.read_points(
            msg, field_names=["x", "y", "z", "intensity"], skip_nans=False
        )
        x = points["x"]
        y = points["y"]
        z = points["z"]
        intensity = points["intensity"]

        if self.args.forward_axis == "-y":
            forward, lateral = -y, x
        elif self.args.forward_axis == "+y":
            forward, lateral = y, -x
        elif self.args.forward_axis == "+x":
            forward, lateral = x, y
        else:
            forward, lateral = -x, -y
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        valid &= ~((x == 0) & (y == 0) & (z == 0))
        valid &= (forward >= self.args.forward_min) & (forward < self.args.forward_max)
        valid &= (lateral >= self.args.lateral_min) & (lateral < self.args.lateral_max)
        valid &= (z >= self.args.z_min) & (z <= self.args.z_max)

        if np.any(valid):
            col = np.floor(
                (forward[valid] - self.args.forward_min) / self.args.resolution
            ).astype(np.int32)
            row_from_bottom = np.floor(
                (lateral[valid] - self.args.lateral_min) / self.args.resolution
            ).astype(np.int32)
            row = self.height - 1 - row_from_bottom
            ok = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
            np.maximum.at(self.bev, (row[ok], col[ok]), intensity[valid][ok])

        self.frames += 1
        if self.frames % 5 == 0 or self.frames == self.args.capture_frames:
            self.get_logger().info(f"captured_frames={self.frames}/{self.args.capture_frames}")


def capture_live_bev(args):
    out_path = Path(args.bev_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started_here = not rclpy.ok()
    if started_here:
        rclpy.init()
    node = LiveBevCapture(args)
    try:
        while rclpy.ok() and node.frames < args.capture_frames:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        if started_here and rclpy.ok():
            rclpy.shutdown()

    if node.frames <= 0:
        raise RuntimeError(f"No frames received from {args.topic}. Is the LiDAR driver running?")

    image = np.clip(node.bev / max(args.intensity_max, 1e-6), 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_path), image)
    print(f"live_bev_image: {out_path}")
    return out_path


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def fit_to_screen(image, max_width=1800, max_height=950):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale == 1.0:
        return image.copy(), scale
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def pick_points(
    image_path,
    title,
    count=4,
    max_width=1800,
    max_height=950,
    fullscreen=True,
):
    image = load_image(image_path)
    display, scale = fit_to_screen(image, max_width, max_height)
    points = []

    def redraw():
        canvas = display.copy()
        for idx, (x, y) in enumerate(points, start=1):
            dx = int(round(x * scale))
            dy = int(round(y * scale))
            cv2.circle(canvas, (dx, dy), 6, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(idx),
                (dx + 8, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(title, canvas)

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= count:
            return
        orig_x = int(round(x / scale))
        orig_y = int(round(y / scale))
        points.append([orig_x, orig_y])
        print(f"{title} point {len(points)}: [{orig_x}, {orig_y}]")
        redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if fullscreen:
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(title, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if len(points) == count:
            break
        if key in (27, ord("q")):
            cv2.destroyWindow(title)
            raise SystemExit("Point picking cancelled.")
        if key == ord("u") and points:
            removed = points.pop()
            print(f"{title} undo: {removed}")
            redraw()
        if key == ord("r"):
            points.clear()
            print(f"{title} reset")
            redraw()

    cv2.destroyWindow(title)
    return points


def main():
    args = parse_args()
    track_path = Path(args.track_map)
    if args.live_bev:
        bev_path = capture_live_bev(args)
    elif args.make_bev:
        bev_path = make_bev(args)
    else:
        bev_path = Path(args.bev_image)
    points_path = Path(args.points_json)
    points_path.parent.mkdir(parents=True, exist_ok=True)

    print("Pick the same physical 4 points in the same order on both images.")
    print("Recommended order: top-left, top-right, bottom-right, bottom-left of the visible ROI.")

    if args.gui:
        from alignment_picker_gui import pick_corresponding_points

        bev_bgr = load_image(bev_path)
        track_bgr = load_image(track_path)
        selected = pick_corresponding_points(
            bev_bgr,
            track_bgr,
            count=4,
            marker_radius=max(2, args.marker_radius),
            fullscreen=args.fullscreen,
        )
        if selected is None:
            raise SystemExit("Point picking cancelled.")
        bev_points, track_points = selected
    elif args.bev_first:
        bev_points = pick_points(
            bev_path, "bev_image_points", max_width=args.display_width,
            max_height=args.display_height, fullscreen=args.fullscreen
        )
        track_points = pick_points(
            track_path, "track_map_points", max_width=args.display_width,
            max_height=args.display_height, fullscreen=args.fullscreen
        )
    else:
        track_points = pick_points(
            track_path, "track_map_points", max_width=args.display_width,
            max_height=args.display_height, fullscreen=args.fullscreen
        )
        bev_points = pick_points(
            bev_path, "bev_image_points", max_width=args.display_width,
            max_height=args.display_height, fullscreen=args.fullscreen
        )

    data = {
        "point_order": "Same physical 4 points clicked in order on track_map_points and bev_image_points.",
        "track_map_points": track_points,
        "bev_image_points": bev_points,
    }
    points_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"saved_points_json: {points_path}")

    if args.run_align:
        cmd = [
            "python3",
            str(args.align_script),
            "--track-map",
            str(track_path),
            "--bev-image",
            str(bev_path),
            "--points-json",
            str(points_path),
            "--out-prefix",
            str(args.out_prefix),
            "--image-output-dir",
            str(args.image_output_dir),
            "--config-output-dir",
            str(args.config_output_dir),
        ]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
