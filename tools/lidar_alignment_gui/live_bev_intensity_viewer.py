#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


from layout_rendering import (
    colorize_layout_lines,
    extract_layout_line_mask,
    warp_layout_line_mask,
    blend_layout_lines,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live BEV intensity viewer for Hesai/PointCloud2 data."
    )
    parser.add_argument("--topic", default="/lidar_points")
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
    parser.add_argument("--intensity-max", type=float, default=120.0)
    parser.add_argument(
        "--overlay-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the aligned fixed track map on the live BEV image.",
    )
    parser.add_argument(
        "--track-map",
        default=str(SCRIPT_DIR / "assets" / "track_map.png"),
        help="Must be the same track image the homography was picked on.",
    )
    parser.add_argument(
        "--homography-json",
        default=str(SCRIPT_DIR / "output" / "config" / "track_map_aligned_homography.json"),
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.65)
    parser.add_argument("--line-supersample", type=int, default=4)
    parser.add_argument("--line-min-source-width", type=int, default=5)
    parser.add_argument(
        "--detect-vehicle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect a raised vehicle cluster and classify it against the road mask.",
    )
    parser.add_argument("--vehicle-z-min", type=float, default=-0.15)
    parser.add_argument("--vehicle-z-max", type=float, default=0.18)
    parser.add_argument(
        "--height-mode",
        choices=["floor", "raw-z"],
        default="floor",
        help="Use floor-plane residual height or raw LiDAR z for vehicle filtering.",
    )
    parser.add_argument("--vehicle-height-min", type=float, default=0.03)
    parser.add_argument("--vehicle-height-max", type=float, default=0.35)
    parser.add_argument("--floor-candidate-z-min", type=float, default=-2.0)
    parser.add_argument("--floor-candidate-z-max", type=float, default=0.3)
    parser.add_argument("--floor-plane-threshold", type=float, default=0.03)
    parser.add_argument("--floor-ransac-iters", type=int, default=250)
    parser.add_argument("--floor-max-points", type=int, default=30000)
    parser.add_argument("--min-cluster-pixels", type=int, default=5)
    parser.add_argument("--max-cluster-pixels", type=int, default=500)
    parser.add_argument("--vehicle-mask-dilate", type=int, default=1)
    parser.add_argument(
        "--offroad-buffer-pixels",
        type=int,
        default=8,
        help="Expand the black-road mask by this many BEV pixels for vehicle search.",
    )
    parser.add_argument(
        "--vehicle-road-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only consider vehicle clusters whose centroid is inside the registered black-road mask.",
    )
    parser.add_argument(
        "--vehicle-track-area-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only consider vehicle clusters whose centroid is inside the registered track mat area.",
    )
    parser.add_argument(
        "--prefer-near-previous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer clusters near the previous vehicle centroid after the first detection.",
    )
    parser.add_argument("--max-tracking-jump-pixels", type=float, default=80.0)
    parser.add_argument(
        "--lost-reset-frames",
        type=int,
        default=15,
        help="Clear pending candidate state after this many consecutive missed detections.",
    )
    parser.add_argument(
        "--reacquire-anywhere-after-lost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After enough missed detections, allow reacquiring a vehicle anywhere on the road.",
    )
    parser.add_argument(
        "--confirm-detections",
        type=int,
        default=3,
        help="Require this many consecutive similar detections before reporting ON_ROAD/OFF_ROAD after reacquire.",
    )
    parser.add_argument("--confirm-distance-pixels", type=float, default=20.0)
    parser.add_argument(
        "--reset-on-time-jump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset tracking state when message timestamps move backward, as in rosbag --loop.",
    )
    parser.add_argument(
        "--show-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw recent detected vehicle centroids.",
    )
    parser.add_argument("--trajectory-length", type=int, default=80)
    parser.add_argument("--bbox-thickness", type=int, default=1)
    parser.add_argument("--candidate-bbox-thickness", type=int, default=1)
    parser.add_argument("--center-radius", type=int, default=3)
    parser.add_argument("--trajectory-thickness", type=int, default=1)
    parser.add_argument(
        "--accumulate",
        type=int,
        default=5,
        help="Number of recent frames to max-accumulate. Use 1 for no accumulation.",
    )
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--window-name", default="live_bev_intensity")
    parser.add_argument(
        "--log-csv",
        default="",
        help="CSV path for per-frame vehicle status logging. Empty disables logging.",
    )
    parser.add_argument(
        "--no-log-csv",
        action="store_true",
        help="Disable CSV status logging.",
    )
    parser.add_argument(
        "--log-every-n-frames",
        type=int,
        default=1,
        help="Write one CSV row every N processed frames.",
    )
    return parser.parse_args()


def extract_white_line_mask(track_bgr):
    return extract_layout_line_mask(track_bgr)


def colorize_track_map(track_bgr, white_mask):
    del track_bgr
    return colorize_layout_lines(white_mask)


def road_close_kernel(track_bgr, baseline_width=1181, baseline_kernel=9):
    """Closing kernel scaled to the track image resolution.

    The 9x9 kernel was tuned on the 1181px track.png, where it bridges the stop
    line and the crosswalk so the road ring stays closed. On the 5228px track2.png
    those markings are 4.4x wider, so a fixed 9x9 leaves them as non-road holes and
    a vehicle centroid there would be judged OFF_ROAD.
    """
    scaled = int(round(baseline_kernel * track_bgr.shape[1] / baseline_width))
    return np.ones((max(baseline_kernel, scaled),) * 2, np.uint8)


def extract_road_mask(track_bgr):
    hsv = cv2.cvtColor(track_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    # Use only the black outer road. Do not include green background, gray
    # inner area, or standalone white decoration as drivable road.
    dark_road = (value < 90) & (saturation < 90)
    mask = dark_road.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, road_close_kernel(track_bgr))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def make_vehicle_search_mask(road_mask, buffer_pixels):
    if buffer_pixels <= 0:
        return road_mask.copy()
    kernel_size = buffer_pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(road_mask, kernel, iterations=1)


def pixel_to_world(col, row, args):
    forward = args.forward_min + (col + 0.5) * args.resolution
    lateral = args.lateral_max - (row + 0.5) * args.resolution
    return forward, lateral


class LiveBevIntensityViewer(Node):
    def __init__(self, args):
        super().__init__("live_bev_intensity_viewer")
        self.args = args
        self.width = int(round((args.forward_max - args.forward_min) / args.resolution))
        self.height = int(round((args.lateral_max - args.lateral_min) / args.resolution))
        self.frames = []
        self.trajectory = []
        self.last_vehicle_centroid = None
        self.missed_vehicle_frames = 0
        self.pending_vehicle_centroid = None
        self.pending_vehicle_count = 0
        self.vehicle_confirmed = False
        self.last_msg_time_ns = None
        self.floor_normal = None
        self.floor_d = None
        self.last_log_time = 0.0
        self.processed_frames = 0
        self.csv_file = None
        self.csv_writer = None
        self.init_csv_logger()
        self.track_overlay, self.track_mask, self.road_mask, self.vehicle_search_mask = (
            self.load_track_overlay()
        )
        self.subscription = self.create_subscription(
            PointCloud2, args.topic, self.on_cloud, 10
        )
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"Subscribed to {args.topic}, BEV size={self.width}x{self.height}, "
            f"resolution={args.resolution:.3f}m"
        )

    def init_csv_logger(self):
        if self.args.no_log_csv or not self.args.log_csv:
            return

        path = Path(self.args.log_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        self.csv_file = path.open("a", newline="", encoding="utf-8")
        fieldnames = [
            "wall_time",
            "stamp_sec",
            "stamp_nanosec",
            "status",
            "forward_m",
            "lateral_m",
            "cluster_area_px",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "vehicle_points",
            "confirmed",
            "pending_count",
            "missed_frames",
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if write_header:
            self.csv_writer.writeheader()
            self.csv_file.flush()
        self.get_logger().info(f"CSV logging enabled: {path}")

    def close_csv_logger(self):
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def load_track_overlay(self):
        if not self.args.overlay_track:
            return None, None, None, None

        track_path = Path(self.args.track_map)
        homography_path = Path(self.args.homography_json)
        if not track_path.exists():
            self.get_logger().warn(f"Track map not found: {track_path}")
            return None, None, None, None
        if not homography_path.exists():
            self.get_logger().warn(f"Homography JSON not found: {homography_path}")
            return None, None, None, None

        track_bgr = cv2.imread(str(track_path), cv2.IMREAD_COLOR)
        if track_bgr is None:
            self.get_logger().warn(f"Could not read track map: {track_path}")
            return None, None, None, None

        data = json.loads(homography_path.read_text(encoding="utf-8"))
        homography = np.asarray(data["homography_track_to_bev"], dtype=np.float64)

        # A homography is only valid for the BEV canvas and track map it was picked on.
        # Changing the crop, the resolution or the track image silently shifts the overlay,
        # so refuse instead of drawing a wrong road mask.
        saved_bev = data.get("bev_image_size") or {}
        if saved_bev and (
            int(saved_bev.get("width", -1)) != self.width
            or int(saved_bev.get("height", -1)) != self.height
        ):
            self.get_logger().error(
                f"Homography was made for a {saved_bev.get('width')}x{saved_bev.get('height')} BEV "
                f"but this viewer builds {self.width}x{self.height}. "
                "Match --resolution/--forward-*/--lateral-* to the alignment run, "
                "or redo the 4-point picking."
            )
            return None, None, None, None
        saved_track = data.get("track_image_size") or {}
        if saved_track and (
            int(saved_track.get("width", -1)) != track_bgr.shape[1]
            or int(saved_track.get("height", -1)) != track_bgr.shape[0]
        ):
            self.get_logger().error(
                f"Homography was made for a {saved_track.get('width')}x{saved_track.get('height')} "
                f"track map but --track-map is {track_bgr.shape[1]}x{track_bgr.shape[0]}: "
                f"{track_path}"
            )
            return None, None, None, None

        white_mask = extract_white_line_mask(track_bgr)
        road_mask = extract_road_mask(track_bgr)
        warped_line_alpha = warp_layout_line_mask(
            white_mask,
            homography,
            (self.width, self.height),
            supersample=self.args.line_supersample,
            minimum_source_width=self.args.line_min_source_width,
        )
        warped = colorize_layout_lines(warped_line_alpha)
        warped_road = cv2.warpPerspective(road_mask, homography, (self.width, self.height))
        road = warped_road > 0
        warped_vehicle_search = make_vehicle_search_mask(
            warped_road, self.args.offroad_buffer_pixels
        )
        vehicle_search = warped_vehicle_search > 0

        self.get_logger().info(
            f"Loaded track overlay: {homography_path}, active_pixels={int(np.count_nonzero(warped_line_alpha))}, "
            f"road_pixels={int(road.sum())}, vehicle_search_pixels={int(vehicle_search.sum())}"
        )
        return warped, warped_line_alpha, road, vehicle_search

    def reset_tracking_state(self, reset_floor=False):
        self.frames.clear()
        self.trajectory.clear()
        self.last_vehicle_centroid = None
        self.missed_vehicle_frames = 0
        self.pending_vehicle_centroid = None
        self.pending_vehicle_count = 0
        self.vehicle_confirmed = False
        if reset_floor:
            self.floor_normal = None
            self.floor_d = None

    def check_time_jump(self, msg):
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if (
            self.args.reset_on_time_jump
            and self.last_msg_time_ns is not None
            and stamp_ns + 1_000_000 < self.last_msg_time_ns
        ):
            self.get_logger().info("Message timestamp moved backward; reset tracking state.")
            self.reset_tracking_state(reset_floor=True)
        self.last_msg_time_ns = stamp_ns

    def estimate_floor_plane(self, x, y, z, base_valid):
        candidate = (
            base_valid
            & (z >= self.args.floor_candidate_z_min)
            & (z <= self.args.floor_candidate_z_max)
        )
        pts = np.column_stack([x[candidate], y[candidate], z[candidate]]).astype(np.float64)
        if len(pts) < 1000:
            self.get_logger().warn(f"Not enough floor candidate points: {len(pts)}")
            return False

        rng = np.random.default_rng(11)
        if len(pts) > self.args.floor_max_points:
            pts = pts[rng.choice(len(pts), self.args.floor_max_points, replace=False)]

        best = None
        for _ in range(self.args.floor_ransac_iters):
            sample = pts[rng.choice(len(pts), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, sample[0]))
            distances = np.abs(pts @ normal + d)
            inliers = distances < self.args.floor_plane_threshold
            count = int(inliers.sum())
            if best is None or count > best[0]:
                best = (count, inliers)

        if best is None or best[0] < 1000:
            self.get_logger().warn("Failed to estimate floor plane.")
            return False

        inlier_pts = pts[best[1]]
        centroid = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        normal = vh[-1]
        normal = normal / np.linalg.norm(normal)
        d = -float(np.dot(normal, centroid))
        if normal[2] < 0:
            normal = -normal
            d = -d

        self.floor_normal = normal
        self.floor_d = d
        residual = inlier_pts @ normal + d
        percentiles_cm = np.percentile(residual, [5, 50, 95]) * 100.0
        self.get_logger().info(
            "Floor plane estimated: "
            f"normal={np.round(normal, 5).tolist()}, d={d:.4f}, "
            f"inliers={best[0]}/{len(pts)}, residual_cm_p5_50_95="
            f"{np.round(percentiles_cm, 2).tolist()}"
        )
        return True

    def floor_height(self, x, y, z):
        if self.floor_normal is None or self.floor_d is None:
            return None
        pts = np.column_stack([x, y, z]).astype(np.float64)
        return pts @ self.floor_normal + self.floor_d

    def detect_vehicle(self, forward, lateral, height, base_valid):
        if not self.args.detect_vehicle:
            return None, 0

        if height is None:
            return None, 0

        if self.args.height_mode == "floor":
            min_height = self.args.vehicle_height_min
            max_height = self.args.vehicle_height_max
        else:
            min_height = self.args.vehicle_z_min
            max_height = self.args.vehicle_z_max

        vehicle_valid = base_valid & (height >= min_height) & (height <= max_height)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if not np.any(vehicle_valid):
            return None, 0

        col = np.floor(
            (forward[vehicle_valid] - self.args.forward_min) / self.args.resolution
        ).astype(np.int32)
        row_from_bottom = np.floor(
            (lateral[vehicle_valid] - self.args.lateral_min) / self.args.resolution
        ).astype(np.int32)
        row = self.height - 1 - row_from_bottom
        ok = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        mask[row[ok], col[ok]] = 255

        if self.args.vehicle_mask_dilate > 0:
            size = self.args.vehicle_mask_dilate * 2 + 1
            kernel = np.ones((size, size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        best = None
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.args.min_cluster_pixels or area > self.args.max_cluster_pixels:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label]
            cxi = int(round(cx))
            cyi = int(round(cy))
            if self.args.vehicle_track_area_only:
                if self.vehicle_search_mask is None:
                    continue
                if not (
                    0 <= cyi < self.vehicle_search_mask.shape[0]
                    and 0 <= cxi < self.vehicle_search_mask.shape[1]
                    and bool(self.vehicle_search_mask[cyi, cxi])
                ):
                    continue
            if self.args.vehicle_road_only:
                if self.road_mask is None:
                    continue
                if not (
                    0 <= cyi < self.road_mask.shape[0]
                    and 0 <= cxi < self.road_mask.shape[1]
                    and bool(self.road_mask[cyi, cxi])
                ):
                    continue
            distance = None
            if self.vehicle_confirmed and self.last_vehicle_centroid is not None:
                prev_x, prev_y = self.last_vehicle_centroid
                distance = float(np.hypot(cx - prev_x, cy - prev_y))
                if distance > self.args.max_tracking_jump_pixels:
                    continue
            score = float(area)
            if self.args.prefer_near_previous and distance is not None:
                score = score - distance * 4.0
            candidate = {
                "bbox": (x, y, w, h),
                "centroid": (float(cx), float(cy)),
                "area": area,
                "score": score,
                "distance": distance,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best, int(vehicle_valid.sum())

    def vehicle_status(self, vehicle):
        if vehicle is None:
            return "NO_VEHICLE", None, None, (0, 255, 255)
        if not self.vehicle_confirmed:
            cx, cy = vehicle["centroid"]
            forward, lateral = pixel_to_world(cx, cy, self.args)
            return "CANDIDATE", forward, lateral, (0, 255, 255)

        cx, cy = vehicle["centroid"]
        cxi = int(round(cx))
        cyi = int(round(cy))
        on_road = (
            self.road_mask is not None
            and 0 <= cyi < self.road_mask.shape[0]
            and 0 <= cxi < self.road_mask.shape[1]
            and bool(self.road_mask[cyi, cxi])
        )
        status = "ON_ROAD" if on_road else "OFF_ROAD"
        color = (0, 255, 0) if on_road else (0, 0, 255)
        forward, lateral = pixel_to_world(cx, cy, self.args)
        return status, forward, lateral, color

    def update_confirmation(self, vehicle):
        if vehicle is None:
            self.pending_vehicle_centroid = None
            self.pending_vehicle_count = 0
            return

        cx, cy = vehicle["centroid"]
        if self.vehicle_confirmed and self.last_vehicle_centroid is not None:
            return

        if self.pending_vehicle_centroid is None:
            self.pending_vehicle_centroid = (cx, cy)
            self.pending_vehicle_count = 1
        else:
            px, py = self.pending_vehicle_centroid
            distance = float(np.hypot(cx - px, cy - py))
            if distance <= self.args.confirm_distance_pixels:
                self.pending_vehicle_count += 1
                alpha = 0.5
                self.pending_vehicle_centroid = (
                    px * (1.0 - alpha) + cx * alpha,
                    py * (1.0 - alpha) + cy * alpha,
                )
            else:
                self.pending_vehicle_centroid = (cx, cy)
                self.pending_vehicle_count = 1

        if self.pending_vehicle_count >= max(1, self.args.confirm_detections):
            self.vehicle_confirmed = True
            self.last_vehicle_centroid = self.pending_vehicle_centroid

    def draw_vehicle(self, image, vehicle):
        self.update_confirmation(vehicle)
        status, forward, lateral, color = self.vehicle_status(vehicle)
        if vehicle is None:
            self.missed_vehicle_frames += 1
            if self.missed_vehicle_frames >= self.args.lost_reset_frames:
                self.pending_vehicle_centroid = None
                self.pending_vehicle_count = 0
                self.trajectory.clear()
                if self.args.reacquire_anywhere_after_lost:
                    self.last_vehicle_centroid = None
                    self.vehicle_confirmed = False
            cv2.putText(
                image,
                status,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            return

        cx, cy = vehicle["centroid"]
        cxi = int(round(cx))
        cyi = int(round(cy))
        x, y, w, h = vehicle["bbox"]
        thickness = (
            self.args.candidate_bbox_thickness
            if status == "CANDIDATE"
            else self.args.bbox_thickness
        )
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(image, (cxi, cyi), max(1, self.args.center_radius), color, -1)
        if self.vehicle_confirmed:
            self.last_vehicle_centroid = (cx, cy)
        self.missed_vehicle_frames = 0

        if self.vehicle_confirmed:
            self.trajectory.append((cxi, cyi))
            keep = max(1, self.args.trajectory_length)
            if len(self.trajectory) > keep:
                self.trajectory = self.trajectory[-keep:]
            if self.args.show_trajectory and len(self.trajectory) >= 2:
                pts = np.asarray(self.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    image,
                    [pts],
                    False,
                    color,
                    max(1, self.args.trajectory_thickness),
                    cv2.LINE_AA,
                )

        label = f"{status} f={forward:.2f}m l={lateral:.2f}m area={vehicle['area']}"
        if status == "CANDIDATE":
            label += f" {self.pending_vehicle_count}/{max(1, self.args.confirm_detections)}"
        cv2.putText(
            image,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    def log_frame(self, msg, vehicle, vehicle_points):
        if self.csv_writer is None:
            return
        every = max(1, self.args.log_every_n_frames)
        if self.processed_frames % every != 0:
            return

        status, forward, lateral, _color = self.vehicle_status(vehicle)
        row = {
            "wall_time": f"{time.time():.6f}",
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "status": status,
            "forward_m": "" if forward is None else f"{forward:.4f}",
            "lateral_m": "" if lateral is None else f"{lateral:.4f}",
            "cluster_area_px": "",
            "bbox_x": "",
            "bbox_y": "",
            "bbox_w": "",
            "bbox_h": "",
            "vehicle_points": int(vehicle_points),
            "confirmed": int(self.vehicle_confirmed),
            "pending_count": int(self.pending_vehicle_count),
            "missed_frames": int(self.missed_vehicle_frames),
        }
        if vehicle is not None:
            x, y, w, h = vehicle["bbox"]
            row.update(
                {
                    "cluster_area_px": int(vehicle["area"]),
                    "bbox_x": int(x),
                    "bbox_y": int(y),
                    "bbox_w": int(w),
                    "bbox_h": int(h),
                }
            )
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def on_cloud(self, msg):
        self.check_time_jump(msg)
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
        base_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        base_valid &= ~((x == 0) & (y == 0) & (z == 0))
        base_valid &= (forward >= self.args.forward_min) & (forward < self.args.forward_max)
        base_valid &= (lateral >= self.args.lateral_min) & (lateral < self.args.lateral_max)
        valid = base_valid & (z >= self.args.z_min) & (z <= self.args.z_max)

        if self.args.height_mode == "floor" and self.floor_normal is None:
            self.estimate_floor_plane(x, y, z, base_valid)

        bev = np.zeros((self.height, self.width), dtype=np.float32)
        if np.any(valid):
            col = np.floor(
                (forward[valid] - self.args.forward_min) / self.args.resolution
            ).astype(np.int32)
            row_from_bottom = np.floor(
                (lateral[valid] - self.args.lateral_min) / self.args.resolution
            ).astype(np.int32)
            row = self.height - 1 - row_from_bottom
            ok = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
            np.maximum.at(bev, (row[ok], col[ok]), intensity[valid][ok])

        self.frames.append(bev)
        if len(self.frames) > max(1, self.args.accumulate):
            self.frames.pop(0)
        accum = np.maximum.reduce(self.frames)

        image = np.clip(accum / self.args.intensity_max, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)

        if self.track_overlay is not None and self.track_mask is not None:
            alpha = min(max(self.args.overlay_alpha, 0.0), 1.0)
            image = blend_layout_lines(image, self.track_mask, opacity=alpha)

        if self.args.height_mode == "floor":
            height = self.floor_height(x, y, z)
        else:
            height = z
        vehicle, vehicle_points = self.detect_vehicle(forward, lateral, height, base_valid)
        self.draw_vehicle(image, vehicle)
        self.log_frame(msg, vehicle, vehicle_points)
        self.processed_frames += 1

        if self.args.scale != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=self.args.scale,
                fy=self.args.scale,
                interpolation=cv2.INTER_CUBIC,
            )

        cv2.imshow(self.args.window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            rclpy.shutdown()

        now = time.time()
        if now - self.last_log_time > 2.0:
            status, _forward, _lateral, _color = self.vehicle_status(vehicle)
            self.get_logger().info(
                f"valid_roi_points={int(valid.sum())}, vehicle_points={vehicle_points}, "
                f"vehicle_status={status}, displayed_shape={image.shape[1]}x{image.shape[0]}"
            )
            self.last_log_time = now


def main():
    args = parse_args()
    rclpy.init()
    node = LiveBevIntensityViewer(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        node.close_csv_logger()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
