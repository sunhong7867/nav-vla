"""BEV vehicle detector for the fixed trackside Hesai OT128.

Ported from ``0725_4점정합ver/live_bev_intensity_viewer.py`` so the detection
geometry stays bit-identical to the alignment run that produced
``track_map_aligned_homography.json``. The viewer keeps its debug/GUI role;
this module is the headless core that ``track_pose_node`` drives.

Sensor frame convention (unchanged from the viewer)::

    forward = -y        lateral = x

BEV canvas: ``col`` indexes forward, ``row`` indexes lateral with row 0 at
``lateral_max``. With the shipped crop this builds a 240x320 canvas, which is
what the stored homography was picked on.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    """Geometry and detection thresholds.

    Defaults mirror the 2026-07-25 long-side-centre LiDAR placement. Changing
    any of the crop fields invalidates the stored homography — see
    ``load_track_masks``.
    """

    resolution: float = 0.05
    forward_min: float = 1.3
    forward_max: float = 13.3
    lateral_min: float = -7.0
    lateral_max: float = 9.0
    z_min: float = -2.2
    z_max: float = 0.6

    # floor plane (RANSAC + SVD refine), fitted once because the sensor is static
    height_mode: str = "floor"  # "floor" | "raw_z"
    floor_candidate_z_min: float = -2.0
    floor_candidate_z_max: float = 0.3
    floor_plane_threshold: float = 0.03
    floor_ransac_iters: int = 250
    floor_max_points: int = 30000

    # vehicle cluster gate
    vehicle_height_min: float = 0.03
    vehicle_height_max: float = 0.35
    vehicle_z_min: float = -0.15
    vehicle_z_max: float = 0.18
    min_cluster_pixels: int = 5
    max_cluster_pixels: int = 500
    vehicle_mask_dilate: int = 1
    max_tracking_jump_pixels: float = 80.0
    prefer_near_previous: bool = True
    vehicle_road_only: bool = False
    vehicle_track_area_only: bool = True
    offroad_buffer_pixels: int = 8

    # detection confirmation / dropout
    confirm_frames: int = 3
    confirm_distance_pixels: float = 20.0
    max_missed_frames: int = 15

    # registered track map (optional; only needed for ON_ROAD/OFF_ROAD + geofence)
    track_map: str = ""
    homography_json: str = ""


def pixel_to_world(col, row, cfg):
    """BEV pixel centre -> (forward_m, lateral_m) in the track frame."""
    forward = cfg.forward_min + (col + 0.5) * cfg.resolution
    lateral = cfg.lateral_max - (row + 0.5) * cfg.resolution
    return forward, lateral


def extract_road_mask(track_bgr):
    hsv = cv2.cvtColor(track_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    # Only the black outer road counts as drivable. Green background, the gray
    # inner area and standalone white decoration are not road.
    dark_road = (value < 90) & (saturation < 90)
    mask = dark_road.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _road_close_kernel(track_bgr))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _road_close_kernel(track_bgr, baseline_width=1181, baseline_kernel=9):
    """Closing kernel scaled to the track image resolution.

    The 9x9 kernel was tuned on the 1181px track.png where it bridges the stop
    line and the crosswalk so the road ring stays closed. On the 5228px
    track2.png those markings are 4.4x wider, so a fixed 9x9 leaves them as
    non-road holes and a vehicle centroid there would be judged OFF_ROAD.
    """
    scaled = int(round(baseline_kernel * track_bgr.shape[1] / baseline_width))
    return np.ones((max(baseline_kernel, scaled),) * 2, np.uint8)


def _vehicle_search_mask(road_mask, buffer_pixels):
    if buffer_pixels <= 0:
        return road_mask.copy()
    size = buffer_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(road_mask, kernel, iterations=1)


class BevVehicleDetector:
    """Stateful single-vehicle detector over accumulated BEV occupancy."""

    def __init__(self, cfg: DetectorConfig, logger=None):
        self.cfg = cfg
        self._log = logger
        self.width = int(round((cfg.forward_max - cfg.forward_min) / cfg.resolution))
        self.height = int(round((cfg.lateral_max - cfg.lateral_min) / cfg.resolution))
        self.floor_normal = None
        self.floor_d = None
        self.road_mask = None
        self.search_mask = None
        self.reset(reset_floor=True)

    # ---------------------------------------------------------------- masks

    def load_track_masks(self):
        """Warp the registered track map into the BEV canvas.

        Returns True when a road mask is available. A homography is only valid
        for the canvas and track image it was picked on, so a size mismatch is
        refused rather than silently drawing a shifted road mask.
        """
        cfg = self.cfg
        if not cfg.track_map or not cfg.homography_json:
            return False
        track_path = Path(cfg.track_map)
        homography_path = Path(cfg.homography_json)
        if not track_path.exists():
            self._warn(f"track map not found: {track_path}")
            return False
        if not homography_path.exists():
            self._warn(f"homography JSON not found: {homography_path}")
            return False

        track_bgr = cv2.imread(str(track_path), cv2.IMREAD_COLOR)
        if track_bgr is None:
            self._warn(f"could not read track map: {track_path}")
            return False

        data = json.loads(homography_path.read_text(encoding="utf-8"))
        homography = np.asarray(data["homography_track_to_bev"], dtype=np.float64)

        saved_bev = data.get("bev_image_size") or {}
        if saved_bev and (
            int(saved_bev.get("width", -1)) != self.width
            or int(saved_bev.get("height", -1)) != self.height
        ):
            self._warn(
                f"homography was picked on a {saved_bev.get('width')}x"
                f"{saved_bev.get('height')} BEV but this crop builds "
                f"{self.width}x{self.height}. Match resolution/forward_*/lateral_* "
                "to the alignment run, or redo the 4-point picking."
            )
            return False
        saved_track = data.get("track_image_size") or {}
        if saved_track and (
            int(saved_track.get("width", -1)) != track_bgr.shape[1]
            or int(saved_track.get("height", -1)) != track_bgr.shape[0]
        ):
            self._warn(
                f"homography was picked on a {saved_track.get('width')}x"
                f"{saved_track.get('height')} track map but track_map is "
                f"{track_bgr.shape[1]}x{track_bgr.shape[0]}: {track_path}"
            )
            return False

        road = extract_road_mask(track_bgr)
        warped_road = cv2.warpPerspective(road, homography, (self.width, self.height))
        self.road_mask = warped_road > 0
        self.search_mask = (
            _vehicle_search_mask(warped_road, self.cfg.offroad_buffer_pixels) > 0
        )
        self._info(
            f"track masks loaded: road_px={int(self.road_mask.sum())}, "
            f"search_px={int(self.search_mask.sum())}"
        )
        return True

    # ---------------------------------------------------------------- state

    def reset(self, reset_floor=False):
        self.last_centroid = None
        self.missed_frames = 0
        self.pending_centroid = None
        self.pending_count = 0
        self.confirmed = False
        if reset_floor:
            self.floor_normal = None
            self.floor_d = None

    # ---------------------------------------------------------------- floor

    def estimate_floor_plane(self, x, y, z, base_valid):
        cfg = self.cfg
        candidate = (
            base_valid
            & (z >= cfg.floor_candidate_z_min)
            & (z <= cfg.floor_candidate_z_max)
        )
        pts = np.column_stack([x[candidate], y[candidate], z[candidate]]).astype(np.float64)
        if len(pts) < 1000:
            self._warn(f"not enough floor candidate points: {len(pts)}")
            return False

        rng = np.random.default_rng(11)
        if len(pts) > cfg.floor_max_points:
            pts = pts[rng.choice(len(pts), cfg.floor_max_points, replace=False)]

        best = None
        for _ in range(cfg.floor_ransac_iters):
            sample = pts[rng.choice(len(pts), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, sample[0]))
            inliers = np.abs(pts @ normal + d) < cfg.floor_plane_threshold
            count = int(inliers.sum())
            if best is None or count > best[0]:
                best = (count, inliers)

        if best is None or best[0] < 1000:
            self._warn("failed to estimate floor plane")
            return False

        inlier_pts = pts[best[1]]
        centroid = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        normal = vh[-1] / np.linalg.norm(vh[-1])
        d = -float(np.dot(normal, centroid))
        if normal[2] < 0:
            normal, d = -normal, -d

        self.floor_normal = normal
        self.floor_d = d
        residual_cm = np.percentile(inlier_pts @ normal + d, [5, 50, 95]) * 100.0
        self._info(
            f"floor plane fitted: normal={np.round(normal, 5).tolist()}, d={d:.4f}, "
            f"inliers={best[0]}/{len(pts)}, residual_cm_p5_50_95="
            f"{np.round(residual_cm, 2).tolist()}"
        )
        return True

    def floor_height(self, x, y, z):
        if self.floor_normal is None or self.floor_d is None:
            return None
        return np.column_stack([x, y, z]).astype(np.float64) @ self.floor_normal + self.floor_d

    # ------------------------------------------------------------- detect

    def process(self, x, y, z):
        """Run one cloud through the pipeline.

        Returns a dict with keys ``status``, ``forward``, ``lateral``,
        ``centroid_px``, ``bbox``, ``area``, ``points`` — or a NO_VEHICLE
        record. Coordinates are metres in the track frame.
        """
        cfg = self.cfg
        forward = -y
        lateral = x
        base_valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        base_valid &= ~((x == 0) & (y == 0) & (z == 0))
        base_valid &= (forward >= cfg.forward_min) & (forward < cfg.forward_max)
        base_valid &= (lateral >= cfg.lateral_min) & (lateral < cfg.lateral_max)

        if cfg.height_mode == "floor":
            if self.floor_normal is None:
                self.estimate_floor_plane(x, y, z, base_valid)
            height = self.floor_height(x, y, z)
        else:
            height = z

        cluster, n_points = self._detect(forward, lateral, height, base_valid)
        self._update_confirmation(cluster)
        return self._describe(cluster, n_points)

    def _detect(self, forward, lateral, height, base_valid):
        cfg = self.cfg
        if height is None:
            return None, 0

        if cfg.height_mode == "floor":
            lo, hi = cfg.vehicle_height_min, cfg.vehicle_height_max
        else:
            lo, hi = cfg.vehicle_z_min, cfg.vehicle_z_max

        vehicle_valid = base_valid & (height >= lo) & (height <= hi)
        if not np.any(vehicle_valid):
            return None, 0

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        col = np.floor(
            (forward[vehicle_valid] - cfg.forward_min) / cfg.resolution
        ).astype(np.int32)
        row_from_bottom = np.floor(
            (lateral[vehicle_valid] - cfg.lateral_min) / cfg.resolution
        ).astype(np.int32)
        row = self.height - 1 - row_from_bottom
        ok = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        mask[row[ok], col[ok]] = 255

        if cfg.vehicle_mask_dilate > 0:
            size = cfg.vehicle_mask_dilate * 2 + 1
            kernel = np.ones((size, size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        best = None
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < cfg.min_cluster_pixels or area > cfg.max_cluster_pixels:
                continue
            cx, cy = centroids[label]
            cxi, cyi = int(round(cx)), int(round(cy))
            if cfg.vehicle_track_area_only and not self._inside(self.search_mask, cxi, cyi):
                continue
            if cfg.vehicle_road_only and not self._inside(self.road_mask, cxi, cyi):
                continue
            distance = None
            if self.confirmed and self.last_centroid is not None:
                px, py = self.last_centroid
                distance = float(np.hypot(cx - px, cy - py))
                if distance > cfg.max_tracking_jump_pixels:
                    continue
            score = float(area)
            if cfg.prefer_near_previous and distance is not None:
                score -= distance * 4.0
            candidate = {
                "bbox": (
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ),
                "centroid": (float(cx), float(cy)),
                "area": area,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best, int(vehicle_valid.sum())

    @staticmethod
    def _inside(mask, col, row):
        if mask is None:
            return False
        return (
            0 <= row < mask.shape[0]
            and 0 <= col < mask.shape[1]
            and bool(mask[row, col])
        )

    def _update_confirmation(self, cluster):
        cfg = self.cfg
        if cluster is None:
            self.pending_centroid = None
            self.pending_count = 0
            if self.confirmed:
                self.missed_frames += 1
                if self.missed_frames > cfg.max_missed_frames:
                    self.reset()
            return

        centroid = cluster["centroid"]
        if self.confirmed:
            self.last_centroid = centroid
            self.missed_frames = 0
            return

        if self.pending_centroid is not None:
            moved = float(
                np.hypot(
                    centroid[0] - self.pending_centroid[0],
                    centroid[1] - self.pending_centroid[1],
                )
            )
            self.pending_count = (
                self.pending_count + 1 if moved <= cfg.confirm_distance_pixels else 1
            )
        else:
            self.pending_count = 1
        self.pending_centroid = centroid
        if self.pending_count >= cfg.confirm_frames:
            self.confirmed = True
            self.last_centroid = centroid
            self.missed_frames = 0

    def _describe(self, cluster, n_points):
        if cluster is None:
            return {"status": "NO_VEHICLE", "forward": None, "lateral": None,
                    "centroid_px": None, "bbox": None, "area": 0, "points": n_points}
        cx, cy = cluster["centroid"]
        forward, lateral = pixel_to_world(cx, cy, self.cfg)
        if not self.confirmed:
            status = "CANDIDATE"
        elif self.road_mask is None:
            status = "TRACKED"
        else:
            status = "ON_ROAD" if self._inside(self.road_mask, int(round(cx)), int(round(cy))) else "OFF_ROAD"
        return {"status": status, "forward": forward, "lateral": lateral,
                "centroid_px": (cx, cy), "bbox": cluster["bbox"],
                "area": cluster["area"], "points": n_points}

    # ---------------------------------------------------------------- log

    def _info(self, msg):
        if self._log is not None:
            self._log.info(msg)

    def _warn(self, msg):
        if self._log is not None:
            self._log.warn(msg)
