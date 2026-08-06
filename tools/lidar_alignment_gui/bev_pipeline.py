"""Shared BEV geometry, vehicle tracking, and track-map alignment I/O.

Extracted from live_bev_intensity_viewer.py / track_map_align.py so the
PySide6 studio and the legacy cv2 viewer stay numerically identical.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from layout_rendering import (
    blend_layout_lines,
    colorize_layout_lines,
    extract_layout_line_mask,
    warp_layout_line_mask,
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class BevGeometry:
    resolution: float = 0.05
    forward_axis: str = "-y"
    forward_min: float = 1.3
    forward_max: float = 13.3
    lateral_min: float = -7.0
    lateral_max: float = 9.0
    z_min: float = -2.2
    z_max: float = 0.6
    rotation_deg: float = 0.0

    @property
    def width(self):
        return int(round((self.forward_max - self.forward_min) / self.resolution))

    @property
    def height(self):
        return int(round((self.lateral_max - self.lateral_min) / self.resolution))

    def split_axes(self, x, y):
        if self.rotation_deg:
            rad = np.deg2rad(self.rotation_deg)
            c, s = np.cos(rad), np.sin(rad)
            x, y = x * c + y * s, -x * s + y * c
        if self.forward_axis == "-y":
            return -y, x
        if self.forward_axis == "+y":
            return y, -x
        if self.forward_axis == "+x":
            return x, y
        return -x, -y

    def grid_indices(self, forward, lateral):
        col = np.floor((forward - self.forward_min) / self.resolution).astype(np.int32)
        row_from_bottom = np.floor(
            (lateral - self.lateral_min) / self.resolution
        ).astype(np.int32)
        row = self.height - 1 - row_from_bottom
        ok = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        return col, row, ok

    def rasterize_max(self, forward, lateral, values, valid):
        bev = np.zeros((self.height, self.width), dtype=np.float32)
        if np.any(valid):
            col, row, ok = self.grid_indices(forward[valid], lateral[valid])
            np.maximum.at(bev, (row[ok], col[ok]), values[valid][ok])
        return bev

    def pixel_to_world(self, col, row):
        forward = self.forward_min + (col + 0.5) * self.resolution
        lateral = self.lateral_max - (row + 0.5) * self.resolution
        return forward, lateral

    def pixel_to_sensor(self, col, row):
        """BEV pixel → sensor-frame (x, y) meters, independent of view rotation."""
        forward, lateral = self.pixel_to_world(col, row)
        if self.forward_axis == "-y":
            xr, yr = lateral, -forward
        elif self.forward_axis == "+y":
            xr, yr = -lateral, forward
        elif self.forward_axis == "+x":
            xr, yr = forward, lateral
        else:
            xr, yr = -forward, -lateral
        if self.rotation_deg:
            rad = np.deg2rad(self.rotation_deg)
            c, s = np.cos(rad), np.sin(rad)
            return xr * c - yr * s, xr * s + yr * c
        return xr, yr

    def to_dict(self):
        return {
            "resolution": self.resolution,
            "forward_axis": self.forward_axis,
            "forward_min": self.forward_min,
            "forward_max": self.forward_max,
            "lateral_min": self.lateral_min,
            "lateral_max": self.lateral_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "rotation_deg": self.rotation_deg,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


# ---------------------------------------------------------------------------
# Track-map masks (identical to live_bev_intensity_viewer / track_map_align)
# ---------------------------------------------------------------------------


def road_close_kernel(track_bgr, baseline_width=1181, baseline_kernel=9):
    scaled = int(round(baseline_kernel * track_bgr.shape[1] / baseline_width))
    return np.ones((max(baseline_kernel, scaled),) * 2, np.uint8)


def extract_road_mask(track_bgr):
    hsv = cv2.cvtColor(track_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    dark_road = (value < 90) & (saturation < 90)
    mask = dark_road.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, road_close_kernel(track_bgr))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def make_vehicle_search_mask(road_mask, buffer_pixels):
    if buffer_pixels <= 0:
        return road_mask.copy()
    kernel_size = buffer_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(road_mask, kernel, iterations=1)


# ---------------------------------------------------------------------------
# Overlay loading
# ---------------------------------------------------------------------------


class OverlayError(RuntimeError):
    pass


def _world_to_pixel_float(geometry, x, y):
    """Continuous (sub-pixel, center-convention) BEV pixel of a sensor-frame point."""
    if geometry.rotation_deg:
        rad = np.deg2rad(geometry.rotation_deg)
        c, s = np.cos(rad), np.sin(rad)
        x, y = x * c + y * s, -x * s + y * c
    axis = geometry.forward_axis
    if axis == "-y":
        forward, lateral = -y, x
    elif axis == "+y":
        forward, lateral = y, -x
    elif axis == "+x":
        forward, lateral = x, y
    else:
        forward, lateral = -x, -y
    col = (forward - geometry.forward_min) / geometry.resolution - 0.5
    row = (geometry.lateral_max - lateral) / geometry.resolution - 0.5
    return col, row


def geometry_pixel_affine(src_geometry, dst_geometry):
    """3x3 affine mapping src-geometry BEV pixels to dst-geometry BEV pixels.

    Both grids sample the same sensor plane, so any difference in crop,
    rotation, axis, or resolution is an affine transform of pixel coords.
    """
    reference = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    src = np.float32([_world_to_pixel_float(src_geometry, px, py) for px, py in reference])
    dst = np.float32([_world_to_pixel_float(dst_geometry, px, py) for px, py in reference])
    affine = cv2.getAffineTransform(src, dst)
    return np.vstack([affine, [0.0, 0.0, 1.0]])


@lru_cache(maxsize=4)
def _track_masks_cached(path_str, _mtime):
    track_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if track_bgr is None:
        raise OverlayError(f"트랙맵 이미지를 읽지 못했습니다: {path_str}")
    return track_bgr, extract_layout_line_mask(track_bgr), extract_road_mask(track_bgr)


def load_track_overlay(
    track_map_path,
    homography_json_path,
    geometry,
    line_supersample=4,
    line_min_source_width=5,
    offroad_buffer_pixels=8,
    render_scale=1.0,
):
    """Load and warp the registered track map for the given BEV geometry.

    line_alpha is rendered at render_scale x the BEV canvas so overlay lines
    stay crisp on an upscaled display image; road/search masks stay at canvas
    resolution because detection runs on the BEV grid.

    Returns dict(line_alpha, render_scale, road_mask, search_mask, homography).
    Raises OverlayError with a user-facing reason when unusable.
    """
    track_path = Path(track_map_path)
    homography_path = Path(homography_json_path)
    if not track_path.exists():
        raise OverlayError(f"트랙맵 이미지가 없습니다: {track_path}")
    if not homography_path.exists():
        raise OverlayError("정합(homography) 파일이 아직 없습니다. ‘트랙 정합’을 실행하세요.")

    track_bgr, line_mask, road_mask = _track_masks_cached(
        str(track_path), track_path.stat().st_mtime_ns
    )

    data = json.loads(homography_path.read_text(encoding="utf-8"))
    homography = np.asarray(data["homography_track_to_bev"], dtype=np.float64)

    saved_geom = data.get("bev_geometry")
    if saved_geom:
        # Both BEV grids sample the same sensor plane, so a homography made on
        # one view maps to any other view (crop/rotation/axis/resolution
        # changes included) through a pixel-space affine transform.
        src_geometry = BevGeometry.from_dict(saved_geom)
        homography = geometry_pixel_affine(src_geometry, geometry) @ homography
    else:
        saved_bev = data.get("bev_image_size") or {}
        if saved_bev and (
            int(saved_bev.get("width", -1)) != geometry.width
            or int(saved_bev.get("height", -1)) != geometry.height
        ):
            raise OverlayError(
                f"저장된 정합은 {saved_bev.get('width')}x{saved_bev.get('height')} BEV 기준인데 "
                f"현재 BEV는 {geometry.width}x{geometry.height}입니다. 재정합이 필요합니다."
            )
    saved_track = data.get("track_image_size") or {}
    if saved_track and (
        int(saved_track.get("width", -1)) != track_bgr.shape[1]
        or int(saved_track.get("height", -1)) != track_bgr.shape[0]
    ):
        raise OverlayError(
            f"저장된 정합은 {saved_track.get('width')}x{saved_track.get('height')} 트랙맵 기준인데 "
            f"현재 트랙맵은 {track_bgr.shape[1]}x{track_bgr.shape[0]}입니다. 재정합이 필요합니다."
        )

    size = (geometry.width, geometry.height)
    render_scale = max(1.0, float(render_scale))
    render_size = (
        int(round(geometry.width * render_scale)),
        int(round(geometry.height * render_scale)),
    )
    if render_scale > 1.001:
        scale_matrix = np.array(
            [[render_scale, 0.0, 0.0], [0.0, render_scale, 0.0], [0.0, 0.0, 1.0]]
        )
        render_homography = scale_matrix @ homography
    else:
        render_homography = homography
    supersample = max(2, int(np.ceil(line_supersample / render_scale)))
    line_alpha = warp_layout_line_mask(
        line_mask,
        render_homography,
        render_size,
        supersample=supersample,
        minimum_source_width=line_min_source_width,
    )
    warped_road = cv2.warpPerspective(road_mask, homography, size)
    search = make_vehicle_search_mask(warped_road, offroad_buffer_pixels)

    # 트랙 매트 전체 영역(도로 링의 볼록 껍질 + 0.5 m 여유): 이 밖의
    # 클러스터(벽·커튼·배경 반사)는 차량 후보에서 원천 제외하는 데 쓴다.
    arena = np.zeros_like(warped_road)
    road_points = cv2.findNonZero((warped_road > 0).astype(np.uint8))
    if road_points is not None and len(road_points) >= 3:
        hull = cv2.convexHull(road_points)
        cv2.fillConvexPoly(arena, hull, 255)
        arena = cv2.dilate(arena, np.ones((21, 21), np.uint8))

    return {
        "line_alpha": line_alpha,
        "render_scale": render_scale,
        "road_mask": warped_road > 0,
        "search_mask": search > 0,
        "arena_mask": arena > 0,
        "homography": homography,
    }


# ---------------------------------------------------------------------------
# Alignment computation / persistence (same outputs as track_map_align.py)
# ---------------------------------------------------------------------------


def compute_homography(track_points, bev_points):
    track = np.asarray(track_points, dtype=np.float32)
    bev = np.asarray(bev_points, dtype=np.float32)
    if track.shape != bev.shape or track.shape[0] < 4:
        raise ValueError("트랙/BEV 각각 4개의 대응점이 필요합니다.")
    homography, inliers = cv2.findHomography(track, bev, method=0)
    if homography is None:
        raise RuntimeError("선택한 점으로 호모그래피를 계산하지 못했습니다.")
    return homography, inliers


def render_alignment_preview(bev_bgr, track_bgr, track_points, bev_points, opacity=0.85):
    homography, _ = compute_homography(track_points, bev_points)
    h, w = bev_bgr.shape[:2]
    line_mask = extract_layout_line_mask(track_bgr)
    line_alpha = warp_layout_line_mask(line_mask, homography, (w, h), supersample=4)
    preview = blend_layout_lines(bev_bgr, line_alpha, opacity=opacity)
    for idx, (x, y) in enumerate(bev_points, start=1):
        cv2.circle(preview, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
        cv2.putText(
            preview,
            str(idx),
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def save_alignment(
    track_bgr,
    bev_gray,
    track_points,
    bev_points,
    track_map_path,
    output_dir,
    out_prefix="track_map_aligned",
    overlay_alpha=0.65,
    geometry=None,
):
    """Persist point pairs + homography + preview images. Returns path dict."""
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    config_dir = output_dir / "config"
    image_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    bev_path = image_dir / "alignment_bev_intensity_raw.png"
    cv2.imwrite(str(bev_path), bev_gray)
    bev_bgr = cv2.cvtColor(bev_gray, cv2.COLOR_GRAY2BGR)

    points_path = config_dir / "track_map_alignment_points.json"
    points_path.write_text(
        json.dumps(
            {
                "point_order": (
                    "Same physical 4 points clicked in order on "
                    "track_map_points and bev_image_points."
                ),
                "track_map_points": [[int(round(x)), int(round(y))] for x, y in track_points],
                "bev_image_points": [[int(round(x)), int(round(y))] for x, y in bev_points],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    homography, inliers = compute_homography(track_points, bev_points)
    bev_h, bev_w = bev_bgr.shape[:2]
    line_mask = extract_layout_line_mask(track_bgr)
    road_mask = extract_road_mask(track_bgr)
    line_alpha = warp_layout_line_mask(line_mask, homography, (bev_w, bev_h), supersample=4)
    warped_overlay = colorize_layout_lines(line_alpha)
    warped_road = cv2.warpPerspective(road_mask, homography, (bev_w, bev_h))
    composite = blend_layout_lines(
        bev_bgr, line_alpha, opacity=float(np.clip(overlay_alpha, 0.0, 1.0))
    )

    homography_path = config_dir / f"{out_prefix}_homography.json"
    warped_path = image_dir / f"{out_prefix}_track_warped.png"
    overlay_path = image_dir / f"{out_prefix}_overlay.png"
    road_mask_path = image_dir / f"{out_prefix}_road_mask.png"

    result = {
        "homography_track_to_bev": homography.tolist(),
        "homography_bev_to_track": np.linalg.inv(homography).tolist(),
        "track_map": str(track_map_path),
        "bev_image": str(bev_path),
        "points_json": str(points_path),
        "track_map_points": np.asarray(track_points, dtype=float).tolist(),
        "bev_image_points": np.asarray(bev_points, dtype=float).tolist(),
        "track_image_size": {
            "width": int(track_bgr.shape[1]),
            "height": int(track_bgr.shape[0]),
        },
        "bev_image_size": {"width": int(bev_w), "height": int(bev_h)},
        "inliers": inliers.ravel().astype(int).tolist() if inliers is not None else [],
    }
    if geometry is not None:
        result["bev_geometry"] = geometry.to_dict()
    homography_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cv2.imwrite(str(warped_path), warped_overlay)
    cv2.imwrite(str(overlay_path), composite)
    cv2.imwrite(str(road_mask_path), warped_road)

    return {
        "homography_json": homography_path,
        "points_json": points_path,
        "bev_image": bev_path,
        "overlay_preview": overlay_path,
        "warped_track": warped_path,
        "road_mask": road_mask_path,
    }


# ---------------------------------------------------------------------------
# Vehicle detection / tracking (port of live_bev_intensity_viewer logic)
# ---------------------------------------------------------------------------


@dataclass
class DetectionParams:
    detect_vehicle: bool = True
    height_mode: str = "floor"  # or "raw-z"
    vehicle_z_min: float = -0.15
    vehicle_z_max: float = 0.18
    vehicle_height_min: float = 0.03
    vehicle_height_max: float = 0.35
    floor_candidate_z_min: float = -2.0
    floor_candidate_z_max: float = 0.3
    floor_plane_threshold: float = 0.03
    floor_ransac_iters: int = 250
    floor_max_points: int = 30000
    min_cluster_pixels: int = 3
    max_cluster_pixels: int = 500
    vehicle_mask_dilate: int = 1
    offroad_buffer_pixels: int = 8
    vehicle_road_only: bool = False
    vehicle_track_area_only: bool = True
    # 적응형 배경 차분: 자동 박스 모드의 유물. 정지 차량을 후보에서 지워
    # ‘차량 지정’ 첫 드래그의 씨앗 탐색을 방해하므로 기본 비활성.
    use_background_subtraction: bool = False
    bg_alpha: float = 0.015
    bg_threshold: float = 0.25
    bg_dilate: int = 1
    show_candidates: bool = True
    match_distance_pixels: float = 40.0
    track_lost_frames: int = 25
    max_vehicles: int = 8
    # A cluster split by reflections merges back into one vehicle.
    merge_distance_pixels: float = 8.0
    # Unconfirmed tracks must appear near-consecutively; flickering noise dies.
    pending_lost_frames: int = 2
    # A vehicle that has not moved keeps its track through long dropouts.
    stationary_lost_frames: int = 150
    # A detection reappearing near a dead confirmed track revives its ID.
    revive_distance_pixels: float = 25.0
    # User-designated vehicles: local search, never deleted automatically.
    designated_match_distance_pixels: float = 30.0
    designated_lost_grace: int = 10
    # 오래 정지해 있던 차량은 좁은 반경만 본다 — 옆을 지나가는 차를 잡아채지
    # 않도록. (12px = 0.6 m @ 0.05 m/px)
    stationary_match_distance_pixels: float = 12.0
    # 사람이 차에 붙어 클러스터가 차 크기보다 이만큼 커지면(면적 배수/최대변
    # 배수) 오염으로 보고 위치 갱신을 멈춘다 — 박스가 사람 쪽으로 끌려가는
    # 것을 막는다. 사람이 떨어지면 정상 추적 재개.
    contaminated_area_ratio: float = 2.0
    contaminated_dim_ratio: float = 1.5
    # 사람 필터: 바닥에서 이 높이(m) 이상의 점이 서 있는 셀과 겹치는 클러스터는
    # 사람으로 보고 차량 지정·매칭에서 제외한다. RC카는 0.35 m를 넘지 않는다.
    exclude_tall_objects: bool = True
    # 0.4 m: 차 최대 높이(0.35 m) 바로 위 — 쭈그려 앉은 사람도 머리/몸통이 걸린다.
    tall_object_min_height: float = 0.4
    # 클러스터의 절반 이상이 tall 셀과 겹쳐야 사람으로 본다 — 벽·구조물
    # 근처의 차량이 팽창된 tall 셀에 스치는 오판정을 줄인다.
    tall_overlap_ratio: float = 0.5
    # 15 cm 팽창: 다리-몸통 셀을 묶되, 벽 옆을 지나는 차량까지 tall로
    # 오판정하지 않는 절충값. (떨어진 다리 조각은 크기 게이트가 거른다.)
    tall_mask_dilate: int = 3
    # 정적 점유맵: 수 초 이상 같은 자리에 있는 셀(정지 차량·시설물). 주행하다
    # 소실된 차량의 재획득 대상에서 정적 클러스터를 제외하는 데 쓴다.
    static_prob_alpha: float = 0.05
    # 0.4: 부분 스캔으로 벽이 간헐 점유(~50%)여도 정적으로 잡되, 지나가는
    # 차의 경로 셀(~0.1)은 걸리지 않는 경계값.
    static_prob_threshold: float = 0.4
    reacquire_veto_lost: int = 3
    # 자기 클러스터 크기(EMA) 대비 이 비율 미만의 조각(사람 다리 등)은 매칭 불가.
    min_area_ratio: float = 0.3
    # 한 프레임에 허용하는 최대 이동(px). 벽 호처럼 거대한 바운딩박스를 가진
    # 클러스터를 다리 삼아 먼 곳으로 순간이동하는 것을 물리적으로 차단한다.
    max_step_pixels: float = 16.0
    # 후보 표시/클릭 대상 최소 크기(px, 한 변). 자잘한 노이즈 박스 숨김.
    min_candidate_display_dim: int = 5
    # 팽창 전 원시 점유 셀 최소 개수 — 바닥 반사 스펙클(1~2셀)이 팽창으로
    # 부풀어 후보가 되는 것을 차단. 원거리 차량도 3셀 이상은 나온다.
    min_raw_cells: int = 3
    # Fraction of learning frames a cell must be occupied to become frozen
    # background. Low on purpose: flickering static props still qualify.
    bg_capture_ratio: float = 0.08
    confirm_detections: int = 3
    confirm_distance_pixels: float = 20.0
    show_trajectory: bool = True
    trajectory_length: int = 80
    bbox_thickness: int = 1
    center_radius: int = 3
    trajectory_thickness: int = 1


def build_exclusion_mask(rects_sensor, geometry):
    """센서 좌표(m) 사각형 목록 → 현재 기하의 제외 마스크(bool).

    우클릭으로 등록한 ‘후보 제외 영역’은 센서 좌표로 저장되므로 크롭·회전이
    바뀌어도 같은 물리 위치를 가린다.
    """
    if not rects_sensor:
        return None
    mask = np.zeros((geometry.height, geometry.width), dtype=np.uint8)
    for corners in rects_sensor:
        pts = []
        for wx, wy in corners:
            col, row = _world_to_pixel_float(geometry, float(wx), float(wy))
            pts.append([int(round(col + 0.5)), int(round(row + 0.5))])
        cv2.fillConvexPoly(mask, np.asarray(pts, dtype=np.int32), 255)
    return mask > 0


def _blend_angle(current, target, alpha):
    """EMA on angles with a 180-degree period (boxes are symmetric)."""
    diff = ((target - current + 90.0) % 180.0) - 90.0
    return ((current + alpha * diff + 90.0) % 180.0) - 90.0


def save_background(mask, geometry, config_dir):
    """Persist a learned frozen-background mask with the geometry it was made on."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(config_dir / "background_mask.png"), mask.astype(np.uint8) * 255)
    (config_dir / "background_meta.json").write_text(
        json.dumps({"geometry": geometry.to_dict()}, indent=2), encoding="utf-8"
    )


def load_background(config_dir, geometry):
    """Load the frozen background, warping it if the view geometry changed."""
    config_dir = Path(config_dir)
    mask_path = config_dir / "background_mask.png"
    meta_path = config_dir / "background_meta.json"
    if not mask_path.exists() or not meta_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    try:
        saved = BevGeometry.from_dict(
            json.loads(meta_path.read_text(encoding="utf-8"))["geometry"]
        )
    except Exception:
        return None
    if saved.to_dict() == geometry.to_dict():
        return mask > 127
    matrix = geometry_pixel_affine(saved, geometry)
    warped = cv2.warpPerspective(
        mask, matrix, (geometry.width, geometry.height), flags=cv2.INTER_NEAREST
    )
    return warped > 127


class VehicleTracker:
    """Multi-vehicle detector/tracker over the BEV grid.

    Static scenery is removed by a frozen reference background (learned once
    with begin_background_capture; persists across sessions) or, when none
    exists, by an adaptive EMA background. New clusters — moving or parked —
    are matched to persistent tracks by nearest centroid, so several cars can
    hold boxes at the same time.
    """

    def __init__(self, geometry, params=None, log=None):
        self.geom = geometry
        self.params = params or DetectionParams()
        self.log = log or (lambda message: None)
        self.floor_normal = None
        self.floor_d = None
        self.bg_prob = None
        self.frozen_bg = None
        self.static_prob = None
        self.arena_mask = None
        # 우클릭으로 등록한 후보 제외 영역 (canvas bool, GUI가 재구축).
        self.exclusion_mask = None
        # 실제 프레임 간격 / 공칭 주기 (패킷 손실로 프레임이 건너뛰면 >1).
        # 워커가 매 프레임 갱신하며, 매칭 반경·예측이 이에 비례한다.
        self.frame_gap = 1.0
        self.tracks = []
        self.dormant = []
        self.designated = []
        self.last_candidates = []
        self._next_track_id = 1
        self.focused_id = None
        self._bg_capture_buf = None
        self._bg_capture_seen = 0
        self._bg_capture_target = 0
        self._bg_learned = False

    def reset(self, reset_floor=False):
        self.tracks = []
        self.dormant = []
        self.focused_id = None
        if reset_floor:
            self.floor_normal = None
            self.floor_d = None
            self.bg_prob = None
            self.static_prob = None

    # -- designated vehicles ---------------------------------------------------

    def designate(self, name, rect):
        """Register a user-designated vehicle seeded from clusters inside rect.

        rect is (x0, y0, x1, y1) in BEV-canvas pixels. Returns True when a
        cluster was found inside the rect, False when seeding fell back to the
        rect center.
        """
        x0, y0, x1, y1 = rect
        rect_cx = (x0 + x1) / 2.0
        rect_cy = (y0 + y1) / 2.0
        seed = None
        seed_distance = None
        for candidate in self.last_candidates:
            if candidate.get("tall"):
                continue  # 사람으로 판정된 클러스터는 차량 씨앗이 될 수 없다.
            cx, cy = candidate["centroid"]
            bx, by, bw, bh = candidate["bbox"]
            intersects = not (
                bx > x1 or bx + bw < x0 or by > y1 or by + bh < y0
            )
            if intersects:
                # 박스 중심에 가장 가까운 클러스터를 차량으로 본다 — 박스
                # 가장자리에 걸친 사람(점이 더 많음)을 잡지 않도록.
                distance = float(np.hypot(cx - rect_cx, cy - rect_cy))
                if seed is None or distance < seed_distance:
                    seed = candidate
                    seed_distance = distance
        self.designated = [d for d in self.designated if d["id"] != name]
        size = (
            max(3, int(round(x1 - x0))),
            max(3, int(round(y1 - y0))),
        )
        if seed is not None:
            centroid, area = seed["centroid"], seed["area"]
        else:
            centroid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            area = 0
        bbox = (
            int(round(centroid[0] - size[0] / 2.0)),
            int(round(centroid[1] - size[1] / 2.0)),
            size[0],
            size[1],
        )
        self.designated.append(
            {
                "id": name,
                "centroid": centroid,
                "bbox": bbox,
                "size": size,
                "seed_rect": (x0, y0, x1, y1),
                "acquired": seed is not None,
                "heading": (
                    seed.get("angle") if seed is not None else None
                ) or 0.0,
                "area": area,
                "lost": 0,
                "matched": True,
                "trajectory": [],
            }
        )
        return seed is not None

    def designate_from_candidate(self, name, candidate):
        """클릭한 후보 클러스터를 그대로 씨앗으로 차량 지정 — 드래그보다 정확."""
        self.designated = [d for d in self.designated if d["id"] != name]
        x, y, w, h = candidate["bbox"]
        size = (max(3, w + 2), max(3, h + 2))
        cx, cy = candidate["centroid"]
        self.designated.append(
            {
                "id": name,
                "centroid": (float(cx), float(cy)),
                "bbox": (
                    int(round(cx - size[0] / 2.0)),
                    int(round(cy - size[1] / 2.0)),
                    size[0],
                    size[1],
                ),
                "size": size,
                "seed_rect": (x - 4, y - 4, x + w + 4, y + h + 4),
                "acquired": True,
                # 지정 즉시 자기 발자국 기준점 설정 — 오래 주차된 차를 지정한
                # 직후에도 정적 거부에서 자기 클러스터가 면제되도록.
                "last_still_pos": (float(cx), float(cy)),
                "heading": candidate.get("angle") or 0.0,
                "area": candidate["area"],
                "area_ema": float(candidate["area"]),
                "velocity": (0.0, 0.0),
                "still": 0,
                "lost": 0,
                "matched": True,
                "trajectory": [],
            }
        )

    def candidate_at(self, col, row):
        """이 지점의 후보 클러스터(박스 +2px 안, 중심 최근접)를 반환."""
        nearest, nearest_distance = None, float("inf")
        minimum_dim = max(1, self.params.min_candidate_display_dim)
        for candidate in self.last_candidates:
            bx, by, bw, bh = candidate["bbox"]
            if max(bw, bh) < minimum_dim:
                continue  # 표시되지 않는 노이즈 후보는 클릭 대상도 아니다.
            if not (bx - 2 <= col <= bx + bw + 2 and by - 2 <= row <= by + bh + 2):
                continue
            cx, cy = candidate["centroid"]
            distance = float(np.hypot(cx - col, cy - row))
            if distance < nearest_distance:
                nearest, nearest_distance = candidate, distance
        return nearest

    def remove_designated(self, name):
        self.designated = [d for d in self.designated if d["id"] != name]
        if self.focused_id == name:
            self.focused_id = None

    def _active_pool(self):
        return self.designated if self.designated else self.tracks

    def track_at(self, col, row):
        """Return the track whose box contains (or is nearest to) this point."""
        nearest, nearest_distance = None, 15.0
        for track in self._active_pool():
            x, y, w, h = track["bbox"]
            if x - 2 <= col <= x + w + 2 and y - 2 <= row <= y + h + 2:
                return track
            cx, cy = track["centroid"]
            distance = float(np.hypot(cx - col, cy - row))
            if distance < nearest_distance:
                nearest, nearest_distance = track, distance
        return nearest

    def add_background_rect(self, bbox, pad=2):
        """Permanently mark a canvas-pixel rect as background (오검출 제거)."""
        if self.frozen_bg is None:
            self.frozen_bg = np.zeros((self.geom.height, self.geom.width), dtype=bool)
        x, y, w, h = bbox
        y0 = max(0, y - pad)
        y1 = min(self.geom.height, y + h + pad + 1)
        x0 = max(0, x - pad)
        x1 = min(self.geom.width, x + w + pad + 1)
        self.frozen_bg[y0:y1, x0:x1] = True

        def inside(track):
            cx, cy = track["centroid"]
            return x0 <= cx < x1 and y0 <= cy < y1

        self.tracks = [t for t in self.tracks if not inside(t)]
        self.dormant = [d for d in self.dormant if not inside(d)]

    def select_at(self, col, row):
        """Focus the confirmed track at/near this BEV-canvas point.

        Returns the focused id, or None when the click landed on empty space
        (which also clears the current focus).
        """
        nearest, nearest_distance = None, 15.0
        for track in self._active_pool():
            if not track.get("confirmed", True):
                continue
            x, y, w, h = track["bbox"]
            if x - 2 <= col <= x + w + 2 and y - 2 <= row <= y + h + 2:
                self.focused_id = track["id"]
                return track["id"]
            cx, cy = track["centroid"]
            distance = float(np.hypot(cx - col, cy - row))
            if distance < nearest_distance:
                nearest, nearest_distance = track, distance
        if nearest is not None:
            self.focused_id = nearest["id"]
            return nearest["id"]
        self.focused_id = None
        return None

    # -- frozen background -----------------------------------------------------

    def begin_background_capture(self, frames):
        self._bg_capture_buf = np.zeros(
            (self.geom.height, self.geom.width), dtype=np.float32
        )
        self._bg_capture_seen = 0
        self._bg_capture_target = max(1, int(frames))

    def capturing_background(self):
        return self._bg_capture_target > 0

    def consume_background_learned(self):
        if self._bg_learned:
            self._bg_learned = False
            return True
        return False

    # -- floor plane -----------------------------------------------------------

    def estimate_floor_plane(self, x, y, z, base_valid):
        p = self.params
        candidate = (
            base_valid & (z >= p.floor_candidate_z_min) & (z <= p.floor_candidate_z_max)
        )
        pts = np.column_stack([x[candidate], y[candidate], z[candidate]]).astype(np.float64)
        if len(pts) < 1000:
            self.log(f"바닥 평면 후보 점이 부족합니다: {len(pts)}")
            return False

        rng = np.random.default_rng(11)
        if len(pts) > p.floor_max_points:
            pts = pts[rng.choice(len(pts), p.floor_max_points, replace=False)]

        best = None
        for _ in range(p.floor_ransac_iters):
            sample = pts[rng.choice(len(pts), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, sample[0]))
            distances = np.abs(pts @ normal + d)
            inliers = distances < p.floor_plane_threshold
            count = int(inliers.sum())
            if best is None or count > best[0]:
                best = (count, inliers)

        if best is None or best[0] < 1000:
            self.log("바닥 평면 추정에 실패했습니다.")
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
        self.log(f"바닥 평면 추정 완료: inliers={best[0]}/{len(pts)}")
        return True

    def floor_height(self, x, y, z):
        if self.floor_normal is None or self.floor_d is None:
            return None
        pts = np.column_stack([x, y, z]).astype(np.float64)
        return pts @ self.floor_normal + self.floor_d

    # -- detection -------------------------------------------------------------

    def detect(self, forward, lateral, height, base_valid, road_mask, search_mask):
        """Return (cluster candidates, raised-point count) for this frame."""
        p = self.params
        if not p.detect_vehicle or height is None:
            return [], 0

        if p.height_mode == "floor":
            min_height, max_height = p.vehicle_height_min, p.vehicle_height_max
        else:
            min_height, max_height = p.vehicle_z_min, p.vehicle_z_max

        vehicle_valid = base_valid & (height >= min_height) & (height <= max_height)
        mask = np.zeros((self.geom.height, self.geom.width), dtype=np.uint8)
        if np.any(vehicle_valid):
            col, row, ok = self.geom.grid_indices(
                forward[vehicle_valid], lateral[vehicle_valid]
            )
            mask[row[ok], col[ok]] = 255

        # 정적 점유맵 갱신: 수 초 이상 계속 점유된 셀 = 정지 물체(정지 차량,
        # 시설물). 주행하다 놓친 차량이 이런 정적 클러스터를 잡는 것을 막는다.
        if self.static_prob is None or self.static_prob.shape != mask.shape:
            self.static_prob = np.zeros(mask.shape, dtype=np.float32)
        self.static_prob *= 1.0 - p.static_prob_alpha
        self.static_prob[mask > 0] += p.static_prob_alpha

        # 사람(키 큰 물체) 셀: 차량 높이 위로 점이 서 있는 셀. 이 위의
        # 클러스터는 사람의 다리/몸통이므로 차량 후보에서 배제한다.
        tall_mask = None
        if p.exclude_tall_objects:
            tall_valid = base_valid & (height > p.tall_object_min_height)
            if np.any(tall_valid):
                tall_mask = np.zeros(
                    (self.geom.height, self.geom.width), dtype=np.uint8
                )
                tcol, trow, tok = self.geom.grid_indices(
                    forward[tall_valid], lateral[tall_valid]
                )
                tall_mask[trow[tok], tcol[tok]] = 255
                if p.tall_mask_dilate > 0:
                    size = p.tall_mask_dilate * 2 + 1
                    tall_mask = cv2.dilate(tall_mask, np.ones((size, size), np.uint8))

        if (
            self._bg_capture_target > 0
            and self._bg_capture_buf is not None
            and self._bg_capture_buf.shape == mask.shape
        ):
            self._bg_capture_buf += (mask > 0).astype(np.float32)
            self._bg_capture_seen += 1
            if self._bg_capture_seen >= self._bg_capture_target:
                frozen = self._bg_capture_buf >= max(
                    1.0, p.bg_capture_ratio * self._bg_capture_seen
                )
                size = max(1, p.bg_dilate) * 2 + 3
                frozen = (
                    cv2.dilate(frozen.astype(np.uint8), np.ones((size, size), np.uint8))
                    > 0
                )
                self.frozen_bg = frozen
                self._bg_capture_target = 0
                self._bg_learned = True
                self.log(f"배경 학습 완료: {int(frozen.sum())}셀을 배경으로 제외합니다.")

        if self.designated:
            # Designated mode: matching is local to each vehicle, so background
            # subtraction is unnecessary — and it must not erase parked cars.
            pass
        elif self.frozen_bg is not None and self.frozen_bg.shape == mask.shape:
            mask[self.frozen_bg] = 0
        elif p.use_background_subtraction:
            if self.bg_prob is None or self.bg_prob.shape != mask.shape:
                self.bg_prob = np.zeros(mask.shape, dtype=np.float32)
            self.bg_prob *= 1.0 - p.bg_alpha
            self.bg_prob[mask > 0] += p.bg_alpha
            background = (self.bg_prob >= p.bg_threshold).astype(np.uint8)
            if p.bg_dilate > 0:
                size = p.bg_dilate * 2 + 1
                background = cv2.dilate(background, np.ones((size, size), np.uint8))
            mask[background > 0] = 0

        if not np.any(mask):
            self.last_candidates = []
            return [], int(vehicle_valid.sum())

        # 우클릭으로 등록한 제외 영역: 어떤 모드에서든 후보를 만들지 않는다.
        if (
            self.exclusion_mask is not None
            and self.exclusion_mask.shape == mask.shape
        ):
            mask[self.exclusion_mask] = 0

        raw_occupancy = mask > 0  # 팽창 전 원시 점유 (노이즈 게이트용)
        if p.vehicle_mask_dilate > 0:
            size = p.vehicle_mask_dilate * 2 + 1
            kernel = np.ones((size, size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        # 라벨별 집계를 bincount 한 번씩으로 일괄 계산 — 클러스터마다
        # 픽셀을 뽑던 이전 방식은 노이즈 폭주 시 프레임당 200ms를 먹었다.
        labels_flat = labels.ravel()
        raw_counts = np.bincount(
            labels_flat[raw_occupancy.ravel()], minlength=num_labels
        )
        tall_counts = (
            np.bincount(labels_flat[(tall_mask > 0).ravel()], minlength=num_labels)
            if tall_mask is not None
            else None
        )
        arena_ok = (
            self.arena_mask
            if self.arena_mask is not None and self.arena_mask.shape == mask.shape
            else None
        )
        # 라벨 필터를 전부 벡터로: 면적·원시셀·arena·tall을 한 번에 계산.
        areas_all = stats[:, cv2.CC_STAT_AREA]
        keep = (
            (areas_all >= p.min_cluster_pixels)
            & (areas_all <= p.max_cluster_pixels)
            & (raw_counts >= p.min_raw_cells)
        )
        keep[0] = False  # 배경 라벨
        if arena_ok is not None:
            cyi = np.clip(np.round(centroids[:, 1]).astype(np.int64), 0, arena_ok.shape[0] - 1)
            cxi = np.clip(np.round(centroids[:, 0]).astype(np.int64), 0, arena_ok.shape[1] - 1)
            keep &= arena_ok[cyi, cxi]
        tall_flags = (
            tall_counts >= p.tall_overlap_ratio * np.maximum(areas_all, 1)
            if tall_counts is not None
            else np.zeros(num_labels, dtype=bool)
        )
        kept = np.nonzero(keep)[0]
        # 노이즈 폭주 시 상한: 면적 큰 순 50개 (표시·매칭 모두 충분).
        if len(kept) > 50:
            kept = kept[np.argsort(-areas_all[kept])[:50]]
        else:
            kept = kept[np.argsort(-areas_all[kept])]
        stats_k = stats[kept]
        centroids_k = centroids[kept]
        candidates = [
            {
                "bbox": (int(sx), int(sy), int(sw), int(sh)),
                "centroid": (float(cx), float(cy)),
                "area": int(sa),
                "angle": None,
                "tall": bool(tf),
            }
            for (sx, sy, sw, sh, sa), (cx, cy), tf in zip(
                stats_k, centroids_k, tall_flags[kept]
            )
        ]

        # A car whose returns split into nearby blobs must stay ONE vehicle.
        merged = []
        for candidate in candidates:
            cx, cy = candidate["centroid"]
            target = None
            for existing in merged:
                # 사람(tall)과 차량 조각은 인접해도 합치지 않는다.
                if existing.get("tall") != candidate.get("tall"):
                    continue
                ex, ey = existing["centroid"]
                if float(np.hypot(cx - ex, cy - ey)) <= p.merge_distance_pixels:
                    target = existing
                    break
            if target is None:
                merged.append(dict(candidate))
                continue
            tx, ty, tw, th = target["bbox"]
            x, y, w, h = candidate["bbox"]
            x0, y0 = min(tx, x), min(ty, y)
            x1, y1 = max(tx + tw, x + w), max(ty + th, y + h)
            total = target["area"] + candidate["area"]
            target["centroid"] = (
                (target["centroid"][0] * target["area"] + cx * candidate["area"]) / total,
                (target["centroid"][1] * target["area"] + cy * candidate["area"]) / total,
            )
            target["bbox"] = (x0, y0, x1 - x0, y1 - y0)
            target["area"] = total
        # 지정(드래그) 씨앗 탐색은 필터 전 전체 후보에서 이루어져야 한다 —
        # 주차 공간처럼 도로 링 밖의 차량도 지정할 수 있도록.
        self.last_candidates = merged

        if self.designated:
            # Designated mode tracks vehicles anywhere (parking areas are
            # outside the road ring); the area filters exist only to tame
            # auto-mode false positives.
            return merged[: p.max_vehicles], int(vehicle_valid.sum())

        def _inside(mask_array, candidate):
            cxi = int(round(candidate["centroid"][0]))
            cyi = int(round(candidate["centroid"][1]))
            return (
                mask_array is not None
                and 0 <= cyi < mask_array.shape[0]
                and 0 <= cxi < mask_array.shape[1]
                and bool(mask_array[cyi, cxi])
            )

        filtered = []
        for candidate in merged:
            if p.vehicle_track_area_only and not _inside(search_mask, candidate):
                continue
            if p.vehicle_road_only and not _inside(road_mask, candidate):
                continue
            filtered.append(candidate)
        return filtered[: p.max_vehicles], int(vehicle_valid.sum())

    # -- tracking + drawing ----------------------------------------------------

    def _draw_trajectory(self, image, trajectory, scale, thickness):
        """Trajectory segments: green while on-road, red while off-road."""
        for (x1, y1, _f1), (x2, y2, on2) in zip(trajectory[:-1], trajectory[1:]):
            color = (0, 255, 0) if on2 else (0, 0, 255)
            cv2.line(
                image,
                (int(round(x1 * scale)), int(round(y1 * scale))),
                (int(round(x2 * scale)), int(round(y2 * scale))),
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _on_road(self, centroid, road_mask):
        cxi = int(round(centroid[0]))
        cyi = int(round(centroid[1]))
        return (
            road_mask is not None
            and 0 <= cyi < road_mask.shape[0]
            and 0 <= cxi < road_mask.shape[1]
            and bool(road_mask[cyi, cxi])
        )

    def step(self, image, candidates, road_mask, scale=1.0):
        """Associate candidates with tracks, draw all vehicles, return summary."""
        if self.designated:
            return self._step_designated(image, candidates, road_mask, scale)
        # 지정된 차량이 없으면 아무 박스도 만들지 않는다. 자동 트랙은 사람·
        # 반사 노이즈를 차량으로 오인하므로 (사용자 결정) 표시하지 않는다.
        # 검출 후보는 계속 계산되어 ‘차량 지정’ 드래그의 씨앗으로만 쓰인다.
        self.tracks = []
        return {
            "count": 0,
            "on_road": 0,
            "off_road": 0,
            "lost": 0,
            "focused_id": None,
            "vehicles": [],
        }

    def _step_auto_unused(self, image, candidates, road_mask, scale=1.0):
        p = self.params
        s = float(scale)

        available = list(candidates)
        for track in self.tracks:
            track["matched"] = False
        for track in sorted(
            self.tracks, key=lambda t: (not t["confirmed"], -t["hits"])
        ):
            best_index, best_distance = None, p.match_distance_pixels
            tx, ty = track["centroid"]
            for index, candidate in enumerate(available):
                cx, cy = candidate["centroid"]
                distance = float(np.hypot(cx - tx, cy - ty))
                if distance <= best_distance:
                    best_index, best_distance = index, distance
            if best_index is None:
                continue
            candidate = available.pop(best_index)
            moved = float(
                np.hypot(
                    candidate["centroid"][0] - track["centroid"][0],
                    candidate["centroid"][1] - track["centroid"][1],
                )
            )
            track["still"] = track.get("still", 0) + 1 if moved < 2.0 else 0
            track["centroid"] = candidate["centroid"]
            track["bbox"] = candidate["bbox"]
            track["area"] = candidate["area"]
            track["hits"] += 1
            track["misses"] = 0
            track["matched"] = True
            if not track["confirmed"] and track["hits"] >= max(1, p.confirm_detections):
                track["confirmed"] = True

        for candidate in available:
            # 같은 자리에서 사라졌던 차량이면 새 ID 대신 기존 ID를 부활시킨다.
            revived = None
            for dormant in self.dormant:
                dx, dy = dormant["centroid"]
                cx, cy = candidate["centroid"]
                if float(np.hypot(cx - dx, cy - dy)) <= p.revive_distance_pixels:
                    revived = dormant
                    break
            if revived is not None:
                self.dormant.remove(revived)
            self.tracks.append(
                {
                    "id": revived["id"] if revived is not None else self._next_track_id,
                    "centroid": candidate["centroid"],
                    "bbox": candidate["bbox"],
                    "area": candidate["area"],
                    "hits": 1,
                    "misses": 0,
                    "still": 0,
                    "matched": True,
                    "confirmed": revived is not None
                    or max(1, p.confirm_detections) <= 1,
                    "trajectory": [],
                }
            )
            if revived is None:
                self._next_track_id += 1

        kept = []
        for track in self.tracks:
            if not track["matched"]:
                track["misses"] += 1
            if track["confirmed"]:
                limit = (
                    p.stationary_lost_frames
                    if track.get("still", 0) >= 10
                    else p.track_lost_frames
                )
            else:
                limit = p.pending_lost_frames
            if track["misses"] <= max(1, limit):
                kept.append(track)
            elif track["confirmed"]:
                self.dormant.append(
                    {"id": track["id"], "centroid": track["centroid"]}
                )
                if len(self.dormant) > 30:
                    self.dormant.pop(0)
        self.tracks = kept

        focused = self.focused_id
        if focused is not None and not any(
            t["confirmed"] and t["id"] == focused for t in self.tracks
        ):
            focused = None

        count = on_count = off_count = 0
        vehicles = []
        for track in self.tracks:
            x, y, w, h = track["bbox"]
            top_left = (int(round(x * s)), int(round(y * s)))
            bottom_right = (int(round((x + w) * s)), int(round((y + h) * s)))
            if not track["confirmed"]:
                cv2.rectangle(
                    image, top_left, bottom_right, (0, 140, 140),
                    max(1, int(round(s * 0.5))),
                )
                continue

            cx, cy = track["centroid"]
            on_road = self._on_road(track["centroid"], road_mask)
            color = (0, 255, 0) if on_road else (0, 0, 255)
            count += 1
            if on_road:
                on_count += 1
            else:
                off_count += 1

            track["trajectory"].append((int(round(cx)), int(round(cy)), on_road))
            keep = max(1, p.trajectory_length)
            if len(track["trajectory"]) > keep:
                track["trajectory"] = track["trajectory"][-keep:]

            is_focused = focused is not None and track["id"] == focused
            thickness = max(1, int(round(p.bbox_thickness * s)))
            if is_focused:
                thickness = max(2, thickness * 2)
                pad = max(2, int(round(2 * s)))
                cv2.rectangle(
                    image,
                    (top_left[0] - pad, top_left[1] - pad),
                    (bottom_right[0] + pad, bottom_right[1] + pad),
                    (255, 255, 255),
                    max(1, thickness // 2),
                )
            cv2.rectangle(image, top_left, bottom_right, color, thickness)
            cv2.circle(
                image,
                (int(round(cx * s)), int(round(cy * s))),
                max(1, int(round(p.center_radius * s))),
                color,
                -1,
            )
            cv2.putText(
                image,
                str(track["id"]),
                (top_left[0], top_left[1] - max(2, int(round(2 * s)))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.3 * s),
                color,
                max(1, int(round(s * 0.5))),
                cv2.LINE_AA,
            )
            if (
                p.show_trajectory
                and len(track["trajectory"]) >= 2
                and (focused is None or is_focused)
            ):
                self._draw_trajectory(
                    image, track["trajectory"], s,
                    max(1, int(round(p.trajectory_thickness * s))),
                )

            forward, lateral = self.geom.pixel_to_world(cx, cy)
            sensor_x, sensor_y = self.geom.pixel_to_sensor(cx, cy)
            vehicles.append(
                {
                    "id": track["id"],
                    "forward": forward,
                    "lateral": lateral,
                    "x": sensor_x,
                    "y": sensor_y,
                    "on_road": on_road,
                    "status": "ON" if on_road else "OFF",
                    "focused": is_focused,
                }
            )

        return {
            "count": count,
            "on_road": on_count,
            "off_road": off_count,
            "lost": 0,
            "focused_id": focused,
            "vehicles": vehicles,
        }

    def _step_designated(self, image, candidates, road_mask, scale):
        """지정 차량 최소 추적 (사용자 설계).

        후보(T 포함) 박스가 잘 따라가는 이유는 판정이 없기 때문이다. 지정
        추적도 똑같이 한다: 이전 위치에서 가장 가까운 후보를 그대로 채택해
        그 박스를 하이라이트한다. 게이트는 둘뿐 — 한 프레임 이동 상한
        (순간이동 방지)과 다른 지정 차량이 이미 차지한 클러스터 제외(ID
        맞바꿈 방지). 잘못 잡으면 후보 클릭 한 번으로 재지정한다.
        """
        p = self.params
        s = float(scale)

        available = list(candidates)
        gap = float(np.clip(self.frame_gap, 1.0, 4.0))
        for vehicle in sorted(self.designated, key=lambda d: d["lost"]):
            vx, vy = vehicle.get("velocity", (0.0, 0.0))
            tx = vehicle["centroid"][0] + vx * gap
            ty = vehicle["centroid"][1] + vy * gap
            radius = p.max_step_pixels * gap + min(8.0, vehicle["lost"] * 2.0)
            best_index, best_distance = None, float("inf")
            for index, candidate in enumerate(available):
                cx, cy = candidate["centroid"]
                distance = float(np.hypot(cx - tx, cy - ty))
                if distance > radius:
                    continue
                owned = False
                for other in self.designated:
                    if other is vehicle:
                        continue
                    ox, oy, ow, oh = other["bbox"]
                    if ox - 2 <= cx <= ox + ow + 2 and oy - 2 <= cy <= oy + oh + 2:
                        owned = True
                        break
                if owned:
                    continue
                if distance < best_distance:
                    best_index, best_distance = index, distance

            if best_index is None:
                vehicle["lost"] += 1
                vehicle["matched"] = False
                # 놓치면 제자리 대기 — 근처 재등장 시 재획득, 아니면 재클릭.
                vehicle["velocity"] = (vx * 0.6, vy * 0.6)
                continue

            candidate = available.pop(best_index)
            previous = vehicle["centroid"]
            moved_x = candidate["centroid"][0] - previous[0]
            moved_y = candidate["centroid"][1] - previous[1]
            vehicle["velocity"] = (
                0.5 * vx + 0.5 * moved_x,
                0.5 * vy + 0.5 * moved_y,
            )
            vehicle["centroid"] = candidate["centroid"]
            vehicle["bbox"] = candidate["bbox"]  # 클릭한 그 박스를 그대로 따라간다
            vehicle["area"] = candidate["area"]
            vehicle["lost"] = 0
            vehicle["matched"] = True

        focused = self.focused_id
        if focused is not None and not any(
            d["id"] == focused for d in self.designated
        ):
            focused = None

        count = on_count = off_count = lost_count = 0
        vehicles = []
        for vehicle in self.designated:
            count += 1
            is_lost = vehicle["lost"] > max(1, p.designated_lost_grace)
            cx, cy = vehicle["centroid"]
            on_road = self._on_road(vehicle["centroid"], road_mask)
            if is_lost:
                lost_count += 1
                color = (140, 140, 140)
                status = "LOST"
            else:
                color = (0, 255, 0) if on_road else (0, 0, 255)
                status = "ON" if on_road else "OFF"
                if on_road:
                    on_count += 1
                else:
                    off_count += 1
                vehicle["trajectory"].append(
                    (int(round(cx)), int(round(cy)), on_road)
                )
                keep = max(1, p.trajectory_length)
                if len(vehicle["trajectory"]) > keep:
                    vehicle["trajectory"] = vehicle["trajectory"][-keep:]

            is_focused = focused is not None and vehicle["id"] == focused
            x, y, w, h = vehicle["bbox"]
            top_left = (int(round(x * s)), int(round(y * s)))
            bottom_right = (int(round((x + w) * s)), int(round((y + h) * s)))
            thickness = max(1, int(round(p.bbox_thickness * s)))
            if is_focused:
                thickness = max(2, thickness * 2)
                pad = max(2, int(round(2 * s)))
                cv2.rectangle(
                    image,
                    (top_left[0] - pad, top_left[1] - pad),
                    (bottom_right[0] + pad, bottom_right[1] + pad),
                    (255, 255, 255),
                    max(1, thickness // 2),
                )
            cv2.rectangle(image, top_left, bottom_right, color, thickness)
            if not is_lost:
                cv2.circle(
                    image,
                    (int(round(cx * s)), int(round(cy * s))),
                    max(1, int(round(p.center_radius * s))),
                    color,
                    -1,
                )
            cv2.putText(
                image,
                str(vehicle["id"]),
                (top_left[0], top_left[1] - max(2, int(round(2 * s)))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.3 * s),
                color,
                max(1, int(round(s * 0.5))),
                cv2.LINE_AA,
            )
            if (
                p.show_trajectory
                and not is_lost
                and len(vehicle["trajectory"]) >= 2
                and (focused is None or is_focused)
            ):
                self._draw_trajectory(
                    image, vehicle["trajectory"], s,
                    max(1, int(round(p.trajectory_thickness * s))),
                )

            forward, lateral = self.geom.pixel_to_world(cx, cy)
            sensor_x, sensor_y = self.geom.pixel_to_sensor(cx, cy)
            vehicles.append(
                {
                    "id": vehicle["id"],
                    "forward": forward,
                    "lateral": lateral,
                    "x": sensor_x,
                    "y": sensor_y,
                    "on_road": on_road,
                    "status": status,
                    "focused": is_focused,
                }
            )

        return {
            "count": count,
            "on_road": on_count,
            "off_road": off_count,
            "lost": lost_count,
            "focused_id": focused,
            "vehicles": vehicles,
        }
