#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from layout_rendering import (
    blend_layout_lines,
    colorize_layout_lines,
    extract_layout_line_mask,
    warp_layout_line_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute track-map-to-BEV homography from picked point pairs."
    )
    parser.add_argument("--track-map", required=True)
    parser.add_argument("--bev-image", required=True)
    parser.add_argument("--points-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument(
        "--image-output-dir",
        default="",
        help="Directory for PNG outputs. Defaults to the out-prefix directory.",
    )
    parser.add_argument(
        "--config-output-dir",
        default="",
        help="Directory for JSON outputs. Defaults to the out-prefix directory.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.65)
    return parser.parse_args()


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def extract_white_line_mask(track_bgr):
    return extract_layout_line_mask(track_bgr)


def road_close_kernel(track_bgr, baseline_width=1181, baseline_kernel=9):
    """Closing kernel scaled to the track image resolution.

    Must stay identical to live_bev_intensity_viewer.road_close_kernel, otherwise the
    alignment preview mask and the live ON_ROAD judgement mask disagree.
    """
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


def colorize_track_map(track_bgr, white_mask):
    del track_bgr
    return colorize_layout_lines(white_mask)


def main():
    args = parse_args()
    track_path = Path(args.track_map)
    bev_path = Path(args.bev_image)
    points_path = Path(args.points_json)
    out_prefix = Path(args.out_prefix)
    image_output_dir = (
        Path(args.image_output_dir) if args.image_output_dir else out_prefix.parent
    )
    config_output_dir = (
        Path(args.config_output_dir) if args.config_output_dir else out_prefix.parent
    )
    image_output_dir.mkdir(parents=True, exist_ok=True)
    config_output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = out_prefix.name

    track_bgr = load_image(track_path)
    bev_bgr = load_image(bev_path)
    points = json.loads(points_path.read_text(encoding="utf-8"))

    track_points = np.asarray(points["track_map_points"], dtype=np.float32)
    bev_points = np.asarray(points["bev_image_points"], dtype=np.float32)
    if track_points.shape != bev_points.shape or track_points.shape[0] < 4:
        raise ValueError(
            "points-json must contain at least 4 matching track_map_points and bev_image_points"
        )

    homography, inliers = cv2.findHomography(track_points, bev_points, method=0)
    if homography is None:
        raise RuntimeError("Could not compute homography from the selected points")

    bev_h, bev_w = bev_bgr.shape[:2]
    white_mask = extract_white_line_mask(track_bgr)
    road_mask = extract_road_mask(track_bgr)
    warped_line_alpha = warp_layout_line_mask(
        white_mask, homography, (bev_w, bev_h), supersample=4
    )
    warped_overlay = colorize_layout_lines(warped_line_alpha)
    warped_road = cv2.warpPerspective(road_mask, homography, (bev_w, bev_h))
    alpha = float(np.clip(args.overlay_alpha, 0.0, 1.0))
    composite = blend_layout_lines(bev_bgr, warped_line_alpha, opacity=alpha)

    homography_path = config_output_dir / f"{output_stem}_homography.json"
    warped_path = image_output_dir / f"{output_stem}_track_warped.png"
    overlay_path = image_output_dir / f"{output_stem}_overlay.png"
    road_mask_path = image_output_dir / f"{output_stem}_road_mask.png"

    result = {
        "homography_track_to_bev": homography.tolist(),
        "homography_bev_to_track": np.linalg.inv(homography).tolist(),
        "track_map": str(track_path),
        "bev_image": str(bev_path),
        "points_json": str(points_path),
        "track_map_points": track_points.astype(float).tolist(),
        "bev_image_points": bev_points.astype(float).tolist(),
        "track_image_size": {
            "width": int(track_bgr.shape[1]),
            "height": int(track_bgr.shape[0]),
        },
        "bev_image_size": {"width": int(bev_w), "height": int(bev_h)},
        "inliers": inliers.ravel().astype(int).tolist() if inliers is not None else [],
    }
    homography_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cv2.imwrite(str(warped_path), warped_overlay)
    cv2.imwrite(str(overlay_path), composite)
    cv2.imwrite(str(road_mask_path), warped_road)

    print(f"homography_json: {homography_path}")
    print(f"warped_track: {warped_path}")
    print(f"overlay_preview: {overlay_path}")
    print(f"warped_road_mask: {road_mask_path}")


if __name__ == "__main__":
    main()
