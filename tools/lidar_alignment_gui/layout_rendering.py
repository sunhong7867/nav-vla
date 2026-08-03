"""Clean extraction and supersampled BEV rendering of track-map layout lines."""

import cv2
import numpy as np


_BASELINE_WIDTH = 5228


def _scaled_odd(value, width, minimum=3):
    scaled = max(minimum, int(round(value * width / _BASELINE_WIDTH)))
    return scaled if scaled % 2 else scaled + 1


def extract_layout_line_mask(track_bgr):
    """Return clean white/gray layout lines without green-background decoration.

    The previous 3x3 opening removed narrow reference lines before the map was
    warped. This extractor keeps those lines, limits candidates to the connected
    track/infield surface, closes only tiny source gaps, and removes isolated
    specks without discarding long one-pixel guides.
    """
    hsv = cv2.cvtColor(track_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    width = track_bgr.shape[1]

    line_candidate = ((value >= 145) & (saturation <= 85)).astype(np.uint8) * 255

    # Dark asphalt and gray infield form the central layout surface. Closing
    # bridges the white painted borders so it becomes one connected component.
    surface = ((saturation < 105) & (value < 175)).astype(np.uint8) * 255
    bridge = _scaled_odd(51, width)
    surface = cv2.morphologyEx(
        surface, cv2.MORPH_CLOSE, np.ones((bridge, bridge), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(surface, connectivity=8)
    if count <= 1:
        return line_candidate
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    domain = (labels == largest).astype(np.uint8) * 255
    domain_size = _scaled_odd(41, width)
    domain = cv2.dilate(
        domain,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (domain_size, domain_size)),
    )

    mask = cv2.bitwise_and(line_candidate, domain)
    close_size = _scaled_odd(3, width)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )

    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(mask, connectivity=8)
    )
    cleaned = np.zeros_like(mask)
    min_area = max(6, int(round(18 * width / _BASELINE_WIDTH)))
    min_extent = max(6, int(round(14 * width / _BASELINE_WIDTH)))
    for label in range(1, component_count):
        _x, _y, component_w, component_h, area = component_stats[label]
        if area >= min_area or max(component_w, component_h) >= min_extent:
            cleaned[component_labels == label] = 255
    return cleaned


def warp_layout_line_mask(
    line_mask,
    homography,
    output_size,
    supersample=4,
    minimum_source_width=5,
):
    """Warp lines with supersampling and return an anti-aliased uint8 alpha mask."""
    output_w, output_h = map(int, output_size)
    supersample = max(1, int(supersample))

    # Guarantee that reference lines narrower than the map-to-BEV scale survive.
    source_width = max(1, int(minimum_source_width))
    if source_width % 2 == 0:
        source_width += 1
    prepared = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (source_width, source_width)
        ),
    )

    scale_matrix = np.array(
        [[supersample, 0.0, 0.0], [0.0, supersample, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    high = cv2.warpPerspective(
        prepared,
        scale_matrix @ np.asarray(homography, dtype=np.float64),
        (output_w * supersample, output_h * supersample),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    high = cv2.morphologyEx(
        high,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    alpha = cv2.resize(high, (output_w, output_h), interpolation=cv2.INTER_AREA)

    # Bridge one-pixel rasterization holes. Deliberate lane-dash gaps are much
    # larger than this kernel and remain untouched.
    support = (alpha >= 18).astype(np.uint8) * 255
    support = cv2.morphologyEx(
        support,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    visibility_floor = (support.astype(np.float32) * 0.55).astype(np.uint8)
    return np.maximum(alpha, visibility_floor)


def colorize_layout_lines(alpha, color=(0, 255, 255)):
    """Create a BGR line layer whose brightness follows the anti-aliased mask."""
    weight = alpha.astype(np.float32) / 255.0
    layer = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        layer[..., channel] = np.round(weight * value).astype(np.uint8)
    return layer


def blend_layout_lines(base_bgr, alpha, opacity=0.45, color=(0, 255, 255)):
    """Blend anti-aliased layout lines over a BGR BEV image."""
    opacity = float(np.clip(opacity, 0.0, 1.0))
    weight = (alpha.astype(np.float32) / 255.0 * opacity)[..., None]
    color_image = np.empty_like(base_bgr, dtype=np.float32)
    color_image[...] = np.asarray(color, dtype=np.float32)
    result = base_bgr.astype(np.float32) * (1.0 - weight) + color_image * weight
    return np.clip(result, 0, 255).astype(np.uint8)
