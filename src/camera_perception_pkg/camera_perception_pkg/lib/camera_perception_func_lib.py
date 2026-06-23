import math

import cv2
import numpy as np


def draw_edges(detection_msg, cls_name, color=255):
    height = 480
    width = 640

    for det in detection_msg.detections:
        if getattr(det, "mask", None) and det.mask.height > 0 and det.mask.width > 0:
            height = det.mask.height
            width = det.mask.width
            break

    image = np.zeros((height, width), dtype=np.uint8)

    for det in detection_msg.detections:
        if det.class_name != cls_name or not det.mask.data:
            continue

        pts = np.array(
            [[int(round(p.x)), int(round(p.y))] for p in det.mask.data],
            dtype=np.int32,
        )
        if len(pts) < 2:
            continue
        cv2.fillPoly(image, [pts], color=color)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

    return image


def bird_convert(image, srcmat, dstmat):
    src = np.array(srcmat, dtype=np.float32)
    dst = np.array(dstmat, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, matrix, (width, height))


def roi_rectangle_below(image, cutting_idx=300):
    roi = np.zeros_like(image)
    start_idx = max(0, min(cutting_idx, image.shape[0]))
    roi[start_idx:, :] = image[start_idx:, :]
    return roi


def dominant_gradient(image, theta_limit=70):
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180.0,
        threshold=25,
        minLineLength=25,
        maxLineGap=15,
    )

    if lines is None:
        return 0.0

    angles = []
    weights = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 1e-6:
            continue

        angle_deg = math.degrees(math.atan2(dx, dy))
        if abs(angle_deg) > theta_limit:
            continue

        length = math.hypot(dx, dy)
        angles.append(angle_deg)
        weights.append(length)

    if not angles:
        return 0.0

    return float(np.average(np.asarray(angles), weights=np.asarray(weights)))


def get_lane_center(
    image,
    detection_height,
    detection_thickness=10,
    road_gradient=0.0,
    lane_width=300,
):
    height, width = image.shape[:2]
    center_x = width // 2

    row_center = height - int(round(detection_height))
    half_thickness = max(1, detection_thickness // 2)
    y0 = max(0, row_center - half_thickness)
    y1 = min(height, row_center + half_thickness + 1)

    band = image[y0:y1, :]
    if band.size == 0:
        return center_x

    xs = np.where(band > 0)[1]
    if xs.size > 0:
        left_x = int(xs.min())
        right_x = int(xs.max())
        if right_x - left_x > 20:
            return int((left_x + right_x) / 2)

        lane_half = lane_width / 2.0
        if left_x < center_x:
            return int(left_x + lane_half)
        return int(right_x - lane_half)

    offset = math.tan(math.radians(road_gradient)) * detection_height
    inferred_center = int(center_x + offset)
    return max(0, min(width - 1, inferred_center))
