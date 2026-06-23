import math


def calculate_slope_between_points(point1, point2):
    """Return the path heading angle in degrees, using image y as forward."""
    p1_x, p1_y = point1
    p2_x, p2_y = point2

    dx = float(p1_x) - float(p2_x)
    dy = float(p2_y) - float(p1_y)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    return math.degrees(math.atan2(dx, dy))
