#!/usr/bin/env python3
"""Extract driveable centrelines from the track texture into world coordinates.

The oracle has to steer for itself — that is the whole point of replacing the
lane follower, which steers identically no matter which goal was asked for. To
steer it needs a path, and hand-authoring one around a 150 m ring is both tedious
and a source of silent error. The texture already contains the geometry; this
reads it out.

Two things make it reliable rather than a guess:

* **The image-to-world map is verified, not assumed.** ``track.world`` puts the
  ground plane at yaw 1.57, so the plane's local x maps to world +y and local y
  to world -x, over 58.1364 x 43.71084 m. Projecting all 15 zones through it
  lands the 8 ring zones on dark asphalt and the 7 parking-area zones on grey,
  15/15. (Note the size in ``models/race_track/model.sdf`` is *not* the one in
  effect; the world file carries its own.)
* **The centreline comes from the two ring boundaries, not from thinning.**
  Neither ``cv2.ximgproc`` nor ``skimage`` is installed. A distance-transform
  ridge was tried first and failed badly — with a 5x5 maximum filter the ridge
  is a noisy plateau rather than a one-pixel line, and a greedy walk over it
  connected 33 of 3084 points. The road is an annulus, so it has exactly one
  outer contour and one hole; pairing each outer point with its nearest inner
  point and taking the midpoint is both simpler and robust to the S-curve at the
  bottom, where a radial parameterisation would cross the band twice.

Output ``config/track_paths.json``::

    {"ring_center": [[x, y], ...],   # closed, arc-length ordered, CCW
     "lane1": [...], "lane2": [...], # ring centre offset by +-lane_width/4
     "meta": {...}}

Usage::

    python3 src/nav_vla_pkg/scripts/extract_track_paths.py
    python3 src/nav_vla_pkg/scripts/extract_track_paths.py --spacing 0.35 --preview
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

REPO = os.path.expanduser("~/ROS2_project/nav-vla")
TEXTURE = os.path.join(
    REPO, "src/simulation_pkg/models/race_track/materials/textures/track.png")
OUT = os.path.join(REPO, "src/nav_vla_pkg/config/track_paths.json")

# From track.world: <model name="ground"> pose yaw=1.57, plane size 58.1364 x 43.71084
PLANE_LOCAL_X = 58.1364      # -> world +y
PLANE_LOCAL_Y = 43.71084     # -> world -x

# Kinematic limit of the simulated vehicle: R_min = wheelbase / tan(max_steer).
# ackermann_cmd_adapter_node.py:23 gives L = 2.86 m and policy_node.py:59 caps
# steer at 0.6 rad, so nothing tighter than 4.18 m can be driven. A generated
# path that violates this looks fine on a plot and then the car simply cannot
# follow it, which shows up as a mysterious tracking error rather than as the
# geometry error it is.
SIM_WHEEL_BASE = 2.86
SIM_MAX_STEER = 0.6
MIN_TURN_RADIUS = SIM_WHEEL_BASE / math.tan(SIM_MAX_STEER)
# Two different bars. DRIVABLE is the physics: below R_min the car simply cannot
# follow the path. COMFORT is a preference — the curvature clamp engages between
# the two, which costs tracking accuracy but is not a failure. Reporting one
# number conflated them and marked a perfectly drivable 4.53 m connector as
# infeasible.
RADIUS_MARGIN = 1.25         # comfort target for paths we design freely
DRIVABLE_MARGIN = 1.05       # hard floor: below this, reject


class Frame:
    def __init__(self, w, h):
        self.w, self.h = w, h

    def world_to_uv(self, x, y):
        return ((y + PLANE_LOCAL_X / 2) / PLANE_LOCAL_X * self.w,
                (PLANE_LOCAL_Y / 2 + x) / PLANE_LOCAL_Y * self.h)

    def uv_to_world(self, u, v):
        return (v / self.h * PLANE_LOCAL_Y - PLANE_LOCAL_Y / 2,
                u / self.w * PLANE_LOCAL_X - PLANE_LOCAL_X / 2)

    @property
    def m_per_px(self):
        return 0.5 * (PLANE_LOCAL_X / self.w + PLANE_LOCAL_Y / self.h)


def road_mask(rgb):
    """Dark asphalt, with the dashed lane markings filled back in.

    The dashes are white, so a naive threshold punches holes through the middle
    of the road and the ridge would wander around them. Closing with a kernel
    wider than a dash restores a solid ring.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    dark = ((hsv[..., 2] < 90) & (hsv[..., 1] < 90)).astype(np.uint8)
    k = np.ones((41, 41), np.uint8)
    return cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)


def largest_component(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == keep).astype(np.uint8)


def annulus_centreline(mask):
    """Midpoints between the ring's outer contour and its hole.

    Returns points already ordered, because the outer contour is ordered and each
    midpoint inherits that order.
    """
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None or len(contours) < 2:
        raise RuntimeError(f"expected an annulus, found {len(contours)} contour(s)")
    hier = hier[0]
    outer_i = max((i for i in range(len(contours)) if hier[i][3] < 0),
                  key=lambda i: cv2.contourArea(contours[i]))
    holes = [i for i in range(len(contours)) if hier[i][3] == outer_i]
    if not holes:
        raise RuntimeError("road mask has no hole — the ring is filled in")
    inner_i = max(holes, key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_i].reshape(-1, 2).astype(np.float64)
    inner = contours[inner_i].reshape(-1, 2).astype(np.float64)
    # Nearest inner point for each outer point, in chunks to bound memory.
    mids = np.empty_like(outer)
    for a in range(0, len(outer), 2048):
        blk = outer[a:a + 2048]
        d2 = ((blk[:, None, 0] - inner[None, :, 0]) ** 2 +
              (blk[:, None, 1] - inner[None, :, 1]) ** 2)
        mids[a:a + 2048] = 0.5 * (blk + inner[np.argmin(d2, axis=1)])
    return mids, outer, inner


def resample_closed(poly, spacing):
    """Uniform arc-length resampling of a closed polyline."""
    p = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    targets = np.arange(0.0, total, spacing)
    out = []
    for t in targets:
        i = int(np.searchsorted(s, t) - 1)
        i = max(0, min(i, len(seg) - 1))
        a = (t - s[i]) / max(seg[i], 1e-9)
        out.append(p[i] + a * (p[i + 1] - p[i]))
    return np.asarray(out), total


def smooth_closed(poly, win):
    if win < 3:
        return poly
    k = np.ones(win) / win
    x = np.convolve(np.r_[poly[-win:, 0], poly[:, 0], poly[:win, 0]], k, "same")[win:-win]
    y = np.convolve(np.r_[poly[-win:, 1], poly[:, 1], poly[:win, 1]], k, "same")[win:-win]
    return np.column_stack([x, y])


def offset_closed(poly, dist):
    """Offset a closed polyline along its left normal by `dist` metres."""
    nxt = np.roll(poly, -1, axis=0)
    prv = np.roll(poly, 1, axis=0)
    tan = nxt - prv
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    return poly + nrm * dist


# --- parking approach paths ------------------------------------------------
# Measured from the texture: group A has dividers at y = -11.17 .. +3.37 every
# 3.63 m, spanning x 0.81 -> 6.47, with a back edge at x ~ 6.4 and the x ~ 0.81
# side open. So a car enters driving +x. Group B is four fully closed rectangles
# and cannot be driven into, so it is not used.
BAY_DIVIDERS_Y = [-11.17, -7.54, -3.90, -0.28, 3.37]
BAY_OPEN_X = 0.81
BAY_BACK_X = 6.47

# Vehicle extent, read off prius_hybrid/meshes/hybrid_body.obj at its 0.01 scale.
# The model's forward direction is local -y, and the origin sits between the axles
# (front wheels at -1.41, rear at +1.45), so the nose reaches 2.269 m ahead of the
# pose the simulator reports and the tail 2.349 m behind it.
#
# This matters: the stop point was first set to BAY_BACK_X - 1.2 = 5.27, which
# reads as "1.2 m clear of the back wall" only if the car were a point. With the
# real 4.618 m body the nose lands at 7.54 m, overhanging the bay by 1.07 m —
# which is exactly what was observed in the simulator. The chassis *collision*
# box is only 2.1 m long, so nothing collides and nothing complains.
CAR_NOSE_AHEAD = 2.269
CAR_TAIL_BEHIND = 2.349
# Aisle position is DERIVED from the turn the car has to make, not chosen.
#
# Entering a bay is a 90 deg turn from +y to +x, and a quarter circle of radius r
# displaces r in both axes, so the turn ends at x = AISLE_X + r. For it to finish
# at the bay mouth rather than inside the bay, AISLE_X = BAY_OPEN_X - r.
#
# The first version picked AISLE_X = -2.50 by eye and let r fall out as
# (0.81 + 2.50) * 0.8 = 2.65 m. That is well inside the vehicle's 4.18 m minimum
# radius: the car cannot make the turn, under-steers through it and overshoots in
# +y, ending against the right-hand divider of every bay. The ring connector was
# checked for this and the bay turn-in was not.
#
# The aisle entry sits as far back as the parking area allows (its y extent starts
# at -19.63), because the shared approach length is what the ordinal axis is made
# of and a feasible radius eats into it.
TURN_RADIUS = MIN_TURN_RADIUS * RADIUS_MARGIN     # 5.23 m
AISLE_X = BAY_OPEN_X - TURN_RADIUS                # -4.42 m
# Kept where the ring connector can still reach it: pulling the entry back to
# -19.40 for a longer aisle squeezed the connector's own turn to 3.0 m and made
# it infeasible in turn. The shared prefix does not have to come from the aisle
# alone — the connector is common to all four bays too, so an episode started on
# the ring shares connector + aisle.
AISLE_ENTRY_Y = -17.60
AISLE_EXIT_Y = 6.50


def bay_centres():
    return [(0.5 * (BAY_BACK_X + BAY_OPEN_X), 0.5 * (a + b))
            for a, b in zip(BAY_DIVIDERS_Y, BAY_DIVIDERS_Y[1:])]


def parking_paths(spacing):
    """One path per bay: shared aisle, then a turn-in.

    The aisle segment is byte-identical across all four bays. That is the whole
    point of the ordinal axis — approaching bay 2 and approaching bay 4 look the
    same to a camera until the turn, so the only way to know which one was meant
    is the instruction. An axis whose answer is visible in the image proves
    nothing about language.
    """
    out = {}
    for k, (bx, by) in enumerate(bay_centres(), start=1):
        pts = []
        r = TURN_RADIUS
        if r < MIN_TURN_RADIUS:
            raise RuntimeError(
                f"bay turn radius {r:.2f} m is below the vehicle minimum "
                f"{MIN_TURN_RADIUS:.2f} m — the car cannot follow this path")
        turn_start_y = by - r
        # 1. aisle: stepped at a fixed pitch from a fixed origin, NOT interpolated
        # between endpoints. Interpolating gives each bay its own step size, so no
        # two approaches share a single sample and the "identical until the turn"
        # property — the entire basis of the ordinal axis — silently disappears.
        y = AISLE_ENTRY_Y
        while y < turn_start_y - 1e-9:
            pts.append((AISLE_X, y))
            y += spacing
        # 2. quarter-turn from heading +y to heading +x. Turning right, so the
        # centre is r to the RIGHT of travel, i.e. at x = AISLE_X + r — not on the
        # aisle itself. Putting it on the aisle (the first attempt) left the arc
        # starting r metres away from where the straight ended, and the resulting
        # kink measured 2.64 m of radius on a path meant to hold 5.23 m.
        cx, cy = AISLE_X + r, turn_start_y
        steps = max(3, int((math.pi / 2 * r) / spacing))
        for i in range(1, steps + 1):
            t = (math.pi / 2) * i / steps
            pts.append((cx - r * math.cos(t), cy + r * math.sin(t)))
        # 3. straight into the bay, stopping with the whole body inside it.
        # Feasible origin range is [OPEN + tail, BACK - nose] = [3.16, 4.20];
        # the bay centre 3.64 sits inside it with ~0.5 m clear at each end, and
        # coincides with the Slot1-4 poses already in zone_map.yaml.
        stop_x = 0.5 * (BAY_OPEN_X + BAY_BACK_X)
        lo, hi = BAY_OPEN_X + CAR_TAIL_BEHIND, BAY_BACK_X - CAR_NOSE_AHEAD
        if not (lo <= stop_x <= hi):
            raise RuntimeError(
                f"car does not fit: origin must be in [{lo:.2f}, {hi:.2f}] "
                f"but stop is {stop_x:.2f} (bay {BAY_BACK_X - BAY_OPEN_X:.2f} m deep, "
                f"car {CAR_NOSE_AHEAD + CAR_TAIL_BEHIND:.2f} m long)")
        x0 = pts[-1][0]
        n = max(2, int(abs(stop_x - x0) / spacing))
        for i in range(1, n + 1):
            pts.append((x0 + (stop_x - x0) * i / n, by))
        out[f"bay{k}"] = {
            "nose_x": stop_x + CAR_NOSE_AHEAD,
            "tail_x": stop_x - CAR_TAIL_BEHIND,
            "clear_front_m": BAY_BACK_X - (stop_x + CAR_NOSE_AHEAD),
            "clear_rear_m": (stop_x - CAR_TAIL_BEHIND) - BAY_OPEN_X,
            "path": pts,
            "centre": [bx, by],
            "stop": [stop_x, by],
            "aisle_points": None,
        }
    # how many leading points are common to every bay path
    keys = list(out)
    common = 0
    while True:
        vals = {tuple(np.round(out[k]["path"][common], 3)) for k in keys}
        if len(vals) != 1:
            break
        common += 1
    for k in keys:
        out[k]["aisle_points"] = common
    return out


def hermite(p0, m0, p1, m1, n):
    """Cubic Hermite from p0 (tangent m0) to p1 (tangent m1)."""
    out = []
    for k in range(n + 1):
        t = k / n
        t2, t3 = t * t, t * t * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        out.append((h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0],
                    h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1]))
    return out


def max_curvature(pts):
    """Menger curvature over consecutive triples, in 1/m."""
    worst = 0.0
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        ab = math.hypot(b[0] - a[0], b[1] - a[1])
        bc = math.hypot(c[0] - b[0], c[1] - b[1])
        ca = math.hypot(a[0] - c[0], a[1] - c[1])
        if min(ab, bc, ca) < 1e-6:
            continue
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        worst = max(worst, 2.0 * area2 / (ab * bc * ca))
    return worst


def ring_to_aisle(lane_poly, spacing, back_m=14.0, tangent_scale=1.0):
    """Connector from the ring lane into the aisle entrance.

    A plain quarter circle does not fit: the ring runs along +x where the aisle
    starts and the aisle runs along +y, but the two are only 3.8 m apart in y, so
    a 90 deg arc large enough to make the turn overshoots in x by ~3.9 m. A cubic
    Hermite lets the entry point sit further back along the ring and absorbs the
    mismatch in the tangent magnitudes instead.
    """
    P = np.asarray(lane_poly)
    d = np.hypot(P[:, 0] - AISLE_X, P[:, 1] - AISLE_ENTRY_Y)
    near = int(np.argmin(d))
    start = (near - int(back_m / spacing)) % len(P)
    p0 = tuple(P[start])
    nxt, prv = P[(start + 1) % len(P)], P[(start - 1) % len(P)]
    h0 = math.atan2(nxt[1] - prv[1], nxt[0] - prv[0])
    p1 = (AISLE_X, AISLE_ENTRY_Y)
    span = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    m = span * tangent_scale
    curve = hermite(p0, (m * math.cos(h0), m * math.sin(h0)),
                    p1, (0.0, m), max(8, int(span / spacing) * 2))
    # resample to the common pitch
    out, acc = [curve[0]], 0.0
    for a, b in zip(curve, curve[1:]):
        acc += math.hypot(b[0] - a[0], b[1] - a[1])
        if acc >= spacing:
            out.append(b)
            acc = 0.0
    return out, start, math.degrees(h0)


def signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texture", default=TEXTURE)
    p.add_argument("--out", default=OUT)
    p.add_argument("--spacing", type=float, default=0.35, help="metres between path points")
    p.add_argument("--smooth-win", type=int, default=9)
    p.add_argument("--min-half-width-m", type=float, default=1.2)
    p.add_argument("--preview", default="", help="write an overlay PNG here")
    args = p.parse_args()

    rgb = cv2.cvtColor(cv2.imread(args.texture, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    fr = Frame(w, h)
    print(f"texture {w}x{h}   {fr.m_per_px * 100:.2f} cm/px")

    mask = largest_component(road_mask(rgb))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    road_half_px = float(np.percentile(dist[dist > 0], 99))
    print(f"road area {mask.mean() * 100:.1f}%   half-width p99 "
          f"{road_half_px * fr.m_per_px:.2f} m -> full width "
          f"{2 * road_half_px * fr.m_per_px:.2f} m")

    mids, outer, inner = annulus_centreline(mask)
    print(f"outer contour {len(outer)} px, hole {len(inner)} px -> {len(mids)} midpoints")

    world = np.array([fr.uv_to_world(u, v) for u, v in mids])
    world, total = resample_closed(world, args.spacing)
    world = smooth_closed(world, args.smooth_win)
    if signed_area(world) < 0:            # make it counter-clockwise
        world = world[::-1]
    print(f"ring centreline: {len(world)} points, {total:.1f} m around")

    lane_half = road_half_px * fr.m_per_px / 2.0     # quarter of the full road width
    lane1 = offset_closed(world, +lane_half)
    lane2 = offset_closed(world, -lane_half)

    # Project every zone onto each lane so the oracle can express "drive to T2"
    # as an arc-length target instead of a proximity test. A zone that lands far
    # from the ring is a parking-area zone and is reported, not silently snapped.
    import yaml
    zone_path = os.path.join(REPO, "src/nav_vla_pkg/config/zone_map.yaml")
    zones = {}
    if os.path.exists(zone_path):
        with open(zone_path, "r", encoding="utf-8") as zf:
            raw = (yaml.safe_load(zf) or {}).get("zones", {})
        for zn, zv in raw.items():
            pz = zv.get("pose", {})
            zx, zy = float(pz.get("x", 0.0)), float(pz.get("y", 0.0))
            ent = {"pose": [zx, zy, float(pz.get("yaw", 0.0))]}
            for lname, lpoly in (("lane1", lane1), ("lane2", lane2)):
                d = np.hypot(lpoly[:, 0] - zx, lpoly[:, 1] - zy)
                i = int(np.argmin(d))
                ent[lname] = {"index": i, "s_m": i * args.spacing,
                              "dist_m": float(d[i])}
            ent["on_ring"] = ent["lane1"]["dist_m"] < 4.0
            zones[zn] = ent
        ring = sum(1 for z in zones.values() if z["on_ring"])
        print(f"zones projected: {ring} on the ring, {len(zones) - ring} off it")
        for zn, zv in zones.items():
            tag = "ring" if zv["on_ring"] else "OFF-RING"
            print(f"  {zn:22s} s={zv['lane1']['s_m']:6.1f} m  "
                  f"dist={zv['lane1']['dist_m']:5.2f} m  {tag}")

    want_radius = MIN_TURN_RADIUS * DRIVABLE_MARGIN
    comfort = MIN_TURN_RADIUS * RADIUS_MARGIN
    print(f"vehicle min turn radius {MIN_TURN_RADIUS:.2f} m "
          f"(L={SIM_WHEEL_BASE}, steer<={SIM_MAX_STEER} rad); "
          f"drivable >= {want_radius:.2f} m, comfortable >= {comfort:.2f} m")
    connectors = {}
    for lname, lpoly in (("lane1", lane1), ("lane2", lane2)):
        best = None
        for ts in np.arange(0.3, 2.01, 0.1):
            for back in np.arange(8.0, 46.0, 2.0):
                c, si, h0 = ring_to_aisle(lpoly, args.spacing, float(back), float(ts))
                k = max_curvature(c)
                r = 1.0 / max(k, 1e-9)
                # Prefer the shortest path that clears the radius requirement;
                # only if none does, fall back to the roundest available.
                feasible = r >= want_radius
                score = (0 if feasible else 1, len(c) if feasible else -r)
                if best is None or score < best[0]:
                    best = (score, k, r, c, si, h0, float(back), float(ts), feasible)
        _, k, r, c, si, h0, back, ts, feasible = best
        connectors[lname] = {"path": [list(p) for p in c], "ring_index": si,
                             "ring_heading_deg": h0, "back_m": back,
                             "tangent_scale": ts, "max_curvature": k,
                             "min_radius_m": r, "feasible": bool(feasible)}
        flag = ("OK" if r >= comfort else
                "DRIVABLE (clamp will engage)" if feasible else "INFEASIBLE")
        print(f"connector {lname}: {len(c)} pts from ring idx {si} "
              f"({back:.0f} m back, tangent {ts:.1f}), min radius {r:.2f} m "
              f"= {r / MIN_TURN_RADIUS:.2f}x limit  [{flag}]")
        if not feasible:
            print(f"    the vehicle cannot follow this — needs >= {want_radius:.2f} m")

    parking = parking_paths(args.spacing)
    shared = next(iter(parking.values()))["aisle_points"]
    print(f"parking: {len(parking)} bays, first {shared} points "
          f"({shared * args.spacing:.1f} m) identical across all of them")
    for k, v in parking.items():
        print(f"  {k}: {len(v['path'])} pts, stop at "
              f"({v['stop'][0]:+.2f}, {v['stop'][1]:+.2f})")

    data = {
        "ring_to_aisle": connectors,
        "parking": parking,
        "zones": zones,
        "ring_center": world.tolist(),
        "lane1": lane1.tolist(),
        "lane2": lane2.tolist(),
        "meta": {
            "texture": os.path.relpath(args.texture, REPO),
            "spacing_m": args.spacing,
            "loop_length_m": total,
            "road_full_width_m": 2 * road_half_px * fr.m_per_px,
            "lane_offset_m": lane_half,
            "plane_local_x_m": PLANE_LOCAL_X,
            "plane_local_y_m": PLANE_LOCAL_Y,
            "note": "lane1 = left of centre in the CCW direction, lane2 = right",
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"wrote {args.out}")

    if args.preview:
        ov = rgb.copy()
        for poly, col in ((world, (255, 0, 0)), (lane1, (0, 200, 255)), (lane2, (255, 200, 0))):
            uv = np.array([fr.world_to_uv(x, y) for x, y in poly], np.int32)
            cv2.polylines(ov, [uv], True, col, 6)
        cv2.imwrite(args.preview, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        print(f"wrote {args.preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
