#!/usr/bin/env python3
"""Zones <-> canonical template ('track_map') frame converter.

The studio's zone editor saves zones in the SENSOR frame ("frame":
"lidar_sensor"), which is re-created every time the banner is unrolled —
zones authored that way die with the installation. This tool re-expresses
them in the template frame (track2.png pixels / metres, +y up), which is
installation-invariant, and projects them back for any later installation:

    # once, after authoring zones on the current installation:
    python3 zones_to_track_frame.py to-assets \
        --zones output/config/track_zones.json \
        --homography output/config/track_map_aligned_homography.json \
        --out output/config/track_assets.yaml

    # after every re-installation (new homography, same assets):
    python3 zones_to_track_frame.py to-sensor \
        --assets output/config/track_assets.yaml \
        --homography output/config/<new>_homography.json \
        --out output/config/track_zones.json

Scale defaults to the homography's local Jacobian (BEV px are metric);
pass --m-per-px with the tape-verified value once D2 measures it.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# BEV canvas defaults — must match DetectorConfig / the alignment run when the
# homography JSON predates the bev_geometry field (the 2026-07-25 one does).
DEFAULT_GEOMETRY = {
    "resolution": 0.05,
    "forward_min": 1.3,
    "forward_max": 13.3,
    "lateral_min": -7.0,
    "lateral_max": 9.0,
}


def load_geometry(*dicts):
    """First dict that carries a bev_geometry wins; else defaults."""
    for d in dicts:
        geo = (d or {}).get("bev_geometry")
        if geo and all(k in geo for k in DEFAULT_GEOMETRY):
            return {k: float(geo[k]) for k in DEFAULT_GEOMETRY}
    return dict(DEFAULT_GEOMETRY)


class Chain:
    """sensor metres <-> BEV px <-> template px <-> canonical metres."""

    def __init__(self, homography_data, geometry, m_per_px=0.0):
        self.h_t2b = np.asarray(homography_data["homography_track_to_bev"], float)
        self.h_b2t = np.asarray(homography_data["homography_bev_to_track"], float)
        self.height_px = float(homography_data["track_image_size"]["height"])
        self.width_px = float(homography_data["track_image_size"]["width"])
        self.g = geometry
        if m_per_px > 0.0:
            self.m_per_px = float(m_per_px)
        else:
            g = geometry
            cx = (g["forward_max"] - g["forward_min"]) / (2.0 * g["resolution"])
            cy = (g["lateral_max"] - g["lateral_min"]) / (2.0 * g["resolution"])
            eps = 0.5
            p0 = self._h(self.h_b2t, cx, cy)
            J = np.column_stack([
                (np.array(self._h(self.h_b2t, cx + eps, cy)) - p0) / eps,
                (np.array(self._h(self.h_b2t, cx, cy + eps)) - p0) / eps,
            ])
            det = abs(np.linalg.det(J))
            if det <= 0.0:
                raise ValueError("degenerate homography")
            self.m_per_px = g["resolution"] / math.sqrt(det)

    @staticmethod
    def _h(H, x, y):
        v = H @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]

    def sensor_to_template_px(self, forward, lateral):
        g = self.g
        col = (forward - g["forward_min"]) / g["resolution"] - 0.5
        row = (g["lateral_max"] - lateral) / g["resolution"] - 0.5
        return self._h(self.h_b2t, col, row)

    def template_px_to_sensor(self, px, py):
        col, row = self._h(self.h_t2b, px, py)
        g = self.g
        forward = g["forward_min"] + (col + 0.5) * g["resolution"]
        lateral = g["lateral_max"] - (row + 0.5) * g["resolution"]
        return forward, lateral

    def template_px_to_map_m(self, px, py):
        return px * self.m_per_px, (self.height_px - py) * self.m_per_px

    def map_m_to_template_px(self, x, y):
        return x / self.m_per_px, self.height_px - y / self.m_per_px


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_assets(args):
    zones_data = _read_json(args.zones)
    homo = _read_json(args.homography)
    if zones_data.get("frame") not in (None, "lidar_sensor"):
        sys.exit(f"unexpected zones frame: {zones_data.get('frame')}")
    chain = Chain(homo, load_geometry(homo, zones_data), args.m_per_px)

    zones = []
    for roi in zones_data.get("rois", []):
        tpx, mm = [], []
        for wx, wy in roi.get("world") or []:
            px, py = chain.sensor_to_template_px(wx, wy)
            tpx.append([round(float(px), 1), round(float(py), 1)])
            mx, my = chain.template_px_to_map_m(px, py)
            mm.append([round(float(mx), 3), round(float(my), 3)])
        if tpx:
            zones.append({
                "name": roi["name"],
                "role": roi.get("role", []),
                "geom": roi.get("geom", "line"),
                "template_px": tpx,
                "map_m": mm,
            })

    out = {
        "frame": "track_map",
        "template_size_px": {"width": int(chain.width_px),
                             "height": int(chain.height_px)},
        "m_per_px": round(chain.m_per_px, 6),
        "m_per_px_source": "parameter" if args.m_per_px > 0 else
                           "homography-derived (tape-verify in D2)",
        "source": {"zones_json": str(args.zones),
                   "homography_json": str(args.homography)},
        "zones": zones,
    }
    _write_structured(Path(args.out), out)
    print(f"{len(zones)} zones -> {args.out} "
          f"(scale {chain.m_per_px * 1000:.3f} mm/px)")


def to_sensor(args):
    assets = _read_structured(Path(args.assets))
    homo = _read_json(args.homography)
    chain = Chain(homo, load_geometry(homo), args.m_per_px)

    rois = []
    for z in assets.get("zones", []):
        world, pixels = [], []
        # template_px is authoritative; map_m is the human-readable mirror
        for px, py in z.get("template_px") or []:
            fwd, lat = chain.template_px_to_sensor(px, py)
            world.append([round(float(fwd), 3), round(float(lat), 3)])
        if world:
            rois.append({"name": z["name"], "role": z.get("role", []),
                         "geom": z.get("geom", "line"), "world": world,
                         "pixels": pixels})
    out = {"frame": "lidar_sensor",
           "projected_from": str(args.assets),
           "homography_json": str(args.homography),
           "rois": rois}
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rois)} zones -> {args.out} (sensor frame, this installation)")


def _write_structured(path, data):
    if path.suffix in (".yaml", ".yml"):
        import yaml
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def _read_structured(path):
    if path.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return _read_json(path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    base = Path(__file__).resolve().parent / "output" / "config"

    a = sub.add_parser("to-assets", help="sensor-frame zones -> canonical assets")
    a.add_argument("--zones", default=str(base / "track_zones.json"))
    a.add_argument("--homography",
                   default=str(base / "track_map_aligned_homography.json"))
    a.add_argument("--out", default=str(base / "track_assets.yaml"))
    a.add_argument("--m-per-px", type=float, default=0.0)
    a.set_defaults(fn=to_assets)

    s = sub.add_parser("to-sensor", help="canonical assets -> this installation")
    s.add_argument("--assets", default=str(base / "track_assets.yaml"))
    s.add_argument("--homography",
                   default=str(base / "track_map_aligned_homography.json"))
    s.add_argument("--out", default=str(base / "track_zones.json"))
    s.add_argument("--m-per-px", type=float, default=0.0)
    s.set_defaults(fn=to_sensor)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
