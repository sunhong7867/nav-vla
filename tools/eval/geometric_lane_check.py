#!/usr/bin/env python3
"""Physical lane-keeping check against GEOMETRIC lane centers.

Usage: geometric_lane_check.py MAP.json [MAP2.json ...]

Unlike compare_segments.py (deviation vs the teacher-corpus median line,
which itself cuts corners at junctions by up to ~0.9 m), this measures the
ego's lateral offset from the geometric lane centerline (track_paths.json
lane1/lane2) and reports where |offset| exceeds the physical lane-touch
margin: (lane_width - car_width)/2 = (3.16 - 1.75)/2 = 0.70 m.
"""
import json, math, os, sys, statistics as st

REPO = os.path.expanduser("~/ROS2_project/nav-vla")
paths = json.load(open(f"{REPO}/src/nav_vla_pkg/config/track_paths.json"))
C = paths["ring_center"]; N = len(C)
LANES = {"lane1": paths["lane1"], "lane2": paths["lane2"]}
ZONES = {name: z["lane2"]["index"] for name, z in paths["zones"].items()
         if isinstance(z, dict) and "lane2" in z}
SEG = 20
MARGIN = 0.70


def ni(x, y):
    return min(range(N), key=lambda i: (C[i][0]-x)**2 + (C[i][1]-y)**2)


def lat(x, y, i):
    j = (i+1) % N
    tx, ty = C[j][0]-C[i][0], C[j][1]-C[i][1]
    n = math.hypot(tx, ty) or 1.0
    return (tx*(y-C[i][1]) - ty*(x-C[i][0])) / n


GEO = {ln: [lat(p[0], p[1], i) for i, p in enumerate(LANES[ln])]
       for ln in LANES}


def zone_near(i):
    return min(ZONES.items(),
               key=lambda kv: min(abs(kv[1]-i), N-abs(kv[1]-i)))[0]


def check(path):
    d = json.load(open(path))
    e = [t for t in d if "DIFFERENT" in t["case"]][0]
    print(f"\n### {os.path.basename(path)}  (margin ±{MARGIN} m = 차선 밟음)")
    for key, ln, name in (("track_a", "lane1", "안쪽"),
                          ("track_b", "lane2", "바깥")):
        per = {}
        for x, y in e[key]:
            i = ni(x, y)
            per.setdefault(i // SEG, []).append(lat(x, y, i) - GEO[ln][i])
        worst = 0.0
        bad = []
        meds = []
        for s, v in sorted(per.items()):
            if len(v) < 5:
                continue
            m = st.median(v)
            meds.append(abs(m))
            w = max(v, key=abs)
            if abs(w) > abs(worst):
                worst = w
            if abs(m) > MARGIN:
                bad.append((s, zone_near(s*SEG + SEG//2), m))
        print(f"  {name}: 구간중앙값 최대 {max(meds):.2f}  순간 최대 {worst:+.2f}"
              f"  밟은 구간 {len(bad)}곳"
              + ("".join(f"\n     seg{s:2d} ({z}) {m:+.2f}" for s, z, m in bad)))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
