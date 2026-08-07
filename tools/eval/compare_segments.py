#!/usr/bin/env python3
"""Segment-wise lane-centering comparison of ring-map probes.

Usage:
    compare_segments.py A_map.json TAG_A [B_map.json TAG_B]

Reference line = per-ring-index median of the teacher corpus
(src/nav_vla_pkg/data_v3y/packed_v3y). Deviation sign: + = outward.
With two maps, prints A -> B per segment plus departure check
(any point > 5 m from ring centre = left the track).
"""
import json, math, glob, os, sys, statistics as st

REPO = os.path.expanduser("~/ROS2_project/nav-vla")
paths = json.load(open(f"{REPO}/src/nav_vla_pkg/config/track_paths.json"))
C = paths["ring_center"]; N = len(C)
ZONES = {name: z["lane2"]["index"] for name, z in paths["zones"].items()
         if isinstance(z, dict) and "lane2" in z}
SEG = 20

def ni(x, y):
    return min(range(N), key=lambda i: (C[i][0]-x)**2 + (C[i][1]-y)**2)

def lat(x, y, i):
    j = (i+1) % N
    tx, ty = C[j][0]-C[i][0], C[j][1]-C[i][1]
    n = math.hypot(tx, ty) or 1.0
    return (tx*(y-C[i][1]) - ty*(x-C[i][0])) / n

def build_ref():
    bins = {"lane1": [[] for _ in range(N)], "lane2": [[] for _ in range(N)]}
    for d in glob.glob(f"{REPO}/src/nav_vla_pkg/data_v3y/packed_v3y/*"):
        if not os.path.isdir(d):
            continue
        try:
            meta = json.load(open(d+"/meta.json"))
            lane = (meta.get("intent_slots") or {}).get("lane")
            if lane not in bins:
                continue
            for l in open(d+"/resampled_10hz.jsonl"):
                r = json.loads(l)
                i = ni(r["x"], r["y"])
                bins[lane][i].append(lat(r["x"], r["y"], i))
        except FileNotFoundError:
            pass
    ref = {}
    for lane in bins:
        m = [st.median(v) if v else None for v in bins[lane]]
        for i in range(N):
            if m[i] is None:
                for k in range(1, N):
                    g = [x for x in (m[(i-k) % N], m[(i+k) % N]) if x is not None]
                    if g:
                        m[i] = sum(g)/len(g); break
        ref[lane] = m
    return ref

REF = build_ref()

def zone_near(i):
    best = min(ZONES.items(), key=lambda kv: min(abs(kv[1]-i), N-abs(kv[1]-i)))
    return best[0]

def seg_map(track, lane):
    per = {}
    for x, y in track:
        i = ni(x, y)
        if REF[lane][i] is not None:
            per.setdefault(i // SEG, []).append(lat(x, y, i) - REF[lane][i])
    return {s: st.median(v) for s, v in per.items() if len(v) >= 5}

def dring(x, y):
    return min(math.hypot(C[i][0]-x, C[i][1]-y) for i in range(0, N, 2))

def load(path):
    d = json.load(open(path))
    diff = [t for t in d if "DIFFERENT" in t["case"]][0]
    out = {"lane1": seg_map(diff["track_a"], "lane1"),
           "lane2": seg_map(diff["track_b"], "lane2")}
    worst = {}
    for t in d:
        for k in ("track_a", "track_b"):
            tr = t.get(k)
            if tr:
                worst[(t["case"][:9], k)] = max(dring(x, y) for x, y in tr)
    return out, worst

def summary(m):
    v = [abs(x) for x in m.values()]
    return f"median {st.median(v):.2f} max {max(v):.2f} over0.5 {sum(1 for x in v if x > 0.5)}"

A, wA = load(sys.argv[1]); tagA = sys.argv[2]
B = wB = tagB = None
if len(sys.argv) > 4:
    B, wB = load(sys.argv[3]); tagB = sys.argv[4]

for lane in ("lane1", "lane2"):
    print(f"\n===== {lane} ({'inner' if lane=='lane1' else 'outer'}) "
          f"— signed dev (+=outward), {tagA}" + (f" -> {tagB}" if B else ""))
    segs = sorted(set(A[lane]) | (set(B[lane]) if B else set()))
    for s in segs:
        a = A[lane].get(s)
        line = f"  seg {s:3d} ({zone_near(s*SEG):14s}) " \
               f"{'  --  ' if a is None else f'{a:+6.2f}'}"
        if B:
            b = B[lane].get(s)
            line += f" -> {'  --  ' if b is None else f'{b:+6.2f}'}"
            if a is not None and b is not None:
                if abs(b) < abs(a) - 0.1: line += "  (improved)"
                elif abs(b) > abs(a) + 0.1: line += "  (WORSE)"
        print(line)
    print(f"  {tagA}: {summary(A[lane])}")
    if B:
        print(f"  {tagB}: {summary(B[lane])}")

print(f"\n=== departures (max dist from ring centre; >5 m = left track)")
for tag, w in ((tagA, wA),) + (((tagB, wB),) if B else ()):
    bad = sum(1 for v in w.values() if v > 5)
    print(f"  {tag}: worst {max(w.values()):.2f} m, departures {bad}/{len(w)}")
