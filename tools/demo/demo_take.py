#!/usr/bin/env python3
"""r6 watch-then-avoid live check: does the POLICY slow, hold, pass, narrate?"""
import json, math, sys, time
sys.path.insert(0, "/home/sh/ROS2_project/nav-vla/src/simulation_pkg")
sys.path.insert(0, "/home/sh/ROS2_project/nav-vla/src/nav_vla_pkg")
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
                       QoSDurabilityPolicy)
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Twist
from nav_vla_pkg.gz_reset import SimResetter
from simulation_pkg import basic

TP = json.load(open("/home/sh/ROS2_project/nav-vla/src/nav_vla_pkg/config/track_paths.json"))
LANE = "lane2"
PTS = TP[LANE]; N = len(PTS)
START_I, OB_I = 200, 275          # ~26 m ahead along lane2
SENT = "Cruise in the outer lane, at a normal speed."

def pose_at(i):
    x, y = PTS[i]; nx, ny = PTS[(i+1) % N]
    return x, y, math.atan2(ny-y, nx-x)

sx, sy, syaw = pose_at(START_I)
ox, oy, oyaw = pose_at(OB_I)
basic.load_model("v9_demo_ob", "hatchback_red", (ox, oy, 0.01265, 0, 0, oyaw),
                 skip_if_exists=True)
REG = "/home/sh/ROS2_project/nav-vla/eval_out/demo/obstacles.json"
import os as _os; _os.makedirs(_os.path.dirname(REG), exist_ok=True)
open(REG, "w").write(json.dumps([{"entity": "v9_demo_ob",
    "model": "hatchback_red", "x": ox, "y": oy, "lane": LANE}]))
print(f"obstacle at ({ox:.1f},{oy:.1f}) start ({sx:.1f},{sy:.1f})")

rclpy.init()
n = Node("r6_demo")
latched = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                     history=QoSHistoryPolicy.KEEP_LAST,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, depth=1)
instr = n.create_publisher(String, "/vla/instruction", latched)
cmd = n.create_publisher(Twist, "/cmd_vel", 10)
qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                 history=QoSHistoryPolicy.KEEP_LAST,
                 durability=QoSDurabilityPolicy.VOLATILE, depth=5)
state = {"tf": None, "reason": []}
def tf_cb(m):
    best = None
    for t in m.transforms:
        x, y = t.transform.translation.x, t.transform.translation.y
        d = math.hypot(x - (state["tf"][0] if state["tf"] else sx),
                       y - (state["tf"][1] if state["tf"] else sy))
        if best is None or d < best[0]:
            best = (d, x, y)
    if best and best[0] < 3.0:
        state["tf"] = (best[1], best[2])
n.create_subscription(TFMessage, "/world/default/dynamic_pose/info", tf_cb, qos)
n.create_subscription(String, "/vla/reasoning",
    lambda m: state["reason"].append(json.loads(m.data).get("text","")), 10)

def spin(s):
    end = time.monotonic() + s
    while time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.05)

instr.publish(String(data=""))
spin(0.5)
ok, msg, _ = SimResetter().reset(sx, sy, syaw + math.pi/2, cmd_pub=cmd)
print("reset:", ok, msg)
spin(2.0)
t0 = time.monotonic()
while time.monotonic() - t0 < 15 and instr.get_subscription_count() < 2:
    rclpy.spin_once(n, timeout_sec=0.1)
instr.publish(String(data=SENT))
print("instruction sent to", instr.get_subscription_count(), "subs")

track = []
t0 = time.monotonic()
while time.monotonic() - t0 < 95:
    rclpy.spin_once(n, timeout_sec=0.05)
    if state["tf"]:
        track.append((time.monotonic() - t0,) + state["tf"])
    time.sleep(0.03)
instr.publish(String(data=""))
spin(0.5)

# ---- verdict (r10: body-extent aware — the r9 reversal) ----------------
d_ob = [(t, math.hypot(x-ox, y-oy)) for t, x, y in track]
min_d = min(d[1] for d in d_ob)
# head-on contact check: min center distance while ego is on the
# obstacle's lane AND the obstacle is ahead (combined half-lengths 4.6 m)
def lane_of_pt(x, y):
    d1 = min((px-x)**2+(py-y)**2 for px, py in TP["lane1"][::3])
    d2 = min((px-x)**2+(py-y)**2 for px, py in TP["lane2"][::3])
    return "lane1" if d1 < d2 else "lane2"
headon = 99.0
for i in range(1, len(track)):
    _, x, y = track[i]
    px, py = track[i-1][1], track[i-1][2]
    hx, hy = x-px, y-py
    if lane_of_pt(x, y) == LANE and (ox-x)*hx+(oy-y)*hy >= 0:
        headon = min(headon, math.hypot(x-ox, y-oy))
# standstill: windows where car moved <5 cm over 1 s
stops = 0.0
for i in range(1, len(track)):
    dt = track[i][0] - track[i-1][0]
    dd = math.hypot(track[i][1]-track[i-1][1], track[i][2]-track[i-1][2])
    if dd/max(dt,1e-3) < 0.05:
        stops += dt
def lane_of(x, y):
    d1 = min((px-x)**2+(py-y)**2 for px, py in TP["lane1"][::3])
    d2 = min((px-x)**2+(py-y)**2 for px, py in TP["lane2"][::3])
    return "lane1" if d1 < d2 else "lane2"
lanes = [lane_of(x, y) for _, x, y in track[::20]]
passed = any(math.hypot(x-ox, y-oy) < 8 and
             ((x-ox)*math.cos(oyaw)+(y-oy)*math.sin(oyaw)) > 2 for _, x, y in track)
print(f"\nmin center dist (any): {min_d:.2f} m")
print(f"min HEAD-ON same-lane:  {headon:.2f} m  (CONTACT if < 4.8)")
watch_lines = [r for r in state["reason"] if "watch" in r.lower() or "stopped" in r.lower()]
print(f"watching narration lines: {len(watch_lines)}")
print("VERDICT:", "PASS" if (headon >= 4.8 and passed and stops >= 2.0) else "CHECK",
      f"(headon>=4.8:{headon>=4.8} passed:{passed} stood>=2s:{stops>=2.0})")
print(f"standstill total: {stops:.1f} s")
print(f"lane trace: {' '.join(l[-1] for l in lanes)}")
print(f"passed obstacle: {passed}")
final_lane = lane_of(track[-1][1], track[-1][2]) if track else "?"
print(f"final lane: {final_lane}  (returned: {final_lane == LANE})")
print(f"drove: {sum(math.hypot(track[i][1]-track[i-1][1], track[i][2]-track[i-1][2]) for i in range(1,len(track))):.1f} m")
print("\nreasoning lines:")
for r in state["reason"]:
    print("  ", r[:110])
