"""Deterministic vehicle reset in Gazebo, for counterfactual-pair collection.

The whole counterfactual strategy rests on one operation: put the car back at a
*bit-identical* starting state, then issue a different instruction. If the reset
scatters, every downstream divergence number is measuring the reset, not the
language. GATE G1 exists to measure exactly that — see
``scripts/measure_reset_determinism.py``.

Sequence (order matters, each step fixes a specific failure):

1. publish a zero ``Twist`` — ``set_pose`` moves the body but does **not** zero
   the controller's last command, so the car drives off the instant physics resumes
2. pause physics
3. ``/world/<world>/set_pose``
4. settle: hold zero command for a few wall-clock ms while still paused
5. unpause
6. one direct ``query_world_pose`` — ``WorldPoseStream`` keeps serving its last
   parsed value and would hand back the *pre-teleport* pose
7. wait for N genuinely new camera frames before anyone records

Do **not** use ``/world/<world>/control`` with ``reset: {all: true}`` — it destroys
the dynamically spawned ``ego_vehicle``. Do not use ``load_ego_car_node`` /
``basic.reset_model`` either: they delete and respawn the model, cost seconds per
call, and only support one hard-coded pose.

Residual body velocity is a known open question: ``set_pose`` writes the pose
component, and gz-sim's velocity components are separate. Whether they survive a
paused teleport is what G1 measures. If scatter is dominated by it, the fix is a
longer ``settle_ms`` or an explicit velocity reset — do not guess, measure first.

CLI::

    ros2 run nav_vla_pkg gz_reset --x 3.7 --y 24.5943 --yaw -1.5707
    ros2 run nav_vla_pkg gz_reset --zone Start
"""

import argparse
import math
import subprocess
import sys
import time

from nav_vla_pkg.gz_pose import query_world_pose, resolve_gz_bin

DEFAULT_WORLD = "default"
DEFAULT_MODEL = "ego_vehicle"
DEFAULT_Z = 0.1


def yaw_to_quat_zw(yaw):
    """Yaw about +z -> (z, w). x and y are always 0 for a planar rotation."""
    return math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def _gz_service(gz_bin, service, reqtype, req, timeout_ms=3000, timeout_s=5.0):
    cmd = [
        gz_bin, "service", "-s", service,
        "--reqtype", reqtype,
        "--reptype", "gz.msgs.Boolean",
        "--timeout", str(timeout_ms),
        "--req", req,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        return False, f"gz binary not found: {gz_bin}"
    except subprocess.TimeoutExpired:
        return False, f"gz service timed out: {service} (is the sim running?)"
    if out.returncode != 0:
        return False, (out.stderr or out.stdout or "gz service failed").strip()
    # gz prints "data: true" on success; a false reply means the request was
    # accepted but refused (wrong model name is the usual cause).
    if "false" in out.stdout.lower():
        return False, f"service returned false: {out.stdout.strip()}"
    return True, out.stdout.strip()


def pause_sim(gz_bin, pause, world=DEFAULT_WORLD):
    return _gz_service(
        gz_bin,
        f"/world/{world}/control",
        "gz.msgs.WorldControl",
        f"pause: {'true' if pause else 'false'}",
    )


def set_model_pose(gz_bin, model, x, y, yaw, z=DEFAULT_Z, world=DEFAULT_WORLD):
    """Teleport `model`. Physics should already be paused."""
    qz, qw = yaw_to_quat_zw(yaw)
    req = (
        f'name: "{model}", '
        f"position: {{x: {x:.6f}, y: {y:.6f}, z: {z:.6f}}}, "
        f"orientation: {{x: 0.0, y: 0.0, z: {qz:.9f}, w: {qw:.9f}}}"
    )
    return _gz_service(gz_bin, f"/world/{world}/set_pose", "gz.msgs.Pose", req)


def pose_error(target, actual):
    """(position error [m], heading error [rad]) between two (x, y, yaw)."""
    if actual is None:
        return None, None
    dp = math.hypot(actual[0] - target[0], actual[1] - target[1])
    dyaw = abs(math.atan2(math.sin(actual[2] - target[2]),
                          math.cos(actual[2] - target[2])))
    return dp, dyaw


class SimResetter:
    """Full reset sequence. `cmd_pub` and `frame_waiter` are optional.

    Without them this still teleports correctly, but the caller is responsible
    for zeroing the command and for not recording stale camera frames.
    """

    def __init__(self, gz_bin=None, model=DEFAULT_MODEL, world=DEFAULT_WORLD,
                 z=DEFAULT_Z, settle_ms=150, verify_tol_m=0.05, logger=None):
        self.gz_bin = gz_bin or resolve_gz_bin("")
        self.model = model
        self.world = world
        self.z = z
        self.settle_ms = settle_ms
        self.verify_tol_m = verify_tol_m
        self._log = logger

    def reset(self, x, y, yaw, cmd_pub=None, frame_waiter=None):
        """Returns (ok, message, measured_pose). measured_pose is (x, y, yaw)."""
        target = (x, y, yaw)

        # 1. stop commanding before anything moves
        self._zero_command(cmd_pub)

        # 2. pause
        ok, msg = pause_sim(self.gz_bin, True, self.world)
        if not ok:
            return False, f"pause failed: {msg}", None

        try:
            # 3. teleport
            ok, msg = set_model_pose(
                self.gz_bin, self.model, x, y, yaw, self.z, self.world
            )
            if not ok:
                return False, f"set_pose failed: {msg}", None

            # 4. settle while still paused
            deadline = time.monotonic() + self.settle_ms / 1000.0
            while time.monotonic() < deadline:
                self._zero_command(cmd_pub)
                time.sleep(0.01)
        finally:
            # 5. always resume, even if the teleport failed — leaving the world
            # paused strands every other node.
            ok_resume, msg_resume = pause_sim(self.gz_bin, False, self.world)
            if not ok_resume:
                return False, f"resume failed: {msg_resume}", None

        # 6. verify with a DIRECT query; the pose stream still holds the old value
        actual = query_world_pose(self.gz_bin, self.model)
        dp, dyaw = pose_error(target, actual)
        if actual is None:
            return False, "pose verify failed (no reading)", None
        if dp > self.verify_tol_m:
            return (False,
                    f"pose verify failed: {dp:.3f} m off target "
                    f"(tol {self.verify_tol_m}) — wrong model name?",
                    actual)

        # 7. drop stale camera frames
        if frame_waiter is not None and not frame_waiter():
            return False, "timed out waiting for fresh camera frames", actual

        self._info(
            f"reset to ({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f}deg) — "
            f"err {dp * 100:.1f} cm / {math.degrees(dyaw):.2f} deg"
        )
        return True, "ok", actual

    @staticmethod
    def _zero_command(cmd_pub):
        if cmd_pub is None:
            return
        try:
            from geometry_msgs.msg import Twist
            cmd_pub.publish(Twist())
        except Exception:
            pass

    def _info(self, msg):
        if self._log is not None:
            self._log.info(msg)


def _load_zone(name):
    import os

    import yaml
    path = os.path.expanduser(
        "~/ROS2_project/nav-vla/src/nav_vla_pkg/config/zone_map.yaml"
    )
    with open(path, "r", encoding="utf-8") as f:
        zones = (yaml.safe_load(f) or {}).get("zones", {})
    if name not in zones:
        raise SystemExit(f"unknown zone '{name}'. available: {', '.join(zones)}")
    pose = zones[name].get("pose", {})
    return float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), float(pose.get("yaw", 0.0))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.add_argument("--yaw", type=float, help="radians")
    p.add_argument("--zone", help="reset to a named zone from zone_map.yaml instead of x/y/yaw")
    p.add_argument("--z", type=float, default=DEFAULT_Z)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--world", default=DEFAULT_WORLD)
    p.add_argument("--settle-ms", type=int, default=150)
    p.add_argument("--gz-bin", default="")
    args = p.parse_args(argv)

    if args.zone:
        x, y, yaw = _load_zone(args.zone)
    elif args.x is None or args.y is None:
        p.error("give --zone, or both --x and --y")
    else:
        x, y, yaw = args.x, args.y, (args.yaw or 0.0)

    before = query_world_pose(resolve_gz_bin(args.gz_bin), args.model)
    r = SimResetter(gz_bin=resolve_gz_bin(args.gz_bin), model=args.model,
                    world=args.world, z=args.z, settle_ms=args.settle_ms)
    ok, msg, actual = r.reset(x, y, yaw)

    if before:
        print(f"before: x={before[0]:8.3f} y={before[1]:8.3f} yaw={math.degrees(before[2]):8.2f}deg")
    print(f"target: x={x:8.3f} y={y:8.3f} yaw={math.degrees(yaw):8.2f}deg")
    if actual:
        dp, dyaw = pose_error((x, y, yaw), actual)
        print(f"after : x={actual[0]:8.3f} y={actual[1]:8.3f} yaw={math.degrees(actual[2]):8.2f}deg")
        print(f"error : {dp * 100:.1f} cm / {math.degrees(dyaw):.2f} deg")
    print(("OK: " if ok else "FAIL: ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
