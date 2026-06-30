"""Ground-truth world pose of a Gazebo model via the gz CLI.

`/odom` is DiffDrive wheel-integrated (drifts, ignores manual drags). Bridging
`/world/default/pose/info` loses entity names in this ros_gz version. So we read
the model's true world pose directly with `gz model -m <name> -p` and parse it.
"""

import math
import re
import shutil
import subprocess
import threading
import time


def resolve_gz_bin(explicit=""):
    return explicit or shutil.which("gz") or "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"


def query_world_pose(gz_bin, model, timeout=3.0):
    """Return (x, y, yaw) of `model` in the world frame, or None on failure.

    Parses `gz model -m <model> -p`, whose output contains an XYZ bracket and an
    RPY bracket, e.g.:
        - Pose [ XYZ (m) ] [ RPY (rad) ]:
          [3.700000 24.594300 0.012650]
          [-0.000000 0.000000 -1.570740]
    """
    try:
        out = subprocess.run(
            [gz_bin, "model", "-m", model, "-p"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    triples = []
    for grp in re.findall(r"\[([^\]]+)\]", out.stdout):
        parts = grp.replace(",", " ").split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue  # skip non-numeric brackets like "[ XYZ (m) ]"
        if len(vals) == 3:
            triples.append(vals)
    if len(triples) >= 2:
        x, y, _ = triples[0]
        yaw = triples[1][2]
        return (x, y, yaw)
    return None


class WorldPoseStream:
    """High-rate ground-truth pose by streaming `gz topic -e` on the world
    dynamic_pose topic and parsing the ego model's blocks. Much faster than
    repeated `gz model -p` CLI calls (which cost ~150ms each)."""

    def __init__(self, gz_bin, model, world="default"):
        self.gz_bin = gz_bin
        self.model = model
        self.topic = f"/world/{world}/dynamic_pose/info"
        self.latest = None  # (x, y, yaw)
        self._run = True
        self._proc = None

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._run = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _loop(self):
        while self._run:
            try:
                self._proc = subprocess.Popen(
                    [self.gz_bin, "topic", "-e", "-t", self.topic],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, bufsize=1,
                )
            except FileNotFoundError:
                return
            cur = None
            mode = None
            px = py = oz = ow = None
            for line in self._proc.stdout:
                if not self._run:
                    break
                s = line.strip()
                if s.startswith("name:"):
                    cur = s.split('"')[1] if '"' in s else None
                    mode = None
                    px = py = oz = ow = None
                    continue
                if cur != self.model:
                    continue
                if s.startswith("position"):
                    mode = "pos"
                    continue
                if s.startswith("orientation"):
                    mode = "ori"
                    continue
                try:
                    if mode == "pos" and s.startswith("x:"):
                        px = float(s[2:])
                    elif mode == "pos" and s.startswith("y:"):
                        py = float(s[2:])
                    elif mode == "ori" and s.startswith("z:"):
                        oz = float(s[2:])
                    elif mode == "ori" and s.startswith("w:"):
                        ow = float(s[2:])
                except ValueError:
                    pass
                if None not in (px, py, oz, ow):
                    yaw = math.atan2(2.0 * ow * oz, 1.0 - 2.0 * oz * oz)
                    self.latest = (px, py, yaw)
                    cur = None
                    mode = None
                    px = py = oz = ow = None
            time.sleep(0.2)  # stream ended; brief pause before restart


def list_models(gz_bin, timeout=4.0):
    """Return all model names in the running world via `gz model --list`."""
    try:
        out = subprocess.run([gz_bin, "model", "--list"],
                             capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    names, grab = [], False
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.lower().startswith("available models"):
            grab = True
            continue
        if grab and s.startswith("- "):
            names.append(s[2:].strip())
    return names
