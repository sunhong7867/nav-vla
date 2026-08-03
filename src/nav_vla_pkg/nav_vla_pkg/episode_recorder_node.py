"""Recorder v2 — schema v2 episode capture for VLA training.

Replaces ``data_engine_node.py`` for new collection. The old node stays as the
reference that produced the 619-episode corpus and as the ablation baseline; it
is not modified.

Four things the old recorder did that made its corpus unusable, and what this
does instead:

1. **It rate-limited inside the image callback** (``time.monotonic()`` gating at
   ``record_fps``). Measured result: 4.33 Hz against a 5 Hz target, a **+15.4%
   systematic error**, with no timestamp field to recover it. LeRobot then
   fabricates ``timestamp = frame_index / fps`` and validates against its own
   fabrication, so the error survives silently into training.
   → Here: **no rate limiting.** Everything is recorded at source rate and
   resampled offline onto an exact grid.

2. **It fused image + pose + command at one instant with no timestamps.** Any
   misalignment was unrecoverable.
   → Here: **four independent dense streams, each with its own timestamp**,
   joined offline.

3. **It wrote JPEGs from inside the subscriber callback.** Disk I/O blocked the
   callback, which is itself part of why the rate drifted.
   → Here: callbacks only enqueue; a writer thread encodes and writes.

4. **It rounded to 4 decimals**, destroying the pre-quantization signal.
   → Here: float64 throughout, and the control stream keeps **all three layers**
   (oracle continuous → pre-round float → emitted integer).

Clock
-----
Run with ``use_sim_time:=true``. Frame timestamps come from the image
``header.stamp``, which is simulator time; if the node clock were wall time the
streams would sit in two different time bases and the offline join would be
meaningless. The node refuses to record if it detects that mismatch.

This does *not* require the stepped-world migration — that trap (``/clock``
stops while paused, so ROS timers never fire) only bites when the world is
paused, and nothing here pauses it.

Control
-------
Driven by ``/recorder/command`` (``std_msgs/String``, JSON)::

    {"cmd": "start", "instruction": "take the first opening", "intent_id": "ord_first",
     "intent_slots": {"branch": "first", "bay_group": "A", "target_speed": 1.2},
     "cf_group_id": "g0042", "cf_variant_id": "v1", "seed": 7}
    {"cmd": "stop", "termination": "success"}
    {"cmd": "event", "kind": "settled"}
    {"cmd": "abort", "termination": "operator_abort"}

Status is echoed on ``/recorder/status``.
"""

import fcntl
import json
import math
import os
import queue
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from nav_vla_pkg.gz_pose import WorldPoseStream, query_world_pose, resolve_gz_bin

SCHEMA_VERSION = 2
DEFAULT_OUT = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla_pkg/data_v2")

# Only these may appear in meta.json. A bare success/failure bool is what let the
# old corpus reach 619/619 success with zero recorded recovery behaviour.
TERMINATIONS = {
    "success", "timeout", "collision", "off_track", "stuck", "operator_abort",
}


def git_sha(repo):
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def start_pose_key(x, y, yaw, pos_q=0.05, yaw_q=math.radians(2.0)):
    """Quantized pose identity used to group counterfactual variants.

    Two episodes share a key iff they started from the same cell. Written at
    collection time on purpose — it cannot be reconstructed afterwards once the
    exact spawn pose is gone.
    """
    return "p{:+06d}_{:+06d}_{:+05d}".format(
        int(round(x / pos_q)), int(round(y / pos_q)), int(round(yaw / yaw_q))
    )


class JsonlWriter:
    """Append-only JSONL sink, safe to call from several threads.

    Two things here are load-bearing:

    * **Its own lock.** Callers must never hold the episode lock while writing —
      the pose callback fires at 58 Hz, and holding a shared lock across disk I/O
      starved the frame writer thread badly enough that a third of the frames
      never reached disk.
    * **No per-line flush.** Flushing every line turned each of ~110 writes/s
      into a syscall. Buffered writes with a periodic flush keep the same crash
      guarantee within a second while removing the stall.
    """

    FLUSH_EVERY = 64

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1 << 16)
        self.n = 0
        # Counted here rather than by re-reading the file at stop time: the
        # per-source split is the only way to see that the tf stream died while
        # the CLI fallback kept the total plausible.
        self.n_tf = 0
        self._lock = threading.Lock()

    def write(self, obj):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            if obj.get("src") == "tf":
                self.n_tf += 1
            self.f.write(line)
            self.n += 1
            if self.n % self.FLUSH_EVERY == 0:
                self.f.flush()

    def close(self):
        with self._lock:
            try:
                self.f.flush()
                self.f.close()
            except Exception:
                pass


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__("episode_recorder")

        self.out_dir = self.declare_parameter("out_dir", DEFAULT_OUT).value
        self.image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        self.cmd_topic = self.declare_parameter("cmd_topic", "/cmd_vel").value
        self.oracle_topic = self.declare_parameter("oracle_topic", "/oracle_action").value
        self.model_name = self.declare_parameter("model_name", "ego_vehicle").value
        self.jpeg_quality = int(self.declare_parameter("jpeg_quality", 88).value)
        self.pose_poll_hz = float(self.declare_parameter("pose_poll_hz", 200.0).value)
        self.queue_max = int(self.declare_parameter("queue_max", 600).value)
        self.writer_threads = int(self.declare_parameter("writer_threads", 3).value)
        self.tf_topic = self.declare_parameter(
            "tf_topic", "/world/default/dynamic_pose/info").value
        self.tf_jump_tol = float(self.declare_parameter("tf_jump_tol", 1.0).value)
        self.ego_match_tol = float(self.declare_parameter("ego_match_tol", 2.0).value)
        self.session = self.declare_parameter("session", "").value
        self.repo = self.declare_parameter(
            "repo", os.path.expanduser("~/ROS2_project/nav-vla")).value

        self._check_clock()

        # --- episode state (guarded by _lock) ---------------------------
        self._lock = threading.Lock()
        self._rec = False
        self._ep = None            # dict of open writers + counters
        self._frame_q = queue.Queue(maxsize=self.queue_max)
        self._dropped = 0
        self._latest_cmd = (0.0, 0.0)
        self._latest_cmd_t = 0.0
        self._latest_oracle = {}
        self._latest_oracle_t = 0.0
        self._ego_idx = None
        self._ego_probe = None
        self._last_tf_xy = None
        self._tf_gap = False
        self._reident = threading.Event()
        self._tf_seq = 0
        self._ego_offset = None
        self._enc_s = 0.0
        self._io_s = 0.0
        self._wrote = 0

        session = self.session or time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.out_dir, session)
        os.makedirs(self.session_dir, exist_ok=True)
        self._claim_session()

        self.gz_bin = resolve_gz_bin(self.declare_parameter("gz_bin", "").value)
        self.stream = WorldPoseStream(self.gz_bin, self.model_name).start()

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=30)
        sensor_cg = ReentrantCallbackGroup()
        ctrl_cg = MutuallyExclusiveCallbackGroup()

        self.create_subscription(Image, self.image_topic, self._img_cb,
                                 sensor_qos, callback_group=sensor_cg)
        self.create_subscription(Twist, self.cmd_topic, self._cmd_cb, 20,
                                 callback_group=sensor_cg)
        self.create_subscription(String, self.oracle_topic, self._oracle_cb, 20,
                                 callback_group=sensor_cg)
        self.create_subscription(TFMessage, self.tf_topic, self._tf_cb,
                                 sensor_qos, callback_group=sensor_cg)
        self.create_subscription(TFMessage, self.tf_topic, self._tf_probe_cb,
                                 sensor_qos, callback_group=sensor_cg)
        self.create_subscription(String, "/recorder/command", self._command_cb, 10,
                                 callback_group=ctrl_cg)
        self.status_pub = self.create_publisher(String, "/recorder/status", 10)

        self._run_writer = True
        # Several writers, not one. Measured under a live Gazebo, a single writer
        # needs ~40 ms per frame (12 ms encode + 28 ms I/O, both inflated ~6x over
        # an idle benchmark by CPU contention) and saturates at ~25 Hz — just under
        # the 30 Hz camera, so the queue grows without bound and the tail of every
        # episode is lost. cv2.imencode releases the GIL, so extra threads
        # genuinely parallelise. Frames carry their own seq and timestamp, so
        # out-of-order completion is harmless; the resampler sorts by t.
        for _ in range(max(1, self.writer_threads)):
            threading.Thread(target=self._writer_loop, daemon=True).start()
        threading.Thread(target=self._pose_loop, daemon=True).start()

        self._img_seen = 0
        # Identify early so the CLI fallback never writes into a real episode —
        # mixing two pose sources in one stream puts two different latencies in
        # the same timeline.
        threading.Thread(target=self._identify_ego, daemon=True).start()
        threading.Thread(target=self._reident_loop, daemon=True).start()
        self.create_timer(2.0, self._heartbeat, callback_group=ctrl_cg)
        self.get_logger().info(
            f"episode_recorder ready — session={self.session_dir}, "
            f"image={self.image_topic}, sim_time={self._sim_time}"
        )

    # ------------------------------------------------------------------ clock

    def _check_clock(self):
        self._sim_time = bool(self.get_parameter_or(
            "use_sim_time", rclpy.parameter.Parameter("use_sim_time", value=False)).value)
        self._clock_alive = None
        if not self._sim_time:
            self.get_logger().warn(
                "use_sim_time is FALSE. Frame stamps come from the simulator but "
                "pose/control stamps would come from the wall clock — the offline "
                "join across streams would be meaningless. Relaunch with "
                "--ros-args -p use_sim_time:=true"
            )

    def clock_alive(self, wait_s=3.0):
        """Is the clock actually advancing?

        Checking only the ``use_sim_time`` *parameter* is not enough, and that
        gap already cost one collection run: with the parameter true and no
        ``/clock`` publisher, ``get_clock().now()`` is pinned at 0 forever, so
        every pose and control row was written with ``t = 0.0``. Image stamps
        were fine (they carry the simulator's own header), which made a total
        failure look like a partial success.

        Result is cached once the clock is seen to move.
        """
        if self._clock_alive:
            return True
        if not self._sim_time:
            self._clock_alive = True
            return True
        t0 = self._now()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            time.sleep(0.1)
            if self._now() > t0:
                self._clock_alive = True
                return True
        self._clock_alive = False
        return False

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _stamp(msg):
        s = msg.header.stamp
        return int(s.sec) + int(s.nanosec) * 1e-9

    # --------------------------------------------------------------- callbacks

    def _img_cb(self, msg):
        self._img_seen += 1
        if not self._rec:
            return
        try:
            ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, ch)
            bgr = arr[:, :, 2::-1] if msg.encoding in ("rgb8", "rgba8") else arr[:, :, :3]
            # The message buffer is reused; the writer thread runs later, so copy.
            frame = np.ascontiguousarray(bgr)
        except ValueError:
            return
        t = self._stamp(msg) or self._now()
        try:
            self._frame_q.put_nowait((t, frame, msg.width, msg.height))
        except queue.Full:
            # Never silently drop. A dropped frame invalidates the episode's
            # timing, so it is counted and the episode is failed at close.
            self._dropped += 1

    def _cmd_cb(self, msg):
        self._latest_cmd = (float(msg.linear.x), float(msg.angular.z))
        self._latest_cmd_t = self._now()
        if not self._rec:
            return
        self._write_control()

    def _oracle_cb(self, msg):
        try:
            self._latest_oracle = json.loads(msg.data)
            self._latest_oracle_t = self._now()
        except (json.JSONDecodeError, TypeError):
            pass

    def _write_control(self):
        with self._lock:
            ep = self._ep
        if ep is None:
            return
        if True:
            t = self._latest_cmd_t
            o = dict(self._latest_oracle)
            row = {
                "t": t,
                "v_cmd": self._latest_cmd[0],
                "yaw_rate_cmd": self._latest_cmd[1],
                # layer 1: oracle's continuous output, before any quantization
                "oracle_curvature": o.get("curvature"),
                "oracle_v": o.get("target_speed_mps"),
                "oracle_steer_rad": o.get("steer_angle_rad"),
                # layer 2: the float that would enter motion_planner's round()
                "steer_float": o.get("steer_float"),
                # layer 3: what an integer-command path would emit
                "steer_cmd": o.get("steering_cmd_int"),
                "source": o.get("source", "unknown"),
                "override": o.get("override", "none"),
                # Age of the merged oracle sample. The resampler rejects stale
                # merges rather than pretending the two arrived together.
                "oracle_age_s": (t - self._latest_oracle_t) if self._latest_oracle_t else None,
            }
        ep["control"].write(row)

    # ------------------------------------------------------------------ threads

    def _writer_loop(self):
        """JPEG encode + disk write off the callback thread."""
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self._run_writer:
            try:
                t, frame, w, h = self._frame_q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                ep = self._ep
                if ep is None:
                    continue
                seq = ep["frame_seq"]
                ep["frame_seq"] += 1
            frames_dir = ep["frames_dir"]
            writer = ep["frames"]
            name = f"{seq:06d}.jpg"
            _t0 = time.perf_counter()
            try:
                ok, buf = cv2.imencode(".jpg", frame, params)
                _t1 = time.perf_counter()
                if not ok:
                    continue
                with open(os.path.join(frames_dir, name), "wb") as f:
                    f.write(buf.tobytes())
                _t2 = time.perf_counter()
                self._enc_s += _t1 - _t0
                self._io_s += _t2 - _t1
                self._wrote += 1
            except Exception as exc:
                self.get_logger().error(f"frame write failed: {exc}")
                continue
            writer.write({"t": t, "seq": seq, "file": f"frames/{name}",
                          "enc": "jpg", "w": int(w), "h": int(h)})

    def _tf_cb(self, msg):
        """Native ground-truth pose from the bridged dynamic_pose topic.

        The bridge drops entity names (frame_id comes through empty), so the ego
        vehicle is located once by matching against the authoritative
        ``gz model -p`` reading and then tracked by index. The index is stable
        for the life of the world; it is re-validated whenever the tracked entry
        jumps further than a vehicle could physically move between messages.
        """
        if not self._rec:
            return
        tfs = msg.transforms
        idx = self._ego_idx
        if idx is None or idx >= len(tfs):
            return
        tr = tfs[idx].transform
        x, y = tr.translation.x, tr.translation.y
        if self._last_tf_xy is not None:
            dx = math.hypot(x - self._last_tf_xy[0], y - self._last_tf_xy[1])
            if dx > self.tf_jump_tol:
                # Either the entity order shifted or the car was teleported.
                # Re-identify rather than silently log another entity's pose.
                #
                # Re-identification cannot happen here: it shells out to
                # `gz model -p` and waits up to 3 s, and this callback runs at
                # 58 Hz. It is handed to a worker thread, and until that worker
                # finishes this stream is dark — so the episode is marked as
                # having a hole. Before this was wired up, `_ego_idx` was only
                # ever restored by the *next* start_episode, which meant a
                # collection run lost the tf stream for every second episode
                # while still reporting valid=true.
                self._ego_idx = None
                self._last_tf_xy = None
                self._tf_gap = True
                self._reident.set()
                self.get_logger().warn(
                    f"pose index {idx} jumped {dx:.2f} m — re-identifying ego")
                return
        self._last_tf_xy = (x, y)
        q = tfs[idx].transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._tf_seq += 1
        with self._lock:
            ep = self._ep
        if ep is None:
            return
        ep["poses"].write({"t": self._now(), "seq": self._tf_seq,
                           "x": x, "y": y, "yaw": yaw, "src": "tf"})

    def _identify_ego(self, timeout_s=3.0):
        """Find the ego vehicle's index in the dynamic_pose array."""
        truth = query_world_pose(self.gz_bin, self.model_name)
        if truth is None:
            return False
        self._ego_probe = None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self._ego_probe is None:
            time.sleep(0.05)
        tfs = self._ego_probe
        if not tfs:
            return False
        dists = sorted(
            ((math.hypot(tf.transform.translation.x - truth[0],
                         tf.transform.translation.y - truth[1]), i)
             for i, tf in enumerate(tfs)), key=lambda p: p[0])
        best_d, best = dists[0]
        runner_d = dists[1][0] if len(dists) > 1 else float("inf")
        # The bridge reports the canonical link, which sits a fixed offset from
        # the model origin `gz model -p` returns (measured ~0.6 m on the Prius).
        # So an exact match is not expected; what must hold is that the match is
        # unambiguous. Tracking a consistently offset point on the rigid body is
        # a frame choice, not an error — but tracking the WRONG BODY is fatal.
        if best_d > self.ego_match_tol or runner_d < 3.0 * best_d:
            self.get_logger().error(
                f"cannot identify {self.model_name} in dynamic_pose: closest "
                f"{best_d:.2f} m, runner-up {runner_d:.2f} m "
                f"(need closest <= {self.ego_match_tol} m and 3x separation)")
            return False
        self._ego_offset = best_d
        self._ego_idx = best
        self._last_tf_xy = None
        self.get_logger().info(
            f"ego identified at dynamic_pose index {best}: canonical-link offset "
            f"{best_d * 100:.1f} cm, next entity {runner_d:.1f} m away")
        return True

    def _reident_loop(self):
        """Restore the tf pose stream after a jump, off the callback thread.

        `_identify_ego` blocks on a subprocess and then waits for a probe frame,
        which is far too long to do inside a 58 Hz callback. Retrying here keeps
        the outage to a fraction of a second instead of the rest of the episode.
        """
        while self._run_writer:
            if not self._reident.wait(timeout=0.5):
                continue
            self._reident.clear()
            for _ in range(5):
                if not self._run_writer or self._ego_idx is not None:
                    break
                if self._identify_ego():
                    if self._rec:
                        self.mark_event("pose_reidentified",
                                        {"index": self._ego_idx})
                    break
                time.sleep(0.5)

    def _tf_probe_cb(self, msg):
        self._ego_probe = msg.transforms

    def _pose_loop(self):
        """Second, independent pose source, recorded alongside the TF bridge.

        The two do not agree, and the disagreement is in the one quantity the
        action labels are built from. Driving straight, both give a clean -90 deg
        offset between reported yaw and heading. Driving a 0.31 rad/s turn, the
        model-origin CLI pose gives -79.5 deg (a 10.5 deg slip angle, physically
        ordinary) while the bridged TF pose gives -67.5 deg (22.5 deg, which is
        not). Until that is resolved, **both streams are recorded, tagged by
        `src`**, so the question can be settled from data already on disk instead
        of re-collecting a corpus in the wrong frame.
        """
        period = 1.0 / max(self.pose_poll_hz, 1.0)
        last_seq = -1
        while self._run_writer:
            time.sleep(period)
            if not self._rec:
                continue
            seq = self.stream.seq
            if seq == last_seq:
                continue
            last_seq = seq
            pose = self.stream.latest
            if pose is None:
                continue
            with self._lock:
                ep = self._ep
            if ep is None:
                continue
            ep["poses"].write({"t": self._now(), "seq": seq, "x": pose[0],
                               "y": pose[1], "yaw": pose[2], "src": "cli"})

    # ------------------------------------------------------------------ control

    def _command_cb(self, msg):
        try:
            req = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self._status("error", detail="command is not valid JSON")
            return
        cmd = str(req.get("cmd", "")).lower()
        if cmd == "start":
            self.start_episode(req)
        elif cmd in ("stop", "abort"):
            self.stop_episode(req.get("termination",
                                      "success" if cmd == "stop" else "operator_abort"))
        elif cmd == "event":
            self.mark_event(req.get("kind", "mark"), req.get("data"))
        else:
            self._status("error", detail=f"unknown cmd '{cmd}'")

    def _claim_session(self):
        """Take an exclusive lock on the session directory, or refuse to run.

        A second recorder on the same session is not a harmless duplicate. Both
        subscribe to ``/recorder/command``, so both react to every start, and
        both pick their episode index with ``len(os.listdir(...))`` — which for
        simultaneous starts returns the same number to both. They then interleave
        writes into one ``frames.jsonl`` and one ``poses.jsonl``, and the result
        looks like a healthy episode: meta.json is written, frame counts are
        plausible, nothing reports an error. It is only visible as a stream whose
        timestamps go backwards.

        This happened during the first collection smoke test — a stale node had
        survived a kill that only reached its ``ros2 run`` wrapper — and cost two
        episodes before the duplicate directories gave it away.

        flock is advisory and per-open-file-description, so it is released
        automatically if the process dies, however it dies. No stale lock file
        has to be cleaned up by hand.
        """
        path = os.path.join(self.session_dir, ".session.lock")
        self._lock_fd = open(path, "w")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_fd.close()
            raise SystemExit(
                f"another episode_recorder already owns {self.session_dir}.\n"
                "Two recorders on one session interleave their writes and the "
                "corruption is not visible in meta.json. Stop the other one "
                "(pgrep -f '[e]pisode_recorder_node') or pass a different "
                "-p session:=<name>."
            )
        self._lock_fd.write(f"{os.getpid()}\n")
        self._lock_fd.flush()

    def start_episode(self, req):
        if self._rec:
            self.stop_episode("operator_abort")

        # Refuse rather than write a corpus of t=0.0 rows. This is the exact
        # failure mode this recorder exists to prevent, so it fails closed.
        if not self.clock_alive():
            detail = ("use_sim_time is true but the clock is stuck at "
                      f"{self._now():.3f} — nothing is publishing /clock. "
                      "Every pose/control stamp would be 0.0 and the offline "
                      "join would be impossible. Check that the ros_gz clock "
                      "bridge is running (ros2 topic hz /clock).")
            self.get_logger().error(detail)
            self._status("error", detail=detail)
            return

        if self._ego_idx is None:
            self._identify_ego()

        pose = self.stream.latest
        if pose is None:
            self._status("error", detail="no pose yet — is the sim running?")
            return

        # exist_ok=False, then bump: counting existing directories can hand the
        # same index to two callers, and exist_ok=True would let the second one
        # overwrite the first's episode in place.
        idx = len([d for d in os.listdir(self.session_dir)
                   if d.startswith("ep_")]) if os.path.isdir(self.session_dir) else 0
        while True:
            ep_dir = os.path.join(self.session_dir, f"ep_{idx:04d}")
            try:
                os.makedirs(os.path.join(ep_dir, "frames"), exist_ok=False)
                break
            except FileExistsError:
                idx += 1

        with self._lock:
            self._dropped = 0
            # The car is teleported between episodes, so the last position of the
            # previous episode is not evidence that the entity order changed.
            # Leaving it set made every reset look like a jump.
            self._last_tf_xy = None
            self._tf_gap = False
            self._ep = {
                "dir": ep_dir,
                "frames_dir": os.path.join(ep_dir, "frames"),
                "frame_seq": 0,
                "frames": JsonlWriter(os.path.join(ep_dir, "frames.jsonl")),
                "poses": JsonlWriter(os.path.join(ep_dir, "poses.jsonl")),
                "control": JsonlWriter(os.path.join(ep_dir, "control.jsonl")),
                "events": JsonlWriter(os.path.join(ep_dir, "events.jsonl")),
                "t_start": self._now(),
                "start_pose": list(pose),
                "req": req,
            }
        self._rec = True
        self.mark_event("episode_start", {"pose": list(pose)})
        self._status("recording", episode=os.path.basename(ep_dir))
        self.get_logger().info(
            f"start {os.path.basename(ep_dir)}: "
            f"instruction={req.get('instruction')!r} "
            f"cf_group={req.get('cf_group_id')}"
        )

    def mark_event(self, kind, data=None):
        with self._lock:
            if self._ep is None:
                return
            self._ep["events"].write({"t": self._now(), "kind": kind, "data": data})

    def stop_episode(self, termination):
        if not self._rec:
            return
        self._rec = False
        # Let the writer drain what the callbacks already enqueued.
        deadline = time.monotonic() + 30.0
        while not self._frame_q.empty() and time.monotonic() < deadline:
            time.sleep(0.05)

        termination = termination if termination in TERMINATIONS else "operator_abort"
        with self._lock:
            ep = self._ep
            self._ep = None
        if ep is None:
            return

        req = ep["req"]
        sp = ep["start_pose"]
        meta = {
            "schema_version": SCHEMA_VERSION,
            "instruction": req.get("instruction", ""),
            "intent_id": req.get("intent_id", ""),
            "intent_slots": req.get("intent_slots", {}),
            "termination": termination,
            "cf_group_id": req.get("cf_group_id"),
            "cf_variant_id": req.get("cf_variant_id"),
            # Which single axis this group varies. Without it, pairs from an
            # ordinal group and a speed group are indistinguishable downstream
            # and their very different divergence figures get averaged together.
            "cf_axis": req.get("cf_axis"),
            "start_pose": sp,
            "start_pose_key": start_pose_key(sp[0], sp[1], sp[2]),
            "seed": req.get("seed"),
            "world": req.get("world", "default"),
            "git_sha": git_sha(self.repo),
            "t_start": ep["t_start"],
            "t_end": self._now(),
            "n_frames": ep["frames"].n,
            "n_poses": ep["poses"].n,
            "n_control": ep["control"].n,
            "dropped_frames": self._dropped,
            "use_sim_time": self._sim_time,
            "pose_source": "tf_bridge" if self._tf_seq > 0 else "gz_cli",
            "ego_canonical_offset_m": self._ego_offset,
            "image_topic": self.image_topic,
            "jpeg_quality": self.jpeg_quality,
        }
        # Per-source pose counts. A single `n_poses` hides the failure that
        # matters: the tf bridge stream can be dead for a whole episode while the
        # CLI fallback keeps the total looking healthy, and the action labels are
        # built from tf.
        n_tf = ep["poses"].n_tf
        n_cli = ep["poses"].n - n_tf
        meta["n_poses_tf"] = n_tf
        meta["n_poses_cli"] = n_cli

        dur_m = max(meta["t_end"] - meta["t_start"], 1e-6)
        reasons = []
        if self._dropped:
            # A dropped frame means the timing of this episode is unrecoverable.
            reasons.append(f"{self._dropped} frames dropped (writer backlog)")
        if self._tf_gap:
            reasons.append("tf pose stream had a gap (ego re-identified mid-episode)")
        if n_tf / dur_m < 5.0:
            reasons.append(f"tf pose stream only {n_tf / dur_m:.1f} Hz "
                           f"({n_tf} rows over {dur_m:.1f}s)")
        meta["valid"] = not reasons
        if reasons:
            meta["invalid_reason"] = "; ".join(reasons)

        for k in ("frames", "poses", "control", "events"):
            ep[k].close()
        with open(os.path.join(ep["dir"], "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        dur = meta["t_end"] - meta["t_start"]
        rate = meta["n_frames"] / dur if dur > 0 else 0.0
        self.get_logger().info(
            f"stop {os.path.basename(ep['dir'])}: {termination}, "
            f"{meta['n_frames']} frames in {dur:.2f}s ({rate:.2f} Hz), "
            f"{meta['n_poses']} poses, {meta['n_control']} control"
            + (f", DROPPED {self._dropped}" if self._dropped else "")
        )
        self._status("idle", episode=os.path.basename(ep["dir"]),
                     termination=termination, n_frames=meta["n_frames"],
                     dropped=self._dropped, valid=meta["valid"])

    # ------------------------------------------------------------------ status

    def _status(self, state, **kw):
        payload = {"state": state, "session": os.path.basename(self.session_dir)}
        payload.update(kw)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _heartbeat(self):
        if self._img_seen == 0:
            self.get_logger().warn(
                f"no images on {self.image_topic} — check the sim and the QoS")
        if self._rec:
            w = max(self._wrote, 1)
            self.get_logger().info(
                f"recording: q={self._frame_q.qsize()}/{self.queue_max} "
                f"dropped={self._dropped} wrote={self._wrote} "
                f"enc={self._enc_s / w * 1000:.2f}ms io={self._io_s / w * 1000:.2f}ms "
                f"busy={(self._enc_s + self._io_s):.2f}s")
            self._enc_s = self._io_s = 0.0; self._wrote = 0
        else:
            # An idle recorder used to publish nothing at all, so a driver that
            # waited for status before its first episode waited forever. The
            # session name only lives here, and the collection manifest has to be
            # written beside the episodes it describes.
            self._status("idle")
        self._img_seen = 0

    def destroy_node(self):
        if self._rec:
            self.stop_episode("operator_abort")
        self._run_writer = False
        try:
            self.stream.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = EpisodeRecorder()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
