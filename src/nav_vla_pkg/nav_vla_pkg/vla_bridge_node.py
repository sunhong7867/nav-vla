"""Closed-loop serving bridge: the only thing that touches /cmd_vel during eval.

Two processes, on purpose. ROS2 Jazzy runs on the system Python 3.12 with apt
numpy/opencv; lerobot brings its own torch/transformers/av. Mixing them has cost
this project a day before. So the policy lives behind ZMQ
(``scripts/vla_policy_server.py`` or any stand-in speaking the same contract) and
this node stays pure rclpy.

Wire contract (msgpack over ZMQ REQ/REP)::

    ->  {"jpeg": bytes, "state": [3]f32, "task": str, "seed": int, "req": int}
    <-  {"actions": [[dx, dy, dyaw], ...]}        # CHUNK_LEN rows, ego-frame SE(2)

`seed` travels explicitly because `D_same` — the reproducibility floor every
divergence number is divided by — is only measurable if the sampler can be pinned
from the caller.

Threading
---------
Four callback slots, and the split is load-bearing:

* **sensor** (Reentrant) — image and odom callbacks write into a mutex-guarded
  `LatestObservation`. No encoding, no torch, no blocking.
* **control** (MutuallyExclusive, 10 Hz) — pops one action and publishes. Runs in
  microseconds and never touches the network. If inference were called from here,
  a 50 ms round trip would eat half the control period.
* **instruction** — `/vla/instruction`, RELIABLE + TRANSIENT_LOCAL depth 1, so an
  eval harness that attaches after the instruction was sent still sees it.
* **a plain thread outside the executor** — issues inference whenever the queue
  falls below `refill_at`. This is the only place that blocks.

Chunk splicing
--------------
New chunks are blended into the overlapping tail of the queue with a linear ramp.
Without it the command jumps at every chunk boundary, which shows up as a steering
twitch every `chunk_len` ticks and inflates the `D_same` noise floor — the
denominator of the whole evaluation.

Rules in the control path
-------------------------
Exactly one, and it is instrumented: if no chunk arrives for `watchdog_ms` or the
queue underruns, command `speed=0, steering=hold`. Every activation is counted and
published. **A valid eval run is watchdog activations == 0 and underrun < 1%**;
both numbers are reported with every result. Stopping at a goal is *not* a rule
here — it has to be learned behaviour, present in the demonstrations as a
deceleration ramp.
"""

import json
import math
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
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

SIM_WHEEL_BASE = 2.86      # ackermann_cmd_adapter_node.py:23
SIM_MAX_STEER = 0.6        # policy_node.py:59
MAX_CURVATURE = math.tan(SIM_MAX_STEER) / SIM_WHEEL_BASE


class LatestObservation:
    """Most recent image and state, with the lock the sensor callbacks share."""

    def __init__(self):
        self._lock = threading.Lock()
        self.jpeg = None
        self.stamp = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.steer = 0.0
        self.seq = 0

    def put_image(self, jpeg, stamp):
        with self._lock:
            self.jpeg = jpeg
            self.stamp = stamp
            self.seq += 1

    def put_state(self, speed, yaw_rate, steer):
        with self._lock:
            self.speed, self.yaw_rate, self.steer = speed, yaw_rate, steer

    def snapshot(self):
        with self._lock:
            if self.jpeg is None:
                return None
            return (self.jpeg, [float(self.speed), float(self.yaw_rate),
                                float(self.steer)], self.stamp, self.seq)


class ActionQueue:
    """Pending actions, spliced rather than concatenated."""

    def __init__(self, overlap):
        self._lock = threading.Lock()
        self._q = deque()
        self.overlap = overlap
        self.underruns = 0
        self.popped = 0

    def __len__(self):
        with self._lock:
            return len(self._q)

    def clear(self):
        with self._lock:
            self._q.clear()

    def pop(self):
        with self._lock:
            if not self._q:
                self.underruns += 1
                return None
            self.popped += 1
            return self._q.popleft()

    def splice(self, chunk):
        """Blend `chunk` into the tail with a linear cross-fade.

        The first `overlap` entries are mixed with what is already queued for
        those ticks, weighting the new chunk in gradually. A hard replace makes
        the command jump by the difference between two independent predictions;
        a plain append makes it jump at the seam instead.
        """
        chunk = [list(map(float, a)) for a in chunk]
        with self._lock:
            n = min(self.overlap, len(self._q), len(chunk))
            for i in range(n):
                w = (i + 1) / (n + 1)          # 0 -> 1 across the overlap
                old = self._q[i]
                chunk[i] = [(1.0 - w) * o + w * c for o, c in zip(old, chunk[i])]
            for _ in range(n):
                self._q.popleft()
            # Anything queued past the overlap is a stale prediction; the new
            # chunk was produced from a newer observation and supersedes it.
            self._q.clear()
            self._q.extend(chunk)


class VlaBridge(Node):

    def __init__(self):
        super().__init__("vla_bridge")

        self.endpoint = self.declare_parameter(
            "endpoint", "ipc:///tmp/nav_vla.sock").value
        self.image_topic = self.declare_parameter(
            "image_topic", "/vla_camera/image_raw").value
        self.odom_topic = self.declare_parameter("odom_topic", "/odom").value
        self.rate_hz = float(self.declare_parameter("rate_hz", 10.0).value)
        self.chunk_len = int(self.declare_parameter("chunk_len", 30).value)
        # Chunk commitment is not a tuning nicety — it decides whether the policy
        # turns at all. The turn decision lives in the BACK of each chunk (the
        # front is the shared-aisle prefix), so re-planning too often executes
        # only fronts and the turn is postponed forever. Measured, 4-bay
        # confusion, v2b checkpoint:
        #   refill 0.93 (~0.6 s committed)  every run collapsed to one endpoint
        #   refill 0.7  (~1.2 s)            bay2/bay3 correct, extremes wrong
        #   refill 0.3  (~2.1 s)            8/8 correct bays
        self.refill_at = float(self.declare_parameter("refill_at", 0.3).value)
        self.splice_overlap = int(self.declare_parameter("splice_overlap", 5).value)
        self.stale_factor = float(
            self.declare_parameter("stale_factor", 1.5).value)
        self.jpeg_quality = int(self.declare_parameter("jpeg_quality", 88).value)
        self.seed = int(self.declare_parameter("seed", 0).value)
        self.timeout_ms = int(self.declare_parameter("req_timeout_ms", 2000).value)
        self.max_speed = float(self.declare_parameter("max_speed", 2.0).value)
        # Demo speed shaping (both default OFF). The policy under-runs the
        # commanded tier by ~15-25% and steps at chunk refills; the teacher's
        # lane logic assumes tier speed, so driving slow also degrades
        # steering. scale multiplies the decoded v; slew caps |dv| per action
        # tick (30 Hz) — 0.08 = 2.4 m/s^2, above any learned braking ramp.
        # Watchdog zeroing bypasses both (safety path must stay instant).
        self.speed_scale = float(self.declare_parameter("speed_scale", 1.0).value)
        self.speed_slew = float(self.declare_parameter("speed_slew", 0.0).value)

        self.bridge = CvBridge()
        self.obs = LatestObservation()
        self.queue = ActionQueue(self.splice_overlap)
        self.task = ""
        self._task_lock = threading.Lock()
        self._last_chunk_t = 0.0
        self.watchdog_hits = 0
        self.n_infer = 0
        self.latencies = deque(maxlen=200)
        self._last_cmd = (0.0, 0.0)
        self._running = True

        sensor_cg = ReentrantCallbackGroup()
        control_cg = MutuallyExclusiveCallbackGroup()
        instr_cg = MutuallyExclusiveCallbackGroup()

        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST,
                                durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        # Two subscriptions on one topic, and both are needed.
        #
        # TRANSIENT_LOCAL is what the plan asks for: a harness that attaches after
        # the instruction was published still receives it, instead of the first
        # trial of every run silently driving with an empty task. But DDS matches
        # durability strictly, so a TRANSIENT_LOCAL *subscriber* will not hear a
        # VOLATILE publisher at all — plain `ros2 topic pub` is refused with an
        # "incompatible QoS" warning and nothing arrives.
        #
        # A VOLATILE subscriber alongside accepts both kinds. `_instr_cb` only
        # acts when the text changes, so receiving the same message twice from
        # a transient-local publisher costs nothing.
        instr_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                               history=QoSHistoryPolicy.KEEP_LAST,
                               durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                               depth=1)
        instr_qos_volatile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                        history=QoSHistoryPolicy.KEEP_LAST,
                                        durability=QoSDurabilityPolicy.VOLATILE,
                                        depth=1)

        self.create_subscription(Image, self.image_topic, self._img_cb,
                                 sensor_qos, callback_group=sensor_cg)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb,
                                 sensor_qos, callback_group=sensor_cg)
        self.create_subscription(String, "/vla/instruction", self._instr_cb,
                                 instr_qos, callback_group=instr_cg)
        self.create_subscription(String, "/vla/instruction", self._instr_cb,
                                 instr_qos_volatile, callback_group=instr_cg)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.stat_pub = self.create_publisher(String, "/vla/status", 10)
        # Telemetry only. Every chunk that gets spliced is mirrored here as raw
        # JSON for observers (the narrator, plotting). Nothing in the control
        # path subscribes to it or branches on it — the rule-free contract in
        # the module docstring is untouched.
        self.plan_pub = self.create_publisher(String, "/vla/plan", 5)

        self.create_timer(1.0 / self.rate_hz, self._control,
                          callback_group=control_cg)
        self.create_timer(2.0, self._heartbeat, callback_group=instr_cg)

        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

        self.get_logger().info(
            f"vla_bridge -> {self.endpoint}, image={self.image_topic}, "
            f"{self.rate_hz:.0f} Hz, chunk={self.chunk_len}, "
            f"splice overlap={self.splice_overlap}")

    # ---------------------------------------------------------------- sensors

    def _img_cb(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:                                  # noqa: BLE001
            self.get_logger().warn(f"image conversion failed: {e}")
            return
        ok, buf = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if ok:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.obs.put_image(buf.tobytes(), t)

    def _odom_cb(self, msg):
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        # Steering angle back-solved from the bicycle model, matching the
        # `state` vector the corpus was resampled with.
        steer = math.atan2(w * SIM_WHEEL_BASE, v) if abs(v) > 1e-3 else 0.0
        self.obs.put_state(v, w, steer)

    def _instr_cb(self, msg):
        text = msg.data.strip()
        with self._task_lock:
            changed = text != self.task
            self.task = text
        if changed:
            # Flush rather than let the previous instruction keep steering for up
            # to a chunk. A counterfactual test that switches instruction mid-run
            # would otherwise measure the old one for another 3 seconds.
            self.queue.clear()
            # An instruction change is a trial boundary. Counters must restart
            # with it: "a valid eval run has zero watchdog activations" is a
            # statement about one trial, and cumulative counts from node startup
            # (when there was no instruction and the queue was empty by
            # construction) would condemn every run.
            self.queue.underruns = 0
            self.queue.popped = 0
            self.watchdog_hits = 0
            self._last_chunk_t = 0.0
            self.get_logger().info(f"instruction: {text!r} (queue + counters reset)")

    # ---------------------------------------------------------------- control

    def _control(self):
        # No instruction = hands off. One explicit stop, then silence, so an
        # external tool (reset, teleop, the collection driver) can own /cmd_vel
        # between trials. Before this, the watchdog kept publishing during
        # resets and the car crept 0.15 m off its spawn before verification.
        with self._task_lock:
            task = self.task
        if not task:
            if self._last_cmd != (0.0, 0.0):
                self._publish(0.0, 0.0)
            return

        a = self.queue.pop()
        now = time.monotonic()

        # Staleness is measured against the *horizon the chunk covers*, not
        # against a fixed 500 ms. A chunk of 30 at 10 Hz is three seconds of
        # actions, and the refill thread deliberately waits until 70% of it is
        # consumed before asking for more — so chunks legitimately arrive about
        # once a second. Comparing arrival time to a flat 500 ms fired the
        # watchdog on a queue that was 24/30 full, which then held steering for
        # most ticks and put 59 discontinuities into an otherwise smooth command
        # stream. The failure this must catch is executing past the horizon that
        # was actually predicted.
        horizon_s = self.chunk_len / self.rate_hz
        stale = (self._last_chunk_t > 0.0
                 and now - self._last_chunk_t > horizon_s * self.stale_factor)

        if a is None or stale:
            # The single permitted rule. Counted, published, and fatal to the
            # validity of an eval run if it ever fires.
            self.watchdog_hits += 1
            # Full zero, not "steering hold". The plan's hold wording is about a
            # steering ANGLE on a real Ackermann car; on the sim's twist
            # interface w is a yaw RATE, and holding it means commanding the car
            # to keep rotating with v=0 — measured as 0.15 m of creep during a
            # reset window.
            self._publish(0.0, 0.0, override="watchdog")
            return

        dx, dy, dyaw = a
        dt = 1.0 / self.rate_hz
        # Forward component ONLY. The action's dy is lateral slip of the model
        # reference point (1.554 m ahead of the rear axle — measured during
        # recorder debugging, |dy|/dx up to 0.41 in turns). The vehicle geometry
        # reproduces that slip on its own; folding it into v with hypot()
        # inflated the commanded speed ~8% through turns, cutting curvature
        # k = w/v by the same fraction. Replaying the ORACLE'S OWN actions
        # missed the bay by 4 m laterally — the under-turn signature that was
        # being blamed on the policy.
        v_raw = dx / dt
        w_raw = dyaw / dt
        # The action's path geometry lives in its curvature k = w/v. Any speed
        # shaping below must preserve k (same line, different pace), so w is
        # recomputed from k after v is final — scaling v alone would flatten
        # every turn by the same factor.
        k_raw = w_raw / v_raw if abs(v_raw) > 1e-3 else None
        v = v_raw * self.speed_scale
        # Clamp to what the vehicle can physically execute. This shapes the
        # command, so it is declared in /vla/status rather than applied quietly.
        override = "none"
        v = max(-self.max_speed, min(self.max_speed, v))
        if self.speed_slew > 0.0:
            v_prev = self._last_cmd[0]
            lo, hi = v_prev - self.speed_slew, v_prev + self.speed_slew
            if v < lo or v > hi:
                v = min(max(v, lo), hi)
                override = "slew"
        if k_raw is not None:
            k = k_raw
            if abs(k) > MAX_CURVATURE:
                k = math.copysign(MAX_CURVATURE, k)
                override = "clamp_curvature"
            w = k * v
        else:
            w = w_raw
        self._publish(v, w, override=override)

    def _publish(self, v, w, override="none"):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)
        self._last_cmd = (v, w)
        if override != "none":
            self.stat_pub.publish(String(data=json.dumps(
                {"override": override, "v": v, "w": w,
                 "watchdog_hits": self.watchdog_hits})))

    # -------------------------------------------------------------- inference

    def _infer_loop(self):
        try:
            import msgpack
            import zmq
        except ImportError as e:                                # noqa: BLE001
            # This thread dying silently leaves the node running, the watchdog
            # firing every tick and the car stopped — which reads as "the policy
            # server is down" rather than "a dependency is missing from the
            # interpreter ros2 run uses" (a venv on PYTHONPATH is not it).
            self.get_logger().error(
                f"inference thread cannot start: {e}. Install into the "
                "interpreter `ros2 run` uses: "
                "/usr/bin/python3 -m pip install --user msgpack pyzmq")
            return

        ctx = zmq.Context.instance()
        sock = None

        def connect():
            s = ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            s.connect(self.endpoint)
            return s

        sock = connect()
        while self._running:
            if len(self.queue) >= self.refill_at * self.chunk_len:
                time.sleep(0.005)
                continue
            snap = self.obs.snapshot()
            with self._task_lock:
                task = self.task
            if snap is None or not task:
                time.sleep(0.05)
                continue
            jpeg, state, stamp, seq = snap
            # `tick` is the absolute count of actions already executed. A
            # deterministic server (replay, or any stub generating a continuous
            # signal) needs it to phase-lock: without it the server can only
            # guess how much of the last chunk was consumed, and guessing wrong
            # by one tick puts a step of exactly 2x the nominal into the command
            # stream at every chunk boundary. Measured before it was added: 5
            # such steps per 25 s, indistinguishable from a splicing bug.
            req = msgpack.packb({"jpeg": jpeg, "state": state, "task": task,
                                 "seed": self.seed, "req": self.n_infer,
                                 "tick": self.queue.popped},
                                use_bin_type=True)
            t0 = time.monotonic()
            try:
                sock.send(req)
                rep = msgpack.unpackb(sock.recv(), raw=False)
            except Exception as e:                              # noqa: BLE001
                # A REQ socket that timed out is stuck in the wrong state; it has
                # to be discarded, not retried.
                self.get_logger().warn(f"policy request failed: {e}")
                sock.close(0)
                sock = connect()
                time.sleep(0.2)
                continue
            self.latencies.append((time.monotonic() - t0) * 1000.0)
            actions = rep.get("actions")
            if not actions:
                time.sleep(0.05)
                continue
            self.queue.splice(actions)
            self._last_chunk_t = time.monotonic()
            # Telemetry mirror of the chunk just spliced (pre-blend, as the
            # server sent it). Published from this thread, not the control
            # timer, so a slow subscriber can never delay a control tick.
            self.plan_pub.publish(String(data=json.dumps(
                {"actions": actions, "t_mono": self._last_chunk_t})))
            self.n_infer += 1

    # --------------------------------------------------------------- reporting

    def _heartbeat(self):
        lat = (sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        total = max(self.queue.popped + self.queue.underruns, 1)
        payload = {
            "queue": len(self.queue), "chunks": self.n_infer,
            "latency_ms": round(lat, 1),
            "underrun_pct": round(100.0 * self.queue.underruns / total, 2),
            "watchdog_hits": self.watchdog_hits,
            "task": self.task,
        }
        self.stat_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.get_logger().info(
            f"q={payload['queue']:2d}/{self.chunk_len} chunks={self.n_infer} "
            f"lat={payload['latency_ms']:.1f}ms "
            f"underrun={payload['underrun_pct']:.2f}% "
            f"watchdog={self.watchdog_hits}")

    def destroy_node(self):
        self._running = False
        # Best effort: on Ctrl-C rclpy may already have torn the context down,
        # and a stop command that raises here would mask the real exit reason.
        try:
            if rclpy.ok():
                self.cmd_pub.publish(Twist())
        except Exception:                                       # noqa: BLE001
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = VlaBridge()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
