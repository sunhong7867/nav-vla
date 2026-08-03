#!/usr/bin/env python3
"""A policy server that answers the real wire contract with no model behind it.

The serving path — ZMQ round trip, chunk splicing, the 10 Hz control loop never
being blocked by inference, the watchdog, instruction latching — is the part of
the evaluation harness most likely to be quietly wrong, and none of it needs a
trained checkpoint to exercise. Debugging it later, against a model whose outputs
are also unknown, means two unknowns at once.

So this speaks exactly what ``scripts/vla_policy_server.py`` will speak and
returns something with **known correct answers**, chosen so that the properties
under test are measurable rather than eyeballed:

``--mode sine``
    Steering follows a continuous sine of absolute tick index, so successive
    chunks are two halves of one smooth curve. Any splicing bug is then a
    discontinuity in ``/cmd_vel`` at a chunk boundary — visible, and measurable as
    a jump in the second difference. A hard replace or a plain append both show up.

``--mode straight``
    Constant forward, zero steering. The floor: if this is not smooth, nothing
    downstream will be.

``--mode replay``
    Emits the action chunk of a recorded episode in order. Drives the car along
    something real, which is the closest a stub gets to end-to-end.

``--latency-ms`` injects a delay, because the interesting failure is a slow
server, not an absent one: at 10 Hz with a chunk of 30 the queue drains in three
seconds, and the plan budgets 50-60 ms per call under Gazebo render contention.

Usage::

    python3 src/nav_vla_pkg/scripts/stub_policy_server.py --mode sine
    python3 src/nav_vla_pkg/scripts/stub_policy_server.py --mode sine --latency-ms 120
    python3 src/nav_vla_pkg/scripts/stub_policy_server.py --mode replay \\
        --episode src/nav_vla_pkg/data_v2/smoke_wide/ep_0000
"""

import argparse
import json
import math
import os
import sys
import time

import msgpack
import zmq


def sine_chunk(tick, n, dt, speed, period_s, amp_rad_s):
    """Continuous in absolute time, so chunk boundaries are not special."""
    out = []
    for i in range(n):
        t = (tick + i) * dt
        w = amp_rad_s * math.sin(2.0 * math.pi * t / period_s)
        out.append([speed * dt, 0.0, w * dt])
    return out


def straight_chunk(n, dt, speed):
    return [[speed * dt, 0.0, 0.0] for _ in range(n)]


def load_replay(ep_dir):
    p = os.path.join(ep_dir, "resampled_10hz.jsonl")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found — run resample_episodes.py first")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    acts = [r["action"] for r in rows if r.get("action")]
    if not acts:
        raise SystemExit(f"{p} has no actions")
    return acts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", default="ipc:///tmp/nav_vla.sock")
    p.add_argument("--mode", default="sine",
                   choices=["sine", "straight", "replay"])
    p.add_argument("--chunk-len", type=int, default=30)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--speed", type=float, default=1.2)
    p.add_argument("--period-s", type=float, default=8.0)
    p.add_argument("--amp", type=float, default=0.25, help="rad/s of yaw")
    p.add_argument("--latency-ms", type=float, default=0.0)
    p.add_argument("--chunk-jitter", type=float, default=0.0,
                   help="rad/s offset applied per chunk. A real policy's "
                        "successive predictions disagree; with a perfectly "
                        "continuous stub, splicing has nothing to smooth and "
                        "cannot be tested at all.")
    p.add_argument("--episode", default="")
    args = p.parse_args()

    import random
    rng = random.Random(1234)
    replay = load_replay(args.episode) if args.mode == "replay" else None
    dt = 1.0 / args.fps

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.endpoint)
    print(f"stub policy server on {args.endpoint}  mode={args.mode} "
          f"chunk={args.chunk_len} latency={args.latency_ms:.0f}ms")
    print("Ctrl-C to stop\n")

    tick, served, t_start = 0, 0, time.monotonic()
    try:
        while True:
            req = msgpack.unpackb(sock.recv(), raw=False)
            if args.latency_ms > 0:
                time.sleep(args.latency_ms / 1000.0)
            # Phase-lock to what the bridge has actually executed. Guessing the
            # consumed count instead drifts by a tick per chunk and forges
            # discontinuities that look exactly like a splicing bug.
            tick = int(req.get("tick", tick))
            if args.mode == "sine":
                actions = sine_chunk(tick, args.chunk_len, dt, args.speed,
                                     args.period_s, args.amp)
            elif args.mode == "straight":
                actions = straight_chunk(args.chunk_len, dt, args.speed)
            else:
                # Clamp, never wrap. Wrapping replays the episode's FIRST
                # actions (cruise speed) the moment the recorded ones run out —
                # i.e. the car parks and then drives off through the back wall,
                # which reads as a serving-path failure that never happened.
                actions = [replay[tick + k] if tick + k < len(replay)
                           else [0.0, 0.0, 0.0]
                           for k in range(args.chunk_len)]
            if args.chunk_jitter:
                off = rng.uniform(-args.chunk_jitter, args.chunk_jitter) * dt
                actions = [[a[0], a[1], a[2] + off] for a in actions]
            sock.send(msgpack.packb({"actions": actions}, use_bin_type=True))
            served += 1
            if served % 20 == 0:
                el = time.monotonic() - t_start
                print(f"served {served} chunks in {el:.0f}s "
                      f"({served / el:.1f}/s), last task={req.get('task', '')!r}, "
                      f"jpeg={len(req.get('jpeg', b''))}B, "
                      f"state={[round(v, 3) for v in req.get('state', [])]}")
    except KeyboardInterrupt:
        print(f"\nserved {served} chunks")
    finally:
        sock.close(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
