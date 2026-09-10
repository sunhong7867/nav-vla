#!/usr/bin/env python3
"""SmolVLA behind the wire contract `stub_policy_server.py` already defined.

Runs in the lerobot venv, not the ROS2 interpreter, and holds no rclpy import.
`vla_bridge_node.py` is the only thing that talks to ROS. The split exists
because ROS2 Jazzy needs its apt numpy/opencv and lerobot brings its own
torch/transformers/av; making them agree is a fight with no prize.

    ->  {"jpeg": bytes, "state": [3]f32, "task": str, "seed": int, "req": int, "tick": int}
    <-  {"actions": [[dx, dy, dyaw], ...]}
        (+ "reasoning": str, "reasoning_req": int when --reasoning-every is
         on and the checkpoint carries a reasoning head; the bridge ignores
         unknown keys, so old bridges are unaffected)

Warm-up is not optional
-----------------------
The first forward pass through a compiled model builds CUDA graphs and takes
tens of seconds. If that happens after the harness has already started stepping
the world, the car sits still through the warm-up and every trial begins with a
watchdog stall. So the server warms up on synthetic input **before** it binds
the socket: nothing can ask it for actions until it is ready to answer at speed.

`--compile` is likewise closer to required than optional. Eager inference was
measured at 108 ms (9.2 Hz) against a 10 Hz control loop — every single tick
underruns. Whether compilation reaches the 36 ms the plan projects is reported
at startup rather than assumed; if it does not, that is a finding, and the run
should be planned around the number actually measured on this GPU.

Determinism
-----------
`seed` arrives per request and is applied per request. `D_same` — the
reproducibility floor that every divergence figure is divided by — is only
measurable if two runs can be pinned to the same sampler state. A server that
seeds once at startup cannot provide it.

Usage::

    ~/venv/navvla/bin/python vla_policy_server.py --checkpoint runs/xxx/checkpoints/last
    ~/venv/navvla/bin/python vla_policy_server.py --random-init      # harness test
"""

import argparse
import io
import sys
import time


def has_reasoning_head(checkpoint):
    """True if the checkpoint was trained by reasoning_vla (extra CE head)."""
    import glob
    import os
    try:
        from safetensors import safe_open
        for f in glob.glob(os.path.join(checkpoint, "model*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                if any(k.startswith("reasoning_head") for k in sf.keys()):
                    return True
    except Exception:
        pass
    return False


def build_policy(args, device):
    """Load a checkpoint, or an untrained model when only the path is under test."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if args.random_init:
        # Deliberately untrained. Lets the whole serving path — warm-up,
        # latency, chunk shape, the bridge's splicing — be exercised before a
        # checkpoint exists. Its actions are meaningless and it says so loudly.
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        cfg = SmolVLAConfig()
        cfg.chunk_size = args.chunk_len
        cfg.n_action_steps = args.chunk_len
        # A default config carries NO features, and SmolVLA then rejects every
        # batch with "All image features are missing from the batch" — the image
        # is present, but the model was never told to expect one. Training fills
        # these in from the dataset; a hand-built model has to state them, and
        # they must match what `to_lerobot.py` writes.
        # camera1, matching what training produces after --rename_map.
        args.camera_key = "observation.images.camera1"
        cfg.input_features = {
            args.camera_key: PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(3,)),
        }
        cfg.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        }
        print("!! RANDOM-INIT MODEL: actions are noise. Harness test only.")
        return SmolVLAPolicy(cfg)

    print(f"loading {args.checkpoint}")
    if args.reasoning_every > 0 and has_reasoning_head(args.checkpoint):
        # reasoning_vla lives one dir above scripts/ in the repo layout,
        # but right next to this file in the server's flat code/ snapshot
        import os
        import sys as _sys
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.dirname(here), here):
            if cand not in _sys.path:
                _sys.path.insert(0, cand)
        from reasoning_vla import ReasoningSmolVLAPolicy
        print("reasoning head found — serving reasoning text every "
              f"{args.reasoning_every} requests")
        return ReasoningSmolVLAPolicy.from_pretrained(args.checkpoint)
    if args.reasoning_every > 0:
        print("--reasoning-every set but checkpoint has no reasoning head; "
              "serving actions only")
        args.reasoning_every = 0
    return SmolVLAPolicy.from_pretrained(args.checkpoint)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--random-init", action="store_true",
                   help="untrained model, for testing the serving path")
    p.add_argument("--endpoint", default="ipc:///tmp/nav_vla.sock")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk-len", type=int, default=30)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--camera-key", default="observation.images.front")
    p.add_argument("--reasoning-every", type=int, default=0, metavar="N",
                   help="decode reasoning text on a side thread every Nth "
                        "request (0 = off; needs a reasoning_vla checkpoint)")
    p.add_argument("--reasoning-log", default="",
                   help="jsonl file to append {req, task, reasoning} rows "
                        "(for probes)")
    args = p.parse_args()

    if not args.checkpoint and not args.random_init:
        print("give --checkpoint, or --random-init to test the path without one",
              file=sys.stderr)
        return 2

    import msgpack
    import numpy as np
    import torch
    import zmq
    from PIL import Image

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    policy = build_policy(args, device).to(device).eval()

    # Take the camera key from the checkpoint, never from a default. Training
    # renames `observation.images.front` to `camera1` (smolvla_base was
    # pretrained with three cameras), so a server that hardcoded `front` would
    # hand the policy a batch it does not recognise. Reading it back means the
    # two can never disagree.
    cam_keys = [k for k in policy.config.input_features
                if k.startswith("observation.images.")]
    if cam_keys:
        if args.camera_key not in cam_keys:
            print(f"camera key from checkpoint: {cam_keys[0]} "
                  f"(was going to send {args.camera_key!r})")
        args.camera_key = cam_keys[0]
    print(f"camera key: {args.camera_key}   "
          f"expects {len(cam_keys)} camera(s)")

    # The processor pipeline is not optional plumbing. It tokenizes `task` into
    # `observation.language.tokens` (without it the model raises KeyError on the
    # very first batch) and, more quietly, applies the dataset normalisation the
    # policy was trained under. Hand-building a batch skips both: the KeyError is
    # loud, the un-normalised inputs would not have been.
    from lerobot.policies.factory import make_pre_post_processors
    preproc, postproc = make_pre_post_processors(
        policy.config, None if args.random_init else args.checkpoint)

    if args.compile:
        policy = torch.compile(policy, mode="reduce-overhead")

    def infer(jpeg, state, task, seed):
        torch.manual_seed(int(seed))
        img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        # CHW float in [0,1] with a batch dimension, matching what
        # LeRobotDataset hands the policy at training time.
        t_img = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                 ).unsqueeze(0)
        batch = {
            args.camera_key: t_img,
            "observation.state": torch.tensor(
                state, dtype=torch.float32).unsqueeze(0),
            "task": [task],
        }
        batch = preproc(batch)
        with torch.inference_mode():
            out = policy.predict_action_chunk(batch)
        out = postproc(out)
        return out.squeeze(0).float().cpu().numpy()

    # State width comes from the normalizer stats, not config.input_features:
    # the config carries smolvla_base's pretrained shapes (state [6], ghost
    # cameras) while the stats hold what this run actually trained on, and the
    # normalizer rejects any other width outright.
    state_dim = 3
    try:
        import glob as _glob
        from safetensors import safe_open
        cand = sorted(_glob.glob(
            f"{args.checkpoint}/*preprocessor*normalizer*.safetensors"))
        if cand:
            with safe_open(cand[0], framework="pt") as f:
                state_dim = f.get_tensor("observation.state.mean").shape[0]
        else:
            state_dim = policy.config.input_features[
                "observation.state"].shape[0]
    except Exception:
        pass
    print(f"state dim: {state_dim}")
    zero_state = [0.0] * state_dim

    # Warm up before binding: a client that connects can immediately be served.
    blank = Image.fromarray(
        (np.zeros((480, 640, 3), dtype=np.uint8) + 128))
    buf = io.BytesIO()
    blank.save(buf, format="JPEG")
    dummy = buf.getvalue()
    print(f"warming up ({args.warmup} passes"
          + (", compiled — this builds CUDA graphs and is slow" if args.compile
             else "") + ")")
    lat = []
    for i in range(max(1, args.warmup)):
        t0 = time.monotonic()
        a = infer(dummy, zero_state, "warmup", 0)
        dt = (time.monotonic() - t0) * 1000.0
        lat.append(dt)
        if i == 0:
            print(f"  first pass {dt:.0f} ms, action chunk {a.shape}")
    steady = lat[len(lat) // 2:]
    ms = sum(steady) / len(steady)
    print(f"warm: {ms:.1f} ms/chunk ({1000.0 / ms:.1f} Hz)")
    if ms > 100.0:
        # 10 Hz control with a chunk of 30 tolerates a slow server, but not one
        # slower than the queue drains. Report it rather than discover it as
        # watchdog activations in an eval run.
        print(f"  WARNING: {ms:.0f} ms is above the plan's 50-60 ms budget. "
              "The queue drains in "
              f"{args.chunk_len / 10.0:.1f}s, so this still keeps up, but "
              "measure underrun during the first trial.")

    # ---- reasoning side channel (never on the action path) ----------------
    # The decode is autoregressive (~1-2 s for 60 tokens with the naive
    # re-forward generate), far beyond the 10 Hz budget, so it runs on a
    # worker thread over a one-slot mailbox: the main loop deposits the
    # latest inputs every Nth request and replies immediately with the most
    # recent finished text. CUDA kernels from the two threads serialize per
    # forward, so the action pass waits at most one decode step (~30 ms).
    reasoning = {"text": "", "req": -1}
    r_slot = []
    if args.reasoning_every > 0:
        import json as _json
        import threading

        r_cv = threading.Condition()

        def _reasoning_worker():
            while True:
                with r_cv:
                    while not r_slot:
                        r_cv.wait()
                    jpeg, st, task, rid = r_slot.pop()
                try:
                    img = np.asarray(
                        Image.open(io.BytesIO(jpeg)).convert("RGB"))
                    t_img = (torch.from_numpy(img).permute(2, 0, 1).float()
                             / 255.0).unsqueeze(0)
                    b = {args.camera_key: t_img,
                         "observation.state": torch.tensor(
                             st, dtype=torch.float32).unsqueeze(0),
                         "task": [task]}
                    b = preproc(b)
                    with torch.inference_mode():
                        text = policy.generate_reasoning(b)[0]
                    reasoning["text"], reasoning["req"] = text, rid
                    if args.reasoning_log:
                        with open(args.reasoning_log, "a") as f:
                            f.write(_json.dumps(
                                {"req": rid, "task": task,
                                 "reasoning": text},
                                ensure_ascii=False) + "\n")
                except Exception as e:   # never take the server down
                    print(f"reasoning decode failed: {e}", file=sys.stderr)

        threading.Thread(target=_reasoning_worker, daemon=True).start()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.endpoint)
    print(f"serving on {args.endpoint}\nCtrl-C to stop\n")

    served, t_start, rlat = 0, time.monotonic(), []
    try:
        while True:
            req = msgpack.unpackb(sock.recv(), raw=False)
            t0 = time.monotonic()
            a = infer(req["jpeg"], req.get("state", zero_state),
                      req.get("task", ""), req.get("seed", 0))
            rlat.append((time.monotonic() - t0) * 1000.0)
            rep = {"actions": [[float(x) for x in row] for row in a]}
            if args.reasoning_every > 0:
                if served % args.reasoning_every == 0 and req.get("task"):
                    with r_cv:
                        r_slot[:] = [(req["jpeg"],
                                      req.get("state", zero_state),
                                      req.get("task", ""), req.get("req", 0))]
                        r_cv.notify()
                if reasoning["req"] >= 0:
                    rep["reasoning"] = reasoning["text"]
                    rep["reasoning_req"] = reasoning["req"]
            sock.send(msgpack.packb(rep, use_bin_type=True))
            served += 1
            if served % 20 == 0:
                w = rlat[-20:]
                print(f"served {served}  lat {sum(w) / len(w):.1f} ms  "
                      f"task={req.get('task', '')[:48]!r}")
    except KeyboardInterrupt:
        if rlat:
            rl = sorted(rlat)
            print(f"\nserved {served} chunks   "
                  f"median {rl[len(rl) // 2]:.1f} ms   p95 {rl[int(0.95 * len(rl))]:.1f} ms")
    finally:
        sock.close(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
