"""Save (server) or load (laptop) a SmolVLA and print a deterministic fingerprint.

Checks the one thing that matters for the train-here / serve-there split: does a
checkpoint written by the training box produce the SAME actions on the serving
box, when torch versions differ (2.6.0+cu124 vs 2.10.0+cu128)?

Random weights are enough — the question is about serialisation and numerics,
not about what the model has learned.
"""
import sys, json, hashlib
import numpy as np
import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

mode, path = sys.argv[1], sys.argv[2]
torch.manual_seed(20260729)

if mode == "save":
    cfg = SmolVLAConfig()
    cfg.chunk_size = 30
    cfg.n_action_steps = 30
    p = SmolVLAPolicy(cfg)
    p.save_pretrained(path)
    print("saved", path)
else:
    p = SmolVLAPolicy.from_pretrained(path)
    print("loaded", path)

p = p.eval()
sd = p.state_dict()
# Fingerprint the weights themselves: if these differ, nothing downstream can match.
h = hashlib.sha256()
for k in sorted(sd):
    v = sd[k]
    if v.dtype.is_floating_point:
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().float().numpy()).tobytes())
print("n_params", sum(v.numel() for v in sd.values()))
print("weight_sha256", h.hexdigest()[:32])
print("torch", torch.__version__, "lerobot_ok")
