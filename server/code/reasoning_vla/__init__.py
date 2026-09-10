"""Reasoning-head extension for SmolVLA (pip lerobot subclass, no fork).

Pinned against lerobot 0.4.4 — the subclass reaches into
`VLAFlowMatching.embed_prefix` / `SmolVLMWithExpertModel.forward` internals,
so a version drift is a real break. An import-time check warns loudly.
"""

import warnings

import lerobot

PINNED = "0.4.4"
if getattr(lerobot, "__version__", "?") != PINNED:
    warnings.warn(
        f"reasoning_vla was written against lerobot {PINNED}, found "
        f"{getattr(lerobot, '__version__', '?')} — verify embed_prefix/"
        "vlm_with_expert.forward before trusting a training run.",
        stacklevel=1)

from .reasoning_policy import ReasoningSmolVLAPolicy  # noqa: E402,F401
from .reasoning_dataset import ReasoningDataset       # noqa: E402,F401
