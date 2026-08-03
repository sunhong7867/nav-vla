"""Shared natural-language speed controls.

The user-facing unit is the vehicle's raw motor command (0..250).  Simulation
controllers convert that limit to m/s only at the final Twist boundary.
"""

import re


MIN_SPEED_RAW = 0
MAX_SPEED_RAW = 250
DEFAULT_SPEED_RAW = 150
SPEED_STEP_RAW = 20
SLOW_SPEED_RAW = 70
NORMAL_SPEED_RAW = 150
FAST_SPEED_RAW = 200

_EXPLICIT_SPEED_PATTERNS = (
    re.compile(
        r"(?:속도|speed)\s*(?:를|을)?\s*(?:raw\s*)?"
        r"(?:값\s*)?(?:을|를)?\s*(?:로|=|:)?\s*(\d{1,3})",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\d)(\d{1,3})\s*(?:의\s*)?(?:속도|speed)", re.IGNORECASE),
)
_FASTER_RE = re.compile(
    r"더\s*(?:빠르게|빨리)|(?:속도|speed).*?(?:올려|높여|증가|up)",
    re.IGNORECASE,
)
_SLOWER_RE = re.compile(
    r"더\s*(?:천천히|느리게)|(?:속도|speed).*?(?:줄여|낮춰|감소|down)",
    re.IGNORECASE,
)
_SLOW_RE = re.compile(r"천천히|느리게|저속|\bslow(?:ly)?\b", re.IGNORECASE)
_NORMAL_RE = re.compile(r"보통\s*속도|평소\s*속도|정상\s*속도|\bnormal\b", re.IGNORECASE)
_FAST_RE = re.compile(r"빠르게|빨리|고속|\bfast(?:er)?\b", re.IGNORECASE)


def clamp_speed_raw(value):
    """Return an integer raw command in the public 0..250 range."""
    return max(MIN_SPEED_RAW, min(MAX_SPEED_RAW, int(round(value))))


def parse_speed_raw(text, current_raw=DEFAULT_SPEED_RAW):
    """Parse a common Korean/English speed request, or return ``None``.

    Explicit numbers win over relative and named modes.  This keeps commands
    such as "속도 100으로 줄여" deterministic at 100 instead of applying a step.
    """
    text = str(text or "").strip()
    if not text:
        return None

    for pattern in _EXPLICIT_SPEED_PATTERNS:
        match = pattern.search(text)
        if match:
            return clamp_speed_raw(int(match.group(1)))

    current_raw = clamp_speed_raw(current_raw)
    if _FASTER_RE.search(text):
        return clamp_speed_raw(current_raw + SPEED_STEP_RAW)
    if _SLOWER_RE.search(text):
        return clamp_speed_raw(current_raw - SPEED_STEP_RAW)
    if _SLOW_RE.search(text):
        return SLOW_SPEED_RAW
    if _NORMAL_RE.search(text):
        return NORMAL_SPEED_RAW
    if _FAST_RE.search(text):
        return FAST_SPEED_RAW
    return None


def raw_speed_to_mps(raw_speed, max_mps=5.0, motor_full_scale=255.0):
    """Map the public raw limit to the simulator's linear speed limit."""
    return clamp_speed_raw(raw_speed) / float(motor_full_scale) * float(max_mps)


def limit_twist_to_raw_speed(linear, angular, raw_speed, max_mps=5.0):
    """Clip a learned Twist while retaining its curvature."""
    limit = raw_speed_to_mps(raw_speed, max_mps=max_mps)
    clipped_linear = max(-limit, min(limit, float(linear)))
    if abs(linear) <= 1e-6:
        return clipped_linear, 0.0
    clipped_angular = float(angular) * abs(clipped_linear / float(linear))
    return clipped_linear, clipped_angular
