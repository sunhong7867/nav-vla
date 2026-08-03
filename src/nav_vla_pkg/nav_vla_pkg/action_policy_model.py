"""Small text-conditioned action policy used by nav-vla.

The model predicts a high-level plan:

    text + current_lane -> steps[{action, zone, lane}]

It intentionally stays small so it can be trained quickly on synthetic command
sentences and run locally inside the chat GUI.
"""

import json
import re

import torch
import torch.nn as nn


ACTIONS = [
    "drive_to_zone",
    "drive_direct",
    "change_lane",
    "keep_lane",
    "stop",
    "start",
    "none",
]
LANES = ["default", "lane1", "lane2"]
NULL_ZONE = "__null__"
MAX_STEPS = 2
UNK = "<unk>"
PAD = "<pad>"


TOKEN_RE = re.compile(r"[a-z0-9/()]+")


def tokenize(text):
    base = TOKEN_RE.findall(str(text or "").lower().replace("-", " ").replace("_", " "))
    tokens = []
    for token in base:
        tokens.append(token)
        tokens.extend(part for part in re.split(r"[/()]+", token) if part)
        match = re.fullmatch(r"([a-z]+)([0-9]+)", token)
        if match:
            tokens.extend(match.groups())
    return tokens


def canonical_plan(teacher):
    steps = teacher.get("steps") if isinstance(teacher, dict) else None
    if not isinstance(steps, list):
        steps = []
    normalized = []
    for step in steps[:MAX_STEPS]:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "none")
        zone = step.get("zone")
        lane = str(step.get("lane") or "default")
        if action not in ACTIONS:
            action = "none"
        if lane not in LANES:
            lane = "default"
        normalized.append({
            "action": action,
            "zone": None if zone is None else str(zone),
            "lane": lane,
        })
    if not normalized:
        normalized = [{"action": "none", "zone": None, "lane": "default"}]
    return {"steps": normalized}


def build_vocabs(samples):
    tokens = {PAD, UNK}
    zones = {NULL_ZONE}
    for sample in samples:
        current_lane = sample.get("current_lane", "lane2")
        tokens.update(tokenize(sample.get("text", "")))
        tokens.add(f"current_{current_lane}")
        for step in canonical_plan(sample.get("teacher", {}))["steps"]:
            if step["zone"] is not None:
                zones.add(step["zone"])
    vocab = {token: i for i, token in enumerate(sorted(tokens))}
    zone_vocab = {zone: i for i, zone in enumerate(sorted(zones))}
    return vocab, zone_vocab


def encode_text(text, current_lane, vocab):
    tokens = tokenize(text) + [f"current_{current_lane}"]
    ids = [vocab.get(token, vocab[UNK]) for token in tokens]
    return ids or [vocab[UNK]]


def plan_to_targets(teacher, zone_vocab):
    plan = canonical_plan(teacher)
    steps = plan["steps"]
    count = min(len(steps), MAX_STEPS)
    targets = {
        "count": count - 1,
        "actions": [],
        "zones": [],
        "lanes": [],
    }
    for i in range(MAX_STEPS):
        step = steps[i] if i < count else {"action": "none", "zone": None, "lane": "default"}
        targets["actions"].append(ACTIONS.index(step["action"]))
        targets["zones"].append(zone_vocab.get(step["zone"] or NULL_ZONE, zone_vocab[NULL_ZONE]))
        targets["lanes"].append(LANES.index(step["lane"]))
    return targets


def outputs_to_plan(outputs, zone_vocab):
    inv_zones = {v: k for k, v in zone_vocab.items()}
    count = int(outputs["count"].argmax(dim=-1).item()) + 1
    steps = []
    for i in range(count):
        action = ACTIONS[int(outputs["actions"][i].argmax(dim=-1).item())]
        zone_name = inv_zones[int(outputs["zones"][i].argmax(dim=-1).item())]
        lane = LANES[int(outputs["lanes"][i].argmax(dim=-1).item())]
        steps.append({
            "action": action,
            "zone": None if zone_name == NULL_ZONE else zone_name,
            "lane": lane,
        })
    return {"steps": steps, "reason": "action_policy"}


class ActionPolicyNet(nn.Module):
    def __init__(self, vocab_size, zone_count, emb_dim=64, hidden=128):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean")
        self.backbone = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.count_head = nn.Linear(hidden, MAX_STEPS)
        self.action_heads = nn.ModuleList(nn.Linear(hidden, len(ACTIONS)) for _ in range(MAX_STEPS))
        self.zone_heads = nn.ModuleList(nn.Linear(hidden, zone_count) for _ in range(MAX_STEPS))
        self.lane_heads = nn.ModuleList(nn.Linear(hidden, len(LANES)) for _ in range(MAX_STEPS))

    def forward(self, token_ids, offsets):
        x = self.embedding(token_ids, offsets)
        h = self.backbone(x)
        return {
            "count": self.count_head(h),
            "actions": [head(h) for head in self.action_heads],
            "zones": [head(h) for head in self.zone_heads],
            "lanes": [head(h) for head in self.lane_heads],
        }


class ActionPolicyPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.vocab = ckpt["vocab"]
        self.zone_vocab = ckpt["zone_vocab"]
        self.zones = [zone for zone in self.zone_vocab if zone != NULL_ZONE]
        self.model = ActionPolicyNet(
            len(self.vocab),
            len(self.zone_vocab),
            emb_dim=int(ckpt.get("emb_dim", 64)),
            hidden=int(ckpt.get("hidden", 128)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def predict(self, text, current_lane="lane2"):
        override = self._rule_override(text, current_lane)
        if override is not None:
            override["confidence"] = 1.0
            return override
        ids = encode_text(text, current_lane, self.vocab)
        token_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        offsets = torch.tensor([0], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(token_ids, offsets)
            probs = torch.softmax(outputs["count"], dim=-1)
            confidence = float(probs.max().cpu().item())
        plan = outputs_to_plan({k: v if k == "count" else [x for x in v] for k, v in outputs.items()}, self.zone_vocab)
        plan = self._sanitize_plan(plan, text, current_lane)
        plan["confidence"] = confidence
        return plan

    def predict_json(self, text, current_lane="lane2"):
        return json.dumps(self.predict(text, current_lane), ensure_ascii=False)

    def _rule_override(self, text, current_lane="lane2"):
        compact = self._plain_text(text)
        if not compact:
            return None
        if compact.startswith(("can ", "could ", "would ", "is ", "are ")):
            return {
                "steps": [{"action": "none", "zone": None, "lane": "default"}],
                "reason": "action_policy_safety_question",
            }
        if self._is_standalone_stop_command(compact):
            return {
                "steps": [{"action": "stop", "zone": None, "lane": "default"}],
                "reason": "action_policy_safety_stop",
            }
        if self._is_standalone_start_command(compact):
            return {
                "steps": [{"action": "start", "zone": None, "lane": "default"}],
                "reason": "action_policy_safety_start",
            }
        goal_markers = {
            " at ",
            " to ",
            " until ",
            " t1",
            " t2",
            " t3",
            " t4",
            " m1",
            " m2",
            " m3",
            " start",
            " crosswalk",
            " slot",
            " in",
            " out",
        }
        has_goal_marker = any(marker in f" {compact} " for marker in goal_markers)
        if not has_goal_marker and compact.startswith((
            "stop",
            "please stop",
            "pause",
            "please pause",
            "halt",
            "please halt",
            "brake",
            "please brake",
            "cancel",
            "hold position",
            "do not move",
        )):
            return {
                "steps": [{"action": "stop", "zone": None, "lane": "default"}],
                "reason": "action_policy_safety_stop",
            }
        if not has_goal_marker and compact.startswith((
            "start",
            "please start",
            "resume",
            "please resume",
            "continue",
            "please continue",
            "go ahead",
            "begin",
            "proceed",
            "keep going",
        )):
            return {
                "steps": [{"action": "start", "zone": None, "lane": "default"}],
                "reason": "action_policy_safety_start",
            }
        sequence = self._sequence_override(compact, text, current_lane)
        if sequence is not None:
            return sequence
        if not self._has_sequence_marker(compact):
            zone = self._detect_zone(text)
            if zone is not None and self._has_drive_intent(compact):
                if self._has_direct_intent(compact) or self._is_direct_only_zone(zone):
                    return {
                        "steps": [{"action": "drive_direct", "zone": zone, "lane": "default"}],
                        "reason": "action_policy_zone_guard_direct",
                    }
                lane = self._detect_lane(compact)
                if lane == "default" and self._has_change_lane_intent(compact):
                    lane = "lane1" if current_lane == "lane2" else "lane2"
                return {
                    "steps": [{
                        "action": "drive_to_zone",
                        "zone": zone,
                        "lane": lane,
                    }],
                    "reason": "action_policy_zone_guard_lane",
                }
        return None

    def _sequence_override(self, compact, text, current_lane):
        if not self._has_sequence_marker(compact):
            return None
        if not self._has_change_lane_intent(compact):
            return None
        zones = self._detect_zones_ordered(text)
        if not zones:
            return None

        opposite = "lane1" if current_lane == "lane2" else "lane2"
        change_pos = self._change_lane_pos(compact)
        if change_pos < 0:
            return None

        first_pos, first_zone = zones[0]
        second_zone = None
        for pos, zone in zones[1:]:
            if zone != first_zone and pos > change_pos:
                second_zone = zone
                break

        # "go M2 then change lane" -> go to M2, then change lanes there.
        if change_pos > first_pos:
            steps = [{"action": "drive_to_zone", "zone": first_zone, "lane": "default"}]
            if second_zone is None:
                steps.append({"action": "change_lane", "zone": None, "lane": opposite})
            else:
                steps.append({"action": "drive_to_zone", "zone": second_zone, "lane": opposite})
            return {"steps": steps, "reason": "action_policy_sequence_guard"}

        # "change lane at M2 then go T4" -> reach M2 first, then continue in the new lane.
        if " at " in f" {compact[:first_pos]} " or " after " in f" {compact[:first_pos]} ":
            steps = [{"action": "drive_to_zone", "zone": first_zone, "lane": "default"}]
            if second_zone is None:
                steps.append({"action": "change_lane", "zone": None, "lane": opposite})
            else:
                steps.append({"action": "drive_to_zone", "zone": second_zone, "lane": opposite})
            return {"steps": steps, "reason": "action_policy_sequence_guard"}

        # "change lane then go M2" can be merged into one lane-specific destination.
        return {
            "steps": [{"action": "drive_to_zone", "zone": first_zone, "lane": opposite}],
            "reason": "action_policy_sequence_guard",
        }

    def _sanitize_plan(self, plan, text, current_lane="lane2"):
        steps = []
        for raw in plan.get("steps", []):
            action = raw.get("action")
            zone = raw.get("zone")
            lane = raw.get("lane") if raw.get("lane") in LANES else "default"
            if action in {"start", "stop", "none"}:
                zone = None
                lane = "default"
            elif action in {"change_lane", "keep_lane"}:
                zone = None
                if lane == "default" and action == "change_lane":
                    lane = "lane1" if current_lane == "lane2" else "lane2"
            elif action in {"drive_to_zone", "drive_direct"}:
                if zone is None:
                    zone = self._detect_zone(text)
                if zone is None:
                    action = "none"
                    lane = "default"
                elif action == "drive_direct":
                    lane = "default"
            else:
                action = "none"
                zone = None
                lane = "default"
            steps.append({"action": action, "zone": zone, "lane": lane})
        if not steps:
            steps = [{"action": "none", "zone": None, "lane": "default"}]
        return {"steps": steps, "reason": plan.get("reason", "action_policy")}

    def _detect_zone(self, text):
        compact = self._compact(text)
        tokens = set(tokenize(text))
        candidates = []
        for zone in self.zones:
            for alias in self._zone_aliases(zone):
                alias_compact = self._compact(alias)
                if not alias_compact:
                    continue
                if len(alias_compact) <= 2:
                    if alias_compact in tokens:
                        candidates.append((len(alias_compact), zone))
                elif alias_compact in compact:
                    candidates.append((len(alias_compact), zone))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _detect_zones_ordered(self, text):
        plain = self._plain_text(text)
        found = []
        for zone in self.zones:
            best = None
            for alias in self._zone_aliases(zone):
                alias_plain = self._plain_text(alias)
                if not alias_plain:
                    continue
                match = re.search(rf"(?<!\w){re.escape(alias_plain)}(?!\w)", plain)
                if not match:
                    continue
                candidate = (match.start(), -len(alias_plain), zone)
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                found.append(best)
        found.sort()

        ordered = []
        seen = set()
        for pos, _neg_len, zone in found:
            if zone in seen:
                continue
            seen.add(zone)
            ordered.append((pos, zone))
        return ordered

    def _zone_aliases(self, zone):
        aliases = [zone]
        low = zone.lower()
        if zone == "Start":
            aliases += ["start", "start line", "starting line", "start zone"]
        elif zone == "crosswalk_stop":
            aliases += ["crosswalk_stop", "crosswalk stop", "crosswalk stop line", "crosswalk"]
        elif zone == "T1/M1":
            aliases += ["t1/m1", "t1 m1", "t1m1", "t1", "m1", "t1 line", "m1 line"]
        elif zone == "IN":
            aliases += ["in", "entrance", "in gate", "entry point"]
        elif low.startswith("out("):
            aliases += ["out", "exit", "out gate", "exit point"]
        elif low.startswith("slot"):
            number = zone[4:]
            aliases += [f"slot {number}", f"slot{number}", f"parking slot {number}", f"parking space {number}"]
        else:
            aliases += [low, f"{low} line", f"{low} point"]
        return aliases

    @staticmethod
    def _compact(text):
        return "".join(tokenize(text))

    @staticmethod
    def _plain_text(text):
        return " ".join(TOKEN_RE.findall(str(text or "").lower().replace("-", " ").replace("_", " ")))

    @staticmethod
    def _has_sequence_marker(compact):
        return any(marker in f" {compact} " for marker in (
            " first ",
            " then ",
            " next ",
            " after ",
            " before ",
            " and then ",
        ))

    @staticmethod
    def _has_drive_intent(compact):
        return compact.startswith((
            "go",
            "please go",
            "drive",
            "please drive",
            "head",
            "navigate",
            "move",
            "take",
            "follow",
            "continue",
            "stay",
            "stop at",
            "please stop at",
            "stop by",
            "please stop by",
        )) or any(
            word in f" {compact} "
            for word in (" go ", " drive ", " navigate ", " move ", " stop at ", " stop by ")
        )

    @staticmethod
    def _is_standalone_stop_command(compact):
        target_markers = (" at ", " to ", " until ", " t1", " t2", " t3", " t4", " m1", " m2", " m3",
                          " start line", " crosswalk", " slot", " in ", " out ")
        if any(marker in f" {compact} " for marker in target_markers):
            return False
        return compact in {
            "stop",
            "please stop",
            "stop now",
            "please stop now",
            "pause",
            "halt",
            "brake",
            "cancel",
            "hold position",
            "do not move",
        }

    @staticmethod
    def _is_standalone_start_command(compact):
        target_markers = (" at ", " to ", " until ", " t1", " t2", " t3", " t4", " m1", " m2", " m3",
                          " start line", " crosswalk", " slot", " in ", " out ", " line")
        if any(marker in f" {compact} " for marker in target_markers):
            return False
        if compact in {
            "go",
            "go now",
            "go ahead",
            "drive",
            "drive now",
            "start",
            "start now",
            "please start",
            "resume",
            "resume driving",
            "continue",
            "continue driving",
            "begin",
            "begin driving",
            "proceed",
            "keep going",
        }:
            return True
        return compact.startswith((
            "start drive",
            "start driving",
            "please start drive",
            "please start driving",
        ))

    @staticmethod
    def _has_direct_intent(compact):
        return any(phrase in compact for phrase in (
            "direct",
            "directly",
            "shortest",
            "ignore the lanes",
            "ignore lanes",
            "without following",
            "leave the lane",
            "cut across",
            "straight across",
        ))

    @staticmethod
    def _detect_lane(compact):
        joined = compact.replace(" ", "")
        if any(token in joined for token in ("lane1", "laneone", "firstlane", "innerlane", "leftlane")):
            return "lane1"
        if any(token in joined for token in ("lane2", "lanetwo", "secondlane", "outerlane", "rightlane")):
            return "lane2"
        return "default"

    @staticmethod
    def _has_change_lane_intent(compact):
        return any(phrase in compact for phrase in (
            "change lane",
            "change lanes",
            "switch lane",
            "switch lanes",
            "lane change",
        ))

    @staticmethod
    def _change_lane_pos(compact):
        positions = [
            compact.find(phrase)
            for phrase in (
                "change lane",
                "change lanes",
                "switch lane",
                "switch lanes",
                "lane change",
            )
        ]
        positions = [pos for pos in positions if pos >= 0]
        return min(positions) if positions else -1

    @staticmethod
    def _is_direct_only_zone(zone):
        return str(zone or "") == "IN" or str(zone or "").startswith("OUT(") or str(zone or "").startswith("Slot")
