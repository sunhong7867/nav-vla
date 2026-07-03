"""Generate high-level English command/action samples for nav-vla.

This creates a JSONL dataset for training a high-level action policy:

    text + current_lane -> ordered action steps

It does not run the simulator. It only produces diverse English instruction
examples with teacher JSON labels that match chat_gui_node's action schema.

Usage:
    ros2 run nav_vla action_sentence_generator
    ros2 run nav_vla action_sentence_generator -- --count 600
    python3 src/nav_vla/nav_vla/action_sentence_generator.py --count 600
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import yaml


DEFAULT_MAP_PATH = os.path.expanduser(
    "~/ROS2_project/nav-vla/src/nav_vla/config/zone_map.yaml"
)
DEFAULT_OUT_DIR = os.path.expanduser("~/ROS2_project/nav-vla/src/nav_vla/data_actions")


LANES = ("lane1", "lane2")


def opposite_lane(lane):
    return "lane1" if lane == "lane2" else "lane2"


def load_zones(path):
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return list((data.get("zones") or {}).keys())


def is_direct_only_zone(zone):
    return zone == "IN" or zone.startswith("OUT(") or zone.startswith("Slot")


def is_lane_zone(zone):
    return not is_direct_only_zone(zone)


def zone_phrase(zone):
    return random.choice(zone_phrases(zone))


def zone_phrases(zone):
    if zone == "Start":
        return ["Start", "the start line", "the starting line", "the start zone"]
    if zone == "crosswalk_stop":
        return [
            "crosswalk_stop",
            "crosswalk",
            "the crosswalk",
            "the crosswalk stop",
            "the crosswalk stop line",
        ]
    if zone == "T1/M1":
        return ["T1/M1", "T1", "M1", "T1 M1", "the T1 line", "the M1 line"]
    if zone == "IN":
        return ["IN", "the entrance", "the in gate", "the entry point"]
    if zone.startswith("OUT("):
        return [zone, "OUT", "the exit", "the out gate", "the exit point"]
    if zone.startswith("Slot"):
        number = zone[4:]
        return [zone, f"slot {number}", f"parking slot {number}", f"parking space {number}"]
    return [zone, f"the {zone} line", f"{zone} point"]


def compact_text(text):
    return re.sub(r"\s+", " ", text.strip())


def action_sample(text, steps, current_lane="lane2", category="command"):
    return {
        "text": compact_text(text),
        "current_lane": current_lane,
        "teacher": {
            "steps": steps,
            "reason": category,
        },
        "category": category,
    }


def drive_step(zone, lane="default"):
    return {"action": "drive_to_zone", "zone": zone, "lane": lane}


def direct_step(zone):
    return {"action": "drive_direct", "zone": zone, "lane": "default"}


def change_step(lane):
    return {"action": "change_lane", "zone": None, "lane": lane}


def keep_step(lane):
    return {"action": "keep_lane", "zone": None, "lane": lane}


def stop_step():
    return {"action": "stop", "zone": None, "lane": "default"}


def start_step():
    return {"action": "start", "zone": None, "lane": "default"}


def generate_lane_goal_samples(zones):
    samples = []
    templates_explicit = [
        "go to {place} through {lane_text}",
        "drive to {place} in {lane_text}",
        "follow {lane_text} to {place}",
        "take {lane_text} until {place}",
        "stay in {lane_text} and stop at {place}",
        "move along {lane_text} to {place}",
        "head for {place} using {lane_text}",
        "continue on {lane_text} until you reach {place}",
        "drive along {lane_text} and stop at {place}",
        "navigate to {place} while keeping {lane_text}",
    ]
    lane_words = {
        "lane1": ["lane 1", "lane one", "the first lane", "the inner lane", "the left lane"],
        "lane2": ["lane 2", "lane two", "the second lane", "the outer lane", "the right lane"],
    }
    templates_default = [
        "go {place} lane",
        "go {place} through lane",
        "go {place} by lane",
        "drive {place} lane",
        "drive {place} through lane",
        "take the lane to {place}",
        "follow lane to {place}",
        "go to {place} by lane",
        "drive to {place} along the lane",
        "follow the current lane to {place}",
        "stay in the current lane until {place}",
        "go to {place} without changing lanes",
        "continue in this lane and stop at {place}",
        "take me to {place} on the current lane",
        "drive along the lane until {place}",
    ]
    for zone in zones:
        if not is_lane_zone(zone):
            continue
        for place in zone_phrases(zone):
            for lane in LANES:
                for template in templates_explicit:
                    lane_text = random.choice(lane_words[lane])
                    samples.append(action_sample(
                        template.format(place=place, lane_text=lane_text),
                        [drive_step(zone, lane)],
                        current_lane=random.choice(LANES),
                        category="lane_goal_explicit",
                    ))
            for current_lane in LANES:
                for template in templates_default:
                    samples.append(action_sample(
                        template.format(place=place),
                        [drive_step(zone, "default")],
                        current_lane=current_lane,
                        category="lane_goal_default",
                    ))
    return samples


def generate_direct_samples(zones):
    samples = []
    templates = [
        "go directly to {place}",
        "drive straight to {place}",
        "ignore the lanes and go to {place}",
        "take the shortest path to {place}",
        "drive directly toward {place}",
        "cut across to {place}",
        "go to {place} without following the lane",
        "leave the lane and head to {place}",
        "take a direct route to {place}",
        "go straight across to {place}",
    ]
    for zone in zones:
        for place in zone_phrases(zone):
            for template in templates:
                samples.append(action_sample(
                    template.format(place=place),
                    [direct_step(zone)],
                    current_lane=random.choice(LANES),
                    category="direct_goal",
                ))
    return samples


def generate_direct_only_guard_samples(zones):
    samples = []
    templates = [
        "go to {place}",
        "drive to {place}",
        "take me to {place}",
        "head to {place}",
        "move to {place}",
        "park at {place}",
    ]
    for zone in zones:
        if not is_direct_only_zone(zone):
            continue
        for place in zone_phrases(zone):
            for template in templates:
                samples.append(action_sample(
                    template.format(place=place),
                    [direct_step(zone)],
                    current_lane=random.choice(LANES),
                    category="direct_only_zone",
                ))
    return samples


def generate_lane_change_samples():
    samples = []
    explicit_templates = [
        "change to {lane_text}",
        "switch to {lane_text}",
        "move into {lane_text}",
        "start driving in {lane_text}",
        "keep {lane_text}",
        "follow {lane_text}",
        "stay on {lane_text}",
    ]
    lane_words = {
        "lane1": ["lane 1", "the first lane", "the inner lane", "the left lane"],
        "lane2": ["lane 2", "the second lane", "the outer lane", "the right lane"],
    }
    for lane in LANES:
        for template in explicit_templates:
            text = template.format(lane_text=random.choice(lane_words[lane]))
            action = keep_step(lane) if template.startswith(("keep", "follow", "stay")) else change_step(lane)
            samples.append(action_sample(
                text,
                [action],
                current_lane=opposite_lane(lane),
                category="lane_change_explicit",
            ))
    opposite_templates = [
        "change lanes",
        "switch lanes",
        "move to the other lane",
        "change to the opposite lane",
        "start a lane change",
    ]
    for current_lane in LANES:
        for template in opposite_templates:
            samples.append(action_sample(
                template,
                [change_step(opposite_lane(current_lane))],
                current_lane=current_lane,
                category="lane_change_opposite",
            ))
    return samples


def generate_start_stop_samples():
    samples = []
    stop_texts = [
        "stop",
        "stop the car",
        "stop driving",
        "please stop",
        "stop now",
        "make the car stop",
        "pause driving",
        "cancel the current drive",
        "hold position",
        "come to a stop",
        "brake now",
        "do not move",
    ]
    stop_verbs = ["stop", "pause", "halt", "brake", "cancel"]
    stop_objects = ["the car", "driving", "the vehicle", "the current motion", "right now", "here"]
    for verb in stop_verbs:
        for obj in stop_objects:
            stop_texts.append(f"{verb} {obj}")
            stop_texts.append(f"please {verb} {obj}")
    for text in stop_texts:
        for current_lane in LANES:
            samples.append(action_sample(
                text, [stop_step()], current_lane=current_lane, category="stop"
            ))

    start_texts = [
        "start",
        "start driving",
        "please start",
        "start the car",
        "resume driving",
        "continue driving",
        "go ahead",
        "begin moving",
        "keep going",
        "proceed",
    ]
    start_verbs = ["start", "resume", "continue", "proceed", "begin"]
    start_objects = ["driving", "the car", "the vehicle", "moving", "forward", "now"]
    for verb in start_verbs:
        for obj in start_objects:
            start_texts.append(f"{verb} {obj}")
            start_texts.append(f"please {verb} {obj}")
    for text in start_texts:
        for current_lane in LANES:
            samples.append(action_sample(
                text, [start_step()], current_lane=current_lane, category="start"
            ))
    return samples


def generate_multi_step_samples(zones):
    samples = []
    lane_zones = [zone for zone in zones if is_lane_zone(zone)]
    direct_zones = [zone for zone in zones if is_direct_only_zone(zone)]
    sequence_templates = [
        "go to {first} first, then switch to {lane_text} and continue to {second}",
        "drive to {first}, then take {lane_text} to {second}",
        "reach {first} first, change into {lane_text}, and go to {second}",
        "go to {first} before moving to {second} through {lane_text}",
    ]
    change_templates = [
        "drive to {first}, change lanes there, then go to {second}",
        "go to {first}, switch lanes at that point, then continue to {second}",
        "reach {first}, change to the other lane, then go to {second}",
    ]
    lane_words = {"lane1": "lane 1", "lane2": "lane 2"}
    for i, first in enumerate(lane_zones):
        for j, second in enumerate(lane_zones):
            if first == second:
                continue
            if (i + j) % 2 != 0:
                continue
            lane = LANES[(i + j) % 2]
            current_lane = opposite_lane(lane)
            first_text = zone_phrase(first)
            second_text = zone_phrase(second)
            for template in sequence_templates:
                samples.append(action_sample(
                    template.format(
                        first=first_text,
                        second=second_text,
                        lane_text=lane_words[lane],
                    ),
                    [drive_step(first, "default"), drive_step(second, lane)],
                    current_lane=current_lane,
                    category="waypoint_then_lane_goal",
                ))
            for template in change_templates:
                samples.append(action_sample(
                    template.format(first=first_text, second=second_text),
                    [drive_step(first, "default"), drive_step(second, opposite_lane(current_lane))],
                    current_lane=current_lane,
                    category="waypoint_then_lane_goal",
                ))
    for i, zone in enumerate(direct_zones):
        before = lane_zones[i % len(lane_zones)]
        samples.append(action_sample(
            f"go to {zone_phrase(before)} first, then go directly to {zone_phrase(zone)}",
            [drive_step(before, "default"), direct_step(zone)],
            current_lane=random.choice(LANES),
            category="lane_waypoint_then_direct",
        ))
    return samples


def generate_question_none_samples(zones):
    samples = []
    templates = [
        "can you stop at {place}?",
        "is it possible to drive to {place}?",
        "could the robot reach {place} someday?",
        "can this car go to {place}?",
        "would it be able to stop at {place}?",
    ]
    for zone in zones:
        for place in zone_phrases(zone):
            for template in templates:
                samples.append(action_sample(
                    template.format(place=place),
                    [{"action": "none", "zone": None, "lane": "default"}],
                    current_lane=random.choice(LANES),
                    category="question_none",
                ))
    return samples


def unique_samples(samples):
    seen = set()
    unique = []
    for sample in samples:
        key = (sample["text"].lower(), sample["current_lane"], json.dumps(sample["teacher"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        sample["id"] = f"cmd_{len(unique):05d}"
        unique.append(sample)
    return unique


def build_samples(zones, count=None, seed=7):
    random.seed(seed)
    samples = []
    samples += generate_lane_goal_samples(zones)
    samples += generate_direct_samples(zones)
    samples += generate_direct_only_guard_samples(zones)
    samples += generate_lane_change_samples()
    samples += generate_start_stop_samples()
    samples += generate_multi_step_samples(zones)
    samples += generate_question_none_samples(zones)
    samples = unique_samples(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)
    if count is not None and count > 0:
        samples = samples[: min(count, len(samples))]
    for i, sample in enumerate(samples):
        sample["id"] = f"cmd_{i:05d}"
    return samples


def write_dataset(samples, out_dir):
    out_path = Path(out_dir).expanduser()
    if out_path.name == "data_actions":
        out_path = out_path / time.strftime("session_%Y%m%d_%H%M%S")
    out_path.mkdir(parents=True, exist_ok=True)
    samples_path = out_path / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    summary = {
        "samples": len(samples),
        "path": str(samples_path),
        "categories": {},
    }
    for sample in samples:
        summary["categories"][sample["category"]] = summary["categories"].get(sample["category"], 0) + 1
    with (out_path / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return samples_path, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default=DEFAULT_MAP_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--count", type=int, default=0, help="0 means write all generated samples.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    zones = load_zones(args.map)
    samples = build_samples(zones, count=args.count or None, seed=args.seed)
    samples_path, summary = write_dataset(samples, args.out)
    print(f"wrote {summary['samples']} samples -> {samples_path}")
    for category, n in sorted(summary["categories"].items()):
        print(f"  {category}: {n}")


if __name__ == "__main__":
    main()
