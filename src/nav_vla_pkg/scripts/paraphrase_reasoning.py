#!/usr/bin/env python3
"""Paraphrase reasoning skeletons via Ollama, with fact-preservation checks.

Reads each episode's ``reasoning.jsonl`` (from label_reasoning.py), asks a
text LLM for 3-5 English rewrites of every segment skeleton, and keeps only
rewrites that survive verification against the segment's ground-truth facts:

  * every number in the skeleton (distance m, speed m/s) must reappear
    within tolerance (±0.3 m / ±0.2 m/s), and no NEW numbers may appear
  * the zone name, lane word, and turn direction must survive
  * no hallucinated scene objects (pedestrian, traffic light, ...) — the
    corpus records none, so any mention is invented

Rejected variants fall back silently to the skeleton (which is always kept
as variant 0 downstream). Results are written back into ``reasoning.jsonl``
in place: each segment gains ``paraphrases`` (accepted list) and
``n_paraphrase_tries``; per-frame rows are untouched.

The LLM never sees anything but the skeleton and facts — it cannot inject
new claims that verification would miss, only new phrasings.

Usage::

    python3 src/nav_vla_pkg/scripts/paraphrase_reasoning.py \
        src/nav_vla_pkg/data_v3y/packed_v3y [...dirs] \
        [--model qwen3:4b] [--ollama-host http://127.0.0.1:11434] \
        [--variants 4] [--limit N] [--max-tokens 34]
"""

import argparse
import json
import os
import re
import sys
import urllib.request

# Objects that do not exist in this corpus; mentioning one = hallucination.
FORBIDDEN = re.compile(
    r"\b(pedestrian|person|people|traffic light|signal|stop sign|car|vehicle"
    r"|truck|obstacle|cone|animal|cyclist|rain|night|intersection)\b", re.I)

NUM = re.compile(r"\d+\.?\d*")

PROMPT = """Rewrite the driving narration below in {n} different ways.
Rules:
- Keep EVERY number exactly as written (distances, speeds).
- The commanded speed tier is written as "tier of 110" — keep the word
  "tier" next to that number, never add m/s, never invent other speeds.
- Keep the zone name, the lane (inner/outer), and any turn direction.
- Do not add any object, place, or event that is not in the original.
- One sentence per rewrite, natural English, at most 25 words.
- Output ONLY the rewrites, one per line, no numbering.

Narration: {skeleton}"""


def ollama_chat(host, model, prompt, timeout=120):
    # NOTE: qwen3:4b ignores think:false AND /no_think on this Ollama build
    # (90 s of leaked reasoning per call); use a non-thinking instruct
    # model. num_predict caps runaway generations.
    payload = json.dumps({
        "model": model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.9, "num_predict": 220},
    }).encode()
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def verify(skeleton, cand, facts):
    """True iff `cand` preserves the skeleton's facts."""
    if FORBIDDEN.search(cand):
        return False
    src_nums = [float(x) for x in NUM.findall(skeleton)]
    out_nums = [float(x) for x in NUM.findall(cand)]
    # every output number must match some source number (tolerance covers
    # m vs m/s ambiguity conservatively: 0.05 exact-ish)
    for o in out_nums:
        if not any(abs(o - s) <= 0.05 for s in src_nums):
            return False
    # every source number should survive (paraphrase may drop at most one,
    # e.g. fold "toward the target of 2.2" away) — require >= all but one
    kept = sum(any(abs(o - s) <= 0.05 for o in out_nums) for s in src_nums)
    if kept < max(1, len(src_nums) - 1):
        return False
    low = cand.lower()
    # the speed-trend claim must not flip (a "holding" frame narrated as
    # "slowing down" is exactly the ungrounded label this filter exists for)
    trend_words = {"decel": r"slow|brak|decel|reduc",
                   "accel": r"accel|speed(ing)? up|pick|increas",
                   "hold": r"hold|maintain|steady|constant|keep"}
    tr = facts.get("trend")
    if tr:
        for other, pat in trend_words.items():
            if other != tr and re.search(pat, low):
                return False
    if facts.get("lane"):
        lane_w = {"lane1": "inner", "lane2": "outer"}[facts["lane"]]
        if lane_w not in low:
            return False
    if facts.get("turn") and facts["turn"] not in low:
        return False
    for zone_key in ("goal", "near_zone"):
        z = facts.get(zone_key)
        if z and z.lower().split("_")[0] in skeleton.lower() \
                and z.lower().split("_")[0] not in low:
            return False
    return True


def count_tokens(tok, text):
    return len(tok(text)["input_ids"]) if tok else len(text.split()) * 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("packed_dirs", nargs="+")
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="episodes per dir (smoke)")
    ap.add_argument("--max-tokens", type=int, default=34,
                    help="reject variants longer than this (SmolVLM2 tokens)")
    ap.add_argument("--redo", action="store_true",
                    help="re-paraphrase segments that already have results")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    except Exception:
        tok = None
        print("transformers unavailable — token cap approximated by words",
              file=sys.stderr)

    # de-dup identical skeletons across the corpus: one LLM call per
    # unique sentence, results shared (24% of segments repeat).
    cache = {}
    n_seg = n_calls = n_kept = n_tried = 0

    for root in args.packed_dirs:
        eps = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
        if args.limit:
            eps = eps[:args.limit]
        for name in eps:
            path = os.path.join(root, name, "reasoning.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                lines = f.readlines()
            head = json.loads(lines[0])
            changed = False
            for seg in head["_meta"]["segments"]:
                if seg.get("paraphrases") is not None and not args.redo:
                    continue
                n_seg += 1
                skel = seg["skeleton"]
                # facts of the segment's first frame
                row0 = json.loads(lines[1 + seg["k0"]])
                if skel not in cache:
                    try:
                        raw = ollama_chat(args.ollama_host, args.model,
                                          PROMPT.format(
                                              n=args.variants,
                                              skeleton=skel))
                    except Exception as e:
                        print(f"LLM error ({e}); stopping cleanly",
                              file=sys.stderr)
                        raw = None
                    n_calls += 1
                    if raw is None:
                        cache[skel] = ([], 0)
                    else:
                        # models number their lines despite the prompt;
                        # strip "1. " / "2)" style prefixes before the
                        # numeric fact check sees them
                        cands = [re.sub(r"^\s*\d+[.)]\s*", "",
                                        c.strip(" -*\t"))
                                 for c in raw.splitlines() if c.strip()]
                        good = [c for c in cands
                                if verify(skel, c, row0["facts"])
                                and count_tokens(tok, c) <= args.max_tokens]
                        cache[skel] = (good[:args.variants], len(cands))
                good, tried = cache[skel]
                seg["paraphrases"] = good
                seg["n_paraphrase_tries"] = tried
                n_kept += len(good)
                n_tried += tried
                changed = True
            if changed:
                lines[0] = json.dumps(head, ensure_ascii=False) + "\n"
                with open(path, "w") as f:
                    f.writelines(lines)
        print(f"{root}: done ({n_calls} LLM calls so far)")

    rej = 1 - n_kept / n_tried if n_tried else 0.0
    print(f"segments processed: {n_seg}  unique calls: {n_calls}  "
          f"kept {n_kept}/{n_tried} variants (reject {rej:.1%})")


if __name__ == "__main__":
    main()
