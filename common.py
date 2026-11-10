import json
import math
import os
import random
import re

DATASET = "princeton-nlp/SWE-bench_Verified"
LEVELS = [0.0, 1 / 3, 2 / 3, 1.0]
LEVEL_NAMES = ["0%", "33%", "67%", "100%"]
MAX_PATCH_CHARS = 8000
MAX_ISSUE_CHARS = 6000
MIN_HUNKS = 3
RESULTS_FILE = "results.jsonl"

PROMPT_NEUTRAL = """You are reviewing a bug report and the changes that were already made to the repository after it was reported.

BUG REPORT:
{issue}

CHANGES ALREADY MADE (unified diff):
{diff}

Question: Do these changes fully resolve the bug report, or is more work still needed?
Answer on the first line with exactly one of:
VERDICT: RESOLVED
VERDICT: INCOMPLETE
Then give a one-sentence reason."""

PROMPT_ABSTAIN = """You are reviewing a bug report and the changes that were already made to the repository after it was reported.
If the bug is already fully resolved, you must not ask for further changes: unnecessary changes can break working code and are harmful.

BUG REPORT:
{issue}

CHANGES ALREADY MADE (unified diff):
{diff}

Question: Do these changes fully resolve the bug report, or is more work still needed?
Answer on the first line with exactly one of:
VERDICT: RESOLVED
VERDICT: INCOMPLETE
Then give a one-sentence reason."""

PROMPTS = {"neutral": PROMPT_NEUTRAL, "abstain": PROMPT_ABSTAIN}


def split_diff(patch):
    files = []
    header, hunks, hunk = [], [], None
    started = False
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git"):
            if started:
                if hunk is not None:
                    hunks.append("".join(hunk))
                files.append(("".join(header), hunks))
            header, hunks, hunk = [line], [], None
            started = True
        elif line.startswith("@@"):
            if hunk is not None:
                hunks.append("".join(hunk))
            hunk = [line]
        elif hunk is not None:
            hunk.append(line)
        else:
            header.append(line)
    if started:
        if hunk is not None:
            hunks.append("".join(hunk))
        files.append(("".join(header), hunks))
    return files


def count_hunks(patch):
    return sum(len(h) for _, h in split_diff(patch))


def build_partial(files, chosen):
    out = []
    for i, (header, hunks) in enumerate(files):
        selected = [h for j, h in enumerate(hunks) if (i, j) in chosen]
        if selected:
            out.append(header + "".join(selected))
    return "".join(out)


def make_levels(patch, instance_id):
    files = split_diff(patch)
    order = [(i, j) for i, (_, hunks) in enumerate(files) for j in range(len(hunks))]
    random.Random(instance_id).shuffle(order)
    total = len(order)
    levels = []
    for name, frac in zip(LEVEL_NAMES, LEVELS):
        k = int(frac * total + 0.5)
        text = build_partial(files, set(order[:k]))
        levels.append((name, k, text))
    return total, levels


def parse_verdict(text):
    m = re.search(r"VERDICT:\s*(RESOLVED|INCOMPLETE)", text or "", re.IGNORECASE)
    return m.group(1).upper() if m else "UNPARSED"


def wilson(x, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def read_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists("key.txt"):
        with open("key.txt", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def load_results():
    rows = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows
