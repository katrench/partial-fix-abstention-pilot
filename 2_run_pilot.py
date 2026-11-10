import argparse
import json
import random
import sys
import time

from datasets import load_dataset

from common import (
    DATASET,
    MAX_ISSUE_CHARS,
    MAX_PATCH_CHARS,
    MIN_HUNKS,
    PROMPTS,
    RESULTS_FILE,
    count_hunks,
    load_results,
    make_levels,
    parse_verdict,
    read_key,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=20, help="number of tasks to sample")
parser.add_argument("--model", default="gemini-3.5-flash-lite", help="Gemini model name")
parser.add_argument("--sleep", type=float, default=6.0, help="pause between requests, seconds")
parser.add_argument("--dry-run", action="store_true", help="print one example prompt and exit")
args = parser.parse_args()

print("Loading the dataset...")
ds = load_dataset(DATASET, split="test")

candidates = [
    x
    for x in ds
    if len(x["patch"]) <= MAX_PATCH_CHARS and count_hunks(x["patch"]) >= MIN_HUNKS
]
candidates.sort(key=lambda x: x["instance_id"])
random.Random(0).shuffle(candidates)
tasks = candidates[: args.n]

if len(tasks) < args.n:
    print(f"Only {len(tasks)} suitable tasks found; using all of them.")

jobs = []
for t in tasks:
    total, levels = make_levels(t["patch"], t["instance_id"])
    issue = t["problem_statement"][:MAX_ISSUE_CHARS]
    for level_name, k, diff_text in levels:
        diff = diff_text if diff_text.strip() else "(no changes have been made yet)"
        for prompt_id, template in PROMPTS.items():
            jobs.append(
                {
                    "instance_id": t["instance_id"],
                    "repo": t["repo"],
                    "hunks_total": total,
                    "level": level_name,
                    "k": k,
                    "prompt_id": prompt_id,
                    "prompt": template.format(issue=issue, diff=diff),
                }
            )

print(f"Tasks: {len(tasks)}, total requests: {len(jobs)}")

if args.dry_run:
    example = jobs[len(jobs) // 2]
    print("\n===== EXAMPLE PROMPT (nothing was sent) =====")
    print(
        f"Task: {example['instance_id']}, level: {example['level']}, "
        f"hunks {example['k']} of {example['hunks_total']}, prompt: {example['prompt_id']}\n"
    )
    print(example["prompt"][:3000])
    sys.exit(0)

key = read_key()
if not key:
    print("No API key found. Put it in key.txt (one line) or in the GEMINI_API_KEY variable.")
    sys.exit(1)

from google import genai
from google.genai import types

client = genai.Client(api_key=key)
config = types.GenerateContentConfig(temperature=0.0)

done = {
    (r["instance_id"], r["level"], r["prompt_id"], r.get("model"))
    for r in load_results()
}
todo = [
    j for j in jobs
    if (j["instance_id"], j["level"], j["prompt_id"], args.model) not in done
]
print(f"Already done: {len(jobs) - len(todo)}, remaining: {len(todo)}")


def ask(prompt):
    wait = 10
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=args.model, contents=prompt, config=config
            )
            return resp.text or ""
        except Exception as e:
            print(f"  error ({type(e).__name__}): {str(e)[:150]}")
            print(f"  waiting {wait}s and retrying ({attempt + 1}/6)")
            time.sleep(wait)
            wait *= 2
    return None


with open(RESULTS_FILE, "a", encoding="utf-8") as out:
    for i, job in enumerate(todo, 1):
        raw = ask(job["prompt"])
        if raw is None:
            print("Could not get a response. Stopping; progress is saved, rerun later.")
            break
        verdict = parse_verdict(raw)
        row = {k: v for k, v in job.items() if k != "prompt"}
        row.update({"verdict": verdict, "raw": raw[:500], "model": args.model})
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[{i}/{len(todo)}] {job['instance_id']} {job['level']} "
              f"{job['prompt_id']} -> {verdict}")
        time.sleep(args.sleep)

print("Done. Now run: python 3_make_graph.py")
