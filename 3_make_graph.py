import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import LEVEL_NAMES, PROMPTS, load_results, wilson


def exact_paired_p(b, c):
    """Two-sided exact McNemar test (binomial) for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


rows = load_results()
if not rows:
    print("results.jsonl is empty or missing. Run 2_run_pilot.py first.")
    raise SystemExit(1)

models = sorted({r.get("model", "unknown") for r in rows})

for model in models:
    mrows = [r for r in rows if r.get("model", "unknown") == model]
    n_tasks = len({r["instance_id"] for r in mrows})
    print(f"\n=== Model: {model} | tasks: {n_tasks} ===")
    print("Share of 'RESOLVED' verdicts (the model says the fix is complete).")
    print("Correct answer: INCOMPLETE at 0%, 33%, 67%; RESOLVED at 100% "
          "(partial states may occasionally already suffice).\n")
    print(f"{'prompt':<9} {'level':<6} {'n':<4} {'RESOLVED':<9} 95% CI")

    plt.figure(figsize=(7, 4.5))
    for prompt_id in PROMPTS:
        xs, ps, lo, hi = [], [], [], []
        for idx, level in enumerate(LEVEL_NAMES):
            sel = [
                r for r in mrows
                if r["prompt_id"] == prompt_id and r["level"] == level
                and r["verdict"] != "UNPARSED"
            ]
            n = len(sel)
            x = sum(r["verdict"] == "RESOLVED" for r in sel)
            p, l, h = wilson(x, n)
            print(f"{prompt_id:<9} {level:<6} {n:<4} {p:<9.2f} [{l:.2f}, {h:.2f}]")
            if n:
                xs.append(idx)
                ps.append(p)
                lo.append(p - l)
                hi.append(h - p)
        if xs:
            plt.errorbar(xs, ps, yerr=[lo, hi], marker="o", capsize=4, label=prompt_id)

    print("\nPaired comparison, same tasks (neutral vs abstain prompt):")
    print(f"{'level':<6} {'neutral->RESOLVED only':<24} {'abstain->RESOLVED only':<24} exact p")
    for level in LEVEL_NAMES:
        by_task = {}
        for r in mrows:
            if r["level"] == level and r["verdict"] != "UNPARSED":
                by_task.setdefault(r["instance_id"], {})[r["prompt_id"]] = r["verdict"]
        b = sum(1 for v in by_task.values()
                if v.get("neutral") == "RESOLVED" and v.get("abstain") == "INCOMPLETE")
        c = sum(1 for v in by_task.values()
                if v.get("neutral") == "INCOMPLETE" and v.get("abstain") == "RESOLVED")
        print(f"{level:<6} {b:<24} {c:<24} {exact_paired_p(b, c):.3f}")

    plt.xticks(range(len(LEVEL_NAMES)), LEVEL_NAMES)
    plt.xlabel("Fraction of the real fix already applied to the code")
    plt.ylabel("Share of 'RESOLVED' verdicts")
    plt.title(f"{model}, {n_tasks} SWE-bench Verified tasks")
    plt.ylim(-0.05, 1.05)
    plt.legend(title="prompt")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = "curve.png" if len(models) == 1 else f"curve_{model}.png"
    plt.savefig(fname, dpi=150)
    print(f"\nFigure saved to {fname}")

unparsed = sum(r["verdict"] == "UNPARSED" for r in rows)
if unparsed:
    print(f"\nNote: {unparsed} responses could not be parsed (see the 'raw' field in results.jsonl).")
