from datasets import load_dataset

from common import DATASET, MAX_PATCH_CHARS, MIN_HUNKS, count_hunks

print("Loading the dataset (the first run may take a minute or two)...")
ds = load_dataset(DATASET, split="test")

counts = [count_hunks(x["patch"]) for x in ds]
short = [c for x, c in zip(ds, counts) if len(x["patch"]) <= MAX_PATCH_CHARS]

print()
print("Total tasks:", len(ds))
print("Tasks whose gold patch has 2+ hunks:", sum(c >= 2 for c in counts))
print("Tasks whose gold patch has 3+ hunks:", sum(c >= MIN_HUNKS for c in counts))
print(
    "3+ hunks and patch <= %d characters (usable for the pilot):" % MAX_PATCH_CHARS,
    sum(c >= MIN_HUNKS for c in short),
)
