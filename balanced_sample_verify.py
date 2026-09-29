"""
Step 6A — independent verification of the balanced three-class sample.

Reads the raw dataset and the sample file, independently recomputes the sampling
(seed 6418, fixed method), and asserts every invariant. No network or model calls.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Gift_Cards.jsonl.gz"
SAMPLE = ROOT / "balanced_sample_3class.json"
SEED = 6418
CLASS_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def fail(msg):
    raise SystemExit(f"VERIFY FAIL: {msg}")


def reference_label(rating: float) -> str:
    return "POSITIVE" if rating >= 4 else ("NEUTRAL" if rating == 3 else "NEGATIVE")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # --- source dataset ---
    checksum = sha256_of(DATA)
    records = []
    rating_dist = Counter()
    class_dist = Counter()
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append((i, rec))
            rating_dist[rec["rating"]] += 1
            class_dist[reference_label(rec["rating"])] += 1
    by_idx = {idx: rec for idx, rec in records}

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta, rows = sample["metadata"], sample["rows"]

    # checksum + count
    if meta["source_dataset_sha256"] != checksum:
        fail(f"checksum mismatch: saved {meta['source_dataset_sha256'][:16]} vs {checksum[:16]}")
    if meta["total_dataset_records"] != len(records):
        fail("dataset record count mismatch")

    # --- row-level invariants ---
    if len(rows) != 150:
        fail(f"rows != 150: {len(rows)}")
    sidx = [r["sample_index"] for r in rows]
    if len(set(sidx)) != 150 or sorted(sidx) != list(range(150)):
        fail("sample_index not unique 0..149")
    ridx = [r["source_row_index"] for r in rows]
    if len(set(ridx)) != 150:
        fail("source_row_index not unique")
    class_counts = Counter(r["reference_label"] for r in rows)
    if not all(class_counts[c] == 50 for c in CLASS_ORDER):
        fail(f"class counts != 50 each: {dict(class_counts)}")
    for r in rows:
        src = by_idx[r["source_row_index"]]
        if r["title"] != src["title"]:
            fail(f"title mismatch at sample {r['sample_index']} (source {r['source_row_index']})")
        if r["text"] != src["text"]:
            fail(f"text mismatch at sample {r['sample_index']}")
        if r["rating"] != src["rating"]:
            fail(f"rating mismatch at sample {r['sample_index']}")
        if r["reference_label"] != reference_label(src["rating"]):
            fail(f"reference rule violated at sample {r['sample_index']}")

    # --- rerun sampling with seed 6418 ---
    groups = {c: [] for c in CLASS_ORDER}
    for idx, rec in records:
        groups[reference_label(rec["rating"])].append(idx)
    rng = random.Random(SEED)
    selected = []
    for c in CLASS_ORDER:
        selected.extend(rng.sample(groups[c], 50))
    rng.shuffle(selected)
    if selected != [r["source_row_index"] for r in rows]:
        fail("rerun with seed 6418 produced different selected source indices/order")

    # --- metadata matches recomputed values ---
    if meta["seed"] != SEED:
        fail("seed mismatch")
    if meta["rating_distribution"] != {str(k): v for k, v in sorted(rating_dist.items())}:
        fail("rating distribution mismatch")
    if meta["three_class_distribution"] != {c: class_dist[c] for c in CLASS_ORDER}:
        fail("three-class distribution mismatch")
    sel_by_class = {c: sorted(r["source_row_index"] for r in rows
                              if r["reference_label"] == c) for c in CLASS_ORDER}
    if meta["selected_source_row_indices_by_class"] != sel_by_class:
        fail("selected indices by class mismatch")
    if meta["selected_class_counts"] != {c: class_counts[c] for c in CLASS_ORDER}:
        fail("selected class counts mismatch")

    print("BALANCED SAMPLE VERIFICATION PASSED (no network/model calls).")
    print(f"  rows=150  counts={dict(class_counts)}")
    print(f"  checksum match: {checksum[:16]}...")
    print(f"  rerun(seed={SEED}) == saved order: True")
    print(f"  dataset: {len(records)} records; rating dist {dict(sorted(rating_dist.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
