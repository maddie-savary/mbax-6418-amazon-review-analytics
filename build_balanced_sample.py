"""
Step 6A — build the balanced three-class sample from the full Gift Cards dataset.

Deterministic sampling (documented):
  1. Read every dataset record in file order; assign zero-based source_row_index.
  2. Derive reference class from rating: >=4 POSITIVE, ==3 NEUTRAL, <=2 NEGATIVE.
  3. Group eligible records by reference class.
  4. random.Random(seed=6418).
  5. sample(group, 50) per class in fixed order: POSITIVE, NEUTRAL, NEGATIVE.
  6. Combine; apply one deterministic shuffle with the SAME seeded generator.
  7. Assign sample_index 0..149 after the shuffle.

Rating/reference are LOCAL evaluation fields only and never enter a model request.
The raw full dataset is never written into this file.
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
OUT = ROOT / "balanced_sample_3class.json"
SEED = 6418
PER_CLASS = 50
CLASS_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def reference_label(rating: float) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    checksum = sha256_of(DATA)

    records = []          # (source_row_index, dict)
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

    total_rows = len(records)
    assert total_rows == 152410, f"expected 152410 records, got {total_rows}"

    groups = {c: [] for c in CLASS_ORDER}
    for idx, rec in records:
        groups[reference_label(rec["rating"])].append(idx)

    for c in CLASS_ORDER:
        assert len(groups[c]) >= PER_CLASS, f"class {c} too small: {len(groups[c])}"

    rng = random.Random(SEED)  # fixed seed
    selected = []
    for c in CLASS_ORDER:      # fixed class order
        chosen = rng.sample(groups[c], PER_CLASS)
        selected.extend(chosen)
    rng.shuffle(selected)      # one deterministic shuffle, same generator
    assert len(selected) == 150 and len(set(selected)) == 150

    by_idx = {idx: rec for idx, rec in records}
    rows = []
    for sample_index, idx in enumerate(selected):  # sample_index 0..149 after shuffle
        rec = by_idx[idx]
        rating = rec["rating"]
        rows.append({
            "sample_index": sample_index,
            "source_row_index": idx,
            "title": rec["title"],
            "text": rec["text"],
            "rating": rating,
            "reference_label": reference_label(rating),
        })

    # metadata
    selected_counts = Counter(r["reference_label"] for r in rows)
    sel_by_class = {c: sorted(r["source_row_index"] for r in rows
                              if r["reference_label"] == c) for c in CLASS_ORDER}

    meta = {
        "source_dataset_path": "data/Gift_Cards.jsonl.gz",
        "source_dataset_sha256": checksum,
        "seed": SEED,
        "sampling_algorithm": (
            "read all records in file order -> zero-based source_row_index -> "
            "reference class from rating (>=4 POSITIVE, ==3 NEUTRAL, <=2 NEGATIVE) -> "
            "group by class -> random.Random(6418).sample(group,50) per class in fixed "
            "order POSITIVE, NEUTRAL, NEGATIVE -> combine -> one rng.shuffle -> "
            "assign sample_index 0..149"),
        "fixed_class_order": CLASS_ORDER,
        "total_dataset_records": total_rows,
        "rating_distribution": dict(sorted(rating_dist.items())),
        "three_class_distribution": {c: class_dist[c] for c in CLASS_ORDER},
        "selected_class_counts": {c: selected_counts[c] for c in CLASS_ORDER},
        "selected_source_row_indices_by_class": sel_by_class,
        "data_handling": {
            "rating": "local evaluation field only; never sent to a model",
            "reference_label": "local evaluation field only; never sent to a model",
        },
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "rows": rows}, f, indent=2)
    print(f"wrote {OUT.name}")
    print(f"  total={len(rows)} counts={dict(selected_counts)}")
    print(f"  checksum={checksum[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
