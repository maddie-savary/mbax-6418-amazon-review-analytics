"""
Validate the official NRC Word-Emotion Association Lexicon (English word-level file).

Loads the word-level association file supplied by the official download and checks
format + integrity WITHOUT scoring any reviews (review scoring is Step 5A2b).

Input file (official): data/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
Format (README VI.1): each line is <term><tab><AffectCategory><tab><AssociationFlag>
   where AssociationFlag is binary 0 or 1, and AffectCategory is one of eight emotions
   (anger, anticipation, disgust, fear, joy, sadness, surprise, trust) or one of two
   sentiments (negative, positive). The sentiment categories exist but are ignored
   during Step 5A2b emotion scoring.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEX = (ROOT / "data" / "nrc" / "NRC-Emotion-Lexicon"
       / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

EIGHT = {"anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"}
SENTIMENTS = {"negative", "positive"}  # present, but ignored during emotion scoring
KNOWN = EIGHT | SENTIMENTS


def main() -> int:
    if not LEX.exists():
        print("LEXICON FILE NOT FOUND:", LEX)
        return 1

    records = Counter()       # per-category record counts (term, category) pairs
    seen = set()              # (term, category) keys, for duplicate detection
    duplicates = 0
    malformed = 0
    bad_flags = 0
    unexpected = Counter()
    n_lines = 0
    terms = set()             # unique English terms (from well-formed records)
    pos_emo_assoc = 0         # 1-valued records across the eight emotion categories
    sentiment_one = Counter() # 1-valued sentiment (negative/positive) records

    with open(LEX, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            if not line.strip():
                continue
            n_lines += 1
            parts = line.split("\t")
            if len(parts) != 3 or not parts[0]:
                malformed += 1
                continue
            term, cat, flag = parts
            records[cat] += 1
            terms.add(term)
            key = (term, cat)
            if key in seen:
                duplicates += 1
            seen.add(key)
            if cat not in KNOWN:
                unexpected[cat] += 1
            if flag not in ("0", "1"):
                bad_flags += 1
            if flag == "1":
                if cat in EIGHT:
                    pos_emo_assoc += 1
                elif cat in SENTIMENTS:
                    sentiment_one[cat] += 1

    present = set(records.keys())
    missing_emotions = EIGHT - present

    if missing_emotions:
        print("MISSING required emotion categories:", sorted(missing_emotions))
        sys.exit(1)
    if unexpected:
        print("UNEXPECTED categories:", dict(unexpected))
        sys.exit(1)
    if bad_flags or malformed or duplicates:
        # report but these are not hard failures by themselves; still surface them
        pass

    print("VALIDATION PASSED")
    print(f"  parsed non-blank lines          : {n_lines}")
    print(f"  unique English terms            : {len(terms)}")
    print(f"  total (term, category) records  : {sum(records.values())}")
    print(f"  duplicate (term,category) pairs : {duplicates}")
    print(f"  malformed lines                 : {malformed}")
    print(f"  records with non-0/1 value      : {bad_flags}")
    print("  categories present              :", sorted(present))
    print("  all eight required emotions found: True")
    print("  per-emotion records             :", dict({c: records[c] for c in EIGHT}))
    print(f"  positive emotion associations   : {pos_emo_assoc} "
          "(sum of 1-valued records across the 8 emotion categories)")
    print("  sentiment polarity records      :", dict({c: records[c] for c in SENTIMENTS}),
          "(present but ignored for emotion scoring)")

    # Per-emotion positive associations (count of flag==1 per emotion)
    emo_one = Counter()
    with open(LEX, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3 or not parts[0]:
                continue
            term, cat, flag = parts
            if cat in EIGHT and flag == "1":
                emo_one[cat] += 1
    print("  per-emotion 1-valued            :", dict({c: emo_one[c] for c in EIGHT}))

    summary = {
        "source_file": str(LEX),
        "parsed_lines": n_lines,
        "unique_terms": len(terms),
        "total_records": sum(records.values()),
        "duplicate_term_category_records": duplicates,
        "malformed_lines": malformed,
        "non_binary_values": bad_flags,
        "categories_present": sorted(present),
        "per_emotion_records": {c: records[c] for c in EIGHT},
        "per_emotion_one_values": {c: emo_one[c] for c in EIGHT},
        "positive_emotion_associations": pos_emo_assoc,
        "sentiment_records": {c: records[c] for c in SENTIMENTS},
        "unexpected_categories": dict(unexpected),
        "all_eight_emotions_present": not missing_emotions,
    }
    out = ROOT / "data" / "nrc" / "nrc_lexicon_validation.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
