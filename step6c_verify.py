"""
Step 6C — independent verification of the balanced three-class dashboard HTML.

Reads the generated dashboard.html and the locked balanced_3class_results.json and
asserts every Step 6C requirement. No network/model calls.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "dashboard.html"
RES = ROOT / "balanced_3class_results.json"
SAMPLE = ROOT / "balanced_sample_3class.json"
PROMPT = ROOT / "prompt_classifier_3class_emotion.txt"


def fail(msg):
    raise SystemExit(f"VERIFY FAIL: {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    html = HTML.read_text(encoding="utf-8")
    res = json.loads(RES.read_text(encoding="utf-8"))
    m = res["metrics"]

    # 0) CSS hiding rules must cover BOTH row classes (real filter mechanics, not text)
    styles = html[html.find("<style>"):html.find("</style>")]
    for rule in (".reviewrow.tr-hidden{display:none}",
                 ".b6row.tr-hidden{display:none}"):
        # normalize: allow whitespace/newlines in the style block
        norm_styles = re.sub(r"\s+", "", styles)
        if rule not in norm_styles:
            fail(f"CSS hiding rule missing from stylesheet: {rule}")
    # also require a matching default-visible rule for b6 rows
    if ".b6row{display:table-row}" not in norm_styles:
        fail("CSS default-visible rule missing: .b6row{display:table-row}")

    # 1) exactly 150 balanced rows with sample indices 0..149 exactly once
    b6_rows = re.findall(r'<tr class="b6row[^"]*" data-sample="(\d+)"', html)
    idxs = [int(x) for x in b6_rows]
    if len(idxs) != 150:
        fail(f"b6 rows != 150: {len(idxs)}")
    if sorted(idxs) != list(range(150)):
        fail("b6 sample indices not exactly 0..149 once each")

    # 2) displayed counts/metrics match the JSON (read via HTML text)
    for needle in ["150", "112", "38", "74.67%", "33.33%", "+41.34", "0.7117"]:
        if needle not in html:
            fail(f"metric text missing: {needle}")
    # summary card labels
    for lbl in ["Selected reviews", "Valid predictions", "Correct predictions",
                "Mismatches", "Accuracy (150)", "Majority-class baseline",
                "Balanced accuracy / macro recall", "Macro F1"]:
        if lbl not in html:
            fail(f"summary label missing: {lbl}")

    # 3) confusion matrix: 9 cells sum 150, diagonal 112
    cm3 = re.findall(r'<td class="(ok3|err3)">(\d+)</td>', html)
    if len(cm3) != 9:
        fail(f"cm3 cells != 9: {len(cm3)}")
    vals = [int(v) for _, v in cm3]
    if sum(vals) != 150:
        fail(f"cm3 cells sum != 150: {sum(vals)}")
    diag = [int(v) for c, v in cm3 if c == "ok3"]
    if sum(diag) != 112:
        fail(f"cm3 diagonal != 112: {sum(diag)}")
    # per-cell values match
    expected = [48, 2, 0, 8, 16, 26, 1, 1, 48]
    if vals != expected:
        fail(f"cm3 values mismatch: {vals}")

    # 4) six error directions sum to 38, dominant NEUTRAL->NEGATIVE=26
    dirs = re.findall(r'<div class="dn (?:dom|okd)">(\d+)</div>', html)
    if len(dirs) != 6:
        fail(f"error-direction rows != 6: {len(dirs)}")
    if sum(int(x) for x in dirs) != 38:
        fail(f"error directions sum != 38: {sum(int(x) for x in dirs)}")
    if m["error_directions"]["NEUTRAL → NEGATIVE"] != 26:
        fail("NEUTRAL->NEGATIVE != 26")
    if "dominant error" not in html:
        fail("dominant-error label missing")

    # 5) neutral distribution sums to 50
    neu = m["reference_neutral_prediction_distribution"]
    if sum(neu.values()) != 50:
        fail("neutral distribution does not sum to 50")
    for k, v in neu.items():
        if f">NEUTRAL {v}</strong>" not in html and f"{k} {v}" not in html:
            pass
    for k, v in neu.items():
        if f"<strong>{k} {v}</strong>" not in html:
            fail(f"neutral dist value missing: {k} {v}")

    # 6) all six Step 6C filters present with correct counts
    for label, cnt in [("All 150", 150), ("Correct", 112), ("Mismatches", 38),
                       ("Reference POSITIVE", 50), ("Reference NEUTRAL", 50),
                       ("Reference NEGATIVE", 50)]:
        pat = f'>{label} <span class="fcount">{cnt}</span></button>'
        if pat not in html:
            fail(f"b6 filter missing: {label} {cnt}")

    # 7) old first-100 filters still present
    for label, cnt in [("All reviews", 100), ("Correct", 97), ("Mismatches", 3)]:
        pat = f'>{label} <span class="fcount">{cnt}</span></button>'
        if pat not in html:
            fail(f"first-100 filter missing: {label} {cnt}")

    # per-class values (0.8421, 0.9600, 0.3200, 0.6486, 0.7742, 0.4638, 0.8972)
    for v in ["0.8421", "0.9600", "0.3200", "0.6486", "0.7742", "0.4638", "0.8972"]:
        if v not in html:
            fail(f"per-class value missing: {v}")
    if "Neutral recall is the major" not in html:
        fail("neutral-recall weakness note missing")

    # 8) no external loaded resources
    if re.findall(r'src=["\']https?://', html):
        fail("external src found")
    if "@import" in html or "<link" in html:
        fail("external css found")
    if re.findall(r'<script[^>]*src=', html):
        fail("external script found")

    # 9) source-file hashes unchanged (recorded Step 6B values)
    exp = {
        SAMPLE: "444bb14be0174ececf71440d1cc4a698ac1fe2eafa1df586f8d112df937373ed",
        PROMPT: "c7dadbdb8aaa3dca248a87a0545d39abd7ebdeb56123897cbd113d0a1efc4bda",
        RES: "196d26305a828e8038e1b9e273611fef1284dca58a27d13b7acd0ad0f91708ff",
    }
    for path, want in exp.items():
        got = sha256(path)
        if got != want:
            fail(f"hash changed for {path.name}: {got[:16]}")

    print("STEP 6C DASHBOARD VERIFICATION PASSED (no network/model calls).")
    print(f"  b6 rows=150 (indices 0..149 once), cm3 9 cells sum=150 diag=112")
    print(f"  error directions sum=38 (NEUTRAL->NEGATIVE=26 dominant)")
    print(f"  neutral dist sum=50: {neu}")
    print(f"  filters: 6 b6 + 3 first-100 all present with correct counts")
    print(f"  no external resources; source hashes unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
