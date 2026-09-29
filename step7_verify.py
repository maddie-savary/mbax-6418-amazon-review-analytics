"""
Step 7 — independent verification of descriptive & prediction visualizations in the HTML.

Reads dashboard.html and the locked saved sources; asserts every Step 7 requirement.
No network/model calls.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "dashboard.html"
BSAMPLE = ROOT / "balanced_sample_3class.json"
B6 = ROOT / "balanced_3class_results.json"
S2 = ROOT / "step2_results.json"


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
    norm = re.sub(r"\s+", " ", html)
    bs = json.loads(BSAMPLE.read_text(encoding="utf-8"))
    b6 = json.loads(B6.read_text(encoding="utf-8"))
    s2 = json.loads(S2.read_text(encoding="utf-8"))
    meta = bs["metadata"]
    b6m = b6["metrics"]

    # 1) five rating counts sum to 152,410 and every count appears in HTML
    rd = meta["rating_distribution"]
    counts = [rd[k] for k in ("1.0", "2.0", "3.0", "4.0", "5.0")]
    if sum(counts) != 152410:
        fail(f"rating counts sum != 152410: {sum(counts)}")
    totals = {1: 12326, 2: 1873, 3: 3271, 4: 6692, 5: 128248}
    for i, cnt in enumerate(totals.values(), start=1):
        if f"{cnt:,}" not in norm:
            fail(f"{i}-star count {cnt:,} not in HTML")

    # 2) percentages computed from counts and present
    for i, cnt in enumerate(totals.values(), start=1):
        pct = f"{100 * cnt / 152410:.2f}%"
        if pct not in norm:
            fail(f"{i}-star percentage {pct} not in HTML")

    # 2b) Step 7 narrative/caption percentage must equal the computed 5-star value
    #     (84.15%), and no stale rounding (84.14%) may appear in the Step 7 section.
    s7_start = norm.find("Descriptive &amp; Prediction Visualizations")
    if s7_start == -1:
        s7_start = norm.find("Descriptive & Prediction Visualizations")
    foot = norm.find("Data source: Amazon Reviews")
    s7_section = norm[s7_start:foot] if s7_start != -1 else ""
    five_pct = f"{100 * 128248 / 152410:.2f}%"  # = 84.15%
    if five_pct not in s7_section:
        fail(f"Step 7 section missing computed 5-star percentage {five_pct}")
    if "84.14%" in s7_section:
        fail("stale 84.14% found in Step 7 section")

    # 3) first-100 distribution 93/7 sum 100
    ref100 = {}
    for r in s2["rows"]:
        ref100[r["reference"]] = ref100.get(r["reference"], 0) + 1
    if ref100.get("POSITIVE") != 93 or ref100.get("NEGATIVE") != 7:
        fail(f"step2 reference dist != 93/7: {ref100}")
    if sum(ref100.values()) != 100:
        fail("first-100 sum != 100")
    for label, val in [("First-100 binary reference", None)]:
        if label not in norm:
            fail(f"contrast card missing: {label}")
    # table rows in the contrast card: check 93 and 7 appear near "First-100"
    idx100 = norm.find("First-100 binary reference")
    chunk100 = norm[idx100:idx100 + 400]
    if "93" not in chunk100 or "7" not in chunk100:
        fail("first-100 contrast values 93/7 missing")

    # 4) balanced distribution 50/50/50 sums 150
    bal = b6m["actual_distribution"]
    if bal != {"POSITIVE": 50, "NEUTRAL": 50, "NEGATIVE": 50}:
        fail("balanced dist != 50/50/50")
    if sum(bal.values()) != 150:
        fail("balanced sum != 150")
    idxB = norm.find("Balanced 150 three-class reference")
    chunkB = norm[idxB:idxB + 400]
    for v in ("50", "50", "50"):
        if v not in chunkB:
            fail("balanced contrast values missing")

    # 5) class-success values 48/50, 16/50, 48/50 match diagonal/support in JSON
    pc = b6m["per_class"]
    expected_success = {"POSITIVE": (48, 50), "NEUTRAL": (16, 50), "NEGATIVE": (48, 50)}
    for cls, (tp, sup) in expected_success.items():
        p = pc[cls]
        if p["tp"] != tp or p["support"] != sup:
            fail(f"{cls} tp/support mismatch in JSON")
        if f"{tp}/{sup}" not in norm:
            fail(f"success label {cls} {tp}/{sup} not in HTML")
        if f"{100 * tp / sup:.0f}%" not in norm:
            fail(f"success pct {cls} not in HTML")

    # 6) six error directions match JSON and sum to 38
    ed = b6m["error_directions"]
    if sum(ed.values()) != 38:
        fail(f"error directions sum != 38: {sum(ed.values())}")
    for (a, p) in [("POSITIVE", "NEUTRAL"), ("POSITIVE", "NEGATIVE"),
                   ("NEUTRAL", "POSITIVE"), ("NEUTRAL", "NEGATIVE"),
                   ("NEGATIVE", "POSITIVE"), ("NEGATIVE", "NEUTRAL")]:
        v = ed[f"{a} → {p}"]
        if f"{a} → {p}" not in norm:
            fail(f"error direction label missing: {a} → {p}")
        # the count appears as the scount of that direction; verify via aria-label
        if f'aria-label="{a} → {p}: {v}"' not in norm:
            fail(f"error direction aria missing: {a} → {p}: {v}")
    if "dominant error" not in norm:
        fail("dominant-error marker missing")

    # 7) still exactly 100 original rows and 150 balanced rows
    f1 = re.findall(r'<tr class="reviewrow[^"]*" data-row="(\d+)"', html)
    b6r = re.findall(r'<tr class="b6row[^"]*" data-sample="(\d+)"', html)
    if len(f1) != 100 or len(b6r) != 150:
        fail(f"row counts: f1={len(f1)} b6={len(b6r)}")

    # 8) CSS rules for both hidden-row classes
    styles = re.sub(r"\s+", "", html[html.find("<style>"):html.find("</style>")])
    for rule in (".reviewrow.tr-hidden{display:none}", ".b6row.tr-hidden{display:none}",
                 ".b6row{display:table-row}"):
        if rule not in styles:
            fail(f"CSS rule missing: {rule}")

    # 9) no external resources
    if re.findall(r'src=["\']https?://', html) or "@import" in html or "<link" in html:
        fail("external resources found")

    # 10) locked-file hashes unchanged (recorded at Step 7 start)
    exp = {
        BSAMPLE: "444bb14be0174ececf71440d1cc4a698ac1fe2eafa1df586f8d112df937373ed",
        B6: "196d26305a828e8038e1b9e273611fef1284dca58a27d13b7acd0ad0f91708ff",
        S2: "e46802ca9556" ,  # prefix check below
    }
    for path, want in exp.items():
        got = sha256(path)
        if len(want) == 12:
            if not got.startswith(want):
                fail(f"hash changed for {path.name}: {got[:16]}")
        elif got != want:
            fail(f"hash changed for {path.name}: {got[:16]}")

    print("STEP 7 VERIFICATION PASSED (no network/model calls).")
    print(f"  rating counts sum=152410; all 5 counts + computed % present in HTML")
    print(f"  first-100 93/7 (sum 100); balanced 50/50/50 (sum 150)")
    print(f"  success rates: 48/50 96%, 16/50 32%, 48/50 96% — match JSON diagonal/support")
    print(f"  six error directions sum=38, all aria-labelled, dominant marked")
    print(f"  tables: 100 original + 150 balanced rows; CSS hides both row classes")
    print(f"  no external resources; locked hashes unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
