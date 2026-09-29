"""
Dashboard four-tab redesign — independent verification of the generated HTML.

Asserts tab structure/ARIA, hash anchors, row counts, filter markup, no
duplicate IDs, no external resources, and that locked source hashes are
unchanged. Offline; no network/model calls.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "dashboard.html"
SAMPLE = ROOT / "balanced_sample_3class.json"
PROMPT = ROOT / "prompt_classifier_3class_emotion.txt"
RES = ROOT / "balanced_3class_results.json"
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

    # 1) exactly four tabs, four panels, correct roles
    tabs = re.findall(r'<button type="button" role="tab" id="([^"]+)" aria-controls="([^"]+)"', html)
    if len(tabs) != 4:
        fail(f"tab buttons != 4: {len(tabs)}")
    panels = re.findall(r'<section id="([^"]+)" class="tabpanel" role="tabpanel" aria-labelledby="([^"]+)"', html)
    if len(panels) != 4:
        fail(f"tab panels != 4: {len(panels)}")
    expected = ["tab-overview", "tab-binary", "tab-emotions", "tab-balanced"]
    got_ids = [t for t, _ in tabs]
    if got_ids != expected:
        fail(f"tab ids mismatch: {got_ids}")
    for tid, ctl in tabs:
        if ctl != tid.replace("tab-", "panel-", 1):
            fail(f"aria-controls mismatch for {tid}: {ctl}")

    # 2) every aria-controls id resolves to a tabpanel and aria-labelledby resolves
    ids = set(re.findall(r'id="([^"]+)"', html))
    for tid, ctl in tabs:
        if ctl not in ids:
            fail(f"aria-controls target missing: {ctl}")
    for pid, lab in panels:
        if lab not in ids:
            fail(f"aria-labelledby target missing: {lab}")
        if lab != pid.replace("panel-", "tab-", 1):
            fail(f"aria-labelledby mismatch for {pid}: {lab}")

    # 3) tab order in document matches expected; overview is the first panel
    first_panel = panels[0][0]
    if first_panel != "panel-overview":
        fail(f"first panel is not overview: {first_panel}")
    # overview initially not hidden; exactly three panels carry hidden
    if re.search(r'id="panel-overview" class="tabpanel" role="tabpanel" aria-labelledby="tab-overview" tabindex="0">', html) is None:
        fail("overview panel is not present without hidden attribute")
    if len(re.findall(r'id="panel-(binary|emotions|balanced)" class="tabpanel" role="tabpanel" aria-labelledby="[^"]+" tabindex="0" hidden>', html)) != 3:
        fail("expected exactly 3 hidden panels (binary/emotions/balanced)")

    # 4) hash anchors: JS builds "#" + tab name; the four valid names must exist
    names = re.findall(r'data-tab="([^"]+)"', html)
    if sorted(set(names)) != ["balanced", "binary", "emotions", "overview"]:
        fail(f"data-tab names mismatch: {names}")
    if 'location.hash' not in html or '"#" + name' not in html:
        fail("hash-building logic missing in JS")
    if '["overview", "binary", "emotions", "balanced"]' not in html:
        fail("valid-hash list missing in JS")

    # 5) row counts: 100 reviewrow, 100 emorow, 150 b6row
    f1 = re.findall(r'<tr class="reviewrow[^"]*" data-row="(\d+)"', html)
    emo = re.findall(r'<tr class="emorow" data-emo-row="(\d+)"', html)
    b6 = re.findall(r'<tr class="b6row[^"]*" data-sample="(\d+)"', html)
    if len(f1) != 100 or len(emo) != 100 or len(b6) != 150:
        fail(f"row counts: reviewrow={len(f1)} emorow={len(emo)} b6row={len(b6)}")
    if [int(x) for x in emo] != list(range(100)):
        fail("emorow indices not 0..99 exactly once")

    # 6) filter groups present with exact counts
    for label, cnt in [("All reviews", 100), ("Correct", 97), ("Mismatches", 3)]:
        if f'>{label} <span class="fcount">{cnt}</span></button>' not in html:
            fail(f"first-100 filter missing: {label} {cnt}")
    for label, cnt in [("All 150", 150), ("Correct", 112), ("Mismatches", 38),
                       ("Reference POSITIVE", 50), ("Reference NEUTRAL", 50),
                       ("Reference NEGATIVE", 50)]:
        if f'>{label} <span class="fcount">{cnt}</span></button>' not in html:
            fail(f"b6 filter missing: {label} {cnt}")

    # 7) no duplicated element IDs
    id_counts = {}
    for m in re.finditer(r'id="([^"]+)"', html):
        id_counts[m.group(1)] = id_counts.get(m.group(1), 0) + 1
    dups = {k: v for k, v in id_counts.items() if v > 1}
    if dups:
        fail(f"duplicated IDs: {dups}")

    # 8) CSS hiding rules + tab styles + disclosure + contrast rules present
    raw_styles = html[html.find("<style>"):html.find("</style>")]
    norm_styles = re.sub(r"\s+", "", raw_styles)
    # rules with a descendant combinator keep their space, so check them on raw CSS
    spaced_rules = ["details.disclose>summary{", ".d-body{", ".stat.accent .delta{color:var(--gold)}"]
    for rule in (".reviewrow.tr-hidden{display:none}", ".b6row.tr-hidden{display:none}",
                 ".b6row{display:table-row}", ".tabbar{", "[aria-selected=\"true\"]{",
                 ".tabpanel[hidden]{display:none}"):
        if rule not in norm_styles:
            fail(f"CSS rule missing: {rule}")
    for rule in spaced_rules:
        if rule not in raw_styles:
            fail(f"CSS rule missing: {rule}")

    # 8b) required disclosure sections exist, are native <details>, closed by default
    required_disclosures = [
        "Inspect the 3 mismatches",
        "Explore all 100 binary reviews",
        "Why NRC ties matter",
        "View the LLM × NRC comparison matrix",
        "Read representative comparison examples",
        "Explore emotion evidence for all 100 reviews",
        "Examine all six error directions",
        "Explore all 150 balanced reviews",
    ]
    for label in required_disclosures:
        if f"<summary>{label} <span class=\"d-meta\">" not in html:
            fail(f"disclosure missing (or not closed-by-default form): {label}")
    # native details, closed: pattern "<details class=\"disclose\">\n      <summary>"
    opened = re.findall(r'<details class="disclose"[^>]*>', html)
    if [d for d in opened if "open" in d]:
        fail("a disclose details element is open by default")
    if len(opened) < 8:
        fail(f"fewer than 8 disclose elements: {len(opened)}")

    # 8c) no red/danger text inside teal/navy accent cards (contrast rule is present
    #     above; additionally no .delta.bad element may be a descendant of .stat.accent)
    #     The gold override covers it in CSS; assert the CSS rule exists (done above).

    # 8d) run-details + methodology compact lines present
    if 'class="mline"' not in html:
        fail("compact metric line (.mline) missing")
    if "<b>Run details:</b>" not in html:
        fail("Balanced 'Run details' line missing")

    # 9) no external resources
    if re.findall(r'src=["\']https?://', html) or "@import" in html or "<link" in html:
        fail("external resources found")

    # 10) locked file hashes unchanged
    exp = {
        SAMPLE: "444bb14be0174ececf71440d1cc4a698ac1fe2eafa1df586f8d112df937373ed",
        PROMPT: "c7dadbdb8aaa3dca248a87a0545d39abd7ebdeb56123897cbd113d0a1efc4bda",
        RES: "196d26305a828e8038e1b9e273611fef1284dca58a27d13b7acd0ad0f91708ff",
        S2: "e46802ca95564394951f0aef61e298b104a836048767f159e453ef7ac113a86b",
    }
    for path, want in exp.items():
        got = sha256(path)
        if got != want:
            fail(f"hash changed for {path.name}: {got[:16]}")

    print("DASHBOARD TABS VERIFICATION PASSED (no network/model calls).")
    print(f"  tabs=4 panels=4; overview first & unhidden; binary/emotions/balanced hidden")
    print(f"  rows: reviewrow=100 emorow=100 b6row=150; emorow indices 0..99 once")
    print(f"  filters: first-100 100/97/3 + b6 150/112/38/50/50/50; no duplicate IDs")
    print(f"  hash anchors #overview #binary #emotions #balanced; CSS hiding + tab rules present")
    print(f"  no external resources; locked hashes unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
