"""
Step 5A3 — independent verification of the LLM vs NRC emotion comparison.

Reads the two locked source files (step5a1_results.json, step5a2b_results.json) and
the comparison output (step5a3_comparison.json), recomputes every metric / invariant
independently, and fails loudly on any difference. No network/model calls. No rating
is read or used as emotion ground truth.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLM = ROOT / "step5a1_results.json"
NRC = ROOT / "step5a2b_results.json"
OUT = ROOT / "step5a3_comparison.json"

EMOTIONS = ["ANGER", "ANTICIPATION", "DISGUST", "FEAR",
            "JOY", "SADNESS", "SURPRISE", "TRUST"]
EMO_SET = set(EMOTIONS)


def fail(msg):
    raise SystemExit(f"VERIFY FAIL: {msg}")


def main() -> int:
    llm = json.loads(LLM.read_text(encoding="utf-8"))
    nrc = json.loads(NRC.read_text(encoding="utf-8"))
    cmp = json.loads(OUT.read_text(encoding="utf-8"))
    lrows, nrows, crow = llm["results"], nrc["results"], cmp["rows"]
    m = cmp["metrics"]

    # --- identity / alignment ---
    assert len(lrows) == 100 and len(nrows) == 100 and len(crow) == 100
    for rows in (lrows, nrows, crow):
        idx = [r["row_index"] for r in rows]
        if len(set(idx)) != 100 or sorted(idx) != list(range(100)):
            fail("indices not unique 0..99")
    lmap = {r["row_index"]: r for r in lrows}
    nmap = {r["row_index"]: r for r in nrows}
    cmap = {r["row_index"]: r for r in crow}
    for i in range(100):
        if not (lmap[i]["title"] == nmap[i]["title"] == cmap[i]["title"]):
            fail(f"title alignment row {i}")
        if not (lmap[i]["text"] == nmap[i]["text"] == cmap[i]["text"]):
            fail(f"text alignment row {i}")

    # eligibility + agreement recompute from sources
    for i in range(100):
        L, N, C = lmap[i], nmap[i], cmap[i]
        if not L["valid"]:
            fail(f"row {i} LLM invalid")
        if L["llm_primary_emotion"] not in EMO_SET:
            fail(f"row {i} LLM emotion out of enum")
        if set(N["emotion_scores"].keys()) != EMO_SET:
            fail(f"row {i} NRC score keys")
        elig = N["nrc_primary_emotion"] != "NO_MATCH"
        if C["comparison_eligible"] != elig:
            fail(f"row {i} eligibility mismatch")
        if not elig:
            if C["exact_emotion_agreement"] is not None or C["tie_aware_emotion_agreement"] is not None:
                fail(f"row {i} NO_MATCH should have null agreements")
        else:
            exp_exact = L["llm_primary_emotion"] == N["nrc_primary_emotion"]
            exp_ta = L["llm_primary_emotion"] in N["nrc_top_emotions"]
            if C["exact_emotion_agreement"] != exp_exact:
                fail(f"row {i} exact mismatch")
            if C["tie_aware_emotion_agreement"] != exp_ta:
                fail(f"row {i} tie-aware mismatch")
            # combined fields mirror sources
            if C["llm_primary_emotion"] != L["llm_primary_emotion"]:
                fail(f"row {i} llm emotion not mirrored")
            if C["nrc_primary_emotion"] != N["nrc_primary_emotion"]:
                fail(f"row {i} nrc primary not mirrored")
            if C["nrc_top_emotions"] != N["nrc_top_emotions"]:
                fail(f"row {i} nrc top not mirrored")
            if C["nrc_tie"] != N["nrc_tie"]:
                fail(f"row {i} nrc tie not mirrored")
            if C["nrc_emotion_scores"] != N["emotion_scores"]:
                fail(f"row {i} nrc scores not mirrored")
            if C["nrc_matched_token_count"] != N["matched_token_count"]:
                fail(f"row {i} nrc matched not mirrored")
            # unique max: exact == tie-aware
            if not N["nrc_tie"] and C["exact_emotion_agreement"] != C["tie_aware_emotion_agreement"]:
                fail(f"row {i} unique max exact != tie-aware")

    eligible = [C for C in crow if C["comparison_eligible"]]
    nomatch = [C for C in crow if C["nrc_primary_emotion"] == "NO_MATCH"]
    n_elt = len(eligible)
    n_exact = sum(1 for C in eligible if C["exact_emotion_agreement"])
    n_ta = sum(1 for C in eligible if C["tie_aware_emotion_agreement"])

    if n_exact > n_ta:
        fail("exact > tie-aware (impossible)")
    if m["eligible_comparison_rows"] != n_elt or m["total_rows"] != 100:
        fail("eligible/total mismatch")
    if m["valid_llm_rows"] != 100 or m["nrc_no_match_rows"] != len(nomatch):
        fail("valid/nomatch counts mismatch")
    if m["nrc_covered_rows"] != 100 - len(nomatch):
        fail("covered rows mismatch")
    if m["no_match_rows"] != sorted(r["row_index"] for r in nomatch):
        fail("no_match_rows list mismatch")

    # agreement metrics exact
    def pct(c, d):
        return round(100 * c / d, 2) if d else 0.0
    if m["exact_agreement"] != {"count": n_exact, "eligible_denominator": n_elt,
                                "percentage_pct": pct(n_exact, n_elt)}:
        fail("exact agreement metric mismatch")
    if m["tie_aware_agreement"] != {"count": n_ta, "eligible_denominator": n_elt,
                                    "percentage_pct": pct(n_ta, n_elt)}:
        fail("tie-aware agreement metric mismatch")
    if m["additional_agreements_recovered_by_tie_aware"] != (n_ta - n_exact):
        fail("recovered-count mismatch")

    # disagreement / tie-break lists
    if m["exact_disagreement_rows"] != sorted(C["row_index"] for C in eligible if not C["exact_emotion_agreement"]):
        fail("exact_disagreement list mismatch")
    if m["tie_aware_disagreement_rows"] != sorted(C["row_index"] for C in eligible if not C["tie_aware_emotion_agreement"]):
        fail("tie_aware_disagreement list mismatch")
    tbs = sorted(C["row_index"] for C in eligible
                 if (not C["exact_emotion_agreement"]) and C["tie_aware_emotion_agreement"])
    if m["tie_break_sensitive_rows"] != tbs:
        fail("tie_break_sensitive list mismatch")

    # distributions
    llm_dist = {e: sum(1 for C in crow if C["llm_primary_emotion"] == e) for e in EMOTIONS}
    nrc_dist = {e: sum(1 for C in crow if C["nrc_primary_emotion"] == e) for e in EMOTIONS}
    nrc_dist["NO_MATCH"] = len(nomatch)
    if m["llm_emotion_distribution"] != llm_dist:
        fail("llm distribution mismatch")
    if m["nrc_primary_emotion_distribution"] != nrc_dist:
        fail("nrc distribution mismatch")
    if sum(llm_dist.values()) != 100 or sum(nrc_dist.values()) != 100:
        fail("distribution counts do not sum to total")

    # 8x8 crosstab: rows=LLM, cols=NRC primary, eligible only
    ctab = {le: {ne: 0 for ne in EMOTIONS} for le in EMOTIONS}
    for C in eligible:
        ctab[C["llm_primary_emotion"]][C["nrc_primary_emotion"]] += 1
    if m["cross_tabulation_8x8_llm_by_nrc"] != ctab:
        fail("crosstab mismatch")
    if sum(ctab[le][ne] for le in EMOTIONS for ne in EMOTIONS) != n_elt:
        fail("crosstab cells do not sum to eligible")

    # grouped unique vs tied
    unique_max = [C for C in eligible if not C["nrc_tie"]]
    tied_max = [C for C in eligible if C["nrc_tie"]]
    def g(rows, field):
        k = len(rows); c = sum(1 for r in rows if r[field])
        return {"rows": k, "agreements": c, "denominator": k,
                "percentage_pct": pct(c, k)}
    exp_grouped = {
        "unique_maximum": {"rows": len(unique_max),
                           "exact": g(unique_max, "exact_emotion_agreement"),
                           "tie_aware": g(unique_max, "tie_aware_emotion_agreement")},
        "tied_maximum": {"rows": len(tied_max),
                         "exact": g(tied_max, "exact_emotion_agreement"),
                         "tie_aware": g(tied_max, "tie_aware_emotion_agreement")},
    }
    if m["grouped_by_nrc_structure"] != exp_grouped:
        fail("grouped-by-structure mismatch")
    # unique max: exact == tie-aware in grouped too
    if m["grouped_by_nrc_structure"]["unique_maximum"]["exact"] != \
       m["grouped_by_nrc_structure"]["unique_maximum"]["tie_aware"]:
        fail("unique-max grouped exact != tie-aware")

    # no rating fields in comparison rows
    for C in crow:
        for bad in ("rating", "reference", "predict", "agree_with_rating", "sentiment"):
            if bad in C:
                fail(f"row {C['row_index']} has forbidden field {bad}")

    print("STEP 5A3 VERIFICATION PASSED (no network/model calls).")
    print(f"  total=100 covered={100-len(nomatch)} NO_MATCH={len(nomatch)} eligible={n_elt}")
    print(f"  exact={n_exact}/{n_elt} ({pct(n_exact,n_elt)}%)   "
          f"tie-aware={n_ta}/{n_elt} ({pct(n_ta,n_elt)}%)   recovered={n_ta-n_exact}")
    print("  unique_max:", m["grouped_by_nrc_structure"]["unique_maximum"])
    print("  tied_max:  ", m["grouped_by_nrc_structure"]["tied_maximum"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
