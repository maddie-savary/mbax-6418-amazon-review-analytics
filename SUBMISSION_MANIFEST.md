# MBAX 6418 — Assignment 1 Submission Manifest

Local audit status: **prepared for submission — not yet published to GitHub or
submitted to Canvas.** No repository has been initialized, no commit/push has
occurred, and no credential, prediction, prompt, or result file has changed as a
result of this audit.

## Required deliverables

| Deliverable | File |
|---|---|
| Reusable prompts | `prompt_classifier.txt`, `prompt_classifier_sentiment_emotion.txt`, `prompt_classifier_3class_emotion.txt` |
| Classification / scoring scripts | `classifier.py`, `classifier_sentiment_emotion.py`, `classifier_3class_emotion.py`, `spot_check.py`, `step2.py`, `step5a1.py`, `run_balanced_3class.py` |
| NRC word-list scoring script | `score_nrc_emotions.py` (plus `validate_nrc_lexicon.py`) |
| Dashboard generator | `generate_dashboard.py` |
| Balanced run's raw row-level output | `balanced_3class_results.json` |
| Final self-contained dashboard | `dashboard.html` (inline CSS/JS, no external assets) |
| Concise final report | `README.md` |
| Full development history | `DEVELOPMENT_NOTES.md` (byte-identical archive of the pre-final README) |
| Final dashboard screenshots | the five `screenshots/dashboard_tab_*.png` captures below |

## Authoritative saved result files

| Stage | File |
|---|---|
| Step 1 spot-check | `spot_check_results.json` |
| Step 2 first-100 run | `step2_results.json` |
| Step 5A1 LLM sentiment+emotion | `step5a1_results.json` |
| Step 5A2b NRC word-list scoring | `step5a2b_results.json` |
| Step 5A3 LLM vs NRC comparison | `step5a3_comparison.json` |
| Step 6A balanced sample | `balanced_sample_3class.json` |
| Step 6B balanced three-class run | `balanced_3class_results.json` |
| NRC source metadata | `nrc_source_manifest.json` |

## Final screenshots

The README's "Final dashboard" section embeds two current tabbed-dashboard
screenshots:
- `screenshots/dashboard_tab_overview.png` — Overview tab (headline metrics, star-rating distribution, comparison tables).
- `screenshots/dashboard_tab_balanced.png` — Balanced Three-Class tab (metrics, confusion matrix, disclosures).

All five final tab captures:
- `screenshots/dashboard_tab_overview.png`
- `screenshots/dashboard_tab_binary.png`
- `screenshots/dashboard_tab_emotions.png`
- `screenshots/dashboard_tab_balanced.png`
- `screenshots/dashboard_tabs_mobile.png`

Earlier-step screenshots under `screenshots/` (Steps 3, 4, 5B, 6C, 7) are
historical and documented in `DEVELOPMENT_NOTES.md`; they are no longer
embedded in the final report.

## Files intentionally excluded (git-ignored) and why

| Path | Reason |
|---|---|
| `data/` | Raw dataset `Gift_Cards.jsonl.gz` (~12 MB gz, ~50 MB uncompressed) + NRC lexicon files: large, re-downloadable/licensed; do not commit |
| `.venv/` | Local virtual environment; contains machine-specific absolute paths |
| `.env` / `*.env` | Secret storage; API key must never be committed |
| `__pycache__/`, `*.pyc` | Python bytecode |
| `.DS_Store` | macOS metadata |
| `.Rhistory`, `.RData`, `.Rproj.user/` | Local interactive-session files |
| `*.log`, `*.swp`, `Thumbs.db`, `.idea/`, `.vscode/` | Editor/OS scratch |

## Offline verification commands (no API key, no network)

```bash
python step2_verify.py
python step5a1_verify.py
python validate_nrc_lexicon.py
python step5a2b_verify.py
python step5a3_verify.py
python balanced_sample_verify.py
python step6b_verify.py
python step6c_verify.py
python step7_verify.py
python dashboard_tabs_verify.py
```

All ten passed in the audit against `dashboard.html` and the authoritative JSON
files. Runners that *do* need the endpoint (`step2.py`, `step5a1.py`,
`run_balanced_3class.py`) are checkpointed/resume-safe and require
`SENTIMENT_API_KEY` from the environment; see README "Setup and reproduction".

## Status

- GitHub publication: **not yet performed** (no `git init`, commit, push, or
  remote creation; nothing submitted to Canvas).
- Pending manual step for the student: create the repository (e.g. GitHub
  Classroom), then commit/push and submit the repo link — after the audit is
  explicitly approved.
