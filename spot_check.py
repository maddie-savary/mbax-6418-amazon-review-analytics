"""
Step 1 spot-check runner.

Tests the classifier on FOUR categories the assignment calls out:
  1. clearly positive reviews
  2. clearly negative reviews
  3. terse reviews
  4. title/body conflicting reviews

Each review carries an EXPECTED label chosen by reading the title+text alone
(gift-card domain judgment) plus the real star-rating. The rating is kept ONLY as
local ground-truth for later evaluation (Step 2); it is never sent to the model —
classifier.py sends title and text only.

Run:
    export SENTIMENT_API_KEY="<course-api-key>"
    python spot_check.py
"""
import json
import sys

from classifier import classify

# expected label chosen from title+text alone; rating shown only as a sanity note
SPOT_CASES = [
    # --- clearly positive ---
    {"id": "pos1", "title": "Great gift", "text": "Having Amazon money is always good.",
     "expected": "POSITIVE", "rating": 5.0},
    {"id": "pos2", "title": "amazon gift card",
     "text": "Always the perfect gift. I have never given one and had someone seem or act disappointed. Just the opposite.",
     "expected": "POSITIVE", "rating": 5.0},
    {"id": "pos3", "title": "Gifts for my two granddaughters",
     "text": "They love it !!", "expected": "POSITIVE", "rating": 5.0},
    # --- clearly negative ---
    {"id": "neg1", "title": "Card was invalid void not activated....",
     "text": "I'd give zero stars if possible. After purchasing this gift card on Lightning Deal during Prime Days I will never buy again.",
     "expected": "NEGATIVE", "rating": 1.0},
    {"id": "neg2", "title": "Terrible",
     "text": "I tried to use the card to purchase something on Amazon and it didn't work",
     "expected": "NEGATIVE", "rating": 1.0},
    {"id": "neg3", "title": "The worst place to eat food",
     "text": "The worst place to eat food. It should earn 0 star if the system allows.",
     "expected": "NEGATIVE", "rating": 1.0},
    # --- terse ---
    {"id": "terse1", "title": "Cute!", "text": "That snowman tin is adorable",
     "expected": "POSITIVE", "rating": 5.0},
    {"id": "terse2", "title": "Easy..",
     "text": "Reloading is easy. Shopping is easy. Amazon is easy.",
     "expected": "POSITIVE", "rating": 5.0},
    {"id": "terse3", "title": "Horrible", "text": "Total scam.",
     "expected": "NEGATIVE", "rating": 1.0},
    # --- conflicting title vs body ---
    {"id": "conf1", "title": "Great concept, received in poor condition",
     "text": "The gift card shipped in a padded envelope. Inside the gift card is packaged in a sealed clear plastic and the card was cracked and damaged.",
     "expected": "NEGATIVE", "rating": 1.0},
    {"id": "conf2", "title": "Nice looking", "text": "The tin is nice but the card does not activate.",
     "expected": "NEGATIVE", "rating": 1.0},
    {"id": "conf3", "title": "Perfect gift", "text": "Shipped broken and unusable.",
     "expected": "NEGATIVE", "rating": 1.0},
]


def main() -> int:
    if not SPOT_CASES:
        print("No spot cases."); return 0
    n_all = n_correct = n_valid = 0
    rows = []
    print(f"{'id':7} {'exp':9} {'pred':9} {'valid':6} correct")
    for c in SPOT_CASES:
        r = classify(c["title"], c["text"])
        pred = r["sentiment"]
        valid = r["valid"]
        correct = valid and (pred == c["expected"])
        n_all += 1
        n_valid += 1 if valid else 0
        n_correct += 1 if correct else 0
        # never dump raw to terminal logs/extras beyond this point
        rows.append({
            "id": c["id"], "title": c["title"], "text": c["text"],
            "expected": c["expected"], "predicted": pred,
            "valid": valid, "correct": correct,
            "rating_for_reference_only": c["rating"],
        })
        print(f"{c['id']:7} {c['expected']:9} {str(pred):9} {str(valid):6}"
              f" {'YES' if correct else 'NO '}")

    print(f"\n--- SUMMARY (schema-valid & correct) ---")
    print(f"cases: {n_all}  valid(responses passed validation): {n_valid}/{n_all}"
          f"  correct: {n_correct}/{n_all}  accuracy(valid-only): "
          f"{100*n_correct/max(n_valid,1):.1f}%")

    # Save spot-check JSON. DISTINCTION: the rating is stored locally ONLY as
    # ground-truth for later evaluation (it is labelled rating_ground_truth). The
    # rating NEVER appears in any model request — classifier.py sends title+text only.
    # No API key is ever written here.
    with open("spot_check_results.json", "w") as f:
        json.dump({
            "note": "Step 1 spot-check output. No API key is included. ",
            "data_handling": {
                "model_request_fields": ["title", "text"],
                "rating": "stored locally as ground-truth for evaluation ONLY; "
                          "never sent in a model request",
            },
            "results": rows,
            "summary": {"cases": n_all, "valid": n_valid, "correct": n_correct},
        }, f, indent=2)
    print("wrote spot_check_results.json")
    return 0 if n_valid == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
