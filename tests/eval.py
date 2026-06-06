import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from karaka import extract_karaka, KarakaFrame
from facts import frame_to_facts, document_to_facts
from rules import apply_rules

ROLES = ["karta", "karma", "karana", "sampradana", "apadana", "adhikarana"]
SLEEP_BETWEEN_CALLS = 5  # seconds; gemini-3.5-flash free tier is 15 RPM


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_frame(predicted: KarakaFrame, expected: dict) -> dict:
    """Score karaka role extraction against the gold frame.

    Only counts non-null expected roles — null-null agreement is not rewarded
    because the model can trivially score 100% by outputting all nulls.
    """
    correct = missed = wrong = spurious = 0
    issues = []

    # verb_root
    verb_ok = predicted.verb_root.lower() == expected["verb_root"].lower()
    if not verb_ok:
        issues.append(f"verb_root expected '{expected['verb_root']}', got '{predicted.verb_root}'")

    for role in ROLES:
        exp = expected.get(role)
        pred = getattr(predicted, role)
        exp_lemma = exp["lemma"].lower() if exp else None
        pred_lemma = pred.lemma.lower() if pred else None

        if exp_lemma is not None:
            if pred_lemma == exp_lemma:
                correct += 1
            elif pred_lemma is None:
                missed += 1
                issues.append(f"[missed]   {role}: expected '{exp_lemma}'")
            else:
                wrong += 1
                issues.append(f"[wrong]    {role}: expected '{exp_lemma}', got '{pred_lemma}'")
        elif pred_lemma is not None:
            spurious += 1
            issues.append(f"[spurious] {role}: model produced '{pred_lemma}' (expected null)")

    total_expected = correct + missed + wrong
    total_predicted = correct + wrong + spurious
    recall = correct / total_expected if total_expected else 1.0
    precision = correct / total_predicted if total_predicted else 1.0

    return {
        "verb_ok": verb_ok,
        "correct": correct,
        "total_expected": total_expected,
        "recall": recall,
        "precision": precision,
        "issues": issues,
    }


def score_derived(predicted: list, expected: list) -> dict:
    """Precision / recall / F1 on the set of derived fact tuples."""
    pred_set = {tuple(f) for f in predicted}
    exp_set  = {tuple(f) for f in expected}

    tp = pred_set & exp_set
    fp = pred_set - exp_set
    fn = exp_set  - pred_set

    precision = len(tp) / len(pred_set) if pred_set else (1.0 if not exp_set else 0.0)
    recall    = len(tp) / len(exp_set)  if exp_set  else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    issues = []
    for f in sorted(fn):
        issues.append(f"[not fired] {f[0]}({', '.join(f[1:])})")
    for f in sorted(fp):
        issues.append(f"[spurious]  {f[0]}({', '.join(f[1:])})")

    return {"precision": precision, "recall": recall, "f1": f1, "issues": issues}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gold_path = Path(__file__).parent / "gold_examples.json"
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    examples = gold["examples"]
    n = len(examples)
    results = []

    print(f"\n=== KARAKA-IR EVALUATION  ({n} examples, ~{n * SLEEP_BETWEEN_CALLS}s) ===\n")

    for i, ex in enumerate(examples):
        if i > 0:
            time.sleep(SLEEP_BETWEEN_CALLS)

        snippet = ex["sentence"][:48] + ("..." if len(ex["sentence"]) > 48 else "")
        print(f"[{i+1:02d}/{n}] {ex['id']}  {snippet}", end="  ", flush=True)

        try:
            doc     = extract_karaka(ex["sentence"])
            # Gold examples are single-event; score against the first extracted frame.
            frame   = doc.events[0] if doc.events else None
            base    = frame_to_facts(frame) if frame else []
            derived = apply_rules(base) if frame else []

            fs = score_frame(frame, ex["expected_frame"])
            ds = score_derived(derived, ex["expected_derived"])

            results.append({"id": ex["id"], "ok": True, "frame": fs, "derived": ds})

            perfect = fs["recall"] == 1.0 and fs["precision"] == 1.0 and ds["f1"] == 1.0
            marker  = "OK" if perfect else "~~"
            print(f"{marker}  frame={fs['correct']}/{fs['total_expected']}  derived_f1={ds['f1']:.2f}")

        except Exception as e:
            results.append({"id": ex["id"], "ok": False, "error": str(e)})
            print(f"ERR  {e}")

    # ---- aggregate summary ------------------------------------------------
    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]

    if not ok:
        print("\nNo successful runs.")
        return

    avg_recall    = sum(r["frame"]["recall"]    for r in ok) / len(ok)
    avg_precision = sum(r["frame"]["precision"] for r in ok) / len(ok)
    avg_derived   = sum(r["derived"]["f1"]      for r in ok) / len(ok)
    verb_ok_count = sum(1 for r in ok if r["frame"]["verb_ok"])
    perfect_count = sum(
        1 for r in ok
        if r["frame"]["recall"] == 1.0
        and r["frame"]["precision"] == 1.0
        and r["derived"]["f1"] == 1.0
    )

    print(f"\n=== SUMMARY  ({len(ok)}/{n} completed) ===")
    print(f"  Verb root accuracy :  {verb_ok_count}/{len(ok)}")
    print(f"  Frame recall       :  {avg_recall:.0%}   (non-null roles correctly extracted)")
    print(f"  Frame precision    :  {avg_precision:.0%}   (no spurious role fills)")
    print(f"  Derived fact F1    :  {avg_derived:.2f}   (inference rule accuracy)")
    print(f"  Perfect examples   :  {perfect_count}/{len(ok)}")
    if errors:
        print(f"  Errors             :  {len(errors)}  ({', '.join(r['id'] for r in errors)})")

    # ---- per-example issues -----------------------------------------------
    flagged = [r for r in ok if r["frame"]["issues"] or r["derived"]["issues"]]
    if flagged:
        print("\n=== ISSUES ===")
        for r in flagged:
            print(f"\n  {r['id']}:")
            for line in r["frame"]["issues"]:
                print(f"    {line}")
            for line in r["derived"]["issues"]:
                print(f"    {line}")


if __name__ == "__main__":
    main()
