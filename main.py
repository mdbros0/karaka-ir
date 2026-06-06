import os
from dotenv import load_dotenv

load_dotenv()

from karaka import extract_karaka
from facts import frame_to_facts
from rules import apply_rules


def run_pipeline(sentence: str) -> None:
    print(f"\n{'='*60}")
    print(f"INPUT:  {sentence}")
    print("="*60)

    # Stage 1: neural extraction
    frame = extract_karaka(sentence)
    print("\n[1] KARAKA FRAME")
    print(frame.model_dump_json(indent=2))

    # Stage 2: symbolic representation
    facts = frame_to_facts(frame)
    print("\n[2] FACTS")
    for fact in facts:
        print(f"  {fact[0]}({', '.join(fact[1:])})")

    # Stage 3: inference
    derived = apply_rules(facts)
    print("\n[3] DERIVED")
    if derived:
        for fact in derived:
            print(f"  {fact[0]}({', '.join(fact[1:])})")
    else:
        print("  (no rules fired)")


if __name__ == "__main__":
    sentences = [
        "Maya gave Ravi a book at the library.",
        "Anna bought groceries from the market on Sunday.",
    ]

    for sentence in sentences:
        run_pipeline(sentence)
        print()
