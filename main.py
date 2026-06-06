import os
import time
from dotenv import load_dotenv

load_dotenv()

from karaka import extract_karaka
from facts import frame_to_facts
from rules import apply_rules
from render import render_answer


def run_pipeline(sentence: str, question: str) -> None:
    print(f"\n{'='*60}")
    print(f"INPUT:    {sentence}")
    print(f"QUESTION: {question}")
    print("="*60)

    # Stage 1: neural extraction
    frame = extract_karaka(sentence)
    print("\n[1] KARAKA FRAME")
    print(frame.model_dump_json(indent=2))

    # Stage 2: symbolic representation
    base_facts = frame_to_facts(frame)
    print("\n[2] FACTS")
    for fact in base_facts:
        print(f"  {fact[0]}({', '.join(fact[1:])})")

    # Stage 3: inference
    derived_facts = apply_rules(base_facts)
    print("\n[3] DERIVED")
    if derived_facts:
        for fact in derived_facts:
            print(f"  {fact[0]}({', '.join(fact[1:])})")
    else:
        print("  (no rules fired)")

    # Stage 4: render answer
    result = render_answer(question, base_facts, derived_facts)
    print("\n[4] ANSWER")
    print(f"  {result.answer}")
    print(f"\n  Reasoning: {result.reasoning}")


def interactive_loop() -> None:
    print("\nKaraka-IR — interactive mode  (type 'quit' to exit)\n")
    while True:
        sentence = input("Sentence : ").strip()
        if sentence.lower() in ("quit", "exit", "q", ""):
            break
        question = input("Question : ").strip()
        if question.lower() in ("quit", "exit", "q", ""):
            break
        run_pipeline(sentence, question)
        print()


if __name__ == "__main__":
    import sys

    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_loop()
    else:
        examples = [
            (
                "Maya gave Ravi a book at the library.",
                "Who owns the book now?",
            ),
            (
                "Anna bought groceries from the market on Sunday.",
                "Where did Anna get the groceries from?",
            ),
        ]

        for i, (sentence, question) in enumerate(examples):
            if i > 0:
                time.sleep(5)
            run_pipeline(sentence, question)
            print()
