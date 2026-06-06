import os
import time
from dotenv import load_dotenv

load_dotenv()

from karaka import extract_karaka, KarakaDocument, KarakaFrame
from facts import frame_to_facts
from rules import apply_rules
from render import render_answer

ROLES = ("karta", "karma", "karana", "sampradana", "apadana", "adhikarana")


def _format_frame(frame: KarakaFrame) -> str:
    lines = [f"  {frame.verb_root} ({frame.tense})"]
    for role in ROLES:
        entity = getattr(frame, role)
        if entity is not None:
            lines.append(f"    {role:12s}: {entity.lemma} / {entity.entity_type or 'unknown'}")
    return "\n".join(lines)


def _deduplicate(facts):
    seen = set()
    result = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def run_pipeline(sentence: str, question: str) -> None:
    print(f"\n{'='*60}")
    print(f"INPUT:    {sentence}")
    print(f"QUESTION: {question}")
    print("="*60)

    # Stage 1: neural extraction — one frame per verb
    doc = extract_karaka(sentence)
    n = len(doc.events)
    print(f"\n[1] KARAKA FRAMES  ({n} event{'s' if n != 1 else ''})")
    for i, frame in enumerate(doc.events):
        print(f"\n  --- event {i+1} ---")
        print(_format_frame(frame))

    # Stage 2: per-event facts + rules, then merge
    all_base: list = []
    all_derived: list = []
    for frame in doc.events:
        base = frame_to_facts(frame)
        derived = apply_rules(base)
        all_base.extend(base)
        all_derived.extend(derived)

    all_base    = _deduplicate(all_base)
    all_derived = _deduplicate(all_derived)

    print("\n[2] FACTS")
    for fact in all_base:
        print(f"  {fact[0]}({', '.join(fact[1:])})")

    print("\n[3] DERIVED")
    if all_derived:
        for fact in all_derived:
            print(f"  {fact[0]}({', '.join(fact[1:])})")
    else:
        print("  (no rules fired)")

    # Stage 4: render answer from merged fact pools
    result = render_answer(question, all_base, all_derived)
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
                "Bhoomika studies at the University of Guelph and has a Coach bag.",
                "What does Bhoomika own?",
            ),
        ]

        for i, (sentence, question) in enumerate(examples):
            if i > 0:
                time.sleep(5)
            run_pipeline(sentence, question)
            print()
