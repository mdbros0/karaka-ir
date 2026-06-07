# Karaka-IR

A neurosymbolic AI tool that translates English into a Pāṇinian kāraka representation
for verifiable reasoning.

## What it does

English sentence → LLM extracts a structured kāraka frame (agent, patient, instrument,
recipient, source, locus) → symbolic engine derives new facts via inference rules →
LLM renders a fluent, auditable answer.

The six kāraka roles come from Pāṇini's *Ashtādhyāyī* (~500 BCE) — the world's oldest
formal grammar, and also the world's oldest semantic role labeling system.

## Status

| Phase | Description | State |
|-------|-------------|-------|
| Phase Zero | Environment setup, API key, hello-world | ✅ Complete |
| Phase One | LLM fundamentals + prompt engineering | ✅ Complete |
| Phase Two | First kāraka extractor + rule engine (vertical slice) | ✅ Complete |
| Phase Three | Question-answering loop + gold test set | ✅ Complete |
| Phase Four | Measurement and iteration | 🔄 In progress |
| Phase Five | Polish and self-refinement loop | ⬜ Not started |
| Phase Six | Users and writeup | ⬜ Not started |

## Evaluation Baseline (Phase Four)

30-example gold test set — run `python tests/eval.py` to reproduce.

| Metric | Score |
|--------|-------|
| Verb root accuracy | 30/30 (100%) |
| Frame recall | 100% |
| Frame precision | 100% |
| Derived fact F1 | 1.00 |
| Perfect examples | 30/30 |

## Stack

- Python 3.11+, Pydantic v2, Instructor
- Gemini 2.5 Flash (via google-genai)
- SWI-Prolog / pyswip (Phase Four+)
- Streamlit → Gradio (Phase Three+)
