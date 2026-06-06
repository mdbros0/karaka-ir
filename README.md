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
| Phase One | LLM fundamentals + prompt engineering | 🔄 In progress |
| Phase Two | First kāraka extractor + rule engine (vertical slice) | ⬜ Not started |
| Phase Three | Question-answering loop + gold test set | ⬜ Not started |
| Phase Four | Measurement and iteration | ⬜ Not started |
| Phase Five | Polish and self-refinement loop | ⬜ Not started |
| Phase Six | Users and writeup | ⬜ Not started |

## Stack

- Python 3.11+, Pydantic v2, Instructor
- Gemini 2.5 Flash (via google-genai)
- SWI-Prolog / pyswip (Phase Four+)
- Streamlit → Gradio (Phase Three+)
