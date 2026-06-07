import os
import instructor
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from facts import Fact

load_dotenv()

client = instructor.from_provider(
    "google/gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

ROLES = ("karta", "karma", "karana", "sampradana", "apadana", "adhikarana")

ROLE_LABELS = {
    "karta":      "karta (agent)",
    "karma":      "karma (patient)",
    "karana":     "karana (instrument)",
    "sampradana": "sampradana (recipient)",
    "apadana":    "apadana (source)",
    "adhikarana": "adhikarana (locus)",
}


class Answer(BaseModel):
    answer: str = Field(
        description=(
            "A fluent answer directly addressing the question. "
            "Use one sentence for specific questions; use multiple sentences if the question "
            "is open-ended (e.g. 'tell me about X') and multiple facts are available."
        )
    )
    reasoning: str = Field(description="The specific facts used to arrive at this answer")


def _format_events(frames) -> str:
    """Format KarakaFrames as structured per-event sections so the LLM sees
    who participates in each event, preserving participant-event binding that
    would be lost in a flat deduplicated fact pool."""
    lines = []
    for i, frame in enumerate(frames, 1):
        lines.append(f"  Event {i}: {frame.verb_root} ({frame.tense})")
        for role in ROLES:
            entity = getattr(frame, role)
            if entity is not None:
                lines.append(f"    {ROLE_LABELS[role]}: {entity.lemma} ({entity.entity_type or 'unknown'})")
    return "\n".join(lines) if lines else "  (no events)"


def _format_derived(facts: list[Fact]) -> str:
    return "\n".join(f"  {f[0]}({', '.join(f[1:])})" for f in facts) if facts else "  (none)"


def render_answer(question: str, frames, derived_facts: list[Fact]) -> Answer:
    """Render a fluent answer grounded in the extracted frames and derived facts.

    `frames` is a list of KarakaFrame objects (not a flat fact list) so the LLM
    can see full participant-event binding per event.
    """
    prompt = f"""You are a fact-grounded reasoning system.

EVENTS (extracted from the sentence — each shows who does what):
{_format_events(frames)}

DERIVED FACTS (produced by inference rules):
{_format_derived(derived_facts)}

QUESTION: {question}

Instructions:
- Answer using ONLY the facts above. Do not use outside knowledge.
- If the facts are insufficient to answer, say so explicitly.
- For specific questions (who, what, where, when), answer in one sentence.
- For open-ended questions ('tell me about X', 'describe X'), use as many sentences as needed
  to cover ALL relevant facts — do not omit any event or derived fact that relates to the subject.
- Your reasoning must cite the specific fact(s) that support the answer."""

    return client.create(
        response_model=Answer,
        config={"temperature": 0.0},
        messages=[{"role": "user", "content": prompt}],
    )
