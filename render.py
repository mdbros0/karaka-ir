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


class Answer(BaseModel):
    answer: str = Field(description="A single fluent sentence that directly answers the question")
    reasoning: str = Field(description="The specific facts used to arrive at this answer")


def _format(facts: list[Fact]) -> str:
    return "\n".join(f"  {f[0]}({', '.join(f[1:])})" for f in facts)


def render_answer(question: str, base_facts: list[Fact], derived_facts: list[Fact]) -> Answer:
    prompt = f"""You are a fact-grounded reasoning system.

BASE FACTS (extracted from the sentence):
{_format(base_facts)}

DERIVED FACTS (produced by inference rules):
{_format(derived_facts) if derived_facts else "  (none)"}

QUESTION: {question}

Instructions:
- Answer using ONLY the facts above. Do not use outside knowledge.
- If the facts are insufficient to answer, say so explicitly.
- Your answer must be a single complete sentence.
- Your reasoning must cite the specific fact(s) that support the answer."""

    return client.create(
        response_model=Answer,
        config={"temperature": 0.0},
        messages=[{"role": "user", "content": prompt}],
    )
