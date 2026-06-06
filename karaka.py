import os
import instructor
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

client = instructor.from_provider(
    "google/gemini-3.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)


class Entity(BaseModel):
    """A noun-like participant in an event."""

    lemma: str = Field(description="The base form of the noun, e.g., 'book' not 'books'")
    entity_type: Optional[str] = Field(
        default=None, description="person, object, location, time, etc."
    )


class KarakaFrame(BaseModel):
    """A single event analyzed using Panini's six karaka roles."""

    verb_root: str = Field(description="The base verb, e.g., 'give', 'sell'")
    tense: str = Field(description="past, present, or future")
    karta: Optional[Entity] = Field(
        default=None, description="Agent: who performs the action autonomously"
    )
    karma: Optional[Entity] = Field(
        default=None, description="Patient: what the action most directly affects"
    )
    karana: Optional[Entity] = Field(
        default=None, description="Instrument: the most effective means used"
    )
    sampradana: Optional[Entity] = Field(
        default=None, description="Recipient: for whom the action is done"
    )
    apadana: Optional[Entity] = Field(
        default=None, description="Source: from where or whom something separates"
    )
    adhikarana: Optional[Entity] = Field(
        default=None, description="Locus: where or when the action takes place"
    )


def extract_karaka(sentence: str) -> KarakaFrame:
    return client.create(
        response_model=KarakaFrame,
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this sentence using Panini's karaka theory.

Sentence: "{sentence}"

Identify the six karaka roles:
- karta (agent): who autonomously performs the action
- karma (patient): what the agent most directly affects
- karana (instrument): the most effective means used
- sampradana (recipient): for whom or to whom the action is done
- apadana (source): from where or whom something separates
- adhikarana (locus): where or when the action takes place

Policy for motion verbs (walk, run, flow, travel, move, go):
- The origin ('from X') is apadana.
- The destination ('to X', 'into X') is adhikarana — the locus where the motion ends.
- Do NOT assign the destination to karma for motion verbs.

Leave any role as null if it is not present in the sentence.""",
            }
        ],
    )


if __name__ == "__main__":
    import time

    sentences = [
        "The chef sliced the onion with a sharp knife.",
        "Maya gave Ravi a book yesterday at the library.",
        "The teacher explained the lesson to her students.",
        "Water flows from the mountain to the sea.",
        "Anna bought groceries from the market on Sunday.",
    ]

    for i, sentence in enumerate(sentences):
        if i > 0:
            time.sleep(5)  # gemini-2.0-flash free tier: 15 requests/minute
        print(f"\n>>> {sentence}")
        frame = extract_karaka(sentence)
        print(frame.model_dump_json(indent=2))
        print("---")
