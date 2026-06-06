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
        default=None, description="person, object, location, time, organization, etc."
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


class KarakaDocument(BaseModel):
    """All events extracted from a sentence — one frame per verb."""

    events: list[KarakaFrame] = Field(
        description="One KarakaFrame per distinct verb/event in the sentence, in order of appearance"
    )


def extract_karaka(sentence: str) -> KarakaDocument:
    return client.create(
        response_model=KarakaDocument,
        config={"temperature": 0.0},
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this sentence and extract ALL events using Panini's karaka theory.
Produce one KarakaFrame for each distinct verb or event in the sentence.

Sentence: "{sentence}"

For each event, identify:
- karta (agent): who autonomously performs the action
- karma (patient): what the action most directly affects
- karana (instrument): the most effective means used
- sampradana (recipient): for whom or to whom the action is done
- apadana (source): from where or whom something separates
- adhikarana (locus): where or when the action takes place

Policies:
- Produce one frame per verb — do not merge separate events into one frame.
- Shared participants (e.g., the same person is karta of multiple events) should appear in each relevant frame.
- For motion verbs (walk, run, flow, travel, move, go): origin ('from X') is apadana; destination ('to X') is adhikarana.
- For identity/classification verbs (be, am, is, are): karta is the subject, karma is the predicate noun ('she is a girl' → karta=she, karma=girl).
- Leave any role null if not expressed in that event.""",
            }
        ],
    )


if __name__ == "__main__":
    import time

    sentences = [
        "The chef sliced the onion with a sharp knife.",
        "Maya gave Ravi a book at the library.",
        "Bhoomika studies at the University of Guelph and has a Coach bag.",
    ]

    for i, sentence in enumerate(sentences):
        if i > 0:
            time.sleep(5)
        print(f"\n>>> {sentence}")
        doc = extract_karaka(sentence)
        for j, frame in enumerate(doc.events):
            print(f"  Event {j+1}: {frame.verb_root}")
            print(frame.model_dump_json(indent=2))
        print("---")
