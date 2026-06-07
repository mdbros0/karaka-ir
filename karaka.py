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

    lemma: str = Field(description="The base form of the noun — always singular. Use the head noun alone, stripping quality adjectives ('knife' not 'sharp knife', 'artifact' not 'ancient artifacts', 'gift' not 'birthday gift'). Exception: preserve compound nouns where the modifier is a brand name or category-specifier ('coach bag' not 'bag', 'computer science' not 'science', 'electronics store' not 'store'). Preserve capitalization of proper nouns ('Bhoomika', 'University of Guelph'). Use lowercase for common nouns. Strip possessives ('mother' not 'Priya's mother').")
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
- If a verb has conjoined objects ('has a bag and a cold'), produce one frame per object so each gets its own karma.
- Resolve pronouns to the named entity they refer to. If 'she' refers to 'Bhoomika', use 'Bhoomika' as the lemma, not 'she'. Always prefer the proper name over a pronoun.
- Shared participants should appear in each relevant frame using their resolved name.
- For motion verbs (walk, run, flow, travel, move, go): origin ('from X') is apadana; destination ('to X') is adhikarana.
- For identity/classification verbs (be, am, is, are): karta is the subject, karma is the predicate noun ('she is a girl' → karta=Bhoomika, karma=girl). Origin markers ('from X') on identity verbs are apadana ('is from Faridabad' → verb=be, karta=Bhoomika, apadana=Faridabad).
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
