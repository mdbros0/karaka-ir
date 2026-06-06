from karaka import KarakaFrame, KarakaDocument

# A Fact is a predicate tuple: (name, arg1, arg2, ...)
# e.g. ("karta", "Maya", "person") means karta(Maya, person)
Fact = tuple[str, ...]


def frame_to_facts(frame: KarakaFrame) -> list[Fact]:
    """Convert a KarakaFrame into a flat list of predicate tuples."""
    facts: list[Fact] = []

    facts.append(("action", frame.verb_root, frame.tense))

    for role_name in ("karta", "karma", "karana", "sampradana", "apadana", "adhikarana"):
        entity = getattr(frame, role_name)
        if entity is not None:
            facts.append((role_name, entity.lemma, entity.entity_type or "unknown"))

    return facts


def document_to_facts(doc: KarakaDocument) -> list[Fact]:
    """Merge base facts from all events, removing exact duplicates while preserving order.

    Duplicates arise when the same participant (e.g. karta=Bhoomika) appears in
    multiple frames. We keep the first occurrence.
    """
    seen: set[Fact] = set()
    merged: list[Fact] = []
    for frame in doc.events:
        for fact in frame_to_facts(frame):
            if fact not in seen:
                seen.add(fact)
                merged.append(fact)
    return merged
