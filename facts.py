from karaka import KarakaFrame

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
