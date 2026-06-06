from facts import Fact

TRANSFER_VERBS = {"give", "send", "lend", "donate", "hand"}
PURCHASE_VERBS = {"buy", "purchase", "acquire"}


def apply_rules(facts: list[Fact]) -> list[Fact]:
    """Derive new facts from existing ones using inference rules."""
    index: dict[str, list[tuple[str, ...]]] = {}
    for fact in facts:
        index.setdefault(fact[0], []).append(fact[1:])

    def get(predicate: str) -> tuple[str, ...] | None:
        entries = index.get(predicate)
        return entries[0] if entries else None

    derived: list[Fact] = []

    action = get("action")
    karta = get("karta")
    karma = get("karma")
    sampradana = get("sampradana")
    apadana = get("apadana")
    adhikarana = get("adhikarana")

    # Rule 1: transfer of possession
    # give(karta, karma, sampradana) → owns(sampradana, karma)
    if action and action[0] in TRANSFER_VERBS and karma and sampradana:
        derived.append(("owns", sampradana[0], karma[0]))

    # Rule 2: purchase implies prior ownership by the source
    # buy(karta, karma, apadana) → formerly_owned(apadana, karma)
    if action and action[0] in PURCHASE_VERBS and karma and apadana:
        derived.append(("formerly_owned", apadana[0], karma[0]))

    # Rule 3: karta with entity_type=person → agent is human
    if karta and karta[1] == "person":
        derived.append(("agent_is_human", karta[0]))

    # Rule 4: adhikarana present → record event location or time
    if adhikarana:
        label = "event_time" if adhikarana[1] == "time" else "event_location"
        derived.append((label, adhikarana[0]))

    return derived
