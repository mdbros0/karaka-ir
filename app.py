import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from karaka import extract_karaka, KarakaFrame
from facts import frame_to_facts
from rules import apply_rules
from render import render_answer

ROLES = ("karta", "karma", "karana", "sampradana", "apadana", "adhikarana")

ROLE_LABELS = {
    "karta":      "Kartā · Agent",
    "karma":      "Karma · Patient",
    "karana":     "Karaṇa · Instrument",
    "sampradana": "Sampradāna · Recipient",
    "apadana":    "Apādāna · Source",
    "adhikarana": "Adhikaraṇa · Locus",
}

EXAMPLES = [
    ("Maya gave Ravi a book at the library.", "Who owns the book now?"),
    ("Anna bought groceries from the market on Sunday.", "Where did Anna get the groceries?"),
    (
        "Bhoomika is a girl. She studies at the University of Guelph and has a coach bag. She is from Faridabad.",
        "Tell me about Bhoomika.",
    ),
    ("Water flows from the mountain to the sea.", "Where does the water come from?"),
    ("The foundation donated books to the orphanage.", "Who received the books?"),
    ("The surgeon operated on the patient using a scalpel at the hospital.", "What tool did the surgeon use?"),
]


def _dedup(facts):
    seen, out = set(), []
    for f in facts:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def render_pipeline(sentence: str, question: str) -> None:
    with st.spinner("Extracting kāraka frames…"):
        doc = extract_karaka(sentence)

    n = len(doc.events)
    st.subheader(f"Kāraka Frames  ({n} event{'s' if n != 1 else ''})")

    for i, frame in enumerate(doc.events):
        filled = [(r, getattr(frame, r)) for r in ROLES if getattr(frame, r) is not None]
        label = f"Event {i + 1} — **{frame.verb_root}** ({frame.tense})"
        with st.expander(label, expanded=True):
            if filled:
                cols = st.columns(min(len(filled), 3))
                for j, (role, entity) in enumerate(filled):
                    cols[j % 3].metric(
                        label=ROLE_LABELS[role],
                        value=entity.lemma,
                        delta=entity.entity_type or "unknown",
                    )
            else:
                st.write("_(no roles extracted)_")

    all_base, all_derived = [], []
    for frame in doc.events:
        base = frame_to_facts(frame)
        all_base.extend(base)
        all_derived.extend(apply_rules(base))
    all_base    = _dedup(all_base)
    all_derived = _dedup(all_derived)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Base Facts")
        for f in all_base:
            st.code(f"{f[0]}({', '.join(f[1:])})", language=None)
    with col2:
        st.subheader("Derived Facts")
        if all_derived:
            for f in all_derived:
                st.code(f"{f[0]}({', '.join(f[1:])})", language=None)
        else:
            st.info("No rules fired.")

    with st.spinner("Generating answer…"):
        result = render_answer(question, all_base, all_derived)

    st.subheader("Answer")
    st.success(result.answer)
    with st.expander("Reasoning trace"):
        st.write(result.reasoning)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Karaka-IR", layout="wide")
st.title("Karaka-IR")
st.caption(
    "English sentence → Pāṇinian kāraka extraction → symbolic inference → auditable answer"
)

# ── Sidebar examples ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Examples")
    for sent, q in EXAMPLES:
        label = sent[:55] + ("…" if len(sent) > 55 else "")
        if st.button(label, use_container_width=True, key=sent[:30]):
            st.session_state["_s"] = sent
            st.session_state["_q"] = q
            st.rerun()

# ── Main form ──────────────────────────────────────────────────────────────────
sentence = st.text_area(
    "Sentence",
    value=st.session_state.get("_s", ""),
    placeholder="Maya gave Ravi a book at the library.",
    height=80,
)
question = st.text_input(
    "Question",
    value=st.session_state.get("_q", ""),
    placeholder="Who owns the book now?",
)

if st.button("Analyze", type="primary", use_container_width=True):
    if sentence and question:
        render_pipeline(sentence, question)
    else:
        st.warning("Enter both a sentence and a question.")
