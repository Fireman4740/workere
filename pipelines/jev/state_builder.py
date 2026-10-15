"""Only document text and the two permitted reference mentions enter a model request."""
import json

NOUL_INSTRUCTIONS = (
    "Given the document and the two specified event mentions, is there a causal relationship "
    "between these events? Answer yes only when the context supports causation, not merely "
    "temporal proximity, correlation, or coreference."
)


def build_state(document, event_a, event_b, style="document"):
    def event(entity):
        return {"id": entity.id, "surface": " ".join(document.tokens[i] for i in entity.token_indices),
                "token_indices": list(entity.token_indices), "char_offsets": entity.char_offsets(document),
                "sentence": document.sentence_of(entity.token_indices)}
    state = (f"Document ID: {document.id}\nDocument:\n{document.text}\n\n"
            f"Event A: {json.dumps(event(event_a), ensure_ascii=False)}\n"
            f"Event B: {json.dumps(event(event_b), ensure_ascii=False)}")
    if style == "explicit_context":
        marked = []
        for i, token in enumerate(document.tokens):
            tags = [tag for tag, e in (("A", event_a), ("B", event_b)) if i in e.token_indices]
            marked.append("".join(f"[EVENT_{tag}]" for tag in tags) + token
                          + "".join(f"[/EVENT_{tag}]" for tag in reversed(tags)))
        state += "\n\nSame complete document with only the selected event tokens marked:\n" + " ".join(marked)
        for tag, e in (("A", event_a), ("B", event_b)):
            sentence = document.sentence_of(e.token_indices)
            if sentence is not None:
                start, end = document.sentences[sentence]
                state += f"\nSentence containing event {tag}: " + " ".join(document.tokens[start:end])
    elif style != "document":
        raise ValueError("unknown context style")
    return state


def choice_question():
    return {"relation": {"type": "choice", "instructions":
        "Which causal relationship between these exact event mentions is supported by the document? "
        "Temporal proximity, correlation and coreference alone do not imply causation.",
        "criteria": {"CAUSES": "Event A causes event B.",
                     "CAUSED_BY": "Event B causes event A.",
                     "NO_RELATION": "Neither causal direction is supported by the document."}}}


def questions_for(variant, instructions=None):
    if instructions is not None and (not isinstance(instructions, str) or not instructions.strip()):
        raise ValueError("decision instructions must be a nonempty string")
    if variant == "choice":
        question = choice_question()
        if instructions is not None:
            question["relation"]["instructions"] = instructions
        return question
    if variant == "noul":
        return noul_question(NOUL_INSTRUCTIONS if instructions is None else instructions)
    raise ValueError("variant must be noul or choice")


def noul_question(instructions=NOUL_INSTRUCTIONS):
    return {"causal": {"type": "noul", "instructions": instructions, "criteria": {
        "true": "The document supports that one of these exact event mentions causes the other.",
        "false": "No supported causal link between the two selected mentions; proximity, correlation "
                 "and coreference alone are insufficient."}}}
