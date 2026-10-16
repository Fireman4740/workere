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
    elif style == "tagged":
        marked = []
        for i, token in enumerate(document.tokens):
            tags = [tag for tag, e in (("A", event_a), ("B", event_b)) if i in e.token_indices]
            marked.append("".join(f"[{tag}]" for tag in tags) + token
                          + "".join(f"[/{tag}]" for tag in reversed(tags)))
        state = state.replace(document.text, " ".join(marked), 1)
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


NOUL_CRITERIA = {
    "true": "The document supports that one of these exact event mentions causes the other.",
    "false": "No supported causal link between the two selected mentions; proximity, correlation "
             "and coreference alone are insufficient."}


def noul_question(instructions=NOUL_INSTRUCTIONS, criteria=None):
    return {"causal": {"type": "noul", "instructions": instructions,
                       "criteria": dict(NOUL_CRITERIA if criteria is None else criteria)}}


def questions_from_params(params):
    """The exact question mapping a predictor configured with `params` sends with every pair."""
    variant = params.get("variant", "noul")
    criteria, extra = params.get("criteria"), params.get("extra_questions") or {}
    if criteria is not None and variant != "noul":
        raise ValueError("custom criteria are supported for noul only")
    if criteria is not None and (set(criteria) != {"true", "false"}
                                 or not all(isinstance(v, str) and v.strip() for v in criteria.values())):
        raise ValueError("noul criteria must map true and false to nonempty strings")
    questions = questions_for(variant, params.get("instructions"))
    if criteria is not None:
        questions["causal"]["criteria"] = dict(criteria)
    if not isinstance(extra, dict):
        raise ValueError("extra_questions must be a mapping")
    for key, spec in extra.items():
        if key in questions:
            raise ValueError(f"extra question id collides with the decision question: {key}")
        questions[key] = noul_question(spec["instructions"], spec["criteria"])["causal"]
    return questions


def request_pairs(document, entities, candidates, unordered=False):
    if not unordered:
        return list(candidates)
    by_id = {e.id: e for e in entities}
    out = {}
    for a, b in candidates:
        pair = tuple(sorted((a, b), key=lambda m: (by_id[m].token_indices, m)))
        out.setdefault(pair, None)
    return list(out)
