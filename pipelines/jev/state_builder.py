"""Only document text and the two permitted reference mentions enter a model request."""
import json

NOUL_INSTRUCTIONS = (
    "Given the document and the two specified event mentions, is there a causal relationship "
    "between these events? Answer yes only when the context supports causation, not merely "
    "temporal proximity, correlation, or coreference."
)


def build_state(document, event_a, event_b):
    def event(entity):
        return {"id": entity.id, "surface": " ".join(document.tokens[i] for i in entity.token_indices),
                "token_indices": list(entity.token_indices), "char_offsets": entity.char_offsets(document),
                "sentence": document.sentence_of(entity.token_indices)}
    return (f"Document ID: {document.id}\nDocument:\n{document.text}\n\n"
            f"Event A: {json.dumps(event(event_a), ensure_ascii=False)}\n"
            f"Event B: {json.dumps(event(event_b), ensure_ascii=False)}")


def noul_question(instructions=NOUL_INSTRUCTIONS):
    return {"causal": {"type": "noul", "instructions": instructions, "criteria": {
        "true": "The document supports that one of these exact event mentions causes the other.",
        "false": "No supported causal link between the two selected mentions; proximity, correlation "
                 "and coreference alone are insufficient."}}}
