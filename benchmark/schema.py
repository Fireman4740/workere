"""Versioned, immutable model-facing objects. No gold annotations live in Document."""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Any

PREDICTION_SCHEMA = "workere-prediction/v2"
DOCUMENT_SCHEMA = "workere-document/v2"
PROTOCOLS = ("entity_extraction", "relations_gold", "end_to_end")


def _id(value, what):
    if not isinstance(value, str) or not value or "," in value:
        raise ValueError(f"{what} must be a nonempty string without commas")
    return value


def probability(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                              or not 0 <= value <= 1):
        raise ValueError("confidence must be null or a finite number in [0,1]")
    return value


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    tokens: tuple[str, ...]
    sentences: tuple[tuple[int, int], ...]
    lang: str = "eng"

    def __post_init__(self):
        _id(self.id, "document id")
        if not isinstance(self.tokens, (tuple, list)) or not isinstance(self.sentences, (tuple, list)):
            raise ValueError("tokens and sentences must be arrays")
        object.__setattr__(self, "tokens", tuple(self.tokens))
        object.__setattr__(self, "sentences", tuple(tuple(s) for s in self.sentences))
        if not all(isinstance(t, str) for t in self.tokens):
            raise ValueError("tokens must be strings")
        for start, end in self.sentences:
            if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(self.tokens):
                raise ValueError("invalid sentence token range")

    @property
    def text(self):
        return " ".join(self.tokens)

    @property
    def token_char_offsets(self):
        offset, spans = 0, []
        for token in self.tokens:
            spans.append((offset, offset + len(token)))
            offset += len(token) + 1
        return tuple(spans)

    def sentence_of(self, indices):
        return next((i for i, (start, end) in enumerate(self.sentences)
                     if indices and start <= indices[0] < end), None)

    def to_dict(self):
        return {"schema": DOCUMENT_SCHEMA, "id": self.id, "tokens": list(self.tokens),
                "sentences": [list(s) for s in self.sentences], "lang": self.lang,
                "text": self.text, "token_char_offsets": [list(s) for s in self.token_char_offsets]}


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    token_indices: tuple[int, ...]
    label: str | None = None
    confidence: float | None = None

    def __post_init__(self):
        _id(self.id, "entity id")
        if not isinstance(self.token_indices, (tuple, list)):
            raise ValueError("entity token_indices must be an array")
        object.__setattr__(self, "token_indices", tuple(self.token_indices))
        if (not self.token_indices or any(type(i) is not int or i < 0 for i in self.token_indices)
                or tuple(sorted(set(self.token_indices))) != self.token_indices):
            raise ValueError("entity token_indices must be nonempty, sorted, unique nonnegative integers")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("entity label must be a string or null")
        probability(self.confidence)

    def char_offsets(self, document):
        chars, segments = document.token_char_offsets, []
        for index in self.token_indices:
            if index >= len(chars):
                raise ValueError(f"entity {self.id}: token index out of bounds")
            if segments and index - 1 in self.token_indices:
                segments[-1][1] = chars[index][1]
            else:
                segments.append(list(chars[index]))
        return segments

    def to_dict(self, document):
        return {"id": self.id, "token_indices": list(self.token_indices),
                "char_offsets": self.char_offsets(document), "label": self.label,
                "confidence": self.confidence}


@dataclass(frozen=True, slots=True)
class Relation:
    source: str
    target: str
    label: str
    confidence: float | None = None

    def __post_init__(self):
        _id(self.source, "relation source")
        _id(self.target, "relation target")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("relation label must be a nonempty string")
        probability(self.confidence)

    def to_dict(self):
        return {"source": self.source, "target": self.target, "label": self.label,
                "confidence": self.confidence}


@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    entities: tuple[Entity, ...] = ()
    relations: tuple[Relation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _id(self.id, "prediction document id")
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "relations", tuple(self.relations))
        if not isinstance(self.metadata, dict):
            raise ValueError("prediction metadata must be an object")
        json.dumps(self.metadata, allow_nan=False)

    def validate(self, document):
        if self.id != document.id:
            raise ValueError("prediction document ID mismatch")
        if len({e.id for e in self.entities}) != len(self.entities):
            raise ValueError("duplicate entity IDs")
        for entity in self.entities:
            if not isinstance(entity, Entity):
                raise ValueError("entities must contain Entity objects")
            entity.char_offsets(document)
        if not all(isinstance(r, Relation) for r in self.relations):
            raise ValueError("relations must contain Relation objects")
        return self

    def to_dict(self, document):
        self.validate(document)
        return {"schema": PREDICTION_SCHEMA, "id": self.id,
                "entities": [e.to_dict(document) for e in self.entities],
                "relations": [r.to_dict() for r in self.relations], "metadata": self.metadata}

    @classmethod
    def from_dict(cls, row, document):
        if row.get("schema") != PREDICTION_SCHEMA:
            raise ValueError("unsupported prediction schema")
        allowed = {"schema", "id", "entities", "relations", "metadata"}
        if set(row) - allowed or not {"id", "entities", "relations"} <= set(row):
            raise ValueError("invalid prediction fields")
        entities = []
        for item in row["entities"]:
            if set(item) - {"id", "token_indices", "char_offsets", "label", "confidence"}:
                raise ValueError("unknown entity field")
            entity = Entity(item["id"], tuple(item["token_indices"]), item.get("label"), item.get("confidence"))
            if "char_offsets" in item and item["char_offsets"] != entity.char_offsets(document):
                raise ValueError("character offsets disagree with token indices and canonical text")
            entities.append(entity)
        relations = tuple(Relation(**item) for item in row["relations"])
        return cls(row["id"], tuple(entities), relations, row.get("metadata", {})).validate(document)


@dataclass(frozen=True, slots=True)
class PipelineInput:
    document: Document
    # Only relations_gold exposes these, explicitly marking the mention-oracle protocol.
    reference_entities: tuple[Entity, ...] | None = None
    reference_candidates: tuple[tuple[str, str], ...] | None = None

    def to_dict(self):
        row = self.document.to_dict()
        if self.reference_entities is not None:
            row["oracle"] = {"kind": "reference_mentions",
                             "entities": [e.to_dict(self.document) for e in self.reference_entities],
                             "candidate_pairs": [list(p) for p in self.reference_candidates or ()]}
        return row
