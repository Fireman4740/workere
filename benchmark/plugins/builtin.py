"""Lightweight baselines and read-only adapters. No external model SDK is imported."""
from __future__ import annotations

import hashlib
from pathlib import Path

from benchmark.io import read_json, read_jsonl
from benchmark.labels import CAUSED_BY, CAUSES, NOREL
from benchmark.registry import register
from benchmark.schema import Entity, Prediction, Relation, probability


@register("entity_extractor", "reference")
class ReferenceEntities:
    def extract(self, inputs, context):
        if context.protocol != "relations_gold" or inputs.reference_entities is None:
            raise ValueError("reference extractor requires the explicit relations_gold oracle protocol")
        return inputs.reference_entities


@register("entity_extractor", "lexicon")
class LexiconEntities:
    def __init__(self, terms, confidence=1.0):
        if not isinstance(terms, list) or not all(isinstance(term, str) and term.strip() for term in terms):
            raise ValueError("lexicon terms must be an array of nonempty strings")
        self.terms = [tuple(term.lower().split()) for term in terms]
        self.confidence = probability(confidence)

    def extract(self, inputs, context):
        tokens = tuple(token.lower() for token in inputs.document.tokens)
        spans = {tuple(range(start, start + len(term))) for term in self.terms
                 for start in range(len(tokens) - len(term) + 1)
                 if tokens[start:start + len(term)] == term}
        return tuple(Entity(f"p{i}", span, confidence=self.confidence) for i, span in enumerate(sorted(spans)))


@register("candidate_generator", "historical")
class HistoricalCandidates:
    def generate(self, inputs, entities, context):
        if context.protocol != "relations_gold" or inputs.reference_candidates is None:
            raise ValueError("historical candidates require relations_gold")
        return inputs.reference_candidates


@register("candidate_generator", "all_ordered")
class AllOrderedCandidates:
    def __init__(self, intra_only=False):
        if type(intra_only) is not bool:
            raise ValueError("intra_only must be boolean")
        self.intra_only = intra_only

    def generate(self, inputs, entities, context):
        doc = inputs.document
        return tuple((a.id, b.id) for a in entities for b in entities if a.id != b.id
                     and (not self.intra_only or (doc.sentence_of(a.token_indices) is not None
                          and doc.sentence_of(a.token_indices) == doc.sentence_of(b.token_indices))))


@register("relation_predictor", "heuristic")
class HeuristicRelations:
    def __init__(self, confidence=.9):
        self.confidence = probability(confidence)

    def predict(self, document, entities, candidates, context):
        by_id, out = {e.id: e for e in entities}, []
        for source, target in candidates:
            a, b = by_id[source], by_id[target]
            if document.sentence_of(a.token_indices) != document.sentence_of(b.token_indices):
                continue
            first, second = sorted((a, b), key=lambda e: e.token_indices)
            between = {t.lower() for t in document.tokens[first.token_indices[-1] + 1:second.token_indices[0]]}
            forward = bool(between & {"cause", "causes", "caused", "causing"})
            reverse = "because" in between
            if not (forward or reverse):
                continue
            label = CAUSES if (a is first) == forward else CAUSED_BY
            out.append(Relation(source, target, label, self.confidence))
        return tuple(out)


@register("relation_predictor", "decision_scores", mode="replay")
class DecisionScores:
    """Read generic choice probabilities; JEV outputs can be mapped to this file format."""
    def __init__(self, path, labels=None, missing="error"):
        if missing not in ("error", "norel"):
            raise ValueError("missing must be error or norel")
        self.labels = labels or {"A": CAUSES, "B": CAUSED_BY, "C": NOREL}
        if not isinstance(self.labels, dict) or not all(isinstance(v, str) for v in self.labels.values()):
            raise ValueError("decision label mapping must map option keys to label strings")
        self.missing, self.rows = missing, {}
        for row in read_jsonl(path):
            key = (row["id"], row["source"], row["target"])
            if key in self.rows:
                raise ValueError(f"duplicate decision score: {key}")
            probs = row["probabilities"]
            if set(probs) != set(self.labels) or not probs:
                raise ValueError("decision probability options differ from the label mapping")
            for value in probs.values():
                if value is None:
                    raise ValueError("probabilities cannot be null")
                probability(value)
            if abs(sum(probs.values()) - 1) > 1e-4:
                raise ValueError("choice probabilities must sum to one")
            self.rows[key] = probs

    def predict(self, document, entities, candidates, context):
        out = []
        for source, target in candidates:
            probs = self.rows.get((document.id, source, target))
            if probs is None:
                if self.missing == "error":
                    raise ValueError(f"missing decision probabilities for {document.id}/{source},{target}")
                continue
            # Stable option-key order is the documented tie rule for this producer.
            choice = max(sorted(probs), key=probs.get)
            out.append(Relation(source, target, self.labels[choice], probs[choice]))
        return tuple(out)


@register("postprocessor", "identity")
class Identity:
    def process(self, document, prediction, context):
        return prediction


@register("postprocessor", "relation_threshold")
class RelationThreshold:
    def __init__(self, threshold=.5, missing_confidence="error"):
        probability(threshold)
        if threshold is None or missing_confidence not in ("error", "keep", "drop"):
            raise ValueError("invalid threshold or missing_confidence policy")
        self.threshold, self.missing = threshold, missing_confidence

    def process(self, document, prediction, context):
        kept = []
        for relation in prediction.relations:
            if relation.confidence is None:
                if self.missing == "error":
                    raise ValueError("relation confidence required by threshold postprocessor")
                keep = self.missing == "keep"
            else:
                keep = relation.confidence >= self.threshold
            if keep:
                kept.append(relation)
        return Prediction(prediction.id, prediction.entities, tuple(kept), prediction.metadata)


@register("pipeline", "saved_predictions", mode="replay")
class SavedPredictions:
    def __init__(self, path):
        self.rows = {}
        for row in read_jsonl(path):
            if row["id"] in self.rows:
                raise ValueError("duplicate document in saved predictions")
            self.rows[row["id"]] = row

    def predict(self, inputs, context):
        return Prediction.from_dict(self.rows[inputs.document.id], inputs.document)


@register("pipeline", "legacy_report", mode="replay")
class LegacyReport:
    """Prediction-only view of a report. The engine separately verifies its historical parity."""
    def __init__(self, path):
        payload = read_json(path)
        self.by_id = {row["id"]: [] for row in payload["results"]["per_doc_metrics"]}
        for row in payload["results"]["per_pair_predictions"]:
            source, target = row["pair"].split(",", 1)
            self.by_id[row["id"]].append(Relation(source, target, row["pred"]))
        self.sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def predict(self, inputs, context):
        if context.protocol != "relations_gold" or inputs.reference_entities is None:
            raise ValueError("legacy report adapter requires relations_gold")
        return Prediction(inputs.document.id, inputs.reference_entities,
                          tuple(self.by_id[inputs.document.id]), {"source_report_sha256": self.sha})
