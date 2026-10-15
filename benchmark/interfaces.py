"""Public component contracts; plugins receive raw documents and an execution context only."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from benchmark.schema import Document, Entity, PipelineInput, Prediction, Relation


@dataclass
class Context:
    seed: int
    protocol: str
    document_id: str
    cost_usd: float | None = 0.0
    usage: list[dict] = field(default_factory=list)
    prediction_metadata: dict = field(default_factory=dict)
    usage_sink: object = None
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(f"{self.seed}::{self.document_id}")

    def record_usage(self, *, cost_usd=None, **details):
        if cost_usd is not None and (type(cost_usd) not in (float, int) or not math.isfinite(cost_usd)
                                     or cost_usd < 0):
            raise ValueError("invalid provider cost")
        self.cost_usd = None if cost_usd is None or self.cost_usd is None else self.cost_usd + cost_usd
        self.usage.append({"cost_usd": cost_usd, **details})
        if self.usage_sink is not None:
            self.usage_sink(self.usage[-1])


class EntityExtractor(Protocol):
    def extract(self, inputs: PipelineInput, context: Context) -> tuple[Entity, ...]: ...


class CandidateGenerator(Protocol):
    def generate(self, inputs: PipelineInput, entities: tuple[Entity, ...],
                 context: Context) -> tuple[tuple[str, str], ...]: ...


class RelationPredictor(Protocol):
    def predict(self, document: Document, entities: tuple[Entity, ...],
                candidates: tuple[tuple[str, str], ...], context: Context) -> tuple[Relation, ...]: ...


class PostProcessor(Protocol):
    def process(self, document: Document, prediction: Prediction, context: Context) -> Prediction: ...


class Pipeline(Protocol):
    def predict(self, inputs: PipelineInput, context: Context) -> Prediction: ...
