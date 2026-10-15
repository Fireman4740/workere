"""Composable inference only. No evaluator or gold relation is available to a component."""
from __future__ import annotations

from benchmark.schema import Prediction


class ModularPipeline:
    def __init__(self, extractor, candidates=None, predictor=None):
        self.extractor, self.candidates, self.predictor = extractor, candidates, predictor

    def predict(self, inputs, context):
        entities = tuple(self.extractor.extract(inputs, context))
        Prediction(inputs.document.id, entities).validate(inputs.document)
        if context.protocol == "entity_extraction":
            return Prediction(inputs.document.id, entities)
        candidates = tuple(self.candidates.generate(inputs, entities, context))
        for pair in candidates:
            if len(pair) != 2 or not all(isinstance(x, str) for x in pair):
                raise ValueError("candidate generators must return string ID pairs")
        relations = tuple(self.predictor.predict(inputs.document, entities, candidates, context))
        return Prediction(inputs.document.id, entities, relations, context.prediction_metadata)

    def close(self):
        for component in (self.extractor, self.candidates, self.predictor):
            close = getattr(component, "close", None)
            if close is not None:
                close()


class ProcessedPipeline:
    def __init__(self, pipeline, processors):
        self.pipeline, self.processors = pipeline, processors

    def predict(self, inputs, context):
        prediction = self.pipeline.predict(inputs, context).validate(inputs.document)
        for processor in self.processors:
            prediction = processor.process(inputs.document, prediction, context).validate(inputs.document)
        return prediction

    def close(self):
        for component in (self.pipeline, *self.processors):
            close = getattr(component, "close", None)
            if close is not None:
                close()


def component_specs(config, protocol):
    allowed = {"entity_extractor", "candidate_generator", "relation_predictor", "postprocessors", "monolithic"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("unknown pipeline fields")
    if "monolithic" in config:
        if set(config) - {"monolithic", "postprocessors"}:
            raise ValueError("monolithic and modular pipeline components are mutually exclusive")
        specs = [("pipeline", config["monolithic"])]
    else:
        keys = ["entity_extractor"]
        if protocol != "entity_extraction":
            keys += ["candidate_generator", "relation_predictor"]
        elif {"candidate_generator", "relation_predictor"} & set(config):
            raise ValueError("entity_extraction does not run candidate or relation components")
        if any(key not in config for key in keys):
            raise ValueError(f"pipeline requires {keys}")
        specs = [(key, config[key]) for key in keys]
    processors = config.get("postprocessors", [])
    if not isinstance(processors, list):
        raise ValueError("postprocessors must be an array")
    return specs + [("postprocessor", spec) for spec in processors]


def build_pipeline(config, protocol, registry, policy):
    specs = component_specs(config, protocol)
    # Resolve the entire graph before constructing any plugin (including remote clients).
    plugins = [registry.resolve(kind, spec, policy) for kind, spec in specs]
    built = [plugin.factory(**spec.get("params", {})) for plugin, (_, spec) in zip(plugins, specs)]
    if "monolithic" in config:
        pipeline, offset = built[0], 1
    elif protocol == "entity_extraction":
        pipeline, offset = ModularPipeline(built[0]), 1
    else:
        pipeline, offset = ModularPipeline(*built[:3]), 3
    manifest = [{"kind": kind, "plugin": spec["plugin"], "version": plugin.version,
                 "requires_network": plugin.requires_network, "may_charge": plugin.may_charge,
                 "execution_mode": plugin.mode,
                 "implementation": f"{plugin.factory.__module__}:{plugin.factory.__qualname__}"}
                for (kind, spec), plugin in zip(specs, plugins)]
    return ProcessedPipeline(pipeline, built[offset:]), manifest
