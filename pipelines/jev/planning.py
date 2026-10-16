"""Zero-API request plan. Dataset gold is used only for evaluator-side descriptive counts."""
from collections import Counter

from benchmark.configuration import load_suite
from benchmark.protocols import public_inputs
from pipelines.jev.client import OPENROUTER_ENDPOINT, is_free
from pipelines.jev.state_builder import build_state, questions_from_params, request_pairs


def plan(config):
    cfg = config.data
    if cfg["protocol"] != "relations_gold":
        raise ValueError("JEV v1 requires relation_gold_mentions")
    params = cfg["pipeline"]["relation_predictor"]["params"]
    client = params["client"]
    suite = load_suite(config)
    variant = params.get("variant", "noul")
    if variant == "choice" and client["model"].startswith("respan/span-"):
        raise ValueError("Span does not support Choice; no requests submitted")
    questions = questions_from_params(params)
    demonstrations = None
    if params.get("demonstrations"):
        from pipelines.jev.demonstrations import RetrievedDemonstrations
        demonstrations = RetrievedDemonstrations(**params["demonstrations"])
    scopes, docs, examples, overflows = Counter(), [], [], []
    total_bytes = 0
    for inputs in public_inputs(suite, "relations_gold"):
        doc, refs = inputs.document, {e.id: e for e in inputs.reference_entities}
        queries = request_pairs(doc, inputs.reference_entities, inputs.reference_candidates, params.get("unordered", False))
        for a, b in queries:
            state = build_state(doc, refs[a], refs[b], params.get("context_style", "document"))
            if demonstrations:
                state = demonstrations.prefix(doc, refs[a], refs[b]) + state
            body = {"model": client["model"], "state": state, "questions": questions}
            import json
            bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
            total_bytes += bound
            if bound > client.get("max_context_tokens", 32768):
                overflows.append({"id": doc.id, "pair": [a, b], "conservative_tokens": bound})
            sa, sb = doc.sentence_of(refs[a].token_indices), doc.sentence_of(refs[b].token_indices)
            scopes["unknown" if sa is None or sb is None else "intra" if sa == sb else "inter"] += 1
            if len(examples) < 2:
                examples.append(body)
        docs.append({"id": doc.id, "mentions": len(refs), "calls": len(queries),
                     "ordered_candidates": len(inputs.reference_candidates)})
    n_calls = sum(d["calls"] for d in docs)
    free = is_free(client)
    return {"schema": "workere-jev-plan/v1", "name": cfg["name"], "mode": "dry-run", "api_calls_made": 0,
            "protocol": "relation_gold_mentions", "suite_fingerprint": suite["fingerprint"],
            "model": client["model"], "model_version": client["model_version"],
            "provider": client.get("provider", "openrouter"), "endpoint": client.get("endpoint", OPENROUTER_ENDPOINT),
            "documents": docs, "document_count": len(docs), "calls_without_cache_or_retries": n_calls,
            "max_attempts_with_retries": n_calls * (1 + client.get("retries", 2)), "scopes": dict(scopes),
            "context_overflows": overflows, "request_examples_without_gold_relations": examples,
            "requested_max_documents": cfg["dataset"].get("max_examples"),
            "estimated_upper_input_tokens": total_bytes,
            "estimated_input_cost_usd_at_configured_rate": 0 if free else total_bytes * client.get("input_price_per_million", .042) / 1e6,
            "conservative_reservation_envelope_usd": 0 if free else n_calls * (1 + client.get("retries", 2)) * client.get("max_cost_per_call_usd", .002),
            "paid_budget_authorized_in_config": client.get("budget_usd", 0),
            "historical_parity": "not_demonstrated_without_original_run_files"}
