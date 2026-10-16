"""No-label prerequisite check for adding multiple Noul questions to a request."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import uuid

from benchmark.io import read_json, write_json
from benchmark.protocols import public_inputs
from pipelines.jev.state_builder import build_state


def check_multi_questions(client, questions, path, max_workers):
    suite = read_json(path)
    inputs = list(public_inputs(suite, "relations_gold"))
    if len(inputs) != 1 or inputs[0].document.id != "41_9ecbplus.xml":
        raise ValueError("multi-question preflight must use the already-seen smoke document")
    item = inputs[0]
    entities = {e.id: e for e in item.reference_entities}
    Path("logs").mkdir(exist_ok=True)
    receipt = Path("logs") / ("multi-question-preflight-" + uuid.uuid4().hex[:12])
    lock = threading.Lock()
    with receipt.with_suffix(".jsonl").open("x") as journal:
        def one(pair):
            a, b = pair
            state = build_state(item.document, entities[a], entities[b])
            def record(event):
                with lock:
                    journal.write(json.dumps({"document_id": item.document.id, "source": a, "target": b, **event}) + "\n")
                    journal.flush()
            single = client.decide(state, {"causal": questions["causal"]}, on_attempt=record)
            multi = client.decide(state, questions, on_attempt=record)
            answers = {key: multi["response"]["answers"][key]["noul"] for key in questions}
            return {"source": a, "target": b, "single_causal": single["response"]["answers"]["causal"]["noul"],
                    "answers": answers, "served_model": multi["response"]["model"]}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            rows = list(pool.map(one, item.reference_candidates))
    max_delta = max(abs(r["single_causal"] - r["answers"]["causal"]) for r in rows)
    distinct = sum(len(set(r["answers"].values())) > 1 for r in rows)
    result = {"document_id": item.document.id, "pairs": len(rows), "model": client.identity["model"],
              "served_models": sorted({r["served_model"] for r in rows}), "max_causal_probability_delta": max_delta,
              "pairs_with_distinct_question_answers": distinct, "api_attempts_including_cache_misses": client.calls,
              "conservative_spend_usd": client.spent, "passed": max_delta <= 1e-8 and distinct > 0,
              "note": "Operational question isolation check, not a proof of statistical independence."}
    result["receipt_path"] = str(receipt.with_suffix(".json"))
    write_json(receipt.with_suffix(".json"), {**result, "rows": rows})
    print("MULTI_QUESTION_PREFLIGHT " + json.dumps(result), flush=True)
    if not result["passed"]:
        raise ValueError("multi-question prerequisite failed; DEV-SCREEN was not queried")
    return result
