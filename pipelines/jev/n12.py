"""Frozen N12 factorial 2x2, with mixed fresh generations and pre-gold fusion."""
import base64
import collections
import concurrent.futures
import datetime
import hashlib
import io
import json
import math
import os
import pathlib
import random
import sys
import threading
import urllib.request
import zipfile

P = pathlib.Path("experiments/jev/n12")
ARMS = ("original", "local", "table")


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def f1(counts):
    return 200 * counts[0] / (2 * counts[0] + counts[1] + counts[2]) if counts[0] else 0.


def views(d):
    # Mark each original token before selecting context. No IDs are renumbered.
    starts, ends = {}, {}
    for mid, indices in zip(d["mentions"], d["spans"]):
        for i in indices:
            if i - 1 not in indices: starts.setdefault(i, []).append(mid)
            if i + 1 not in indices: ends.setdefault(i, []).append(mid)
    tokens = ["".join(f"<{mid} " for mid in starts.get(i, [])) + token
              + ">" * len(ends.get(i, [])) for i, token in enumerate(d["tokens"])]
    assert " ".join(tokens) == d["text"]
    selected = set()
    for indices in d["spans"]:
        for i in indices:
            sid = next(s for s, (a, b) in enumerate(d["sentences"]) if a <= i < b)
            selected.update(range(max(0, sid - 1), min(len(d["sentences"]), sid + 2)))
    kept = set(i for s in selected for i in range(*d["sentences"][s]))
    assert all(set(sp) <= kept for sp in d["spans"])
    parts, previous = [], -1
    for sid in sorted(selected):
        if sid > previous + 1: parts.append("[non-mention context omitted]")
        a, b = d["sentences"][sid]; parts.append(" ".join(tokens[a:b])); previous = sid
    if previous < len(d["sentences"]) - 1: parts.append("[non-mention context omitted]")
    local = " ".join(parts)
    table = "\n\nOccurrence inventory (same original IDs; no event identity inferred):\n"
    for mid, sp in zip(d["mentions"], d["spans"]):
        sids = sorted({s for s, (a,b) in enumerate(d["sentences"]) if any(a <= i < b for i in sp)})
        table += json.dumps({"id": mid, "sentences_zero_based": sids,
                             "surface": " ".join(d["tokens"][i] for i in sp)}, ensure_ascii=False) + "\n"
    return {"original": d["text"], "local": local, "table": d["text"] + table}, {
        "doc": d["id"], "total_sentences": len(d["sentences"]),
        "kept_sentences": len(selected), "kept_tokens": len(kept), "total_tokens": len(tokens),
        "all_mentions_retained": True, "local_equals_original": local == d["text"]}


def prepare_for(documents, seeds, reads, arms=ARMS):
    from benchmark.data import parse_annotations
    bank = read("experiments/jev/n8/demo-bank.json")["documents"]
    prompt = read("experiments/jev/n8/prompt.json")
    requests, inventory = [], []
    for d in documents:
        texts, info = views(d); inventory.append(info)
        eligible = [b for b in bank if b["id"].split("_")[0] != d["id"].split("_")[0]]
        for schedule, seed in enumerate(seeds):
            demos = random.Random(f"{seed}::{d['id']}").sample(eligible, reads)
            for slot, demo in enumerate(demos):
                answer = [{"pair": f"{a},{b}", "label": lab}
                          for a, lab, b in parse_annotations(demo["annots"]) if lab != "norel"]
                prefix = [{"role":"system", "content":prompt["system"]},
                          {"role":"user", "content":prompt["user_template"].format(doc_text=demo["text"])},
                          {"role":"assistant", "content":json.dumps(answer)}]
                for arm in arms:
                    messages = prefix + [{"role":"user", "content":prompt["user_template"].format(doc_text=texts[arm])}]
                    body = {"model":"openai/gpt-6-luna:batch", "messages":messages,
                            "reasoning":{"effort":"high"}, "max_tokens":32768}
                    bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
                    logical = f"s{schedule}:{arm}:{d['id']}:{slot}"
                    requests.append({"logical_id":logical, "custom_id":"n12_"+sha(logical.encode()),
                                     "schedule":schedule, "seed":seed, "arm":arm,"doc":d["id"],
                                     "slot":slot,"demo":demo["id"],"body":body,
                                     "body_sha256":sha(canonical(body)),"input_utf8_bound":bound})
    assert len(requests) == len({r["logical_id"] for r in requests})
    for r in requests:
        assert r["demo"].split("_")[0] != r["doc"].split("_")[0]
        assert r["body"]["messages"][:3] == next(
            x["body"]["messages"][:3] for x in requests
            if (x["doc"], x["schedule"], x["slot"]) == (r["doc"],r["schedule"],r["slot"]))
    return requests, inventory


def prepare():
    protocol = read(P / "protocol.json")
    requests, inventory = prepare_for(read(P/"public.json"), protocol["schedule_seeds"], 3)
    assert len(requests) == 72
    assert all(r["input_utf8_bound"] + 32768 <= 65536 for r in requests)
    assert all(r["input_utf8_bound"] * .1 / 1e6 + 32768 * .5 / 1e6 <= .025 for r in requests)
    return requests


def validate_content(response, allowed):
    assert isinstance(response, dict) and response["model"] in {
        "openai/gpt-6-luna", "openai/gpt-6-luna:batch", "openai/gpt-6-luna-20260922"}
    assert response.get("provider") == "OpenAI"
    assert len(response["choices"]) == 1 and response["choices"][0]["finish_reason"] == "stop"
    items = json.loads(response["choices"][0]["message"]["content"])
    assert isinstance(items, list)
    labels = {}
    for item in items:
        assert isinstance(item, dict) and set(item) == {"pair", "label"}
        assert isinstance(item["pair"], str) and item["label"] in {"causes", "causedby", "norel"}
        pair = tuple(x.strip() for x in item["pair"].split(","))
        assert pair in allowed
        labels[pair] = item["label"]  # Preserve historical last-label-wins within a read.
    return labels


def freeze_predictions(requests, events):
    docs = {d["id"]: d for d in read(P / "public.json")}
    by = {e["logical_id"]: e for e in events}
    if len(by) != len(events) or set(by) != {r["logical_id"] for r in requests}:
        return None, [{"logical_coverage": "requires exactly all72unique logical events"}]
    labels, failures, groups, ids = {}, [], collections.defaultdict(set), []
    for r in requests:
        e = by[r["logical_id"]]
        try:
            assert e["api_call"] is True and e["cache_hit"] is False and e["error"] is None
            assert e["custom_id"] == r["custom_id"] and e["request"] == r["body"]
            assert isinstance(e["batch_id"], str) and e["batch_id"]
            groups[r["doc"]].add(e["batch_id"])
            response_id = e["response"].get("id")
            assert isinstance(response_id, str) and response_id
            ids.append(response_id)
            labels[r["logical_id"]] = validate_content(
                e["response"], {tuple(p) for p in docs[r["doc"]]["pair_list"]})
        except Exception as exc:
            failures.append({"logical_id": r["logical_id"], "type": type(exc).__name__,
                             "message": str(exc)[:1000]})
    if len(groups) != 4 or any(len(bs) != 1 for bs in groups.values()) or len(set.union(*groups.values())) != 4:
        failures.append({"batch_coverage": "requires four distinct document batches mixing all18logical slots"})
    if len(ids) != 72 or len(set(ids)) != 72:
        failures.append({"response_identity": "requires72distinct nonempty served responseIDs"})
    if failures:
        return None, failures
    predictions = []
    for schedule in (0, 1):
        for arm in ARMS:
            for doc, d in docs.items():
                passes = [labels[f"s{schedule}:{arm}:{doc}:{i}"] for i in range(3)]
                scores = [{"a": a, "b": b, "luna": sum(
                    ps.get((a, b), "norel") != "norel" for ps in passes) / 3}
                          for a, b in sorted(d["pair_list"])]
                predictions.append({"schedule": schedule, "arm": arm, "doc": doc, "scores": scores})
    return predictions, []


def freeze_decisions(predictions, d1_rows):
    d1 = {(r["doc"], r["a"], r["b"]): r["d1"] for r in d1_rows}
    scores = {(r["schedule"], r["arm"], r["doc"], v["a"], v["b"]): v["luna"]
              for r in predictions for v in r["scores"]}
    decisions = []
    for (schedule, arm, doc, a, b), value in sorted(scores.items()):
        fused = (value + scores[schedule, arm, doc, b, a] + d1[doc, a, b] + d1[doc, b, a]) / 4
        decisions.append({"schedule": schedule, "arm": arm, "doc": doc, "a": a, "b": b,
                          "fused_score": fused, "positive": fused >= .39})
    return decisions


def evaluate(decisions, controls, draws=10000):
    gold = {(r["doc"],r["a"],r["b"]):r["gold"] for r in controls}
    per = {str(s): {a:collections.defaultdict(lambda:[0,0,0]) for a in ARMS} for s in (0,1)}
    assert len(decisions) == len(gold) * 6
    for r in decisions:
        k = r["doc"],r["a"],r["b"]; y,pred = gold[k],r["positive"]
        assert y == gold[k[0],k[2],k[1]]
        counts=per[str(r["schedule"])][r["arm"]][r["doc"].split("_")[0]]
        counts[0]+=int(pred and y);counts[1]+=int(pred and not y);counts[2]+=int(not pred and y)
    topics=sorted(per["0"]["original"])
    def score(s,a,ts):
        c=[sum(per[str(s)][a][t][i] for t in ts) for i in range(3)]
        return {"TP_FP_FN":c,"F1":f1(c)}
    systems={str(s):{a:score(s,a,topics) for a in ARMS} for s in (0,1)}
    deltas={a:[systems[str(s)][a]["F1"]-systems[str(s)]["original"]["F1"] for s in (0,1)] for a in ARMS[1:]}
    return {"status":"COMPLETE_EXPLORATORY_SMOKE", "systems":systems,
            "schedule_delta_vs_original_points":deltas,
            "mean_delta_vs_original_points":{a:sum(ds)/2 for a,ds in deltas.items()},
            "documents":4,"topics":topics,"ordered_candidates":len(gold),"per_topic_counts":per,
            "continuation_candidates":[],"decision":"no promotion on four exposed documents; assess integrity and costs only"}


def catalog(root):
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as response:
        data = json.load(response)
    save(root / "catalog-response.json", data)
    rows = [r for r in data["data"] if r["id"] == "openai/gpt-6-luna"]
    assert len(rows) == 1 and rows[0].get("canonical_slug") == "openai/gpt-6-luna-20260922"
    prices = rows[0]["pricing"]
    for name, ceiling in (("prompt", 1e-7), ("completion", 5e-7)):
        value = float(prices[name])
        assert math.isfinite(value) and 0 <= value <= ceiling


def export(root):
    files = {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix not in {".sqlite", ".lock", ".tmp"}
             and not p.name.endswith((".sqlite-wal", ".sqlite-shm")) and p.name != "manifest.json"}
    save(root / "manifest.json", {"run": os.environ.get("ORX_RUN_ID"),
                                "commit": os.environ.get("ORX_COMMIT"), "files_sha256": files})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in list(files) + ["manifest.json"]:
            archive.write(root / name, name)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    size = 6000
    print("N12_EVIDENCE_BEGIN " + json.dumps({"zip_sha256": sha(buffer.getvalue()),
          "base64_chars": len(encoded), "chunks": (len(encoded) + size - 1) // size}), flush=True)
    for i in range(0, len(encoded), size):
        print("N12_EVIDENCE_CHUNK " + json.dumps({"index": i // size, "data": encoded[i:i + size]}), flush=True)
    print("N12_EVIDENCE_END", flush=True)


def account_costs(root):
    batches = {}
    for path in (root / "batches").glob("*/luna-batch-jobs/final-retrieval.json"):
        batch = read(path)
        cost = (batch.get("usage") or {}).get("cost")
        if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
            batches[batch["id"]] = cost
    known = sum(batches.values())
    unknown = 18 * (4 - len(batches))
    return {"known_provider_reported_usd": known,
            "conservative_usd": known + unknown * .025,
            "unaccounted_logical_slots": unknown, "batch_totals": batches,
            "allocation": "perrequest and perarm cost is not directly measured in mixed batches"}


def main():
    requests = prepare()
    pre = {"gate": "PASS", "logical_slots": 72, "documents": 4, "batches": 4,
           "requests_per_batch": 18, "views": [views(d)[1] for d in read(P / "public.json")], "distinct_request_bodies": len({r["body_sha256"] for r in requests}),
           "max_input_bound": max(r["input_utf8_bound"] for r in requests),
           "full_reservation_usd": 1.8, "hard_cap_usd": 1.8, "paid_retries": 0}
    print("N12_PREFLIGHT " + json.dumps(pre), flush=True)
    if "--preflight" in sys.argv:
        save(P / "preflight-frozen.json", pre)
        save(P / "requests-frozen.json", requests)
        return
    run = os.environ["ORX_RUN_ID"]
    root = pathlib.Path("logs/n12") / run
    assert not root.exists(), "no automatic run resumption or paid retry"
    root.mkdir(parents=True)
    for name in ("protocol.json", "public.json", "selection.json", "preflight-frozen.json",
                 "requests-frozen.json", "d1-public.json"):
        save(root / name, read(P / name))
    for source in (pathlib.Path(__file__), pathlib.Path("pipelines/jev/n12_transport.py"),
                   P / "test_n12.py",
                   pathlib.Path("prompts/esl_standard_eci_lean_coref.yaml"),
                   pathlib.Path("experiments/jev/n8/demo-bank.json"),
                   pathlib.Path("experiments/jev/n8/prompt.json")):
        (root / source.name).write_bytes(source.read_bytes())
    save(root / "source-identities.json", {"ORX_RUN_ID": run, "ORX_COMMIT": os.environ.get("ORX_COMMIT"),
         "controls_sha256": sha((P / "controls.json").read_bytes()),
         "public_sha256": sha((P / "public.json").read_bytes())})
    events, lock = [], threading.Lock()
    try:
        assert canonical(requests) == canonical(read(P / "requests-frozen.json"))
        from pipelines.jev.spend import SharedBudget
        budget = SharedBudget(path=str(root / "campaign-budget.sqlite"), scope="N12-" + run,
                              limit_usd=1.8, initial_spend_usd=0)
        ticket = budget.reserve(1.8)
        save(root / "reservation-before-calls.json", budget.status())
        catalog(root)
        groups = collections.defaultdict(list)
        for row in requests:
            groups[row["doc"]].append(row)
        for doc, rows in groups.items():
            random.Random("N12::mixed-batch::20261028::" + doc).shuffle(rows)
        save(root / "batch-order-before-calls.json", {
            doc: [r["logical_id"] for r in rows] for doc, rows in groups.items()})

        def record(event):
            with lock:
                events.append(event)
                with (root / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

        def generate(item):
            from pipelines.jev.n12_transport import FreshLogicalBatchClient
            doc, rows = item
            client = FreshLogicalBatchClient(model="openai/gpt-6-luna:batch",
                model_version="gpt-6-luna-20260922", mode="live",
                cache_path=str(root / "batches" / doc / "unused-response-cache.sqlite"),
                retries=0, max_calls=18, budget_usd=.45, max_cost_per_call_usd=.025,
                input_price_per_million=.1, max_context_tokens=65536,
                max_output_tokens=32768, reasoning_effort="high", timeout=600,
                requests_per_minute=6000)
            try:
                return client.decide_logical(rows, record)
            except Exception as exc:
                return {"doc": doc, "status": "failed", "type": type(exc).__name__,
                        "message": str(exc)[:1000]}
            finally:
                client.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(generate, groups.items()))
        save(root / "group-status.json", statuses)
        accounting = account_costs(root)
        known, upper = accounting["known_provider_reported_usd"], accounting["conservative_usd"]
        save(root / "costs.json", accounting)
        budget.settle(ticket, upper)
        save(root / "costs.json", {**accounting, "budget": budget.status()})
        predictions, failures = freeze_predictions(requests, events)
        if len(statuses) != 4 or any(g["status"] != "completed" for g in statuses):
            predictions = None
            failures.append({"failed_groups": [g for g in statuses if g["status"] != "completed"]})
        save(root / "validity.json", {"failures": failures, "answers": len(events)})
        if predictions is None:
            result = {"status": "BLOCKED_VALIDITY", "contrasts": None,
                      "failures": failures, "read_gold": False}
        else:
            save(root / "predictions-before-gold.json", predictions)
            d1 = read(P / "d1-public.json")
            decisions = freeze_decisions(predictions, d1)
            save(root / "decisions-before-gold.json", decisions)
            save(root / "freeze-before-gold.json", {
                "events_sha256": sha((root / "events.jsonl").read_bytes()),
                "predictions_sha256": sha((root / "predictions-before-gold.json").read_bytes()),
                "decisions_sha256": sha((root / "decisions-before-gold.json").read_bytes()),
                "controls_opened": False,
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            controls = read(P / "controls.json")
            assert sha((P / "controls.json").read_bytes()) == read(root / "source-identities.json")["controls_sha256"]
            assert [{k: r[k] for k in ("doc", "a", "b", "d1")} for r in controls] == d1
            save(root / "controls.json", controls)
            result = evaluate(decisions, controls)
        result.update(study="N12", cost_known_usd=known, cost_conservative_usd=upper,
                      planned_request_slots=72, recorded_response_events=len(events),
                      human_annotations=0, training=False)
        save(root / "result.json", result)
        print("N12_RESULT " + json.dumps(result), flush=True)
    except Exception as exc:
        if not (root / "costs.json").exists():
            # Unknown POST outcomes retain every unknown logical-slot reservation.
            accounting = account_costs(root)
            accounting["global_reservation_retained"] = True
            save(root / "costs.json", accounting)
        result = {"study": "N12", "status": "BLOCKED_CONTROLLER", "contrasts": None,
                  "type": type(exc).__name__, "message": str(exc)[:2000],
                  "recorded_response_events": len(events), "no_paid_retry": True}
        save(root / "controller-failure.json", result)
        print("N12_FAILURE " + json.dumps(result), flush=True)
        raise
    finally:
        export(root)


if __name__ == "__main__":
    main()
