"""Frozen N9 factorial 2x2, with mixed fresh generations and pre-gold fusion."""
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

P = pathlib.Path("experiments/jev/n9")
ARMS = ("R0N0", "R0N1", "R1N0", "R1N1")
CONTRAST_WEIGHTS = {
    "retrieval_main": (-.5, -.5, .5, .5),
    "negatives_main": (-.5, .5, -.5, .5),
    "interaction": (1., -1., -1., 1.),
    "R0N1_vs_R0N0": (-1., 1., 0., 0.),
    "R1N0_vs_R0N0": (-1., 0., 1., 0.),
    "R1N1_vs_R0N0": (-1., 0., 0., 1.),
}


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


def prepare():
    import yaml
    from benchmark.schema import Document, Entity
    from benchmark.data import parse_annotations
    from pipelines.jev.llm import marked_text
    from pipelines.jev.contrastive import select_demonstrations, negative_pairs

    protocol = read(P / "protocol.json")
    bank = read("experiments/jev/round4/demo-bank.json")["documents"]
    prompt = yaml.safe_load(pathlib.Path("prompts/esl_standard_eci_lean_coref.yaml").read_text())
    requests = []
    for d in read(P / "public.json"):
        doc = Document(d["id"], tuple(d["tokens"]), tuple(tuple(x) for x in d["sentences"]), d["lang"])
        entities = tuple(Entity(m, tuple(sp)) for m, sp in zip(d["mentions"], d["spans"]))
        eligible = [b for b in bank if b["id"].split("_")[0] != d["id"].split("_")[0]
                    and any(lab != "norel" for _, lab, _ in parse_annotations(b["annots"]))
                    and negative_pairs(b)]
        assert len(eligible) >= 3
        structural = select_demonstrations(doc, entities, bank, 3)
        for schedule, seed in enumerate(protocol["schedule_seeds"]):
            random_demos = random.Random(f"{seed}::{doc.id}").sample(eligible, 3)
            for retrieval, demos in enumerate((random_demos, structural)):
                for negatives in (0, 1):
                    arm = f"R{retrieval}N{negatives}"
                    for slot, demo in enumerate(demos):
                        answer = [{"pair": f"{a},{b}", "label": lab}
                                  for a, lab, b in parse_annotations(demo["annots"]) if lab != "norel"]
                        if negatives:
                            answer += [{"pair": f"{x},{y}", "label": "norel"}
                                       for a, b in negative_pairs(demo) for x, y in ((a, b), (b, a))]
                        messages = [
                            {"role": "system", "content": prompt["system"]},
                            {"role": "user", "content": prompt["user_template"].format(
                                doc_text=demo["text"], pair_lines="", doc_id=demo["id"])},
                            {"role": "assistant", "content": json.dumps(answer)},
                            {"role": "user", "content": prompt["user_template"].format(
                                doc_text=marked_text(doc, entities), pair_lines="", doc_id=doc.id)},
                        ]
                        body = {"model": "openai/gpt-6-luna:batch", "messages": messages,
                                "reasoning": {"effort": "high"}, "max_tokens": 32768}
                        bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
                        assert bound + 32768 <= 65536
                        assert bound * .1 / 1e6 + 32768 * .5 / 1e6 <= .025
                        logical = f"s{schedule}:{arm}:{doc.id}:{slot}"
                        requests.append({"logical_id": logical, "custom_id": "n9_" + sha(logical.encode()),
                                         "schedule": schedule, "seed": seed, "arm": arm,
                                         "doc": doc.id, "slot": slot, "demo": demo["id"],
                                         "eligible_demo_ids": [b["id"] for b in eligible],
                                         "negative_pairs_in_demo": [list(p) for p in negative_pairs(demo)],
                                         "explicit_negative_rows": 2 * len(negative_pairs(demo)) if negatives else 0,
                                         "body": body, "body_sha256": sha(canonical(body)),
                                         "input_utf8_bound": bound})
    assert len(requests) == len({r["logical_id"] for r in requests}) == 240
    assert len({r["custom_id"] for r in requests}) == 240
    validate_pairing(requests)
    return requests


def validate_pairing(requests):
    by = {r["logical_id"]: r for r in requests}
    for row in requests:
        if row["arm"].endswith("N1"):
            continue
        other_id = row["logical_id"].replace(row["arm"], row["arm"][:-1] + "1", 1)
        other = by[other_id]
        assert row["demo"] == other["demo"]
        assert row["negative_pairs_in_demo"] == other["negative_pairs_in_demo"]
        left, right = row["body"]["messages"], other["body"]["messages"]
        assert len(left) == len(right) == 4
        assert [left[i] for i in (0, 1, 3)] == [right[i] for i in (0, 1, 3)]
        positive = json.loads(left[2]["content"])
        negative = [{"pair": f"{x},{y}", "label": "norel"}
                    for a, b in row["negative_pairs_in_demo"] for x, y in ((a, b), (b, a))]
        assert json.loads(right[2]["content"]) == positive + negative
        assert all(it["label"] != "norel" for it in positive)
        assert row["explicit_negative_rows"] == 0 and other["explicit_negative_rows"] == len(negative) > 0


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
        return None, [{"logical_coverage": "requires exactly all240unique logical events"}]
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
    if len(groups) != 10 or any(len(bs) != 1 for bs in groups.values()) or len(set.union(*groups.values())) != 10:
        failures.append({"batch_coverage": "requires ten distinct document batches mixing all24logical slots"})
    if len(ids) != 240 or len(set(ids)) != 240:
        failures.append({"response_identity": "requires240distinct nonempty served responseIDs"})
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


def evaluate(decisions, controls, draws=20000):
    gold = {(r["doc"], r["a"], r["b"]): r["gold"] for r in controls}
    per = {str(s): {a: collections.defaultdict(lambda: [0, 0, 0]) for a in ARMS} for s in (0, 1)}
    seen = set()
    for r in decisions:
        key = r["doc"], r["a"], r["b"]
        assert key in gold and gold[key] == gold[key[0], key[2], key[1]]
        unique = (r["schedule"], r["arm"]) + key
        assert unique not in seen
        seen.add(unique)
        y, pred = gold[key], r["positive"]
        counts = per[str(r["schedule"])][r["arm"]][r["doc"].split("_")[0]]
        counts[0] += int(pred and y)
        counts[1] += int(pred and not y)
        counts[2] += int(not pred and y)
    assert len(seen) == len(gold) * 8
    topics = sorted(per["0"]["R0N0"])

    def total(schedule, arm, sample):
        return [sum(per[str(schedule)][arm][topic][i] for topic in sample) for i in range(3)]

    def contrasts(sample):
        condition_f1 = [[f1(total(s, a, sample)) for a in ARMS] for s in (0, 1)]
        return {name: [sum(w * v for w, v in zip(weights, values)) for values in condition_f1]
                for name, weights in CONTRAST_WEIGHTS.items()}

    systems = {str(s): {a: {"TP_FP_FN": total(s, a, topics), "F1": f1(total(s, a, topics))}
                       for a in ARMS} for s in (0, 1)}
    observed = contrasts(topics)
    boot = {name: [] for name in CONTRAST_WEIGHTS}
    rng = random.Random(202610289)
    for _ in range(draws):
        sampled = rng.choices(topics, k=len(topics))
        for name, deltas in contrasts(sampled).items():
            boot[name].append(sum(deltas) / 2)
    # Nominal 99.1666667% percentile intervals, Bonferroni six-contrast family.
    lo, hi = math.floor(draws * .05 / 12), draws - math.floor(draws * .05 / 12) - 1
    results = {}
    for name, values in boot.items():
        values.sort()
        deltas = observed[name]
        ci = [values[lo], values[hi]]
        results[name] = {"schedule_delta_points": deltas, "mean_delta_points": sum(deltas) / 2,
                         "simultaneous_CI_points": ci}
    candidates = [a for a in ARMS[1:] if
                  results[a + "_vs_R0N0"]["mean_delta_points"] >= 1 and
                  results[a + "_vs_R0N0"]["simultaneous_CI_points"][0] > 0 and
                  min(results[a + "_vs_R0N0"]["schedule_delta_points"]) > 0]
    return {"status": "COMPLETE", "systems": systems, "contrasts": results,
            "continuation_candidates": candidates, "topics": topics,
            "documents": len({r["doc"] for r in controls}), "ordered_candidates": len(gold),
            "bootstrap": {"draws": draws, "seed": 202610289, "family_size": 6,
                          "same_topic_draw_all_conditions_schedules": True,
                          "tail_probability": .05 / 12, "percentile_indices_zero_based": [lo, hi],
                          "nominal_interval_percent": 100 * (1 - .05 / 6)},
            "per_topic_counts": per,
            "LOTO_mean_contrasts_points": {t: {n: sum(ds) / 2 for n, ds in
                contrasts([x for x in topics if x != t]).items()} for t in topics},
            "interpretation": "exposed development diagnosis; exploratory candidate only, no production promotion"}


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
    print("N9_EVIDENCE_BEGIN " + json.dumps({"zip_sha256": sha(buffer.getvalue()),
          "base64_chars": len(encoded), "chunks": (len(encoded) + size - 1) // size}), flush=True)
    for i in range(0, len(encoded), size):
        print("N9_EVIDENCE_CHUNK " + json.dumps({"index": i // size, "data": encoded[i:i + size]}), flush=True)
    print("N9_EVIDENCE_END", flush=True)


def account_costs(root):
    batches = {}
    for path in (root / "batches").glob("*/luna-batch-jobs/final-retrieval.json"):
        batch = read(path)
        cost = (batch.get("usage") or {}).get("cost")
        if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
            batches[batch["id"]] = cost
    known = sum(batches.values())
    unknown = 24 * (10 - len(batches))
    return {"known_provider_reported_usd": known,
            "conservative_usd": known + unknown * .025,
            "unaccounted_logical_slots": unknown, "batch_totals": batches,
            "allocation": "perrequest and perarm cost is not directly measured in mixed batches"}


def main():
    requests = prepare()
    pre = {"gate": "PASS", "logical_slots": 240, "documents": 10, "batches": 10,
           "requests_per_batch": 24, "distinct_request_bodies": len({r["body_sha256"] for r in requests}),
           "max_input_bound": max(r["input_utf8_bound"] for r in requests),
           "full_reservation_usd": 6., "hard_cap_usd": 6.25, "paid_retries": 0}
    print("N9_PREFLIGHT " + json.dumps(pre), flush=True)
    if "--preflight" in sys.argv:
        save(P / "preflight-frozen.json", pre)
        save(P / "requests-frozen.json", requests)
        return
    run = os.environ["ORX_RUN_ID"]
    root = pathlib.Path("logs/n9") / run
    assert not root.exists(), "no automatic run resumption or paid retry"
    root.mkdir(parents=True)
    for name in ("protocol.json", "public.json", "selection.json", "preflight-frozen.json",
                 "requests-frozen.json", "d1-public.json"):
        save(root / name, read(P / name))
    for source in (pathlib.Path(__file__), pathlib.Path("pipelines/jev/n9_transport.py"),
                   P / "audit_n9.py", P / "test_n9.py",
                   pathlib.Path("prompts/esl_standard_eci_lean_coref.yaml"),
                   pathlib.Path("experiments/jev/round4/demo-bank.json"),
                   pathlib.Path("pipelines/jev/contrastive.py")):
        (root / source.name).write_bytes(source.read_bytes())
    save(root / "source-identities.json", {"ORX_RUN_ID": run, "ORX_COMMIT": os.environ.get("ORX_COMMIT"),
         "controls_sha256": sha((P / "controls.json").read_bytes()),
         "public_sha256": sha((P / "public.json").read_bytes())})
    events, lock = [], threading.Lock()
    try:
        assert canonical(requests) == canonical(read(P / "requests-frozen.json"))
        from pipelines.jev.spend import SharedBudget
        budget = SharedBudget(path=str(root / "campaign-budget.sqlite"), scope="N9-" + run,
                              limit_usd=6.25, initial_spend_usd=0)
        ticket = budget.reserve(6.)
        save(root / "reservation-before-calls.json", budget.status())
        catalog(root)
        groups = collections.defaultdict(list)
        for row in requests:
            groups[row["doc"]].append(row)
        for doc, rows in groups.items():
            random.Random("N9::mixed-batch::20261028::" + doc).shuffle(rows)
        save(root / "batch-order-before-calls.json", {
            doc: [r["logical_id"] for r in rows] for doc, rows in groups.items()})

        def record(event):
            with lock:
                events.append(event)
                with (root / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

        def generate(item):
            from pipelines.jev.n9_transport import FreshLogicalBatchClient
            doc, rows = item
            client = FreshLogicalBatchClient(model="openai/gpt-6-luna:batch",
                model_version="gpt-6-luna-20260922", mode="live",
                cache_path=str(root / "batches" / doc / "unused-response-cache.sqlite"),
                retries=0, max_calls=24, budget_usd=.61, max_cost_per_call_usd=.025,
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
        if len(statuses) != 10 or any(g["status"] != "completed" for g in statuses):
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
        result.update(study="N9", cost_known_usd=known, cost_conservative_usd=upper,
                      planned_request_slots=240, recorded_response_events=len(events),
                      human_annotations=0, training=False)
        save(root / "result.json", result)
        print("N9_RESULT " + json.dumps(result), flush=True)
    except Exception as exc:
        if not (root / "costs.json").exists():
            # Unknown POST outcomes retain every unknown logical-slot reservation.
            accounting = account_costs(root)
            accounting["global_reservation_retained"] = True
            save(root / "costs.json", accounting)
        result = {"study": "N9", "status": "BLOCKED_CONTROLLER", "contrasts": None,
                  "type": type(exc).__name__, "message": str(exc)[:2000],
                  "recorded_response_events": len(events), "no_paid_retry": True}
        save(root / "controller-failure.json", result)
        print("N9_FAILURE " + json.dumps(result), flush=True)
        raise
    finally:
        export(root)


if __name__ == "__main__":
    main()
