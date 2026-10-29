"""Independent, standard-library-only replay. No project imports or network.

Usage: python audit_n9.py extracted-evidence-directory
"""
import collections
import hashlib
import json
import math
import pathlib
import random
import sys


def read(path):
    return json.loads(path.read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def run(folder):
    manifest = read(folder / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        assert digest((folder / name).read_bytes()) == expected, name
    protocol = read(folder / "protocol.json")
    requests = read(folder / "requests-frozen.json")
    assert len(requests) == len({r["logical_id"] for r in requests}) == 240
    assert len({r["custom_id"] for r in requests}) == 240
    expected = {r["logical_id"]: r for r in requests}
    for r in requests:
        assert r["body_sha256"] == digest(canon(r["body"]))
        assert r["custom_id"] == "n9_" + digest(r["logical_id"].encode())
        assert r["body"]["model"] == "openai/gpt-6-luna:batch"
        assert r["body"]["reasoning"] == {"effort": "high"} and r["body"]["max_tokens"] == 32768
        assert len(json.dumps(r["body"], ensure_ascii=False).encode()) + 1024 == r["input_utf8_bound"]
        assert r["input_utf8_bound"] + 32768 <= 65536
        assert r["demo"] in r["eligible_demo_ids"] and r["demo"].split("_")[0] != r["doc"].split("_")[0]
        if r["arm"].endswith("N0"):
            other = expected[r["logical_id"].replace(r["arm"], r["arm"][:-1] + "1", 1)]
            assert other["demo"] == r["demo"]
            left, right = r["body"]["messages"], other["body"]["messages"]
            assert [left[i] for i in (0, 1, 3)] == [right[i] for i in (0, 1, 3)]
            positives = json.loads(left[2]["content"])
            added = [{"pair": f"{a},{b}", "label": "norel"}
                     for x, y in r["negative_pairs_in_demo"] for a, b in ((x, y), (y, x))]
            assert json.loads(right[2]["content"]) == positives + added
            assert r["explicit_negative_rows"] == 0 and other["explicit_negative_rows"] == len(added) > 0
    raw_events = (folder / "events.jsonl").read_bytes()
    events = [json.loads(line) for line in raw_events.splitlines()]
    public = {d["id"]: d for d in read(folder / "public.json")}
    labels, batches, served_ids = {}, {}, set()
    payload_rows, raw_rows = {}, {}
    statuses = read(folder / "group-status.json")
    assert len(statuses) == 10 and all(s["status"] == "completed" for s in statuses)
    for doc in public:
        path = folder / "batches" / doc / "luna-batch-jobs"
        payload, final = read(path / "submission-payload.json"), read(path / "final-retrieval.json")
        assert payload["model"] == "openai/gpt-6-luna" and len(payload["requests"]) == 24
        assert final["status"] == "completed" and len(final["results"]) == 24
        cost = final["usage"]["cost"]
        assert type(cost) in (int, float) and math.isfinite(cost) and 0 <= cost <= .6 + 1e-12
        assert final["id"] not in batches
        batches[final["id"]] = cost
        for p in payload["requests"]:
            assert p["custom_id"] not in payload_rows
            payload_rows[p["custom_id"]] = (doc, p["body"])
        for r in final["results"]:
            assert r["custom_id"] not in raw_rows
            raw_rows[r["custom_id"]] = (final["id"], r)
    assert len(raw_rows) == len(payload_rows) == 240
    for event in events:
        key = event["logical_id"]
        assert key in expected and key not in labels
        r = expected[key]
        assert event["custom_id"] == r["custom_id"] and event["request"] == r["body"]
        assert event["api_call"] is True and event["cache_hit"] is False and event["error"] is None
        assert payload_rows[r["custom_id"]] == (r["doc"], {k: v for k, v in r["body"].items() if k != "model"})
        bid, raw = raw_rows[r["custom_id"]]
        assert not raw.get("error") and raw["response"]["status_code"] == 200
        assert event["batch_id"] == bid and event["response"] == raw["response"]["body"]
        assert event["provider_reported_batch_cost_usd"] == batches[bid]
        response = event["response"]
        assert response["model"] in {"openai/gpt-6-luna", "openai/gpt-6-luna:batch", "openai/gpt-6-luna-20260922"}
        assert response["provider"] == "OpenAI" and len(response["choices"]) == 1
        assert response["choices"][0]["finish_reason"] == "stop"
        assert isinstance(response["id"], str) and response["id"] and response["id"] not in served_ids
        served_ids.add(response["id"])
        allowed = {tuple(p) for p in public[r["doc"]]["pair_list"]}
        items = json.loads(response["choices"][0]["message"]["content"])
        assert isinstance(items, list)
        labels[key] = {}
        for item in items:
            assert isinstance(item, dict) and set(item) == {"pair", "label"}
            assert item["label"] in {"causes", "causedby", "norel"}
            pair = tuple(x.strip() for x in item["pair"].split(","))
            assert pair in allowed
            labels[key][pair] = item["label"]
    assert set(labels) == set(expected) and len(served_ids) == 240 and len(batches) == 10
    reservation = read(folder / "reservation-before-calls.json")
    assert reservation["accounted_or_reserved_usd"] == 6 and reservation["limit_usd"] == 6.25
    freeze = read(folder / "freeze-before-gold.json")
    assert freeze["controls_opened"] is False
    assert freeze["events_sha256"] == digest(raw_events)
    for stem in ("predictions", "decisions"):
        assert freeze[stem + "_sha256"] == digest((folder / (stem + "-before-gold.json")).read_bytes())
    controls = read(folder / "controls.json")
    assert read(folder / "source-identities.json")["controls_sha256"] == digest((folder / "controls.json").read_bytes())
    d1_rows = read(folder / "d1-public.json")
    assert [{k: r[k] for k in ("doc", "a", "b", "d1")} for r in controls] == d1_rows
    gold = {(r["doc"], r["a"], r["b"]): r["gold"] for r in controls}
    d1 = {(r["doc"], r["a"], r["b"]): r["d1"] for r in d1_rows}
    predictions = {(r["schedule"], r["arm"], r["doc"], v["a"], v["b"]): v["luna"]
                   for r in read(folder / "predictions-before-gold.json") for v in r["scores"]}
    frozen = {(r["schedule"], r["arm"], r["doc"], r["a"], r["b"]): r
              for r in read(folder / "decisions-before-gold.json")}
    arms = ("R0N0", "R0N1", "R1N0", "R1N1")
    per = {str(s): {a: collections.defaultdict(lambda: [0, 0, 0]) for a in arms} for s in (0, 1)}
    assert len(predictions) == len(frozen) == 8 * len(gold)
    for schedule in (0, 1):
        for arm in arms:
            scores = {}
            for doc, a, b in gold:
                value = sum(labels[f"s{schedule}:{arm}:{doc}:{slot}"].get((a, b), "norel") != "norel"
                            for slot in range(3)) / 3
                assert predictions[schedule, arm, doc, a, b] == value
                scores[doc, a, b] = value
            for k, y in gold.items():
                rev = (k[0], k[2], k[1])
                assert gold[rev] == y
                fused = (scores[k] + scores[rev] + d1[k] + d1[rev]) / 4
                pred = fused >= .39
                row = frozen[(schedule, arm) + k]
                assert row["fused_score"] == fused and row["positive"] == pred
                count = per[str(schedule)][arm][k[0].split("_")[0]]
                count[0] += int(pred and y)
                count[1] += int(pred and not y)
                count[2] += int(not pred and y)
    topics = sorted(per["0"]["R0N0"])

    def score(schedule, arm, sample):
        c = [sum(per[str(schedule)][arm][t][i] for t in sample) for i in range(3)]
        return 200 * c[0] / (2 * c[0] + c[1] + c[2]) if c[0] else 0.

    def contrasts(sample):
        output = collections.defaultdict(list)
        for schedule in (0, 1):
            a, b, c, d = [score(schedule, arm, sample) for arm in arms]
            for name, value in (("retrieval_main", (-a-b+c+d)/2),
                                ("negatives_main", (-a+b-c+d)/2),
                                ("interaction", a-b-c+d),
                                ("R0N1_vs_R0N0", b-a),
                                ("R1N0_vs_R0N0", c-a),
                                ("R1N1_vs_R0N0", d-a)):
                output[name].append(value)
        return output

    observed = contrasts(topics)
    rng, boot = random.Random(202610289), {n: [] for n in observed}
    for _ in range(20000):
        sample = rng.choices(topics, k=len(topics))
        for name, deltas in contrasts(sample).items():
            boot[name].append(sum(deltas)/2)
    result = read(folder / "result.json")
    assert result["status"] == "COMPLETE" and result["bootstrap"]["percentile_indices_zero_based"] == [83, 19916]
    summary = {}
    for name, values in boot.items():
        values.sort()
        expected_row = {"schedule_delta_points": observed[name],
                        "mean_delta_points": sum(observed[name])/2,
                        "simultaneous_CI_points": [values[83], values[19916]]}
        for field, expected_value in expected_row.items():
            actual = result["contrasts"][name][field]
            left = actual if isinstance(actual, list) else [actual]
            right = expected_value if isinstance(expected_value, list) else [expected_value]
            assert all(abs(x-y) < 1e-10 for x, y in zip(left, right)), (name, field)
        summary[name] = expected_row
    for schedule in (0, 1):
        for arm in arms:
            count = [sum(per[str(schedule)][arm][t][i] for t in topics) for i in range(3)]
            assert result["systems"][str(schedule)][arm]["TP_FP_FN"] == count
            assert abs(result["systems"][str(schedule)][arm]["F1"] - score(schedule, arm, topics)) < 1e-10
    candidates = [a for a in arms[1:] if summary[a+"_vs_R0N0"]["mean_delta_points"] >= 1
                  and summary[a+"_vs_R0N0"]["simultaneous_CI_points"][0] > 0
                  and min(summary[a+"_vs_R0N0"]["schedule_delta_points"]) > 0]
    assert candidates == result["continuation_candidates"]
    total_cost = sum(batches.values())
    assert abs(total_cost - result["cost_known_usd"]) < 1e-10
    return {"audit": "PASS", "raw_answers": 240, "distinct_served_response_ids": 240,
            "distinct_document_batches": 10, "raw_provider_body_reconciliation": True,
            "distinct_bodies": len({r["body_sha256"] for r in requests}),
            "contrasts": summary, "continuation_candidates": candidates,
            "provider_reported_cost_usd": total_cost,
            "files_verified": len(manifest["files_sha256"]), "API_calls_in_audit": 0}


if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1])
    output = run(folder)
    print(json.dumps(output, indent=2))
    (folder.parent / "independent-audit.json").write_text(json.dumps(output, indent=2))
