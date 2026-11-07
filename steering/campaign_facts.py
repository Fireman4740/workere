"""C2: source-anchored Qwen event cards and longer per-block thinking.

The C1 local-thinking requests and outputs are reused, never regenerated. Each
document's cards are extracted once with the same frozen model. True cards and
within-document permutations are compared at 2048 tokens; plain and true-card
conditions at 4096 test the budget and interaction. Evaluation labels and gold
coreference are never passed to either model stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

from steering import campaign_blocks as c


ARMS = (
    ("cards_2048", "cards", 2048),
    ("permuted_2048", "permuted", 2048),
    ("plain_4096", "plain", 4096),
    ("cards_4096", "cards", 4096),
)
FIELDS = ("actor", "affected", "evidence", "status")
STATUSES = ("asserted", "negated", "planned", "uncertain", "unknown")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def feature_schema(size):
    fields = {"actor": {"type": "string"}, "affected": {"type": "string"},
              "evidence": {"type": "string"}, "status": {"type": "string", "enum": list(STATUSES)}}
    item = {"type": "object", "properties": fields, "required": list(FIELDS), "additionalProperties": False}
    keys = [str(i) for i in range(size)]
    return {"type": "object", "properties": {key: item for key in keys},
            "required": keys, "additionalProperties": False}


def unique_object(items):
    keys = [k for k, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate JSON keys")
    return dict(items)


def parse_features(text, size):
    try:
        obj = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {str(i) for i in range(size)}:
        return None
    for item in obj.values():
        if not isinstance(item, dict) or set(item) != set(FIELDS):
            return None
        if not all(isinstance(item[k], str) for k in FIELDS):
            return None
        if item["status"] not in STATUSES:
            return None
    return obj


def clean_card(card, document):
    """Keep copied text only if an exact source span exists; status needs evidence."""
    plain = c.TAG.sub(lambda match: match.group(2), document)
    result, rejected = {}, 0
    for field in ("actor", "affected", "evidence"):
        value = card[field].strip()
        if value and len(value) <= 160 and (value in plain or value in document):
            result[field] = value
        else:
            rejected += bool(value)
            result[field] = ""
    result["status"] = card["status"] if result["evidence"] else "unknown"
    return result, int(rejected)


def event_surfaces(document, needed):
    found = {}
    for match in c.TAG.finditer(document):
        event_id, span = match.groups()
        if event_id in needed and event_id not in found:
            found[event_id] = span
    if set(found) != set(needed):
        raise ValueError(f"event tags missing: {sorted(set(needed)-set(found))}")
    return found


def permute_cards(cards, dataset, doc_id):
    ids = sorted(cards)
    if len(ids) < 2:
        return dict(cards)
    shift = random.Random(f"{c.SEED}::{dataset}::{doc_id}::cards").randrange(1, len(ids))
    return {event_id: dict(cards[ids[(i + shift) % len(ids)]]) for i, event_id in enumerate(ids)}


def feature_messages(text, event_ids, surfaces):
    events = "\n".join(f"{i}: <{event_id} {surfaces[event_id]}>" for i, event_id in enumerate(event_ids))
    system = ("Extract information only from the source text. For each listed event, return an actor, "
              "an affected participant, a short evidence span, and a status. Copy each non-empty actor, "
              "affected participant and evidence span verbatim from the source; use an empty string "
              "if the source does not specify it. Status must be asserted, negated, planned, uncertain, "
              "or unknown. Do not infer causal links or use knowledge outside this document.")
    user = (f"Source document:\n{text}\n\nEvents:\n{events}\n\nReturn one JSON object keyed by "
            "event index, each value containing actor, affected, evidence, and status.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def source_bundle(source):
    manifest = json.loads((source / "manifest.json").read_text())
    assert c.sha(source / "requests.jsonl") == manifest["request_sha256"], "C1 request hash mismatch"
    for name, expected in manifest["source_hashes"].items():
        assert c.sha(c.DATA / name) == expected, f"dataset differs from C1: {name}"
    all_reqs = c.jsonl(source / "requests.jsonl")
    outputs = c.jsonl(source / "generations.jsonl")
    assert len(outputs) == len(all_reqs) == manifest["requests"]
    assert {r["key"] for r in outputs} == {r["key"] for r in all_reqs}
    selected = [r for r in all_reqs if r["condition"] == "local_think"]
    assert len(selected) * 4 == len(all_reqs)
    return manifest, selected


def config():
    result = c.engine_config()
    result.update(feature_tokens=int(os.environ.get("FEATURE_TOKENS", "1536")),
                  long_think_tokens=int(os.environ.get("LONG_THINK_TOKENS", "4096")))
    assert result["think_tokens"] == 2048 and result["long_think_tokens"] == 4096
    return result


def prepare(out, source):
    parent, parent_reqs = source_bundle(source)
    ctb, _, esc, _ = c.inputs()
    data = {"CTB": ctb, "ESC": esc}
    requests = []
    for dataset in ("CTB", "ESC"):
        for doc_id in parent["selected"][dataset]:
            doc = data[dataset][doc_id]
            event_ids = sorted({event for pair in c.pairs(doc) for event in pair})
            surfaces = event_surfaces(doc["text"], set(event_ids))
            for start in range(0, len(event_ids), c.BLOCK_SIZE):
                ids = event_ids[start:start+c.BLOCK_SIZE]
                request = dict(dataset=dataset, doc_id=doc_id, event_ids=ids,
                               messages=feature_messages(doc["text"], ids, surfaces))
                request["key"] = signature(request)
                requests.append(request)
    assert len({r["key"] for r in requests}) == len(requests)
    path = out / "feature_requests.jsonl"
    serialized = "".join(canonical(r) + "\n" for r in requests)
    if path.exists():
        assert path.read_text() == serialized, "feature request identity mismatch"
    else:
        path.write_text(serialized)
    source_files = ("manifest.json", "requests.jsonl", "generations.jsonl", "report.json", "model_identity.json")
    manifest = dict(parent_run=source.name, selected=parent["selected"], source_hashes={n:c.sha(source / n) for n in source_files},
                    feature_request_sha256=c.sha(path), feature_requests=len(requests),
                    parent_local_requests=len(parent_reqs), engine=config(),
                    arms=[dict(name=n, card_mode=mode, think_tokens=budget) for n, mode, budget in ARMS],
                    scope="CTB20 and ESC12 previously used for development; no gold in model requests",
                    commit=os.environ.get("ORX_COMMIT"))
    if (out / "manifest.json").exists():
        old = json.loads((out / "manifest.json").read_text())
        assert {k:v for k,v in old.items() if k != "commit"} == {k:v for k,v in manifest.items() if k != "commit"}
    c.dump(out / "manifest.json", manifest)
    print("PREPARED", json.dumps({k:v for k,v in manifest.items() if k != "source_hashes"}), flush=True)


def resume(out, source):
    current, previous = json.loads((out / "manifest.json").read_text()), json.loads((source / "manifest.json").read_text())
    for key in ("source_hashes", "feature_request_sha256", "engine", "arms"):
        assert current[key] == previous[key], f"resume {key} mismatch"
    if (source / "model_identity.json").exists():
        assert json.loads((source / "model_identity.json").read_text()) == c.model_identity(), "checkpoint identity differs"
    for name in ("feature_generations.jsonl", "requests.jsonl", "generations.jsonl"):
        path = source / name
        if path.exists():
            assert not (out / name).exists(), f"destination already has {name}"
            shutil.copy2(path, out / name)
    c.dump(out / "resume.json", {"source": str(source), "identity_verified": True})
    print("RESUME verified source, feature prompts, engine and checkpoint", flush=True)


def complete(out):
    if not (out / "requests.jsonl").exists():
        print("COVERAGE no pair requests yet", flush=True)
        return False
    requests = c.jsonl(out / "requests.jsonl")
    expected = {r["key"] for r in requests}
    done = c.load_completed(out / "generations.jsonl", expected)
    print(f"COVERAGE completed={len(done)} expected={len(expected)}", flush=True)
    return len(expected) > 0 and done == expected


def clean_cards(out):
    reqs = c.jsonl(out / "feature_requests.jsonl")
    rows = c.jsonl(out / "feature_generations.jsonl")
    by_key = {r["key"]: r for r in rows}
    assert len(by_key) == len(rows) == len(reqs)
    assert set(by_key) == {r["key"] for r in reqs}
    ctb, _, esc, _ = c.inputs()
    data = {"CTB": ctb, "ESC": esc}
    cards = defaultdict(dict)
    quality = defaultdict(Counter)
    costs, costs_by_dataset = Counter(), defaultdict(Counter)
    for req in reqs:
        row = by_key[req["key"]]
        parsed = parse_features(row["answer"], len(req["event_ids"])) if row["status"] == "valid" else None
        quality[req["dataset"]]["feature_chunks"] += 1
        quality[req["dataset"]]["invalid_chunks"] += int(parsed is None)
        for field in ("input_tokens", "output_tokens", "wall_seconds_share"):
            costs[field] += row[field]
            costs_by_dataset[req["dataset"]][field] += row[field]
        for i, event_id in enumerate(req["event_ids"]):
            raw = parsed[str(i)] if parsed is not None else dict(actor="", affected="", evidence="", status="unknown")
            clean, rejected = clean_card(raw, data[req["dataset"]][req["doc_id"]]["text"])
            key = (req["dataset"], req["doc_id"])
            assert event_id not in cards[key]
            cards[key][event_id] = clean
            quality[req["dataset"]]["events"] += 1
            quality[req["dataset"]]["events_with_evidence"] += int(bool(clean["evidence"]))
            quality[req["dataset"]]["events_with_actor"] += int(bool(clean["actor"]))
            quality[req["dataset"]]["events_with_affected"] += int(bool(clean["affected"]))
            quality[req["dataset"]]["rejected_unanchored_fields"] += rejected
    encoded = {f"{dataset}::{doc}": mapping for (dataset, doc), mapping in sorted(cards.items())}
    c.dump(out / "event_cards.json", encoded)
    c.dump(out / "feature_quality.json", {"quality": {k:dict(v) for k,v in quality.items()}, "cost": dict(costs),
           "cost_by_dataset": {k:dict(v) for k,v in costs_by_dataset.items()},
           "note": "Source-span presence is verified; event attribution and status are model predictions. Extraction cost is shared by the card arms."})
    print("FEATURE_QUALITY", json.dumps({k:dict(v) for k,v in quality.items()}), "COST", json.dumps(dict(costs)), flush=True)
    return cards


def add_cards(messages, card_map, pair_list):
    ids = sorted({event for pair in pair_list for event in pair})
    assert set(ids) <= set(card_map), "event card coverage mismatch"
    rows = [f"<{event}> {canonical(card_map[event])}" for event in ids]
    messages = [dict(message) for message in messages]
    text = messages[-1]["content"]
    marker = "\n\nPairs:\n"
    location = text.rfind(marker)
    assert location >= 0, "original C1 pair prompt was altered"
    messages[-1]["content"] = (text[:location] + "\n\nExtracted event notes (may be incomplete):\n" +
                                "\n".join(rows) + text[location:])
    return messages


def build_pair_requests(out, source, cards):
    _, parent_reqs = source_bundle(source)
    requests = []
    for original in parent_reqs:
        card_set = cards[(original["dataset"], original["doc_id"])]
        shuffled = permute_cards(card_set, original["dataset"], original["doc_id"])
        for name, mode, budget in ARMS:
            messages = original["messages"] if mode == "plain" else add_cards(
                original["messages"], card_set if mode == "cards" else shuffled, original["pairs"])
            request = dict(dataset=original["dataset"], doc_id=original["doc_id"], block=original["block"],
                           condition=name, card_mode=mode, think_tokens=budget, schedule=original["schedule"],
                           pairs=original["pairs"], c1_key=original["key"], messages=messages)
            request["key"] = signature(request)
            requests.append(request)
    requests.sort(key=lambda r:(r["condition"], r["dataset"], r["schedule"], r["doc_id"], r["block"]))
    assert len({r["key"] for r in requests}) == len(requests)
    content = "".join(canonical(r)+"\n" for r in requests)
    path = out / "requests.jsonl"
    if path.exists():
        assert path.read_text() == content, "resume pair requests differ from event-card cache"
    else:
        path.write_text(content)
    print("PAIR_REQUESTS", len(requests), "sha256", c.sha(path), flush=True)
    return requests


def infer(out, source):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from offline import ENGINE_EXTRA

    identity = c.model_identity()
    assert identity == json.loads((source / "model_identity.json").read_text()), "checkpoint differs from C1"
    c.dump(out / "model_identity.json", identity)
    cfg = config()
    print("ENGINE_CONFIG", json.dumps(dict(model=c.MODEL, vllm=__import__("vllm").__version__,
          checkpoint_config_sha256=identity["config_sha256"], chunked_prefill=True, prefix_caching=False, **cfg)), flush=True)
    started = time.monotonic()
    llm = LLM(model=c.MODEL, tokenizer=c.MODEL, trust_remote_code=True, max_model_len=cfg["max_model_len"],
              max_num_seqs=cfg["batch"], max_num_batched_tokens=cfg["max_num_batched_tokens"],
              enable_chunked_prefill=True, enable_prefix_caching=False, gpu_memory_utilization=cfg["gpu_util"],
              enforce_eager=False, disable_log_stats=True,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}, **ENGINE_EXTRA)
    print(f"== engine loaded seconds={time.monotonic()-started:.1f}", flush=True)
    tok = llm.get_tokenizer()
    def render(messages, thinking):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking, preserve_thinking=True)
    def params(schema, budget):
        return SamplingParams(temperature=0, top_p=.95, top_k=20, max_tokens=budget,
                              structured_outputs=StructuredOutputsParams(json=schema))
    smoke_messages = [{"role":"user", "content":"Return an event object with empty fields and status unknown."}]
    smoke = llm.generate([render(smoke_messages, False)], params(feature_schema(1), cfg["feature_tokens"]), use_tqdm=False)[0].outputs[0]
    if smoke.finish_reason != "stop" or parse_features(smoke.text, 1) is None:
        raise RuntimeError(f"event JSON smoke failed: {smoke.finish_reason} {smoke.text!r}")
    smoke = llm.generate([render([{"role":"user", "content":"Return a JSON decision 0=N."}], True) +
                          "The answer is N.\n</think>\n\n"], params(c.schema(1), cfg["answer_tokens"]), use_tqdm=False)[0].outputs[0]
    if smoke.finish_reason != "stop" or c.parse_answer(smoke.text, 1) is None:
        raise RuntimeError(f"pair JSON smoke failed: {smoke.finish_reason} {smoke.text!r}")
    print("SMOKE passed: event schema, pair schema and thinking template", flush=True)
    features = c.jsonl(out / "feature_requests.jsonl")
    feature_path = out / "feature_generations.jsonl"
    feature_done = c.load_completed(feature_path, {r["key"] for r in features})
    todo = [r for r in features if r["key"] not in feature_done]
    with feature_path.open("a", buffering=1) as sink:
        for start in range(0, len(todo), cfg["batch"]):
            batch = todo[start:start+cfg["batch"]]
            texts = [render(r["messages"], False) for r in batch]
            lengths = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
            if any(n + cfg["feature_tokens"] > cfg["max_model_len"] for n in lengths):
                raise RuntimeError("event extraction input would exceed model context")
            tick = time.monotonic()
            outputs = llm.generate(texts, [params(feature_schema(len(r["event_ids"])), cfg["feature_tokens"]) for r in batch], use_tqdm=False)
            seconds = time.monotonic()-tick
            assert len(outputs) == len(batch)
            for req, output, n in zip(batch, outputs, lengths):
                ans = output.outputs[0]
                valid = ans.finish_reason == "stop" and parse_features(ans.text, len(req["event_ids"])) is not None
                sink.write(json.dumps(dict(key=req["key"], status="valid" if valid else "invalid_answer",
                    answer=ans.text, finish=ans.finish_reason, input_tokens=n, output_tokens=len(ans.token_ids),
                    wall_seconds_share=seconds/len(batch)), ensure_ascii=False) + "\n")
                sink.flush()
            print(f"EVENT_FEATURES {start+len(batch)}/{len(todo)} batch_seconds={seconds:.1f}", flush=True)
    cards = clean_cards(out)
    extracted = json.loads((out / "feature_quality.json").read_text())["quality"]
    chunks = sum(v["feature_chunks"] for v in extracted.values())
    invalid = sum(v["invalid_chunks"] for v in extracted.values())
    if invalid > max(1, int(.10 * chunks)):
        raise RuntimeError(f"too many invalid feature chunks ({invalid}/{chunks}); stop before pair inference")
    requests = build_pair_requests(out, source, cards)
    path = out / "generations.jsonl"
    completed = c.load_completed(path, {r["key"] for r in requests})
    with path.open("a", buffering=1) as sink:
        for name, _, budget in ARMS:
            for dataset in ("CTB", "ESC"):
                for schedule in (0, 1):
                    group = [r for r in requests if r["condition"] == name and r["dataset"] == dataset
                             and r["schedule"] == schedule and r["key"] not in completed]
                    group_start = time.monotonic()
                    print(f"== start {name} {dataset} order={schedule} remaining={len(group)}", flush=True)
                    for pos in range(0, len(group), cfg["batch"]):
                        batch = group[pos:pos+cfg["batch"]]
                        tick = time.monotonic()
                        prompts = [render(r["messages"], True) for r in batch]
                        lengths = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
                        eligible = [i for i,n in enumerate(lengths) if n+budget+cfg["answer_tokens"]+16 <= cfg["max_model_len"]]
                        think_params = SamplingParams(temperature=0, top_p=.95, top_k=20, max_tokens=budget,
                                                      stop=["</think>"], include_stop_str_in_output=False)
                        traces = llm.generate([prompts[i] for i in eligible], think_params, use_tqdm=False) if eligible else []
                        assert len(traces) == len(eligible)
                        trace_map = {i:res.outputs[0] for i,res in zip(eligible, traces)}
                        answer_prompts = [prompts[i] + trace_map[i].text + "\n</think>\n\n" for i in eligible]
                        answer_lens = [len(tok.encode(p, add_special_tokens=False)) for p in answer_prompts]
                        if any(n+cfg["answer_tokens"] > cfg["max_model_len"] for n in answer_lens):
                            raise RuntimeError("answer prompt exceeds context after thinking")
                        outputs = llm.generate(answer_prompts,
                            [params(c.schema(len(batch[i]["pairs"])), cfg["answer_tokens"]) for i in eligible],
                            use_tqdm=False) if eligible else []
                        assert len(outputs) == len(eligible)
                        answer_map = {i:res.outputs[0] for i,res in zip(eligible, outputs)}
                        answer_lengths = dict(zip(eligible, answer_lens))
                        seconds = time.monotonic()-tick
                        for i, req in enumerate(batch):
                            trace, answer = trace_map.get(i), answer_map.get(i)
                            valid = bool(answer and answer.finish_reason == "stop" and c.parse_answer(answer.text, len(req["pairs"])) is not None)
                            row = dict(key=req["key"], dataset=dataset, doc_id=req["doc_id"], condition=name,
                                schedule=schedule, block=req["block"], status="valid" if valid else
                                ("context_overflow" if answer is None else "invalid_answer"),
                                prompt_tokens=lengths[i], total_input_tokens=(lengths[i] if trace else 0)+answer_lengths.get(i,0),
                                think_tokens=len(trace.token_ids) if trace else 0,
                                answer_tokens=len(answer.token_ids) if answer else 0,
                                thinking_forced=bool(trace and trace.stop_reason != "</think>"),
                                thinking_finish=trace.finish_reason if trace else None,
                                answer_finish=answer.finish_reason if answer else None,
                                thought=trace.text if trace else None, answer=answer.text if answer else None,
                                batch_seconds=seconds, batch_size=len(batch), wall_seconds_share=seconds/len(batch))
                            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                            sink.flush()
                            completed.add(req["key"])
                        if (pos//cfg["batch"])%5 == 0 or pos+len(batch) == len(group):
                            print(f"[{name} {dataset} order={schedule}] {pos+len(batch)}/{len(group)} batch={seconds:.1f}s", flush=True)
                    print(f"== completed {name} {dataset} order={schedule} seconds={time.monotonic()-group_start:.1f}", flush=True)
                    live_result(out, dataset, name, schedule)


def gold_and_universe(dataset, doc):
    universe = {(str(a), str(b)) for a,b in doc["pair_list"] if a != b}
    gold = ({(a,b) for a,label,b in doc["gold"] if label != "norel"} if dataset == "CTB" else
            {tuple(p.split(",",1)) for p in doc["gold"]})
    assert gold <= universe
    return gold, universe


def prediction_groups(requests, rows):
    by_key = {r["key"]:r for r in rows}
    assert len(by_key) == len(rows)
    predictions = defaultdict(lambda: defaultdict(set))
    oriented = defaultdict(lambda: defaultdict(set))
    quality, costs = defaultdict(Counter), defaultdict(Counter)
    for req in requests:
        result = by_key[req["key"]]
        group = (req["dataset"], req["condition"], req["schedule"])
        quality[group][result["status"]] += 1
        quality[group]["forced_thinking"] += int(result["thinking_forced"])
        for field in ("prompt_tokens", "total_input_tokens", "think_tokens", "answer_tokens", "wall_seconds_share"):
            costs[group][field] += result[field]
        answer = c.parse_answer(result["answer"], len(req["pairs"])) if result["status"] == "valid" else None
        if answer is None:
            continue
        for i,(a,b) in enumerate(req["pairs"]):
            if answer[str(i)] != "N":
                predictions[group][req["doc_id"]].update(((a,b),(b,a)))
                oriented[group][req["doc_id"]].add((a,b) if answer[str(i)] == "A" else (b,a))
    return predictions, oriented, quality, costs


def live_result(out, dataset, condition, schedule):
    requests = [r for r in c.jsonl(out / "requests.jsonl") if r["dataset"] == dataset
                and r["condition"] == condition and r["schedule"] == schedule]
    rows = {r["key"]:r for r in c.jsonl(out / "generations.jsonl")}
    if any(r["key"] not in rows for r in requests): return
    predictions, _, _, _ = prediction_groups(requests, [rows[r["key"]] for r in requests])
    manifest = json.loads((out / "manifest.json").read_text())
    ctb, _, esc, _ = c.inputs()
    data = ctb if dataset == "CTB" else esc
    per = {}
    for doc_id in manifest["selected"][dataset]:
        gold, universe = gold_and_universe(dataset, data[doc_id])
        per[doc_id] = c.counts(predictions[(dataset,condition,schedule)][doc_id], gold, universe)
    print("PARTIAL_RESULT", dataset, condition, f"order={schedule}", json.dumps(c.metric(per)), flush=True)


def historical_baselines(selected, ctb, reference, esc, work):
    from pair_verifier import cached_votes
    from eci.parse import parse_triples
    bases = {"CTB":defaultdict(dict), "ESC":defaultdict(dict)}
    predictions = {"CTB":defaultdict(dict), "ESC":defaultdict(dict)}
    ids = selected["CTB"]
    pool = cached_votes([work / "out/N1/gen.jsonl", work / "out/P3/gen.jsonl"], ctb, ids)
    for doc_id in ids:
        qwen = {tuple(k.split(",",1)) for k,v in reference["qwen"][doc_id].items() if v != "norel"}
        pooled = set()
        for (a,b),v in pool[doc_id].items():
            if v >= .5: pooled.update(((a,b),(b,a)))
        gold, universe = gold_and_universe("CTB", ctb[doc_id])
        bases["CTB"]["qwen7"][doc_id] = c.counts(qwen, gold, universe)
        bases["CTB"]["n1_p3"][doc_id] = c.counts(pooled, gold, universe)
        predictions["CTB"]["qwen7"][doc_id] = qwen & universe
        predictions["CTB"]["n1_p3"][doc_id] = pooled & universe
    ref = json.loads((c.DATA / "esc/baseline.json").read_text())
    p2 = defaultdict(list)
    for row in c.jsonl(work / "out/P2/gen.jsonl"):
        if row["doc_id"] not in selected["ESC"]: continue
        found = None
        if row.get("finish_reason") in (None,"stop") and "</think>" in row.get("text", ""):
            try:
                found = {(a,b) for a,label,b in parse_triples(row["text"].rsplit("</think>",1)[1]) if label != "norel"}
            except ValueError: pass
        p2[row["doc_id"]].append(found)
    for doc_id in selected["ESC"]:
        assert len(p2[doc_id]) == 3
        valid = [x for x in p2[doc_id] if x is not None]
        gold, universe = gold_and_universe("ESC", esc[doc_id])
        for name in ("luna7", "fusion", "qwen_p2_sym"):
            pred = set()
            for a,b in universe:
                ab, ba = ref[doc_id][f"{a},{b}"], ref[doc_id][f"{b},{a}"]
                if name == "luna7": positive = ab[0] >= 4/7 - 1e-9
                elif name == "fusion": positive = (ab[0]+ba[0]+ab[1]+ba[1])/4 >= .39
                else: positive = bool(valid) and (sum((a,b) in x for x in valid)+sum((b,a) in x for x in valid))/(2*len(valid)) >= .5
                if positive: pred.add((a,b))
            bases["ESC"][name][doc_id] = c.counts(pred, gold, universe)
            predictions["ESC"][name][doc_id] = pred
    return bases, predictions


def score(out, source, work):
    from compare import paired_delta
    requests, rows = c.jsonl(out / "requests.jsonl"), c.jsonl(out / "generations.jsonl")
    expected = {r["key"] for r in requests}
    assert c.load_completed(out / "generations.jsonl", expected) == expected
    manifest = json.loads((out / "manifest.json").read_text())
    parent_report = json.loads((source / "report.json").read_text())
    predictions, oriented, quality, costs = prediction_groups(requests, rows)
    ctb, reference, esc, _ = c.inputs()
    data = {"CTB":ctb, "ESC":esc}
    baselines, baseline_predictions = historical_baselines(manifest["selected"], ctb, reference, esc, work)
    for dataset in manifest["selected"]:
        for name, per in baselines[dataset].items():
            original = parent_report["baselines"][dataset][name]
            assert c.metric(per) == original, f"baseline reproduction mismatch: {dataset} {name}"
    report = dict(scope=manifest["scope"], parent_run=manifest["parent_run"], selected=manifest["selected"],
                  feature_quality=json.loads((out / "feature_quality.json").read_text()),
                  baselines=parent_report["baselines"], conditions={}, contrasts={}, combinations={})
    per_groups = {}
    for dataset, ids in manifest["selected"].items():
        for condition, mode, budget in ARMS:
            for schedule in (0,1):
                group=(dataset,condition,schedule)
                per={}
                for doc_id in ids:
                    gold, universe = gold_and_universe(dataset,data[dataset][doc_id])
                    per[doc_id] = c.counts(predictions[group][doc_id],gold,universe)
                per_groups[group] = per
                parent = parent_report["conditions"][f"{dataset}/local_think/order{schedule}"]["per_document"]
                comparisons={name:paired_delta(per,base,5000,seed=c.SEED) for name,base in baselines[dataset].items()}
                comparisons["c1_local_think_2048"] = paired_delta(per,parent,5000,seed=c.SEED)
                result = dict(**c.metric(per), quality=dict(quality[group]), cost=dict(costs[group]),
                              paired_vs_baselines=comparisons, per_document=per)
                if dataset == "CTB":
                    direction = {}
                    for doc_id in ids:
                        expected = {(a,b) if label == "causes" else (b,a) for a,label,b in ctb[doc_id]["gold"]
                                    if label in ("causes", "causedby")}
                        guessed = oriented[group][doc_id]
                        direction[doc_id] = dict(tp=len(guessed & expected), fp=len(guessed - expected),
                                                 fn=len(expected - guessed))
                    result["directed_ctb"] = c.metric(direction)
                if mode != "plain":
                    result["shared_feature_cost"] = report["feature_quality"]["cost_by_dataset"][dataset]
                key=f"{dataset}/{condition}/order{schedule}"
                report["conditions"][key]=result
                print("RESULT",key,json.dumps({k:v for k,v in result.items() if k != "per_document"}),flush=True)
        for schedule in (0,1):
            for label,new,old in (("cards_vs_permuted_2048","cards_2048","permuted_2048"),
                                  ("cards_vs_plain_4096","cards_4096","plain_4096"),
                                  ("cards_4096_vs_2048","cards_4096","cards_2048")):
                key=f"{dataset}/{label}/order{schedule}"
                report["contrasts"][key]=paired_delta(per_groups[(dataset,new,schedule)],per_groups[(dataset,old,schedule)],5000,seed=c.SEED)
                print("CONTRAST",key,json.dumps(report["contrasts"][key]),flush=True)
        reference_name = "n1_p3" if dataset == "CTB" else "qwen_p2_sym"
        for condition, _, _ in ARMS:
            for rule in ("consensus_only", "add_consensus", "filter_if_both_reject", "majority_of_three"):
                per, added_tp, added_fp = {}, 0, 0
                for doc_id in ids:
                    first = predictions[(dataset,condition,0)][doc_id]
                    second = predictions[(dataset,condition,1)][doc_id]
                    old = baseline_predictions[dataset][reference_name][doc_id]
                    common, either = first & second, first | second
                    chosen = {"consensus_only":common, "add_consensus":old | common,
                              "filter_if_both_reject":old & either,
                              "majority_of_three":(old & either) | common}[rule]
                    gold, universe = gold_and_universe(dataset,data[dataset][doc_id])
                    added_tp += len((chosen-old) & gold)
                    added_fp += len(((chosen-old) & universe) - gold)
                    per[doc_id] = c.counts(chosen,gold,universe)
                key=f"{dataset}/{condition}/{rule}"
                report["combinations"][key] = dict(**c.metric(per), added_tp=added_tp, added_fp=added_fp,
                    vs_baseline=paired_delta(per,baselines[dataset][reference_name],5000,seed=c.SEED),
                    scope="predefined exploratory rule on reused development docs; not a held-out confirmation")
                print("COMBINATION",key,json.dumps(report["combinations"][key]),flush=True)
    c.dump(out / "report.json",report)
    print("FINAL",json.dumps(dict(requests=len(requests),source=manifest["parent_run"],report=str(out / "report.json"))),flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action",required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out",type=Path,required=True); p.add_argument("--source",type=Path,required=True)
    p = sub.add_parser("resume"); p.add_argument("--out",type=Path,required=True); p.add_argument("--source",type=Path,required=True)
    p = sub.add_parser("infer"); p.add_argument("--out",type=Path,required=True); p.add_argument("--source",type=Path,required=True)
    p = sub.add_parser("complete"); p.add_argument("--out",type=Path,required=True)
    p = sub.add_parser("score"); p.add_argument("--out",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--work",type=Path,required=True)
    args=parser.parse_args()
    if args.action=="prepare": prepare(args.out,args.source)
    elif args.action=="resume": resume(args.out,args.source)
    elif args.action=="infer": infer(args.out,args.source)
    elif args.action=="score": score(args.out,args.source,args.work)
    else: raise SystemExit(0 if complete(args.out) else 1)


if __name__ == "__main__": main()
