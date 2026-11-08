"""C3: gold-free routing of cached disagreements to paired frozen-Qwen judges.

Direct and evidence-based judgments use the same per-pair thinking budget and
both event orders. Only complete final JSON is scored. Invalid judgments abstain
and retain the cached vote. No thresholds, prompts or routes use evaluation gold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import time

from steering import campaign_blocks as c
from steering import campaign_facts as f
from steering.compare import paired_delta

ARMS = ("direct", "evidence")
FIELDS = ("event_a_quote", "event_b_quote", "link_quote", "alternative")
PRIMARY = "consensus_repair"


def unordered(values):
    return {tuple(sorted(p)) for p in values}


def symmetric(values):
    return {p for a, b in values for p in ((a, b), (b, a))}


def route_pairs(universe, base, judgments):
    """Route if any valid local judgment disagrees with the baseline, without labels."""
    return {p for p in universe if any(p in view and view[p] != (p in base) for view in judgments)}


def answer_schema(arm):
    properties = {"label": {"type": "string", "enum": ["A", "B", "N"]}}
    if arm == "evidence":
        properties.update({key: {"type": "string"} for key in FIELDS})
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def parse_final(text, arm):
    try:
        obj = json.loads(text, object_pairs_hook=f.unique_object)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(answer_schema(arm)["required"]):
        return None
    if obj["label"] not in ("A", "B", "N"):
        return None
    if arm == "evidence" and not all(isinstance(obj[key], str) for key in FIELDS):
        return None
    return obj


def grounded(obj, context, presented):
    """Check copied spans and mention identity, not the semantic truth of causality."""
    for key, event in zip(("event_a_quote", "event_b_quote"), presented):
        quote = obj[key].strip()
        if not quote or len(quote) > 800 or quote not in context:
            return False
        if event not in {m.group(1) for m in c.TAG.finditer(quote)}:
            return False
    support = obj["link_quote"].strip()
    if obj["label"] != "N" and not support:
        return False
    if support and (len(support) > 1200 or support not in context):
        return False
    return bool(obj["alternative"].strip())


def canonical_label(label, reverse):
    if reverse and label in ("A", "B"):
        return {"A": "B", "B": "A"}[label]
    return label


def merge_decision(baseline, labels, rule):
    """Two valid, direction-consistent votes required to alter a baseline decision."""
    assert len(labels) == 2
    if None in labels or labels[0] != labels[1]:
        return baseline
    positive = labels[0] != "N"
    return baseline or positive if rule == "additions_only" else positive


def messages(rules, context, presented, arm):
    system = (rules + "\n\nClassify ONLY the requested pair. A means event A causes event B, "
              "B means event B causes event A, N means no annotated causal relation. "
              "Distinguish causation from temporal order, shared topic and mere association. "
              "Use the source and its annotation rules, not prior knowledge about this document. "
              "Return one JSON object with a label field containing A, B or N.")
    if arm == "evidence":
        system += ("\nEvaluate the relation using text evidence specific to both events. "
                   "Consider the strongest non-causal explanation and why it is or is not adequate. "
                   "Also return event_a_quote and event_b_quote: short exact source substrings "
                   "including the corresponding complete event tags; preserve tags verbatim. "
                   "Return link_quote: an exact source substring supporting the causal judgment, "
                   "or an empty string if none is supported; implicit contextual causation is allowed "
                   "when justified by the text and the annotation rules. Return alternative: a brief "
                   "description of the competing interpretation. A copied quote alone is not proof "
                   "of causation. Do not relax the annotation rules.")
    user = f"Document context:\n{context}\n\nRequested pair: A=<{presented[0]}>, B=<{presented[1]}>."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def bundle(source):
    manifest = json.loads((source / "manifest.json").read_text())
    report = json.loads((source / "report.json").read_text())
    assert manifest["selected"] == report["selected"]
    requests = [r for r in c.jsonl(source / "requests.jsonl") if r["condition"] == "plain_4096"]
    expected = {r["key"] for r in requests}
    rows = [r for r in c.jsonl(source / "generations.jsonl") if r["key"] in expected]
    assert len(rows) == len(expected) == len(requests), "C2 plain_4096 arm incomplete or duplicated"
    assert len({r["key"] for r in rows}) == len(rows)
    by_key = {r["key"]: r for r in rows}
    judgments = defaultdict(dict)
    for req in requests:
        row = by_key[req["key"]]
        answer = c.parse_answer(row.get("answer"), len(req["pairs"])) if row["status"] == "valid" else None
        if answer is None:
            continue  # An invalid block does not count as a negative judgment for routing.
        group = (req["dataset"], req["doc_id"], req["schedule"])
        for i, pair in enumerate(req["pairs"]):
            key = tuple(sorted(pair))
            assert key not in judgments[group]
            judgments[group][key] = answer[str(i)] != "N"
    return manifest, report, judgments


def write_jsonl(path, rows):
    content = "".join(f.canonical(row) + "\n" for row in rows)
    if path.exists():
        assert path.read_text() == content, f"frozen inputs changed: {path.name}"
    else:
        path.write_text(content)


def prepare(out, source, work):
    out.mkdir(parents=True, exist_ok=True)
    parent, report, judgments = bundle(source)
    ctb, reference, esc, rules = c.inputs()
    docs = {"CTB": ctb, "ESC": esc}
    baselines, baseline_predictions = f.historical_baselines(parent["selected"], ctb, reference, esc, work)
    requests, routes, diagnostics, base_snapshot = [], [], {}, {}
    for dataset, ids in parent["selected"].items():
        reference_name = "n1_p3" if dataset == "CTB" else "qwen_p2_sym"
        for name, per in baselines[dataset].items():
            assert c.metric(per) == report["baselines"][dataset][name], "historical baseline changed"
        base_snapshot[dataset] = {name: {d: sorted(pred[d]) for d in ids}
                                  for name, pred in baseline_predictions[dataset].items()}
        diag = Counter()
        oracle, baseline_counts = {}, baselines[dataset][reference_name]
        for doc_id in ids:
            doc = docs[dataset][doc_id]
            universe = set(c.pairs(doc))
            base = unordered(baseline_predictions[dataset][reference_name][doc_id])
            views = [judgments[(dataset, doc_id, order)] for order in (0, 1)]
            routed = route_pairs(universe, base, views)
            diag["all_candidate_links"] += len(universe)
            diag["routed_links"] += len(routed)
            diag["addition_candidates"] += len(routed - base)
            diag["removal_candidates"] += len(routed & base)
            for pair in sorted(routed):
                context = c.windows(doc["text"], set(pair))
                routes.append(dict(dataset=dataset, doc_id=doc_id, pair=list(pair), baseline=pair in base))
                for arm in ARMS:
                    for reverse in (False, True):
                        presented = pair[::-1] if reverse else pair
                        req = dict(dataset=dataset, doc_id=doc_id, pair=list(pair), arm=arm,
                                   reverse=reverse, presented=list(presented), context=context,
                                   messages=messages(rules[dataset], context, presented, arm))
                        req["key"] = f.signature(req)
                        requests.append(req)
            # Diagnostics use labels only after routing/prompts are fixed; never used to select a run.
            gold, directed_universe = f.gold_and_universe(dataset, doc)
            gold_links = unordered(gold)
            diag["routed_missing_true_links"] += len((routed - base) & gold_links)
            diag["routed_existing_false_links"] += len((routed & base) - gold_links)
            diag["missed_gold_outside_route"] += len(gold_links - base - routed)
            oracle[doc_id] = c.counts(symmetric((base - routed) | (gold_links & routed)), gold, directed_universe)
        diagnostics[dataset] = dict(**diag, baseline=c.metric(baseline_counts),
            oracle_route_repair=c.metric(oracle),
            warning="oracle uses gold; unattainable diagnostic ceiling, never a predictive result or routing input")
        print("ROUTING_DIAGNOSTIC", dataset, json.dumps(diagnostics[dataset]), flush=True)
    # Alternate arms across corpus/order strata to avoid a single fixed temporal condition order.
    requests.sort(key=lambda r: (r["dataset"], r["reverse"],
                                 r["arm"] if not r["reverse"] else ("0" if r["arm"] == "evidence" else "1"),
                                 r["doc_id"], r["pair"]))
    assert len({r["key"] for r in requests}) == len(requests) == 4 * len(routes)
    write_jsonl(out / "requests.jsonl", requests)
    write_jsonl(out / "routes.jsonl", routes)
    if not requests:
        (out / "generations.jsonl").touch(exist_ok=True)
    c.dump(out / "baselines.json", base_snapshot)
    c.dump(out / "routing_diagnostic.json", diagnostics)
    upstream_cost = {}
    for dataset, ids in parent["selected"].items():
        names = ("N1", "P3") if dataset == "CTB" else ("P2",)
        upstream_rows = [row for name in names for row in c.jsonl(work / f"out/{name}/gen.jsonl")
                         if row.get("doc_id") in ids]
        upstream_cost[dataset] = dict(
            vote=dict(readings=len(upstream_rows), output_tokens=sum(r.get("num_tokens",0) for r in upstream_rows),
                      recorded_initial_input_tokens=sum(r.get("prompt_tokens",0) for r in upstream_rows),
                      wall_seconds=None, note="Initial inputs omit forced-answer re-prefill; per-row seconds are cumulative batch timestamps, not additive."),
            c2_plain_4096={f"order{order}":report.get("conditions",{}).get(f"{dataset}/plain_4096/order{order}",{}).get("cost")
                           for order in (0,1)})
    c.dump(out / "upstream_cost.json",upstream_cost)
    provenance = {f"parent/{name}": c.sha(source / name) for name in (
        "manifest.json", "requests.jsonl", "generations.jsonl", "report.json", "model_identity.json")}
    provenance.update({f"baseline/{name}": c.sha(work / f"out/{name}/gen.jsonl") for name in ("N1", "P3", "P2")})
    for name in ("ctb-suite.json", "ctb-prompt.json", "reference.json", "esc/test-s40.json", "esc/prompt.json", "esc/baseline.json"):
        provenance[f"data/{name}"] = c.sha(c.DATA / name)
    manifest = dict(parent_run=source.name, selected=parent["selected"], source_hashes=provenance,
                    requests=len(requests), routed_links=len(routes), arms=list(ARMS), engine=c.engine_config(),
                    request_sha256=c.sha(out / "requests.jsonl"), routes_sha256=c.sha(out / "routes.jsonl"),
                    baseline_sha256=c.sha(out / "baselines.json"), commit=os.environ.get("ORX_COMMIT"),
                    primary=PRIMARY, scope="CTB20/ESC12 reused development; no gold routing, fitting or weight training")
    if (out / "manifest.json").exists():
        old = json.loads((out / "manifest.json").read_text())
        assert {k:v for k,v in old.items() if k != "commit"} == {k:v for k,v in manifest.items() if k != "commit"}
    c.dump(out / "manifest.json", manifest)
    print("PREPARED", json.dumps({k:v for k,v in manifest.items() if k != "source_hashes"}), flush=True)


def resume(out, source):
    current = json.loads((out / "manifest.json").read_text())
    previous = json.loads((source / "manifest.json").read_text())
    for key in ("request_sha256", "routes_sha256", "baseline_sha256", "engine", "arms", "source_hashes"):
        assert current[key] == previous[key], f"resume {key} differs"
    if (source / "generations.jsonl").exists():
        assert json.loads((source / "model_identity.json").read_text()) == c.model_identity()
        assert not (out / "generations.jsonl").exists()
        shutil.copy2(source / "generations.jsonl", out / "generations.jsonl")
    print("RESUME exact inputs, engine and checkpoint verified", flush=True)


def infer(out, source):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from offline import ENGINE_EXTRA
    identity = c.model_identity()
    assert identity == json.loads((source / "model_identity.json").read_text()), "checkpoint changed"
    c.dump(out / "model_identity.json", identity)
    cfg = c.engine_config()
    print("ENGINE_CONFIG", json.dumps(dict(model=c.MODEL, vllm=__import__("vllm").__version__, **cfg)), flush=True)
    tick = time.monotonic()
    llm = LLM(model=c.MODEL, tokenizer=c.MODEL, trust_remote_code=True, max_model_len=cfg["max_model_len"],
              max_num_seqs=cfg["batch"], max_num_batched_tokens=cfg["max_num_batched_tokens"],
              enable_chunked_prefill=True, enable_prefix_caching=False, gpu_memory_utilization=cfg["gpu_util"],
              enforce_eager=False, disable_log_stats=True,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}, **ENGINE_EXTRA)
    load_seconds = time.monotonic() - tick
    c.dump(out / "load_cost.json", dict(seconds=load_seconds))
    tok = llm.get_tokenizer()
    def render(msgs, thinking=True):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking, preserve_thinking=True)
    def params(arm):
        return SamplingParams(temperature=0, top_p=.95, top_k=20, max_tokens=cfg["answer_tokens"],
                              structured_outputs=StructuredOutputsParams(json=answer_schema(arm)))
    for arm in ARMS:
        smoke_prompt = [{"role":"user", "content":'Return label N. Other fields, if required, may be empty strings.'}]
        result = llm.generate([render(smoke_prompt, False)], params(arm), use_tqdm=False)[0].outputs[0]
        if result.finish_reason != "stop" or parse_final(result.text, arm) is None:
            raise RuntimeError(f"{arm} final-schema smoke failed: {result.finish_reason}")
    print("SMOKE passed: both final JSON schemas", flush=True)
    requests = c.jsonl(out / "requests.jsonl")
    done = c.load_completed(out / "generations.jsonl", {r["key"] for r in requests})
    todo = [r for r in requests if r["key"] not in done]
    with (out / "generations.jsonl").open("a", buffering=1) as sink:
        for pos in range(0, len(todo), cfg["batch"]):
            batch = todo[pos:pos+cfg["batch"]]
            tick = time.monotonic()
            prompts = [render(r["messages"]) for r in batch]
            lengths = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
            eligible = [i for i,n in enumerate(lengths) if n+cfg["think_tokens"]+cfg["answer_tokens"]+16 <= cfg["max_model_len"]]
            thinking = SamplingParams(temperature=0, top_p=.95, top_k=20, max_tokens=cfg["think_tokens"],
                                      stop=["</think>"], include_stop_str_in_output=False)
            traces = llm.generate([prompts[i] for i in eligible], thinking, use_tqdm=False) if eligible else []
            assert len(traces) == len(eligible)
            trace_map = {i:x.outputs[0] for i,x in zip(eligible,traces)}
            final_prompts = [prompts[i]+trace_map[i].text+"\n</think>\n\n" for i in eligible]
            final_lengths = [len(tok.encode(p, add_special_tokens=False)) for p in final_prompts]
            assert all(n+cfg["answer_tokens"] <= cfg["max_model_len"] for n in final_lengths)
            finals = llm.generate(final_prompts, [params(batch[i]["arm"]) for i in eligible], use_tqdm=False) if eligible else []
            assert len(finals) == len(eligible)
            answer_map = {i:x.outputs[0] for i,x in zip(eligible,finals)}
            final_lens = dict(zip(eligible,final_lengths))
            seconds = time.monotonic()-tick
            for i, req in enumerate(batch):
                trace, answer = trace_map.get(i), answer_map.get(i)
                parsed = parse_final(answer.text, req["arm"]) if answer and answer.finish_reason == "stop" else None
                row = dict(key=req["key"], dataset=req["dataset"], arm=req["arm"], reverse=req["reverse"],
                    status="valid" if parsed else ("context_overflow" if answer is None else "invalid_answer"),
                    thought=trace.text if trace else None, answer=answer.text if answer else None,
                    thinking_forced=bool(trace and trace.stop_reason != "</think>"),
                    thinking_finish=trace.finish_reason if trace else None, answer_finish=answer.finish_reason if answer else None,
                    prompt_tokens=lengths[i], total_input_tokens=(lengths[i] if trace else 0)+final_lens.get(i,0),
                    think_tokens=len(trace.token_ids) if trace else 0, answer_tokens=len(answer.token_ids) if answer else 0,
                    wall_seconds_share=seconds/len(batch))
                sink.write(json.dumps(row, ensure_ascii=False)+"\n")
                sink.flush()
            if pos//cfg["batch"] % 5 == 0 or pos+len(batch) == len(todo):
                print("PROGRESS", json.dumps(dict(done=len(done)+pos+len(batch), total=len(requests),
                    dataset=batch[-1]["dataset"], arm=batch[-1]["arm"], reverse=batch[-1]["reverse"], batch_seconds=seconds)), flush=True)
    print("INFERENCE complete", len(requests), flush=True)


def score(out):
    manifest = json.loads((out / "manifest.json").read_text())
    for name, key in (("requests.jsonl","request_sha256"), ("routes.jsonl","routes_sha256"), ("baselines.json","baseline_sha256")):
        assert c.sha(out / name) == manifest[key]
    requests = c.jsonl(out / "requests.jsonl")
    assert c.load_completed(out / "generations.jsonl", {r["key"] for r in requests}) == {r["key"] for r in requests}
    rows = {r["key"]:r for r in c.jsonl(out / "generations.jsonl")}
    predictions, quality, costs = {}, defaultdict(Counter), defaultdict(Counter)
    for req in requests:
        row = rows[req["key"]]
        group = (req["dataset"], req["arm"])
        quality[group][row["status"]] += 1
        quality[group]["forced_thinking"] += int(row["thinking_forced"])
        for field in ("prompt_tokens", "total_input_tokens", "think_tokens", "answer_tokens", "wall_seconds_share"):
            costs[group][field] += row[field]
        obj = parse_final(row["answer"], req["arm"]) if row["status"] == "valid" else None
        raw = canonical_label(obj["label"], req["reverse"]) if obj else None
        anchored = bool(obj and (req["arm"] == "direct" or grounded(obj, req["context"], req["presented"])))
        if obj and not anchored:
            quality[group]["unanchored_answer"] += 1
        key = (req["dataset"], req["doc_id"], tuple(req["pair"]), req["arm"], req["reverse"])
        assert key not in predictions
        predictions[key] = (raw if anchored else None, raw)
    ctb, _, esc, _ = c.inputs()
    docs = {"CTB":ctb, "ESC":esc}
    baseline_json = json.loads((out / "baselines.json").read_text())
    routes = defaultdict(set)
    for route in c.jsonl(out / "routes.jsonl"):
        routes[(route["dataset"],route["doc_id"])].add(tuple(route["pair"]))
    report = dict(scope=manifest["scope"], primary=PRIMARY, parent_run=manifest["parent_run"],
                  selected=manifest["selected"], conditions={}, baselines={}, contrasts={},
                  upstream_cost=json.loads((out / "upstream_cost.json").read_text()),
                  routing=json.loads((out / "routing_diagnostic.json").read_text()))
    per_conditions = {}
    for dataset, ids in manifest["selected"].items():
        bases = baseline_json[dataset]
        base_name = "n1_p3" if dataset == "CTB" else "qwen_p2_sym"
        base_per = {}
        for name, saved in bases.items():
            per = {}
            for doc_id in ids:
                gold, universe = f.gold_and_universe(dataset, docs[dataset][doc_id])
                per[doc_id] = c.counts({tuple(p) for p in saved[doc_id]}, gold, universe)
            base_per[name] = per
            report["baselines"][f"{dataset}/{name}"] = c.metric(per)
        for arm in ARMS:
            variants = ((PRIMARY,0), ("additions_only",0))
            if arm == "evidence":
                variants += (("raw_consensus_repair",1),)
            for variant, field_index in variants:
                per, changes = {}, Counter()
                for doc_id in ids:
                    old = unordered(bases[base_name][doc_id])
                    chosen = set(old)
                    for pair in routes[(dataset,doc_id)]:
                        labels = [predictions[(dataset,doc_id,pair,arm,reverse)][field_index] for reverse in (False,True)]
                        changes["abstained_routes"] += int(None in labels)
                        changes["disagreed_routes"] += int(None not in labels and labels[0] != labels[1])
                        if merge_decision(pair in old, labels, variant): chosen.add(pair)
                        else: chosen.discard(pair)
                    gold, universe = f.gold_and_universe(dataset,docs[dataset][doc_id])
                    gold_links = unordered(gold)
                    changes["added_tp_links"] += len((chosen-old) & gold_links)
                    changes["added_fp_links"] += len((chosen-old)-gold_links)
                    changes["removed_tp_links"] += len((old-chosen) & gold_links)
                    changes["removed_fp_links"] += len((old-chosen)-gold_links)
                    per[doc_id] = c.counts(symmetric(chosen),gold,universe)
                key = f"{dataset}/{arm}/{variant}"
                per_conditions[key] = per
                result = dict(**c.metric(per), changes_unordered=dict(changes), quality=dict(quality[(dataset,arm)]),
                              incremental_cost=dict(costs[(dataset,arm)]), per_document=per,
                              vs_baselines={n:paired_delta(per,b,5000,seed=c.SEED) for n,b in base_per.items()})
                report["conditions"][key] = result
                print("RESULT",key,json.dumps({k:v for k,v in result.items() if k != "per_document"}),flush=True)
        key = f"{dataset}/evidence_vs_direct/{PRIMARY}"
        contrast = paired_delta(per_conditions[f"{dataset}/evidence/{PRIMARY}"],
                                per_conditions[f"{dataset}/direct/{PRIMARY}"],5000,seed=c.SEED)
        report["contrasts"][key] = contrast
        print("CONTRAST",key,json.dumps(contrast),flush=True)
    c.dump(out / "report.json", report)
    print("FINAL",json.dumps(dict(requests=len(requests),report=str(out / "report.json"),
        scope=manifest["scope"],cost_note="Incremental adjudication only; baseline/C2 generation costs are sunk and must be included for deployment.")),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","infer","score","resume","complete"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--work",type=Path)
    args = parser.parse_args()
    if args.action == "prepare": prepare(args.out,args.source,args.work)
    elif args.action == "infer": infer(args.out,args.source)
    elif args.action == "score": score(args.out)
    elif args.action == "resume": resume(args.out,args.source)
    else: raise SystemExit(0 if c.complete(args.out) else 1)


if __name__ == "__main__":
    main()
