"""C1: paired context/thinking factorial for exhaustive ECI pair blocks.

Frozen Qwen, no gradients. CTB20 and ESC12 are chosen by document metadata;
their labels are never used to construct inference requests. All candidates,
including candidates absent from previous votes, occur once per schedule and
condition. JSON is decoded separately after any bounded reasoning; only that
final JSON is scored. Partial or invalid blocks abstain but stay in the F1
denominator. Outputs persist on a mounted volume and can resume by identity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = "/app/models/Qwen3.8-27B-W4A16-AutoRound"
SEED = 20261107
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "12"))
CONDITIONS = (
    ("local_no_think", "local", False),
    ("full_no_think", "full", False),
    ("local_think", "local", True),
    ("full_think", "full", True),
)
TAG = re.compile(r"<([^\s<>]+)\s+([^<>]+)>")
BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9<\"'])|\n+")


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(doc: dict) -> list[tuple[str, str]]:
    return sorted({tuple(sorted((str(a), str(b)))) for a, b in doc["pair_list"] if a != b})


def select_docs(ctb: dict[str, dict], reference: dict, esc: list[dict]) -> dict[str, list[str]]:
    """Length-stratified CTB and 12 ESC topics, using no annotation labels."""
    eligible = sorted(reference["qwen_ids"], key=lambda d: (len(ctb[d]["tokens"]), d))
    rng = random.Random(SEED)
    chosen = []
    for k in range(5):
        group = eligible[k * len(eligible) // 5:(k + 1) * len(eligible) // 5]
        chosen.extend(rng.sample(group, 4))
    topics = defaultdict(list)
    for doc in esc:
        topics[doc["id"].split("_")[0]].append(doc)
    rng = random.Random(SEED)
    selected_topics = rng.sample(sorted(topics), 12)
    selected_esc = []
    for index, topic in enumerate(selected_topics):
        options = sorted(topics[topic], key=lambda d: (len(d["text"]), d["id"]))
        selected_esc.append(options[0 if index % 2 == 0 else -1]["id"])
    assert len(set(chosen)) == 20 and len(set(selected_esc)) == 12
    assert len({d.split("_")[0] for d in selected_esc}) == 12
    return {"CTB": chosen, "ESC": selected_esc}


def windows(text: str, ids: set[str]) -> str:
    """Union of mention sentences and neighbours; exact substrings of source text.

    ESC's stored document text has no sentence offsets. The deterministic
    punctuation splitter is an approximation; include every occurrence of an ID.
    The same splitter is applied to both datasets, without consulting labels.
    """
    starts = [0] + [match.end() for match in BOUNDARY.finditer(text)]
    ends = starts[1:] + [len(text)]
    sentences = [text[a:b] for a, b in zip(starts, ends)]
    found, include = set(), set()
    for i, sentence in enumerate(sentences):
        hits = {m.group(1) for m in TAG.finditer(sentence)} & ids
        if hits:
            found |= hits
            include.update(range(max(0, i - 1), min(len(sentences), i + 2)))
    if found != ids:
        raise ValueError(f"source text lacks requested event tags: {sorted(ids-found)}")
    runs = []
    for i in sorted(include):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return "\n[... omitted text ...]\n".join("".join(sentences[i] for i in run) for run in runs)


def schema(size: int) -> dict:
    names = [str(i) for i in range(size)]
    return {"type": "object", "properties": {n: {"type": "string", "enum": ["A", "B", "N"]} for n in names},
            "required": names, "additionalProperties": False}


def parse_answer(text: str, size: int) -> dict[str, str] | None:
    """Only accept an entire final answer with exactly one decision per pair."""
    def no_duplicates(items):
        keys = [k for k, _ in items]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate pair indices")
        return dict(items)
    try:
        obj = json.loads(text, object_pairs_hook=no_duplicates)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != {str(i) for i in range(size)}:
        return None
    return obj if all(v in ("A", "B", "N") for v in obj.values()) else None


def make_messages(rules: str, context: str, block: list[tuple[str, str]], order: list[int]) -> list[dict]:
    question = "\n".join(f"{i}: A=<{block[i][0]}>, B=<{block[i][1]}>" for i in order)
    system = (rules + "\n\nOnly classify the listed pairs. Give every listed pair exactly one "
              "answer in a JSON object mapping its index to A, B, or N. "
              "A means A causes B; B means B causes A; N means no annotated causal relation. "
              "Use the event tags in the text. Do not invent extra pair indices.")
    user = (f"Document context:\n{context}\n\nPairs:\n{question}\n\n"
            "Return the JSON object with one A, B, or N value for each listed index.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def inputs():
    ctb = {d["id"]: d for d in json.loads((DATA / "ctb-suite.json").read_text())["documents"]}
    reference = json.loads((DATA / "reference.json").read_text())
    esc_rows = json.loads((DATA / "esc/test-s40.json").read_text())
    esc = {d["id"]: d for d in esc_rows}
    ctb_rules = json.loads((DATA / "ctb-prompt.json").read_text())["system"].split("# Output Format")[0].strip()
    esc_rules = json.loads((DATA / "esc/prompt.json").read_text())["system"].split(
        "If there are no mentions in the text")[0].strip()
    return ctb, reference, esc, {"CTB": ctb_rules, "ESC": esc_rules}


def prepare(out: Path) -> None:
    ctb, reference, esc, rules = inputs()
    selected = select_docs(ctb, reference, list(esc.values()))
    docs = {"CTB": ctb, "ESC": esc}
    requests = []
    for dataset in ("CTB", "ESC"):
        for doc_id in selected[dataset]:
            doc = docs[dataset][doc_id]
            all_pairs = pairs(doc)
            tags = {m.group(1) for m in TAG.finditer(doc["text"])}
            assert {x for p in all_pairs for x in p} <= tags, (dataset, doc_id)
            for block_index, start in enumerate(range(0, len(all_pairs), BLOCK_SIZE)):
                block = all_pairs[start:start + BLOCK_SIZE]
                local = windows(doc["text"], {x for p in block for x in p})
                for condition, context_name, thinking in CONDITIONS:
                    for schedule in (0, 1):
                        order = list(range(len(block)))
                        if schedule:
                            random.Random(f"{SEED}::{dataset}::{doc_id}::{block_index}").shuffle(order)
                        context = local if context_name == "local" else doc["text"]
                        record = dict(dataset=dataset, doc_id=doc_id, block=block_index,
                                      condition=condition, context=context_name, thinking=thinking,
                                      schedule=schedule, pairs=block,
                                      messages=make_messages(rules[dataset], context, block, order))
                        record["key"] = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                        requests.append(record)
    requests.sort(key=lambda r: (r["condition"], r["dataset"], r["schedule"], r["doc_id"], r["block"]))
    assert len({r["key"] for r in requests}) == len(requests)
    serialized = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in requests)
    request_path = out / "requests.jsonl"
    if request_path.exists():
        assert request_path.read_text() == serialized, "Existing requests differ; refusing unsafe resume"
    else:
        request_path.write_text(serialized)
    data_paths = [DATA / "ctb-suite.json", DATA / "reference.json", DATA / "ctb-prompt.json",
                  DATA / "esc/test-s40.json", DATA / "esc/baseline.json", DATA / "esc/prompt.json"]
    manifest = dict(model=MODEL, seed=SEED, block_size=BLOCK_SIZE, selected=selected,
                    conditions=[dict(name=n, context=c, thinking=t) for n, c, t in CONDITIONS],
                    schedules=[0, 1], requests=len(requests),
                    candidate_pairs={name: sum(len(pairs(docs[name][i])) for i in selected[name]) for name in selected},
                    source_hashes={str(p.relative_to(DATA)): sha(p) for p in data_paths},
                    request_sha256=sha(request_path), engine=engine_config(), commit=os.environ.get("ORX_COMMIT"),
                    scope="reused CTB47 and ESC TEST-S40 development data; no gold in requests")
    if (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text())
        assert {k:v for k,v in previous.items() if k != "commit"} == {
            k:v for k,v in manifest.items() if k != "commit"}, "existing manifest differs"
    dump(out / "manifest.json", manifest)
    print("PREPARED", json.dumps({k: v for k, v in manifest.items() if k not in ("source_hashes", "conditions")}), flush=True)


def resume(out: Path, source: Path) -> None:
    current, previous = json.loads((out / "manifest.json").read_text()), json.loads((source / "manifest.json").read_text())
    assert current["request_sha256"] == previous["request_sha256"], "resume request identity mismatch"
    assert current["source_hashes"] == previous["source_hashes"], "resume source identity mismatch"
    assert current["engine"] == previous["engine"], "resume engine settings mismatch"
    if (source / "generations.jsonl").exists():
        assert (source / "model_identity.json").exists(), "resume model identity unavailable"
        assert json.loads((source / "model_identity.json").read_text()) == model_identity(), "resume checkpoint identity mismatch"
        assert not (out / "generations.jsonl").exists(), "destination already contains generations"
        shutil.copy2(source / "generations.jsonl", out / "generations.jsonl")
    dump(out / "resume.json", {"source": str(source), "request_identity_verified": True})
    print("RESUME verified request and source identities", flush=True)


def load_completed(path: Path, expected: set[str]) -> set[str]:
    if not path.exists():
        return set()
    # A host restart can interrupt the last append. Preserve all complete
    # records; discard only a malformed final fragment in the destination.
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        try:
            json.loads(raw[boundary:].decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            with path.open("r+b") as handle:
                handle.truncate(boundary)
            print("RECOVERY discarded incomplete final JSONL record", flush=True)
        else:
            with path.open("ab") as handle:
                handle.write(b"\n")
            print("RECOVERY terminated complete final JSONL record", flush=True)
    rows = jsonl(path)
    seen = [row["key"] for row in rows]
    if len(set(seen)) != len(seen) or not set(seen) <= expected:
        raise ValueError("generation cache has duplicate or unexpected keys")
    return set(seen)


def complete(out: Path) -> bool:
    requests = jsonl(out / "requests.jsonl")
    expected = {r["key"] for r in requests}
    done = load_completed(out / "generations.jsonl", expected)
    print(f"COVERAGE completed={len(done)} expected={len(expected)}", flush=True)
    return done == expected


def engine_config():
    return dict(max_model_len=int(os.environ.get("MAX_MODEL_LEN", "32768")),
                max_num_batched_tokens=int(os.environ.get("MAX_BATCHED_TOKENS", "8192")),
                batch=int(os.environ.get("BATCH", "8")), gpu_util=float(os.environ.get("GPU_UTIL", "0.90")),
                think_tokens=int(os.environ.get("THINK_TOKENS", "2048")),
                answer_tokens=int(os.environ.get("ANSWER_TOKENS", "384")))


def model_identity():
    from importlib.metadata import version
    root = Path(MODEL)
    files = sorted(p for p in root.glob("*.safetensors") if p.is_file())
    if not files:
        raise RuntimeError(f"no model weight files under {root}")
    return {"model": MODEL, "config_sha256": sha(root / "config.json"),
            "tokenizer_config_sha256": sha(root / "tokenizer_config.json"),
            "tokenizer_sha256": sha(root / "tokenizer.json"), "vllm_version": version("vllm"),
            "weights": [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]}


def live_score(out: Path, dataset: str, condition: str, schedule: int) -> None:
    """Print a completed group's micro-F1 using every document in that dataset."""
    requests = [r for r in jsonl(out / "requests.jsonl") if r["dataset"] == dataset
                and r["condition"] == condition and r["schedule"] == schedule]
    results = {r["key"]: r for r in jsonl(out / "generations.jsonl")}
    if any(r["key"] not in results for r in requests):
        return
    ctb, _, esc, _ = inputs()
    all_docs = ctb if dataset == "CTB" else esc
    selected = json.loads((out / "manifest.json").read_text())["selected"][dataset]
    prediction = defaultdict(set)
    for req in requests:
        row = results[req["key"]]
        answer = parse_answer(row["answer"], len(req["pairs"])) if row["status"] == "valid" else None
        if answer:
            for i, (a, b) in enumerate(req["pairs"]):
                if answer[str(i)] != "N":
                    prediction[req["doc_id"]].update(((a, b), (b, a)))
    per = {}
    for doc_id in selected:
        doc = all_docs[doc_id]
        universe = {(str(a), str(b)) for a, b in doc["pair_list"] if a != b}
        gold = ({(a, b) for a, label, b in doc["gold"] if label != "norel"} if dataset == "CTB"
                else {tuple(p.split(",", 1)) for p in doc["gold"]})
        per[doc_id] = counts(prediction[doc_id], gold, universe)
    print("PARTIAL_RESULT", dataset, condition, f"order={schedule}", json.dumps(metric(per)), flush=True)


def infer(out: Path) -> None:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from offline import ENGINE_EXTRA

    requests = jsonl(out / "requests.jsonl")
    expected = {r["key"] for r in requests}
    destination = out / "generations.jsonl"
    completed = load_completed(destination, expected)
    cfg = engine_config()
    identity = model_identity()
    dump(out / "model_identity.json", identity)
    print("ENGINE_CONFIG", json.dumps(dict(model=MODEL, model_config_sha256=identity["config_sha256"],
          vllm=__import__("vllm").__version__, chunked_prefill=True, prefix_caching=False, **cfg)), flush=True)
    engine_started = time.monotonic()
    llm = LLM(model=MODEL, tokenizer=MODEL, trust_remote_code=True, max_model_len=cfg["max_model_len"],
              max_num_seqs=cfg["batch"], max_num_batched_tokens=cfg["max_num_batched_tokens"],
              enable_chunked_prefill=True, enable_prefix_caching=False, gpu_memory_utilization=cfg["gpu_util"],
              enforce_eager=False, disable_log_stats=True,
              compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}, **ENGINE_EXTRA)
    print(f"== engine loaded seconds={time.monotonic()-engine_started:.1f}", flush=True)
    tok = llm.get_tokenizer()
    def render(messages, thinking):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking, preserve_thinking=True)
    def output_params(size):
        return SamplingParams(temperature=0, top_p=0.95, top_k=20, max_tokens=cfg["answer_tokens"],
                              structured_outputs=StructuredOutputsParams(json=schema(size)))
    # Before the long run, check constrained output and both chat-template paths on GPU.
    smoke = [{"role": "user", "content": "Output a JSON object whose key 0 has value N."}]
    for thinking in (False, True):
        prefix = render(smoke, thinking)
        if thinking:
            p = SamplingParams(temperature=0, max_tokens=32, stop=["</think>"], include_stop_str_in_output=False)
            thought = llm.generate([prefix], p, use_tqdm=False)[0].outputs[0]
            if thought.finish_reason not in ("length", "stop"):
                raise RuntimeError(f"thinking smoke failed: {thought.finish_reason}")
            prefix += thought.text + "\n</think>\n\n"
        result = llm.generate([prefix], output_params(1), use_tqdm=False)[0].outputs[0]
        if result.finish_reason != "stop" or parse_answer(result.text, 1) is None:
            raise RuntimeError(f"structured JSON smoke failed for thinking={thinking}: {result.finish_reason} {result.text!r}")
    print("SMOKE passed: both thinking templates and constrained JSON", flush=True)
    done_so_far = len(completed)
    with destination.open("a", buffering=1) as sink:
        for condition, _, thinking in CONDITIONS:
            for dataset in ("CTB", "ESC"):
                for schedule in (0, 1):
                    group = [r for r in requests if r["condition"] == condition and r["dataset"] == dataset
                             and r["schedule"] == schedule and r["key"] not in completed]
                    started_group = time.monotonic()
                    profile_blocks, profile_seconds = 0, 0.0
                    print(f"== start {condition} {dataset} order={schedule} remaining={len(group)}", flush=True)
                    for pos in range(0, len(group), cfg["batch"]):
                        batch = group[pos:pos + cfg["batch"]]
                        started = time.monotonic()
                        prefixes = [render(r["messages"], thinking) for r in batch]
                        input_lens = [len(tok.encode(p, add_special_tokens=False)) for p in prefixes]
                        required = cfg["answer_tokens"] + (cfg["think_tokens"] + 16 if thinking else 0)
                        eligible = [i for i, n in enumerate(input_lens) if n + required <= cfg["max_model_len"]]
                        traces = {}
                        if thinking and eligible:
                            p = SamplingParams(temperature=0, top_p=0.95, top_k=20, max_tokens=cfg["think_tokens"],
                                               stop=["</think>"], include_stop_str_in_output=False)
                            for index, result in zip(eligible, llm.generate([prefixes[i] for i in eligible], p, use_tqdm=False)):
                                traces[index] = result.outputs[0]
                        answer_prefixes = []
                        for i in eligible:
                            trace = traces.get(i)
                            answer_prefixes.append(prefixes[i] if trace is None else prefixes[i] + trace.text + "\n</think>\n\n")
                        answer_lens = [len(tok.encode(p, add_special_tokens=False)) for p in answer_prefixes]
                        if any(n + cfg["answer_tokens"] > cfg["max_model_len"] for n in answer_lens):
                            raise RuntimeError("answer prompt exceeds context after bounded thinking")
                        outputs = llm.generate(answer_prefixes, [output_params(len(batch[i]["pairs"])) for i in eligible],
                                               use_tqdm=False) if eligible else []
                        answers = {i: result.outputs[0] for i, result in zip(eligible, outputs)}
                        answer_lengths = dict(zip(eligible, answer_lens))
                        elapsed = time.monotonic() - started
                        for i, request in enumerate(batch):
                            trace, answer = traces.get(i), answers.get(i)
                            parsed = parse_answer(answer.text, len(request["pairs"])) if answer else None
                            valid = bool(answer and answer.finish_reason == "stop" and parsed is not None)
                            record = dict(key=request["key"], dataset=dataset, doc_id=request["doc_id"],
                                          condition=condition, schedule=schedule, block=request["block"],
                                          status="valid" if valid else ("context_overflow" if answer is None else "invalid_answer"),
                                          prompt_tokens=input_lens[i], think_tokens=len(trace.token_ids) if trace else 0,
                                          total_input_tokens=(input_lens[i] if trace else 0) + answer_lengths.get(i, 0),
                                          answer_tokens=len(answer.token_ids) if answer else 0,
                                          thinking_forced=bool(trace and trace.stop_reason != "</think>"),
                                          thinking_finish=trace.finish_reason if trace else None,
                                          answer_finish=answer.finish_reason if answer else None,
                                          thought=trace.text if trace else None,
                                          answer=answer.text if answer else None,
                                          batch_seconds=elapsed, batch_size=len(batch), wall_seconds_share=elapsed/len(batch))
                            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                            sink.flush()
                            completed.add(request["key"])
                        done_so_far += len(batch)
                        profile_blocks += len(batch)
                        profile_seconds += elapsed
                        if profile_blocks >= 50 and profile_blocks - len(batch) < 50:
                            print(f"PROFILE {condition} {dataset} order={schedule} blocks={profile_blocks} "
                                  f"seconds={profile_seconds:.1f} blocks_per_second={profile_blocks/profile_seconds:.4f}", flush=True)
                        if (pos // cfg["batch"]) % 5 == 0 or pos + len(batch) == len(group):
                            print(f"[{condition} {dataset} order={schedule}] {pos+len(batch)}/{len(group)} "
                                  f"total={done_so_far}/{len(requests)} batch={elapsed:.1f}s", flush=True)
                    print(f"== completed {condition} {dataset} order={schedule} seconds={time.monotonic()-started_group:.1f}", flush=True)
                    live_score(out, dataset, condition, schedule)


def counts(pred: set[tuple[str, str]], gold: set[tuple[str, str]], universe: set[tuple[str, str]]) -> dict:
    pred, gold = pred & universe, gold & universe
    return dict(tp=len(pred & gold), fp=len(pred - gold), fn=len(gold - pred))


def metric(per_doc: dict[str, dict]) -> dict:
    total = {k: sum(row[k] for row in per_doc.values()) for k in ("tp", "fp", "fn")}
    tp, fp, fn = (total[k] for k in ("tp", "fp", "fn"))
    return dict(**total, precision=100*tp/max(tp+fp, 1), recall=100*tp/max(tp+fn, 1),
                f1=200*tp/max(2*tp+fp+fn, 1), documents=len(per_doc))


def score(out: Path, work: Path) -> None:
    from compare import paired_delta
    from pair_verifier import cached_votes
    from eci.parse import parse_triples

    requests, outputs = jsonl(out / "requests.jsonl"), jsonl(out / "generations.jsonl")
    expected = {r["key"] for r in requests}
    assert load_completed(out / "generations.jsonl", expected) == expected, "incomplete run; do not publish a final score"
    by_key = {r["key"]: r for r in outputs}
    manifest = json.loads((out / "manifest.json").read_text())
    ctb, reference, esc, _ = inputs()
    docs = {"CTB": ctb, "ESC": esc}
    selected = manifest["selected"]
    gold, universe = {}, {}
    for dataset in selected:
        gold[dataset], universe[dataset] = {}, {}
        for doc_id in selected[dataset]:
            doc = docs[dataset][doc_id]
            universe[dataset][doc_id] = {(str(a), str(b)) for a, b in doc["pair_list"] if a != b}
            if dataset == "CTB":
                gold[dataset][doc_id] = {(a, b) for a, label, b in doc["gold"] if label != "norel"}
            else:
                gold[dataset][doc_id] = {tuple(p.split(",", 1)) for p in doc["gold"]}
            assert gold[dataset][doc_id] <= universe[dataset][doc_id], (dataset, doc_id)
    prediction = defaultdict(lambda: defaultdict(set))
    oriented = defaultdict(lambda: defaultdict(set))
    quality = defaultdict(Counter)
    costs = defaultdict(Counter)
    for req in requests:
        result = by_key[req["key"]]
        group = (req["dataset"], req["condition"], req["schedule"])
        quality[group][result["status"]] += 1
        quality[group]["forced_thinking"] += int(result["thinking_forced"])
        for k in ("prompt_tokens", "total_input_tokens", "think_tokens", "answer_tokens", "wall_seconds_share"):
            costs[group][k] += result[k]
        answer = parse_answer(result["answer"], len(req["pairs"])) if result["status"] == "valid" else None
        if answer is None:
            continue
        for i, (a, b) in enumerate(req["pairs"]):
            if answer[str(i)] != "N":
                prediction[group][req["doc_id"]].update(((a, b), (b, a)))
                oriented[group][req["doc_id"]].add((a, b) if answer[str(i)] == "A" else (b, a))
    baselines = {"CTB": {}, "ESC": {}}
    ids = selected["CTB"]
    qwen7 = {d: {tuple(k.split(",", 1)) for k, v in reference["qwen"][d].items() if v != "norel"} for d in ids}
    pool = cached_votes([work / "out/N1/gen.jsonl", work / "out/P3/gen.jsonl"], ctb, ids)
    qwen_pool = {}
    for d in ids:
        qwen_pool[d] = set()
        for (a, b), vote in pool[d].items():
            if vote >= .5:
                qwen_pool[d].update(((a, b), (b, a)))
    for name, preds in (("qwen7", qwen7), ("n1_p3", qwen_pool)):
        baselines["CTB"][name] = {d: counts(preds[d], gold["CTB"][d], universe["CTB"][d]) for d in ids}
    esc_base = json.loads((DATA / "esc/baseline.json").read_text())
    p2 = defaultdict(list)
    p2_path = work / "out/P2/gen.jsonl"
    for row in jsonl(p2_path):
        if row["doc_id"] not in selected["ESC"]:
            continue
        found = None
        if row.get("finish_reason") in (None, "stop") and "</think>" in row.get("text", ""):
            try:
                found = {(a, b) for a, lab, b in parse_triples(row["text"].rsplit("</think>", 1)[1]) if lab != "norel"}
            except ValueError:
                pass
        p2[row["doc_id"]].append(found)
    for d in selected["ESC"]:
        assert len(p2[d]) == 3, (d, len(p2[d]))
    for name in ("luna7", "fusion", "qwen_p2_sym"):
        per = {}
        for d in selected["ESC"]:
            pred = set()
            valid = [x for x in p2[d] if x is not None]
            for a, b in universe["ESC"][d]:
                ab, ba = esc_base[d][f"{a},{b}"], esc_base[d][f"{b},{a}"]
                if name == "luna7":
                    positive = ab[0] >= 4/7 - 1e-9
                elif name == "fusion":
                    positive = (ab[0] + ba[0] + ab[1] + ba[1]) / 4 >= .39
                else:
                    positive = bool(valid) and (sum((a, b) in x for x in valid) + sum((b, a) in x for x in valid)) / (2*len(valid)) >= .5
                if positive:
                    pred.add((a, b))
            per[d] = counts(pred, gold["ESC"][d], universe["ESC"][d])
        baselines["ESC"][name] = per
    report = dict(scope=manifest["scope"], selected=selected, conditions={}, baselines={}, factorial_effects={},
                  baseline_hashes={str(p): sha(p) for p in (work / "out/N1/gen.jsonl", work / "out/P3/gen.jsonl", p2_path)})
    per_groups = {}
    for dataset in selected:
        report["baselines"][dataset] = {name: metric(per) for name, per in baselines[dataset].items()}
        for condition, _, _ in CONDITIONS:
            for schedule in (0, 1):
                group = (dataset, condition, schedule)
                per = {d: counts(prediction[group][d], gold[dataset][d], universe[dataset][d]) for d in selected[dataset]}
                per_groups[group] = per
                comparisons = {name: paired_delta(per, base, 5000, seed=SEED) for name, base in baselines[dataset].items()}
                result = dict(**metric(per), quality=dict(quality[group]), cost=dict(costs[group]),
                              paired_vs_baselines=comparisons, per_document=per)
                if dataset == "CTB":
                    directed = {}
                    for d in selected[dataset]:
                        true_edges = {(a, b) if label == "causes" else (b, a) for a, label, b in ctb[d]["gold"]
                                      if label in ("causes", "causedby")}
                        predicted_edges = oriented[group][d]
                        directed[d] = dict(tp=len(predicted_edges & true_edges), fp=len(predicted_edges - true_edges),
                                           fn=len(true_edges - predicted_edges))
                    result["directed_ctb"] = metric(directed)
                name = f"{dataset}/{condition}/order{schedule}"
                report["conditions"][name] = result
                print("RESULT", name, json.dumps({k:v for k,v in result.items() if k != "per_document"}), flush=True)
        for schedule in (0, 1):
            for label, new, old in (("full_minus_local_no_think", "full_no_think", "local_no_think"),
                                    ("full_minus_local_think", "full_think", "local_think"),
                                    ("think_minus_no_local", "local_think", "local_no_think"),
                                    ("think_minus_no_full", "full_think", "full_no_think")):
                key = f"{dataset}/{label}/order{schedule}"
                report["factorial_effects"][key] = paired_delta(per_groups[(dataset, new, schedule)],
                                                                 per_groups[(dataset, old, schedule)], 5000, seed=SEED)
                print("FACTOR", key, json.dumps(report["factorial_effects"][key]), flush=True)
    dump(out / "report.json", report)
    print("BASELINES", json.dumps(report["baselines"]), flush=True)
    print("FINAL", json.dumps(dict(requests=len(requests), selected=selected, report=str(out / "report.json"))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "infer", "complete"):
        p = sub.add_parser(action)
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("resume")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare": prepare(args.out)
    elif args.action == "infer": infer(args.out)
    elif args.action == "resume": resume(args.out, args.source)
    elif args.action == "score": score(args.out, args.work)
    else: raise SystemExit(0 if complete(args.out) else 1)


if __name__ == "__main__":
    main()
