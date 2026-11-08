"""C3 over a persistent local vLLM completion API, without loading GPU weights.

Keep the native two-stage token protocol. Do not import native C3 generations:
FP8 KV, FP16 recurrent state, prefix caching and scheduling define a new arm.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import urllib.parse

from steering import campaign_adjudicate as a

PROTOCOL = "c3-raw-completions-batch-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def ordered_choices(response, count):
    choices = response.get("choices", [])
    if len(choices) != count or {c.get("index") for c in choices} != set(range(count)):
        raise ValueError("API returned missing, duplicate or unexpected completion indices")
    return sorted(choices, key=lambda x: x["index"])


def payload(model, prompts, max_tokens, arm=None):
    body = dict(model=model, prompt=prompts, temperature=0, top_p=.95, top_k=20,
                max_tokens=max_tokens, n=1, stream=False, add_special_tokens=False,
                return_token_ids=True, skip_special_tokens=True)
    if arm is None:
        body.update(stop=["</think>"], include_stop_str_in_output=False)
    else:
        # Native-vLLM extension, not an OpenRouter request.
        body["structured_outputs"] = {"json": a.answer_schema(arm)}
    return body


class Client:
    def __init__(self, base, key, cache, identity, timeout=900):
        self.base, self.key, self.cache = base.rstrip("/"), key, cache
        self.identity, self.timeout = identity, timeout
        cache.mkdir(parents=True, exist_ok=True)

    def call(self, path, body=None, cache=False):
        key = digest(dict(base=self.base, path=path, body=body, identity=self.identity))
        target = self.cache / (key + ".json")
        if cache and target.exists():
            stored = json.loads(target.read_text())
            assert stored["request_hash"] == key
            return stored["response"], True, stored["elapsed_seconds"]
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        url = urllib.parse.urljoin(self.base + "/", path.lstrip("/"))
        req = urllib.request.Request(url,
            data=canonical(body).encode() if body is not None else None, headers=headers)
        tick = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                obj = json.load(response)
        except urllib.error.HTTPError as exc:
            # Headers may contain credentials; record only the response body.
            detail = exc.read().decode("utf-8", errors="replace")[:3000]
            raise RuntimeError(f"API HTTP {exc.code}: {detail}") from None
        elapsed = time.monotonic() - tick
        if cache:
            record = dict(request_hash=key, request=body, response=obj, elapsed_seconds=elapsed)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(canonical(record) + "\n")
            temporary.replace(target)
        return obj, False, elapsed


def batches(requests, size):
    """Every multi-prompt POST has a single final-answer schema."""
    result, current = [], []
    for req in requests:
        if current and (len(current) == size or req["arm"] != current[0]["arm"]):
            result.append(current)
            current = []
        current.append(req)
    if current:
        result.append(current)
    return result


def generation_rows(requests, prompts, traces, finals, tokenizer, elapsed, request_cost):
    rows = []
    for req, prompt, trace, final in zip(requests, prompts, traces, finals, strict=True):
        thought = trace["text"]
        final_prompt = tokenizer.encode(prompt + thought + "\n</think>\n\n", add_special_tokens=False)
        prompt_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        parsed = a.parse_final(final["text"], req["arm"]) if final["finish_reason"] == "stop" else None
        rows.append(dict(key=req["key"], dataset=req["dataset"], arm=req["arm"], reverse=req["reverse"],
            status="valid" if parsed else "invalid_answer", thought=thought, answer=final["text"],
            thinking_forced=trace.get("stop_reason") != "</think>",
            thinking_finish=trace["finish_reason"], answer_finish=final["finish_reason"],
            prompt_tokens=prompt_count, total_input_tokens=prompt_count+len(final_prompt),
            think_tokens=len(trace.get("token_ids") or tokenizer.encode(thought, add_special_tokens=False)),
            answer_tokens=len(final.get("token_ids") or tokenizer.encode(final["text"], add_special_tokens=False)),
            wall_seconds_share=elapsed/len(requests), protocol=PROTOCOL,
            api_batch_usage_share={k:v/len(requests) for k,v in request_cost.items() if isinstance(v,(int,float))}))
    return rows


def execute_batch(client, tokenizer, requests, cfg, model):
    arm = requests[0]["arm"]
    assert all(r["arm"] == arm for r in requests)
    prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False,
        add_generation_prompt=True, enable_thinking=True, preserve_thinking=True) for r in requests]
    token_prompts = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    if any(len(p)+cfg["think_tokens"]+cfg["answer_tokens"]+16 > cfg["max_model_len"] for p in token_prompts):
        raise ValueError("Frozen pair exceeds the declared context; refusing silent truncation")
    tick = time.monotonic()
    first, hit1, elapsed1 = client.call("/completions", payload(model, token_prompts, cfg["think_tokens"]), cache=True)
    traces = ordered_choices(first, len(requests))
    final_prompts = [tokenizer.encode(p+t["text"]+"\n</think>\n\n", add_special_tokens=False)
                     for p,t in zip(prompts,traces,strict=True)]
    second, hit2, elapsed2 = client.call("/completions", payload(model,final_prompts,cfg["answer_tokens"],arm), cache=True)
    finals = ordered_choices(second, len(requests))
    usage = {}
    for obj in (first,second):
        for name,value in obj.get("usage",{}).items():
            if isinstance(value,(int,float)):
                usage[name] = usage.get(name,0)+value
    rows = generation_rows(requests,prompts,traces,finals,tokenizer,elapsed1+elapsed2,usage)
    return rows, dict(seconds=time.monotonic()-tick, cache_hits=int(hit1)+int(hit2),
                      usage=usage, historical_request_seconds=elapsed1+elapsed2)


def run(out, native, server_file):
    from transformers import AutoTokenizer
    cfg = a.c.engine_config()
    server = json.loads(server_file.read_text())
    identity = a.c.model_identity()
    for field in ("config_sha256","tokenizer_sha256","tokenizer_config_sha256","vllm_version"):
        assert server["model_identity"][field] == identity[field], f"server/client model differs: {field}"
    assert server["weights_signature"] == digest(identity["weights"]), "server weights metadata differ"
    model = os.environ.get("API_MODEL","qwen3.8-27b")
    base = os.environ["API_BASE_URL"]
    contract = dict(protocol=PROTOCOL,server=server,base_url=base,served_model=model,
                    max_tokens_thinking=cfg["think_tokens"],max_tokens_answer=cfg["answer_tokens"],
                    raw_token_prompts=True,batch_size=int(os.environ.get("API_BATCH_SIZE","16")),
                    workers=int(os.environ.get("API_WORKERS","2")),
                    native_reference_run=native.name,native_outputs_reused=False)
    contract_file = out/"api_contract.json"
    if contract_file.exists():
        assert json.loads(contract_file.read_text()) == contract, "API contract changed on resume"
    a.c.dump(contract_file,contract)
    # Same scientific requests/baselines as native C3, but independent responses.
    manifest = json.loads((out/"manifest.json").read_text())
    original = json.loads((native/"manifest.json").read_text())
    for field in ("request_sha256","routes_sha256","baseline_sha256","source_hashes","arms"):
        assert manifest[field] == original[field], f"native/API scientific inputs differ: {field}"
    a.c.dump(out/"model_identity.json",identity)
    key_file = Path(os.environ["API_KEY_FILE"])
    key = key_file.read_text().strip() if key_file.exists() else ""
    client = Client(base,key,out/"http-cache",digest(contract))
    models,_,_ = client.call("/models")
    assert model in {m["id"] for m in models["data"]}, "wrong served model name"
    tokenizer = AutoTokenizer.from_pretrained(a.c.MODEL,trust_remote_code=True)
    # Verify server rendering/tokenizer alignment without a generation.
    test_text = tokenizer.apply_chat_template([{"role":"user","content":"Return JSON label N."}],
        tokenize=False,add_generation_prompt=True,enable_thinking=True,preserve_thinking=True)
    tokenized,_,_ = client.call("/../tokenize",dict(model=model,prompt=test_text,add_special_tokens=False))
    assert tokenized["tokens"] == tokenizer.encode(test_text,add_special_tokens=False), "server tokenizer mismatch"
    smoke = []
    for arm in a.ARMS:
        msgs = [{"role":"user","content":"Return label N. All other fields may be empty strings."}]
        text = tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        result,_,_ = client.call("/completions",payload(model,[tokenizer.encode(text,add_special_tokens=False)],128,arm))
        choice = ordered_choices(result,1)[0]
        assert choice["finish_reason"] == "stop" and a.parse_final(choice["text"],arm), "final JSON schema smoke failed"
        smoke.append(dict(arm=arm,response=result))
    # Stop delimiter and raw completion semantics must survive the HTTP surface.
    check,_,_ = client.call("/completions",payload(model,[tokenizer.encode(test_text,add_special_tokens=False)],64))
    assert "</think>" not in ordered_choices(check,1)[0]["text"], "thinking stop leaked into output"
    a.c.dump(out/"smoke.json",dict(tokenizer_match=True,schema=smoke,thinking=check))
    print("SMOKE passed: model identity, tokenizer, raw thinking, both final JSON schemas",flush=True)
    if os.environ.get("ORX_NOTIFY_FILE"):
        with open(os.environ["ORX_NOTIFY_FILE"],"a") as notify:
            notify.write("C3 API: tests modèle/tokenizer/JSON passés ; génération batch démarrée.\n")
    print("API_CONTRACT",canonical(contract),flush=True)
    requests = a.c.jsonl(out/"requests.jsonl")
    done = a.c.load_completed(out/"generations.jsonl",{r["key"] for r in requests})
    todo = [r for r in requests if r["key"] not in done]
    chunks = batches(todo,contract["batch_size"])
    started = time.monotonic()
    with (out/"generations.jsonl").open("a",buffering=1) as sink, \
         (out/"http-batches.jsonl").open("a",buffering=1) as ledger, \
         ThreadPoolExecutor(max_workers=contract["workers"]) as pool:
        futures = {pool.submit(execute_batch,client,tokenizer,chunk,cfg,model):chunk for chunk in chunks}
        completed = len(done)
        try:
            for future in as_completed(futures):
                rows, cost = future.result()
                for row in rows:
                    sink.write(canonical(row)+"\n")
                sink.flush()
                completed += len(rows)
                ledger.write(canonical(dict(keys=[r["key"] for r in rows],**cost))+"\n")
                print("PROGRESS",canonical(dict(done=completed,total=len(requests),
                    dataset=rows[-1]["dataset"],arm=rows[-1]["arm"],batch_seconds=cost["seconds"],
                    cumulative_wall_seconds=time.monotonic()-started)),flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    a.c.dump(out/"api_wall_cost.json",dict(inference_seconds=time.monotonic()-started,
              startup_and_server_loading_included=False,native_responses_reused=False))
    a.score(out)
    report = json.loads((out/"report.json").read_text())
    report["api_contract"] = contract
    report["api_wall_cost"] = json.loads((out/"api_wall_cost.json").read_text())
    a.c.dump(out/"report.json",report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--server",type=Path,required=True)
    args = parser.parse_args()
    run(args.out,args.native,args.server)
