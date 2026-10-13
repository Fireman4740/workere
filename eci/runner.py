"""One run, end to end — and a queue of runs.

  1. config -> run directory (config snapshot, git state), MLflow run started with its params
  2. evaluated documents loaded and decorated (syntax block, coreference hint)
  3. few-shot pool loaded, every document's demo sets planned, missing CoTs synthesized
  4. documents processed concurrently; each finished document is appended to the checkpoint
  5. report built from the checkpoint, written, histograms and graphs logged

A resumed run (same directory) skips the documents already in its checkpoint, so a crash costs
only the documents in flight. The plan is seeded, so a resumed document sees the demos it
would have seen in the first attempt.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml
from langchain_core.messages import AIMessage, HumanMessage

from eci import chains, graphs, parse, prompts as P, report as R, scope, syntax, tools as T, voting
from eci.config import load_config, n_folds, n_passes, prompt_spec
from eci.data import Doc, label_set, load_eval_docs, to_binary_undirected
from eci.fewshot import FewShotBank
from eci.labels import BINARY_LABELS, CAUSAL, NOREL, NOREL_VARIANTS
from eci.llm import (CallMeta, ChatClient, Ledger, SqliteLLMCache, TraceWriter, cost_usd,
                     make_chat_model)
from eci.tracking import RunDir, Tracker, git_state

ModelFactory = Callable[..., object]


def default_factory(cfg: dict) -> ModelFactory:
    """One run's models. They share one HTTP client that this run owns and closes (`aclose`), so
    no connection outlives the run's event loop (see make_chat_model)."""
    from openai import DefaultAsyncHttpxClient
    http = DefaultAsyncHttpxClient()

    def make(model_id: str, temperature: float, reasoning_effort, cache):
        return make_chat_model(model_id=model_id, temperature=temperature,
                               reasoning_effort=reasoning_effort, base_url=cfg["model"]["base_url"],
                               provider_order=cfg["model"]["provider_order"], cache=cache,
                               http_async_client=http)
    make.aclose = http.aclose
    return make


class Runner:
    def __init__(self, cfg: dict, run_dir: RunDir, *, factory: Optional[ModelFactory] = None,
                 retry_failed: bool = False):
        self.cfg = cfg
        self.run_dir = run_dir
        self.prompt = prompt_spec(cfg)
        self.ds = cfg["datasets"][cfg["active_dataset"]]
        self.factory = factory or default_factory(cfg)
        self.retry_failed = retry_failed
        self.ledger = Ledger()
        self.trace = TraceWriter(run_dir.file(".traces.jsonl"))
        self.tracker = Tracker(cfg, run_dir)
        self.bank: Optional[FewShotBank] = None
        self.llm_cache = (SqliteLLMCache(cfg["llm_cache"]["path"], cfg["llm_cache"]["namespace"])
                          if cfg["llm_cache"]["enabled"] else None)
        self.chains = (chains.ChainSource(cfg["coref_hint"]["source"], cfg["coref_hint"]["esl_root"])
                       if cfg["coref_hint"]["enabled"] else None)

    # ── clients ───────────────────────────────────────────────────────────────

    def _clients(self) -> None:
        cfg, exp, m = self.cfg, self.cfg["experiment"], self.cfg["model"]
        rt = cfg["runtime"]
        self.slots = asyncio.Semaphore(exp["concurrency"])
        common = dict(ledger=self.ledger, trace=self.trace, timeout=exp["timeout"],
                      max_tool_rounds=rt["max_tool_rounds"],
                      mlflow_traces=cfg["mlflow"]["enabled"] and cfg["mlflow"]["log_traces"])
        # Inference is never served from the LLM cache: reported metrics come from live calls.
        self.infer = ChatClient(self.factory(m["default_model_id"], m["temperature"], m["reasoning_effort"], False),
                                slots=self.slots, **common)
        self.cot = ChatClient(self.factory(m["default_model_id"], m["temperature"], m["reasoning_effort"],
                                           self.llm_cache), slots=self.slots, cache=self.llm_cache, **common)
        # Tool sub-calls run inside an orchestrator call that holds a slot: no shared budget.
        self.tool_llm = ChatClient(self.factory(m["default_model_id"], 0.0, None, self.llm_cache),
                                   slots=None, **common)
        self.extractor_llm = ChatClient(self.factory(cfg["tools_config"]["eci_extractor"]["model_id"], 0.0,
                                                     None, self.llm_cache), slots=None, **common)
        self.extractor_slots = asyncio.Semaphore(cfg["tools_config"]["eci_extractor"]["concurrency"])
        self.encoder_preds = T.encoder_loader(cfg["encoder"]["path"])

    def _tools(self, doc: Doc, user_prompt: str):
        exp = self.cfg["experiment"]
        if not (exp["enable_tools"] and exp["tools"]):
            return None
        return T.build(list(exp["tools"]), T.ToolContext(
            doc=doc, user_prompt=user_prompt, cfg=self.cfg, sub_llm=self.tool_llm,
            extractor_llm=self.extractor_llm, fewshot=self.bank, encoder_preds=self.encoder_preds,
            extractor_slots=self.extractor_slots))

    def decorate(self, docs: List[Doc]) -> None:
        """Blocks appended after the text wherever a document is shown."""
        syntax.annotate(docs, self.cfg["syntax"]["level"], self.cfg["syntax"]["discourse"] or [])
        if self.chains is not None:
            for d in docs:
                d.coref_block = chains.coref_block(d, self.chains.clusters(d))

    # ── one document ──────────────────────────────────────────────────────────

    def _user_prompt(self, doc: Doc, pair_ids, block, wl_block, gc_block) -> str:
        return P.test_prompt(self.prompt, doc, pair_ids,
                             [block["instruction"] if block else None, wl_block, gc_block],
                             binary_undirected=self.cfg["data"]["binary_undirected"],
                             constrain=self.ds["constrain_to_pair_list"])

    async def _pass(self, msgs, meta: CallMeta, tools):
        """One pass: (final text, triples, intermediate step answers). A multi-step prompt makes
        one call per step, the answer parsed at the last one."""
        exp = self.cfg["experiment"]
        step_answers = []
        for i, step in enumerate(self.prompt.steps or []):
            out, _ = await self.infer.call(msgs, CallMeta(**{**vars(meta), "block": f"{meta.block or ''}step{i}"}),
                                           retries=exp["retries"], retry_wait=exp["retry_wait"], tools=tools)
            step_answers.append(out[-1].content)
            msgs = msgs + [AIMessage(content=out[-1].content), HumanMessage(content=step["prompt"])]
        out, triples = await self.infer.call(msgs, meta, retries=exp["retries"], retry_wait=exp["retry_wait"],
                                             tools=tools, validate=lambda ai: parse.parse_triples(ai.content))
        return out[-1].content, triples, step_answers

    async def process(self, doc: Doc) -> dict:
        cfg, data = self.cfg, self.cfg["data"]
        split_on = cfg["mention_split"]["enabled"]
        blocks = scope.split_blocks(doc) if split_on else [None]
        wl = cfg["mention_whitelist"]
        keep = scope.whitelist_ids(doc, wl["mode"], wl["path"]) if wl["enabled"] else None
        wl_block = scope.whitelist_instruction(keep, doc) if keep is not None else None
        gc_block = scope.count_instruction(scope.gold_count(doc)) if cfg["gold_count"]["enabled"] else None

        block_prompts = [self._user_prompt(doc, doc.pair_list, b, wl_block, gc_block) for b in blocks]
        user_prompt = block_prompts[0] if not split_on else "\n\n".join(
            f"===== block: {b['kind']} =====\n{p}" for b, p in zip(blocks, block_prompts))
        demo_sets = (self.bank.demo_sets(doc) if cfg["few_shot"]["enabled"] and cfg["few_shot"]["systematic"]
                     else None)
        reprompt = user_prompt if (cfg["reprompt"]["systematic"] and not self.prompt.steps) else ""
        tools = self._tools(doc, user_prompt)
        n = n_passes(cfg)
        gold = doc.gold
        if data["vote_to_binary"]:
            gold = [t for t in to_binary_undirected(doc.gold) if t[1] != NOREL]

        def prompt_for(run_idx: int, block_idx: int) -> str:
            # Reshuffling only changes anything when the prompt lists the pairs.
            if not (data["reshuffle_per_resample"] and doc.pair_list and self.ds["constrain_to_pair_list"]):
                return block_prompts[block_idx]
            ids = list(doc.pair_list)
            random.Random(f"{doc.id}::{run_idx}").shuffle(ids)
            return self._user_prompt(doc, ids, blocks[block_idx], wl_block, gc_block)

        tags = {"gold_triples": json.dumps(doc.gold)}
        calls = [self._pass(P.messages(self.prompt.system,
                                       demo_sets[r % len(demo_sets)] if demo_sets else None,
                                       prompt_for(r, b), reprompt),
                            CallMeta(role="inference", doc_id=doc.id, pass_idx=r,
                                     block=blocks[b]["kind"] if split_on else None, tags=tags), tools)
                 for r in range(n) for b in range(len(blocks))]
        base = {"id": doc.id, "doc_idx": doc.idx, "lang": doc.lang, "gold_triples": gold,
                "mention_sentence": doc.sentence_of, "first_token": doc.first_token,
                "surfaces": doc.surfaces, "whitelist": None, "leakage": None}
        try:
            outs = await asyncio.gather(*calls)
        except Exception as e:
            print(f"[run] document {doc.id} failed: {type(e).__name__}: {e}")
            return {**base, "pred_triples": [], "pair_stats": {}, "justifications": {},
                    "failure": f"{type(e).__name__}: {e}", "offspec_labels": 0,
                    "context": {"system_prompt": self.prompt.system, "user_prompt": user_prompt}}

        # Re-unite each pass's block calls into one document-level prediction.
        per_pass, leakage = [], None
        if split_on:
            leakage = {k: {"kept": 0, "dropped": 0} for k in scope.BLOCK_KINDS}
            for r in range(n):
                combined = []
                for b, block in enumerate(blocks):
                    tr = outs[r * len(blocks) + b][1]
                    if cfg["mention_split"]["filter_out_of_block"]:
                        tr, dropped = scope.filter_block(tr, block)
                        leakage[block["kind"]]["kept"] += len(tr)
                        leakage[block["kind"]]["dropped"] += dropped
                    combined += tr
                per_pass.append(combined)
        else:
            per_pass = [o[1] for o in outs]

        whitelist = None
        if keep is not None:
            emitted = sum(len(t) for t in per_pass)
            filtered = [scope.enforce_whitelist(t, keep) for t in per_pass]
            whitelist = {"emitted": emitted, "out_of_scope": sum(d for _, d in filtered)}
            if wl["enforce"]:
                per_pass = [t for t, _ in filtered]

        valid = CAUSAL | NOREL_VARIANTS
        offspec = sum(1 for tr in per_pass for _, l, _ in tr if str(l).lower() not in valid)
        rng = random.Random(f"{cfg['seed']}::{doc.id}::vote")
        if len(per_pass) > 1:
            final, stats = voting.aggregate(per_pass, cfg["experiment"]["resampling"]["tie_breaking"], rng)
        else:
            final, stats = per_pass[0], voting.single_pass_stats(per_pass[0])
        if data["binary_undirected"]:
            final = [(min(s, t), l, max(s, t)) for s, l, t in final]
        if data["vote_to_binary"]:
            final = voting.votes_to_binary(stats, data["binary_default"], data["novote_norel"])

        justifications: Dict[str, List[str]] = {}
        for text, _, _ in outs:
            for key, why in parse.parse_justifications(text).items():
                justifications.setdefault(key, []).append(why)

        return {**base, "pred_triples": final, "pair_stats": stats, "justifications": justifications,
                "failure": None, "whitelist": whitelist, "leakage": leakage, "offspec_labels": offspec,
                "context": {"system_prompt": self.prompt.system, "user_prompt": user_prompt,
                            "step_responses": [s for _, _, s in outs] if self.prompt.steps else []}}

    # ── the run ───────────────────────────────────────────────────────────────

    async def run(self) -> dict:
        cfg = self.cfg
        os.environ["LANGCHAIN_TRACING_V2"] = cfg["experiment"]["tracing"]
        os.environ["LANGCHAIN_PROJECT"] = cfg["experiment"]["tracing_name"]
        git = git_state()
        if not self.run_dir.file(".config.yaml").exists():
            self.run_dir.write_config(cfg)
        self.run_dir.write_git(git)
        print(f"Run {self.run_dir.stem}  ({self.run_dir.path})")
        print(f"dataset={self.ds['name']} split={self.ds['split']} prompt={self.prompt.name} "
              f"model={cfg['model']['default_model_id']} passes={n_passes(cfg)}")
        self.tracker.start(git)
        t0 = time.monotonic()
        try:
            report = await self._run()
        except BaseException:
            self.tracker.abort()
            print(f"\nRun interrupted. Finished documents are in {self.run_dir.checkpoint}.\n"
                  f"Resume with:  python run.py --resume {self.run_dir.path}", file=sys.stderr)
            raise
        finally:
            close = getattr(self.factory, "aclose", None)   # test factories own no connections
            if close is not None:
                await close()
        print(f"Done in {time.monotonic() - t0:.0f}s")
        return report

    async def _run(self) -> dict:
        cfg = self.cfg
        exp = cfg["experiment"]
        self._clients()
        docs = await asyncio.to_thread(load_eval_docs, cfg)
        folds = n_folds(cfg)
        for d in docs:
            d.fold = d.idx % folds if folds > 1 else -1
        await asyncio.to_thread(self.decorate, docs)
        print(f"{len(docs)} documents" + (f" (shuffled, seed {cfg['seed']})" if cfg["data"]["shuffle_docs"] else ""))

        if cfg["few_shot"]["enabled"]:
            self.bank = FewShotBank(cfg, self.prompt)
            await asyncio.to_thread(self.bank.load, self.decorate)
            self.bank.build_plan(docs)
            dump = cfg["few_shot"]["cot_generation"]["dump_dir"]
            await self.bank.pregenerate(
                self.cot, lambda demo: self._tools(demo, ""), exp["retries"], exp["retry_wait"],
                Path(dump) / self.run_dir.stem if dump else None)

        done = self.run_dir.load_docs()
        if self.retry_failed:
            done = {k: v for k, v in done.items() if not v["failure"]}
        todo = [d for d in docs if d.id not in done]
        if done:
            print(f"resuming: {len(done)} documents already finished, {len(todo)} to go")
        print(f"concurrency: {exp['concurrency']} LLM calls in flight, {exp['doc_concurrency']} documents open")

        doc_slots = asyncio.Semaphore(exp["doc_concurrency"])
        graph_dir = self.run_dir.path / "graphs"
        count = len(done)

        async def one(doc: Doc):
            nonlocal count
            async with doc_slots:
                res = await self.process(doc)
            self.run_dir.append_doc(res)
            done[doc.id] = res
            count += 1
            fold = f"[fold {doc.fold + 1}/{folds}] " if folds > 1 else ""
            print(f"{fold}[{count}/{len(docs)}] {doc.id}" + (" FAILED" if res["failure"] else ""), flush=True)
            self.tracker.progress(count)
            if cfg["mlflow"]["log_causal_graphs"] and not res["failure"]:
                svg = await asyncio.to_thread(graphs.render_graph, graph_dir, doc.id, doc.surfaces,
                                              res["gold_triples"], res["pred_triples"], doc.text)
                if svg:
                    self.tracker.graph(svg)

        await asyncio.gather(*[one(d) for d in todo])
        return self._finish([done[d.id] for d in docs if d.id in done], len(docs))

    def _finish(self, results: List[dict], total: int) -> dict:
        cfg = self.cfg
        labels = BINARY_LABELS if (cfg["data"]["binary_undirected"] or cfg["data"]["vote_to_binary"]) else label_set(cfg)
        report = R.build(results, total, cfg, labels)
        if cfg["datasets"][cfg["active_dataset"]]["kfold"]["enabled"]:
            report["kfold_n_folds"] = n_folds(cfg)
        if cfg["mention_whitelist"]["enabled"]:
            wls = [r["whitelist"] for r in results if r["whitelist"]]
            emitted = sum(w["emitted"] for w in wls)
            oos = sum(w["out_of_scope"] for w in wls)
            report["mention_whitelist"] = {"documents": len(wls), "emitted": emitted, "out_of_scope": oos,
                                           "out_of_scope_rate": oos / emitted if emitted else 0.0}
            print(f"mention whitelist: {oos}/{emitted} emitted pairs out of scope "
                  f"({report['mention_whitelist']['out_of_scope_rate']:.1%})")
        if cfg["mention_split"]["enabled"]:
            leak = {}
            for kind in scope.BLOCK_KINDS:
                kept = sum(r["leakage"][kind]["kept"] for r in results if r["leakage"])
                dropped = sum(r["leakage"][kind]["dropped"] for r in results if r["leakage"])
                leak[kind] = {"emitted": kept + dropped, "in_block": kept, "out_of_block": dropped,
                              "out_of_block_rate": dropped / (kept + dropped) if kept + dropped else 0.0}
            e = sum(v["emitted"] for v in leak.values())
            o = sum(v["out_of_block"] for v in leak.values())
            leak["overall"] = {"emitted": e, "in_block": e - o, "out_of_block": o,
                               "out_of_block_rate": o / e if e else 0.0,
                               "documents": sum(1 for r in results if r["leakage"])}
            report["mention_split_leakage"] = leak
        report["offspec_labels"] = sum(r["offspec_labels"] for r in results)

        usage = self.ledger.as_dict()
        if self.llm_cache is not None:
            usage["llm_cache"] = {"hits": self.llm_cache.hits, "misses": self.llm_cache.misses}
        cost = cost_usd(self.ledger, cfg["model"]["default_model_id"], cfg["model"]["base_url"])
        report["usage"] = {**usage, "cost_usd": cost}

        path = self.run_dir.write_report(cfg, report, git_state(), "success", None)
        figures = graphs.histograms(report, cfg["active_dataset"], self.run_dir.path / "figures",
                                    f"{self.run_dir.stem}.")
        self.tracker.finish(report, usage, cost,
                            [path, self.run_dir.file(".traces.jsonl"), self.run_dir.file(".config.yaml")] + figures,
                            "FINISHED")
        t = usage["total"]
        print(R.headline(report))
        print(f"LLM calls {t['calls']} (+{t['failed_attempts']} failed attempts)  "
              f"tokens in {t['input_tokens']:,} out {t['output_tokens']:,}  "
              f"cost {'$%.4f' % cost if cost is not None else 'n/a'}"
              + (f"  | providers {', '.join(f'{p} {n}' for p, n in usage['providers'].items())}"
                 if usage["providers"] else "")
              + (f"  | llm cache {self.llm_cache.stats()}" if self.llm_cache else ""))
        print(f"report: {path}")
        return report


# ── Entry points ──────────────────────────────────────────────────────────────

def run_config(cfg: dict, factory: Optional[ModelFactory] = None) -> dict:
    run_dir = RunDir.create(cfg["runtime"]["logdir"])
    return asyncio.run(Runner(cfg, run_dir, factory=factory).run())


def resume(path: str, retry_failed: bool, factory: Optional[ModelFactory] = None) -> dict:
    run_dir = RunDir(Path(path))
    cfg = run_dir.read_config()
    return asyncio.run(Runner(cfg, run_dir, factory=factory, retry_failed=retry_failed).run())


def run_queue(path: str, only: Optional[List[str]] = None, config_path=None,
              factory: Optional[ModelFactory] = None) -> List[str]:
    """Named experiments run one after another; a failure is reported and the queue goes on.
    Returns the names that failed. Entries: {name, overrides}; any other top-level key of the
    file (e.g. a `_defaults` anchor block) is ignored."""
    with open(path, encoding="utf-8") as f:
        entries = yaml.safe_load(f)["experiments"]
    if only:
        missing = set(only) - {e["name"] for e in entries}
        if missing:
            raise SystemExit(f"not in the queue: {sorted(missing)}")
        entries = [e for e in entries if e["name"] in set(only)]
    failed = []
    t_all = time.time()
    for i, entry in enumerate(entries, 1):
        print(f"{'=' * 60}\n[{i}/{len(entries)}] {entry['name']}\n"
              f"{yaml.dump(entry.get('overrides') or {}, default_flow_style=False).rstrip()}\n{'=' * 60}")
        t0 = time.time()
        try:
            cfg = load_config(config_path, overrides=entry.get("overrides") or {}) if config_path \
                else load_config(overrides=entry.get("overrides") or {})
            run_config(cfg, factory)
            print(f"[{entry['name']}] done in {time.time() - t0:.0f}s\n")
        except Exception as e:
            print(f"[{entry['name']}] FAILED after {time.time() - t0:.0f}s: {e}", file=sys.stderr)
            traceback.print_exc()
            failed.append(entry["name"])
    print(f"queue finished in {time.time() - t_all:.0f}s — {len(entries) - len(failed)}/{len(entries)} succeeded")
    if failed:
        print(f"failed: {failed}", file=sys.stderr)
    return failed
