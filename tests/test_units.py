"""Unit tests: config strictness, parsing, voting, scope, coherence, rationales, the LLM cache,
connections across the runs of a queue.
Run: python -m unittest tests.test_units -v"""
from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from eci import parse, voting
from eci.config import DEFAULT_CONFIG, ConfigError, deep_merge, flatten_params, load_config, overrides_from_agentere
from eci.data import Doc, doc_from_row, to_binary_undirected
from eci.fewshot import parse_rationale
from eci.labels import normalize
from eci.scope import count_instruction, gold_count, split_blocks, whitelist_ids, whitelist_instruction
from eci.tools import check_coherence


class Config(unittest.TestCase):
    def test_unknown_override_key_fails(self):
        with self.assertRaises(ConfigError):
            deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})

    def test_enum_typos_fail_before_anything_runs(self):
        for ov in ({"syntax": {"level": "mention"}}, {"few_shot": {"selection": "bm25"}},
                   {"data": {"binary_undirected": True, "vote_to_binary": True}}):
            with self.assertRaises(ConfigError):
                load_config(overrides=ov)

    def test_params_skip_execution_sections(self):
        params = flatten_params(load_config())
        self.assertIn("dataset.prompt", params)
        self.assertFalse(any(k.startswith(("runtime.", "mlflow.", "rule_sets")) for k in params))

    def test_agentere_snapshot_import(self):
        base = yaml.safe_load(DEFAULT_CONFIG.read_text())
        snap = {"seed": 7, "prompt": {"system": "x"}, "coref_injection": {"enabled": True},
                "experiment": {"retries": 5, "unknown_key": 1}, "mlflow": {"experiment_name": "agentere"}}
        ov, dropped = overrides_from_agentere(snap, base)
        self.assertEqual(ov, {"seed": 7, "experiment": {"retries": 5}})
        self.assertEqual(sorted(dropped), ["coref_injection", "experiment.unknown_key"])


class Parsing(unittest.TestCase):
    def test_last_populated_array_wins_over_stray_empty(self):
        text = 'I considered [] first.\n[{"pair": "1,2", "label": "causes"}]\nThen [] again.'
        self.assertEqual(parse.parse_triples(text), [("1", "causes", "2")])

    def test_unpaired_quote_in_reasoning_does_not_hide_answer(self):
        text = 'He said "nothing will happen.\n[{"pair": "e1,e2", "label": "causedby"}]'
        self.assertEqual(parse.parse_triples(text), [("e1", "causedby", "e2")])

    def test_empty_answer_and_missing_answer(self):
        self.assertEqual(parse.parse_triples("No relations. []"), [])
        with self.assertRaises(parse.ParseError):
            parse.parse_triples("no json here")

    def test_labels_are_normalised(self):
        self.assertEqual(normalize(" Caused by "), "causedby")
        self.assertEqual(parse.parse_triples('[{"pair":"1,2","label":"Causes"}]'), [("1", "causes", "2")])
        self.assertEqual(parse.parse_triples('[{"pair":"1,2","label":"Causes"}]', normalize_labels=False),
                         [("1", "Causes", "2")])

    def test_justifications(self):
        text = '[{"pair":"1,2","label":"causes","why":"because"}]'
        self.assertEqual(parse.parse_justifications(text), {"1,2": "because"})


class Voting(unittest.TestCase):
    def test_plurality_with_implicit_norel_and_one_vote_per_pass(self):
        passes = [[("a", "causes", "b"), ("a", "causes", "b"), ("a", "causes", "b")],   # looping pass
                  [("a", "causes", "b")], []]
        final, stats = voting.aggregate(passes, "norel", random.Random(0))
        self.assertEqual(final, [("a", "causes", "b")])
        self.assertEqual(stats["a,b"]["vote_counts"], {"causes": 2, "norel": 1})

    def test_three_way_tie_goes_to_norel(self):
        passes = [[("a", "causes", "b")], [("a", "causedby", "b")], []]
        final, _ = voting.aggregate(passes, "norel", random.Random(0))
        self.assertEqual(final, [])

    def test_votes_to_binary_is_not_an_or(self):
        stats = {"a,b": {"vote_counts": {"causes": 1, "norel": 2}},
                 "b,a": {"vote_counts": {"causedby": 1, "norel": 2}}}
        self.assertEqual(voting.votes_to_binary(stats, "norel", True), [])      # 2 causal vs 4 norel
        lone = {"a,b": {"vote_counts": {"causes": 1}}}
        self.assertEqual(voting.votes_to_binary(lone, "norel", True), [])       # 1 vs implicit 1
        self.assertEqual(voting.votes_to_binary(lone, "norel", False), [("a", "causal", "b")])
        self.assertEqual(voting.votes_to_binary({"a,a": {"vote_counts": {"causes": 3}}}, "norel", True), [])


def _doc(**kw) -> Doc:
    base = dict(id="d", idx=0, text="", gold=[], lang="eng", mentions=["e1", "e2", "e3"],
                surfaces={"e1": "fire", "e2": "spread", "e3": "evacuated"},
                sentence_of={"e1": 0, "e2": 0, "e3": 1}, first_token={}, pair_list=[],
                tokens=[], spans=[], sentences=[])
    base.update(kw)
    return Doc(**base)


class Scope(unittest.TestCase):
    def test_split_partitions_the_pair_space(self):
        from eci.scope import in_block
        doc = _doc()
        blocks = split_blocks(doc)
        for a in doc.mentions:
            for b in doc.mentions:
                if a != b:
                    self.assertEqual(sum(in_block(a, b, bl) for bl in blocks), 1)

    def test_oracle_whitelist_and_count(self):
        doc = _doc(gold=[("e1", "causes", "e2"), ("e2", "causedby", "e1")])
        self.assertEqual(whitelist_ids(doc, "oracle", ""), ["e1", "e2"])
        self.assertIn("<e1 fire>, <e2 spread>", whitelist_instruction(["e1", "e2"], doc))
        self.assertIsNone(whitelist_instruction(["e1", "e2", "e3"], doc))   # restricts nothing
        self.assertEqual(gold_count(doc), 1)
        self.assertIn("exactly 2 entries", count_instruction(1))


class Coherence(unittest.TestCase):
    rules = yaml.safe_load(DEFAULT_CONFIG.read_text())["rule_sets"]["esl"]

    def test_rules_fire_in_the_label_space_models_emit(self):
        out = check_coherence([{"pair": "1,2", "label": "causes"}], self.rules)
        self.assertIn("1 incoherence violations", out)   # missing "2,1 causedby"
        ok = check_coherence([{"pair": "1,2", "label": "causes"}, {"pair": "2,1", "label": "causedby"}], self.rules)
        self.assertIn("No incoherence found", ok)

    def test_transitivity(self):
        pairs = [{"pair": "1,2", "label": "causes"}, {"pair": "2,1", "label": "causedby"},
                 {"pair": "2,3", "label": "causes"}, {"pair": "3,2", "label": "causedby"}]
        self.assertIn("'1'->'3' should be 'causes'", check_coherence(pairs, self.rules))


class Data(unittest.TestCase):
    def test_row_parsing(self):
        row = {"id": "x", "text": "The <e0 fire> <e1 spread> .", "annots": "<e0 fire> causes <e1 spread>",
               "tokens": ["The", "fire", "spread", "."], "spans": [[1], [2]], "mentions": ["e0", "e1"],
               "sentences": [[0, 4]]}
        d = doc_from_row(row, 3, text_field="text", ann_field="annots", valid_labels=None,
                         binary_undirected=False, shuffle_pair_list=False)
        self.assertEqual(d.gold, [("e0", "causes", "e1")])
        self.assertEqual(d.surfaces, {"e0": "fire", "e1": "spread"})
        self.assertEqual(d.pair_list, [("e0", "e1"), ("e1", "e0")])
        self.assertEqual(to_binary_undirected([("b", "causes", "a"), ("a", "causedby", "b")]),
                         [("a", "causal", "b")])


class Rationales(unittest.TestCase):
    def test_header_answer_and_steps(self):
        tmp = Path(tempfile.mkdtemp()) / "x.txt"
        tmp.write_text("Text: 1_15ecbplus.xml\nthe document...\nAnswer:\nstep one\n--- Step 2 ---\nstep two\n")
        self.assertEqual(parse_rationale(tmp), ("1_15ecbplus.xml", ["step one", "step two"]))
        tmp.write_text("Text: doc\nno answer line")
        with self.assertRaises(ValueError):
            parse_rationale(tmp)


class LLMCache(unittest.TestCase):
    def test_replay_carries_no_usage(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration
        from eci.llm import SqliteLLMCache
        cache = SqliteLLMCache(str(Path(tempfile.mkdtemp()) / "c.sqlite"))
        live = AIMessage(content="hi", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
                         response_metadata={"token_usage": {"cost": 0.1}})
        cache.update("p", "llm", [ChatGeneration(message=live)])
        self.assertEqual(live.usage_metadata["input_tokens"], 5)       # the live object is untouched
        (hit,) = cache.lookup("p", "llm")
        self.assertIsNone(hit.message.usage_metadata)
        self.assertTrue(hit.message.response_metadata["cache_hit"])
        self.assertIsNone(cache.lookup("p", "other-model"))
        self.assertEqual((cache.hits, cache.misses), (1, 1))


class _KeepAliveChat(BaseHTTPRequestHandler):
    """A minimal OpenAI-compatible endpoint that keeps connections open, as OpenRouter does."""
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                           "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": "ok"}}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class QueueConnections(unittest.TestCase):
    def test_consecutive_runs_do_not_reuse_a_closed_loop(self):
        """Each queue entry is its own asyncio.run. langchain-openai's shared client kept the first
        run's keep-alive connections, and the second run failed with "Event loop is closed"."""
        from eci.runner import default_factory
        server = ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveChat)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        cfg = {"model": {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "provider_order": []}}
        os.environ.setdefault("OPENROUTER_API_KEY", "unused-by-the-local-server")

        async def one_run():
            factory = default_factory(cfg)
            model = factory("m", 0.0, None, None)
            try:
                return await asyncio.gather(*[model.ainvoke("hi") for _ in range(5)], return_exceptions=True)
            finally:
                close = getattr(factory, "aclose", None)   # as Runner.run does
                if close is not None:
                    await close()

        for run in range(3):
            outcomes = asyncio.run(one_run())
            self.assertEqual([type(o).__name__ for o in outcomes], ["AIMessage"] * 5, f"run {run + 1}: {outcomes}")


if __name__ == "__main__":
    unittest.main()
