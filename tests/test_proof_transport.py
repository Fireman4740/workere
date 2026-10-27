"""Offline technical tests. No benchmark labels enter any model request."""

import itertools
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from proof_transport import client
from proof_transport.core import (
    VARIANTS,
    build_registry,
    decide,
    select_additions,
    validate_registry,
)
from proof_transport.data import (
    document_from_raw,
    load_historical_groups,
    public_bundle,
    select_roots,
)
from proof_transport.evaluation import blind_audit, evaluate
from proof_transport.fixtures import (
    a2_response,
    citation,
    frame,
    raw_fixture,
    receipt,
    smoke,
    source_certificate,
)
from proof_transport.pipeline import (
    DEFAULT_CONFIG,
    collect_items,
    extraction_requests,
    load_plan,
    prepare,
    read_json,
    read_jsonl,
    validate_config,
    write_once,
)
from proof_transport.schema import (
    Citation,
    Invalid,
    compare_fact,
    digest,
    pair,
    validate_all_citations,
    validate_certificate,
    validate_root,
)


class Case(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.suite, self.controls = raw_fixture()
        self.bundle = public_bundle(self.suite, self.controls)
        self.doc = self.bundle["documents"]["SYNTHETIC_SUCCESS"]
        self.roots = select_roots(self.bundle)
        self.root = self.roots[0]
        self.cert = source_certificate(self.doc, self.root)
        self.extracted = {
            self.root["id"]: {
                "source_id": self.root["id"],
                "verdict": "SUPPORTED",
                "reason": "fixture",
                "certificate": self.cert,
            }
        }
        self.verified = {
            self.root["id"]: {
                "source_id": self.root["id"],
                "source_certificate_hash": digest(self.cert),
                "verdict": "SUPPORTED",
                "reason": "fixture",
                "alternative_ids": ["ap"],
                "justification": "Fixture source verified.",
                "citations": self.cert["evidence"],
            }
        }
        self.registry = build_registry(
            self.bundle, self.roots, self.extracted, self.verified
        )
        self.proposal = self.registry["proposals"][0]
        self.groups = {self.doc["id"]: {"status": "AVAILABLE", "groups": [["a", "ap"]]}}

    def verdict(self, variant, raw):
        return decide(
            variant, self.proposal, self.registry, self.bundle, raw, self.groups
        )

    def good(self, variant):
        if variant == "A0":
            return {
                "candidate_id": self.proposal["id"],
                "identity": "SAME",
                "justification": "Fixture identity.",
            }
        if variant == "A1":
            return {
                "candidate_id": self.proposal["id"],
                "alternative_frame": frame(self.doc),
                "discriminating_fields": ["time", "place"],
                "justification": "Fixture compatibility.",
            }
        return a2_response(self.doc, self.cert, self.proposal["id"])


class ExactCitations(Case):
    def test_exact_quote(self):
        Citation.parse(citation(self.doc, "blast"), self.doc)

    def test_unicode_codepoints_not_bytes(self):
        d = {"text": "Été — 🔥 feu"}
        Citation.parse({"start": 6, "end": 7, "quote": "🔥"}, d)
        with self.assertRaises(Invalid):
            Citation.parse({"start": 10, "end": 14, "quote": "🔥"}, d)

    def test_wrong_quote(self):
        with self.assertRaises(Invalid):
            Citation.parse({"start": 0, "end": 2, "quote": "xx"}, self.doc)

    def test_bad_boundaries(self):
        for start, end in [(-1, 1), (0, 100000), (3, 3), (4, 1), (True, 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(Invalid):
                Citation.parse({"start": start, "end": end, "quote": "On"}, self.doc)

    def test_shifted_offset(self):
        s = citation(self.doc, "blast")
        s["start"] += 1
        with self.assertRaises(Invalid):
            Citation.parse(s, self.doc)

    def test_unknown_citation_fields(self):
        with self.assertRaises(Invalid):
            Citation.parse({**citation(self.doc, "blast"), "gold": True}, self.doc)

    def test_nested_citations_checked(self):
        v = {
            "nested": [
                {"deeper": citation(self.doc, "blast")},
                {"deeper": {"start": 0, "end": 2, "quote": "bad"}},
            ]
        }
        with self.assertRaises(Invalid):
            validate_all_citations(v, self.doc)

    def test_citation_without_quote_invalid(self):
        with self.assertRaises(Invalid):
            validate_all_citations({"start": 0, "end": 2}, self.doc)


class DataBoundary(Case):
    def test_fusion_all_four_terms(self):
        rows = deepcopy(self.controls)
        for r in rows:
            if [r["a"], r["b"]] == ["a", "b"]:
                r.update(luna=0.8, d1=0.6)
            if [r["a"], r["b"]] == ["b", "a"]:
                r.update(luna=0.4, d1=0.2)
        b = public_bundle(self.suite, rows)
        r = next(x for x in b["scores"][self.doc["id"]] if x["pair"] == ["a", "b"])
        self.assertAlmostEqual(r["fusion"], 0.5)

    def test_no_reverse_fill(self):
        with self.assertRaises(Invalid):
            public_bundle(self.suite, self.controls[:-1])

    def test_duplicate_ordered_score(self):
        with self.assertRaises(Invalid):
            public_bundle(self.suite, self.controls + [self.controls[0]])

    def test_nonfinite_and_boolean_scores(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.9"):
            rows = deepcopy(self.controls)
            rows[0]["d1"] = value
            with self.subTest(value=value), self.assertRaises(Invalid):
                public_bundle(self.suite, rows)

    def test_poison_gold_cannot_change_public_or_selection(self):
        s, r = deepcopy(self.suite), deepcopy(self.controls)
        s["documents"][0].update(
            gold="POISON GOLD",
            relations={"causes": [[0, 2]]},
            annots="POISON",
            coref_clusters=[[0, 1, 2]],
            text="WRONG BALISES",
        )
        for x in r:
            x["gold"] = "POISON GOLD"
        safe = public_bundle(s, r)
        self.assertEqual(self.bundle, safe)
        self.assertEqual(self.roots, select_roots(safe))
        self.assertNotIn("POISON", json.dumps(safe))

    def test_numeric_id_coercion_forbidden(self):
        s = deepcopy(self.suite)
        s["documents"][0]["mentions"][0] = 1
        with self.assertRaises(Invalid):
            public_bundle(s, self.controls)

    def test_surface_mismatch(self):
        s = deepcopy(self.suite["documents"][0])
        s["surfaces"] = {"a": "BLAST", "b": "evacuation.", "ap": "explosion"}
        with self.assertRaises(Invalid):
            document_from_raw(s)

    def test_negative_duplicate_and_out_of_bounds_spans(self):
        for span in ([-1], [999], [1, 1], [3, 2]):
            s = deepcopy(self.suite["documents"][0])
            s["spans"][0] = span
            with self.subTest(span=span), self.assertRaises(Invalid):
                document_from_raw(s)

    def test_discontinuous_span_is_not_forged_contiguous_quote(self):
        s = deepcopy(self.suite["documents"][0])
        s["spans"][0] = [0, 2]
        d = document_from_raw(s)
        self.assertEqual(d["mentions"]["a"]["surface"], "On the")
        self.assertEqual(len(d["mentions"]["a"]["segments"]), 2)

    def test_sources_deterministic_without_labels(self):
        self.assertEqual(select_roots(self.bundle), select_roots(self.bundle))
        self.assertTrue(
            all(r["score"] >= 0.39 and r["generation"] == 0 for r in self.roots)
        )

    def test_unknowns_never_agree(self):
        f = {"state": "UNKNOWN", "value": None, "citations": []}
        self.assertEqual(compare_fact(f, f), "UNKNOWN")

    def test_participant_roles_are_structured_not_bag_of_names(self):
        from proof_transport.schema import validate_frame

        f = frame(self.doc)
        f["participants"] = {
            "state": "OBSERVED",
            "value": {"agent": ["depot"], "patient": ["people"]},
            "citations": [citation(self.doc, "North depot")],
        }
        validate_frame(f, self.doc)
        g = deepcopy(f)
        g["participants"]["value"] = {"agent": ["people"], "patient": ["depot"]}
        self.assertEqual(compare_fact(f["participants"], g["participants"]), "CONFLICT")
        f["participants"]["value"] = "depot people"
        with self.assertRaises(Invalid):
            validate_frame(f, self.doc)

    def test_native_t2_preserves_but_does_not_copy_in_group_scores(self):
        from proof_transport.historical import native_t2

        groups = {self.doc["id"]: {"status": "AVAILABLE", "groups": [["a", "ap", "b"]]}}
        out = native_t2(self.bundle, groups)
        self.assertEqual(
            out["documents"][self.doc["id"]]["positive_pairs"], [["a", "b"]]
        )
        groups[self.doc["id"]]["groups"] = [["a", "ap"]]
        out = native_t2(self.bundle, groups)
        self.assertEqual(
            {tuple(p) for p in out["documents"][self.doc["id"]]["positive_pairs"]},
            {("a", "b"), ("ap", "b")},
        )

    def test_missing_groups_not_empty_groups(self):
        g = load_historical_groups(self.path, self.bundle["documents"])
        self.assertEqual(g[self.doc["id"]]["status"], "UNAVAILABLE")
        (self.path / "predictions.jsonl").write_text(
            json.dumps({"id": self.doc["id"], "metadata": {"coref_groups": []}}) + "\n"
        )
        g = load_historical_groups(self.path, self.bundle["documents"])
        self.assertEqual(g[self.doc["id"]]["status"], "AVAILABLE")
        self.assertEqual(g[self.doc["id"]]["groups"], [])

    def test_invalid_cached_group_ids_and_overlap(self):
        for groups in ([["unknown"]], [["a", "ap"], ["ap", "b"]], [[1, 2]]):
            (self.path / "predictions.jsonl").write_text(
                json.dumps({"id": self.doc["id"], "metadata": {"coref_groups": groups}})
                + "\n"
            )
            self.assertEqual(
                load_historical_groups(self.path, self.bundle["documents"])[
                    self.doc["id"]
                ]["status"],
                "UNAVAILABLE",
            )


class SourceAndRegistry(Case):
    def test_valid_source(self):
        validate_certificate(self.cert, self.root, self.doc)
        validate_registry(self.bundle, self.roots, self.registry)

    def test_orientation_cannot_come_from_sorted_pair(self):
        c = deepcopy(self.cert)
        c["cause"] = c["effect"]
        with self.assertRaises(Invalid):
            validate_certificate(c, self.root, self.doc)

    def test_source_must_be_initial_positive_generation_zero(self):
        for value in (1, False):
            r = deepcopy(self.root)
            r["generation"] = value
            with self.assertRaises(Invalid):
                validate_root(r, self.bundle)
        r = deepcopy(self.root)
        r["score"] = 0.38
        with self.assertRaises(Invalid):
            validate_root(r, self.bundle)

    def test_source_quote_missing_mention(self):
        c = deepcopy(self.cert)
        c["evidence"] = [citation(self.doc, "blast")]
        with self.assertRaises(Invalid):
            validate_certificate(c, self.root, self.doc)

    def test_missing_extraction_not_silent_negative(self):
        with self.assertRaises(Invalid):
            build_registry(self.bundle, self.roots, {}, self.verified)

    def test_source_false_or_ambiguous_abstains_without_deleting_baseline(self):
        e = deepcopy(self.extracted)
        e[self.root["id"]]["verdict"] = "ABSTAIN"
        reg = build_registry(self.bundle, self.roots, e, {})
        self.assertEqual(reg["proposals"], [])
        row = next(
            r for r in self.bundle["scores"][self.doc["id"]] if r["pair"] == ["a", "b"]
        )
        self.assertEqual(row["fusion"], 0.9)

    def test_wrong_source_hash_does_not_verify(self):
        v = deepcopy(self.verified)
        v[self.root["id"]]["source_certificate_hash"] = "wrong"
        reg = build_registry(self.bundle, self.roots, self.extracted, v)
        self.assertEqual(reg["proposals"], [])

    def test_proposal_limit_ids_and_duplicates(self):
        for alternatives in (["ap", "a", "b"], ["ap", "ap"], ["missing"], ["b"]):
            v = deepcopy(self.verified)
            v[self.root["id"]]["alternative_ids"] = alternatives
            reg = build_registry(self.bundle, self.roots, self.extracted, v)
            self.assertEqual(reg["proposals"], [])
            self.assertEqual(reg["sources"][self.root["id"]]["status"], "ABSTAIN")

    def test_registry_tamper_detected(self):
        r = deepcopy(self.registry)
        r["proposals"][0]["alternative"] = "a"
        with self.assertRaises(Invalid):
            validate_registry(self.bundle, self.roots, r)

    def test_evidence_tamper_detected(self):
        r = deepcopy(self.registry)
        r["sources"][self.root["id"]]["certificate"]["evidence"][0]["quote"] = "bad"
        with self.assertRaises(Invalid):
            validate_registry(self.bundle, self.roots, r)


class VariantGates(Case):
    def test_all_good_variants_accept(self):
        for v in VARIANTS:
            with self.subTest(v=v):
                self.assertEqual(
                    self.verdict(v, None if v == "A3" else self.good(v))["verdict"],
                    "ACCEPT",
                )

    def test_missing_response_is_pending_not_negative(self):
        self.assertEqual(self.verdict("A0", None)["verdict"], "PENDING")

    def test_a0_unknown_abstains(self):
        r = self.good("A0")
        r["identity"] = "UNKNOWN"
        self.assertEqual(self.verdict("A0", r)["verdict"], "ABSTAIN")

    def test_a0_no_certificate_and_wrong_ids_rejected(self):
        for mutation in ({"candidate_id": "wrong"}, {"certificate": {}}):
            r = self.good("A0")
            r.update(mutation)
            self.assertEqual(self.verdict("A0", r)["verdict"], "ABSTAIN")

    def test_a1_two_unknown_fields_do_not_count(self):
        r = self.good("A1")
        for f in ("time", "place"):
            r["alternative_frame"][f] = {
                "state": "UNKNOWN",
                "value": None,
                "citations": [],
            }
        self.assertEqual(self.verdict("A1", r)["verdict"], "ABSTAIN")

    def test_a1_one_agreement_insufficient(self):
        r = self.good("A1")
        r["discriminating_fields"] = ["time"]
        self.assertEqual(self.verdict("A1", r)["verdict"], "ABSTAIN")

    def test_a1_conflicting_incident_blocks(self):
        r = self.good("A1")
        r["alternative_frame"]["incident"]["value"] = "another_incident"
        self.assertEqual(self.verdict("A1", r)["verdict"], "ABSTAIN")

    def test_a1_scope_mismatch_blocks(self):
        r = self.good("A1")
        r["alternative_frame"]["scope"]["world"] = "counterfactual"
        self.assertEqual(self.verdict("A1", r)["verdict"], "ABSTAIN")

    def test_a2_exact_citation_but_changed_witness_blocks(self):
        r = self.good("A2")
        r["evidence_hash"] = digest([citation(self.doc, "blast")])
        self.assertIn("EVIDENCE_CHANGED", self.verdict("A2", r)["reason"])

    def test_a2_compatibility_is_not_binding(self):
        r = self.good("A2")
        r["binding_kind"] = "FACTUAL_COMPATIBILITY"
        self.assertEqual(self.verdict("A2", r)["verdict"], "ABSTAIN")

    def test_a2_missing_binding(self):
        r = self.good("A2")
        r["binding"] = []
        self.assertEqual(self.verdict("A2", r)["verdict"], "ABSTAIN")

    def test_a2_false_quote(self):
        r = self.good("A2")
        r["binding"][0]["quote"] = "another explosion"
        self.assertEqual(self.verdict("A2", r)["verdict"], "ABSTAIN")

    def test_a2_each_unknown_obligation_abstains(self):
        for f in self.good("A2")["obligations"]:
            r = self.good("A2")
            r["obligations"][f] = "UNKNOWN"
            self.assertEqual(self.verdict("A2", r)["verdict"], "ABSTAIN")

    def test_a2_incomplete_scan_and_residual(self):
        for key, value in (
            ("covered_mentions", ["a"]),
            ("covered_sentences", [0]),
            ("residual", "UNKNOWN"),
        ):
            r = self.good("A2")
            r["competitors"][key] = value
            self.assertEqual(self.verdict("A2", r)["verdict"], "ABSTAIN")

    def test_a2_plausible_other_incident_blocks(self):
        r = self.good("A2")
        r["competitors"]["rivals"] = [
            {
                "mention_id": None,
                "citations": [citation(self.doc, "blast")],
                "outcome": "PLAUSIBLE",
                "reason": "A competing occurrence cannot be ruled out.",
            }
        ]
        self.assertEqual(self.verdict("A2", r)["verdict"], "ABSTAIN")

    def test_a2_source_and_relation_modal_scope(self):
        for key in ("alternative_frame", "relation_scope"):
            r = self.good("A2")
            scope = r[key]["scope"] if key == "alternative_frame" else r[key]
            scope["modality"] = "possible"
            scope["world"] = "hypothetical"
            self.assertEqual(self.verdict("A2", r)["verdict"], "ABSTAIN")

    def test_a2_new_causal_premises_are_not_transport(self):
        r = self.good("A2")
        r["new_causal_premises"] = True
        self.assertEqual(self.verdict("A2", r)["verdict"], "ABSTAIN")

    def test_preventive_scope_not_blanket_rejected(self):
        # Structural permissibility, not a semantic claim about this fixture text.
        for s in [
            self.cert["relation_scope"],
            *[f["scope"] for f in self.cert["frames"].values()],
        ]:
            s.update(
                world="planned",
                modality="intended_prevention",
                operators=["intended", "prevent"],
            )
        self.verified[self.root["id"]]["source_certificate_hash"] = digest(self.cert)
        self.registry = build_registry(
            self.bundle, self.roots, self.extracted, self.verified
        )
        self.proposal = self.registry["proposals"][0]
        r = self.good("A2")
        r["alternative_frame"]["scope"] = deepcopy(self.cert["frames"]["a"]["scope"])
        r["relation_scope"] = deepcopy(self.cert["relation_scope"])
        self.assertEqual(self.verdict("A2", r)["verdict"], "ACCEPT")

    def test_a3_missing_is_unavailable_not_abstain(self):
        self.groups[self.doc["id"]] = {
            "status": "UNAVAILABLE",
            "reason": "MISSING_T2_CACHE",
        }
        self.assertEqual(self.verdict("A3", None)["verdict"], "UNAVAILABLE")

    def test_binding_must_cover_alternative_mention(self):
        r = self.good("A2")
        r["binding"] = [citation(self.doc, "blast")]
        self.assertIn("ALTERNATIVE_NOT_IN_BINDING", self.verdict("A2", r)["reason"])

    def test_a3_does_not_propagate_inside_one_group(self):
        self.groups[self.doc["id"]]["groups"] = [["a", "b", "ap"]]
        self.assertEqual(self.verdict("A3", None)["verdict"], "ABSTAIN")

    def test_a3_does_not_join_new_chains(self):
        self.groups[self.doc["id"]]["groups"] = [["a", "b"]]
        self.assertEqual(self.verdict("A3", None)["verdict"], "ABSTAIN")


class SelectionCaps(Case):
    def selections(self, proposals):
        reg = {"proposals": proposals}
        decisions = [
            {
                "variant": v,
                "candidate_id": p["id"],
                "root_id": p["root_id"],
                "doc": p["doc"],
                "target": p["target"],
                "verdict": "ACCEPT",
                "reason": "fixture",
                "selected": False,
            }
            for v in VARIANTS
            for p in proposals
        ]
        return decisions, reg

    def many(self, roots=6, duplicate_targets=False):
        mids = (
            ["b"]
            + [f"c{i}" for i in range(roots)]
            + (["ap"] if duplicate_targets else [f"ap{i}" for i in range(roots)])
        )
        tokens = ["event" for _ in mids]
        raw = {
            "id": "SYNTHETIC_MANY",
            "tokens": tokens,
            "mentions": mids,
            "spans": [[i] for i in range(len(mids))],
            "sentences": [[0, len(mids)]],
        }
        cs = [
            {
                "doc": raw["id"],
                "a": a,
                "b": b,
                "luna": 0.9
                if a == "b" and b.startswith("c") or b == "b" and a.startswith("c")
                else 0.0,
                "d1": 0.9
                if a == "b" and b.startswith("c") or b == "b" and a.startswith("c")
                else 0.0,
            }
            for a, b in itertools.permutations(mids, 2)
        ]
        bundle = public_bundle({"documents": [raw]}, cs)
        props = []
        for i in range(roots):
            cause, alternative = f"c{i}", "ap" if duplicate_targets else f"ap{i}"
            props.append(
                {
                    "id": f"p{i}",
                    "root_id": f"r{i}",
                    "doc": raw["id"],
                    "cause": cause,
                    "effect": "b",
                    "alternative": alternative,
                    "target": list(pair(alternative, "b")),
                    "root_score": 0.9,
                    "generation": 1,
                    "reusable_as_source": False,
                }
            )
        return bundle, props

    def test_five_per_document_for_every_arm(self):
        bundle, props = self.many()
        ds, reg = self.selections(props)
        selected = select_additions(ds, reg, bundle)
        for v in VARIANTS:
            self.assertEqual(
                sum(d["selected"] for d in selected if d["variant"] == v), 5
            )
            self.assertEqual(
                sum(
                    d["reason"] == "DOCUMENT_CAP" for d in selected if d["variant"] == v
                ),
                1,
            )

    def test_one_per_source(self):
        bundle, props = self.many(2)
        props[1]["root_id"] = props[0]["root_id"]
        ds, reg = self.selections(props)
        selected = select_additions(ds, reg, bundle)
        for v in VARIANTS:
            self.assertEqual(
                sum(d["selected"] for d in selected if d["variant"] == v), 1
            )

    def test_duplicate_targets_once(self):
        bundle, props = self.many(2, True)
        ds, reg = self.selections(props)
        selected = select_additions(ds, reg, bundle)
        for v in VARIANTS:
            self.assertEqual(
                sum(d["selected"] for d in selected if d["variant"] == v), 1
            )

    def test_no_recursive_reuse(self):
        ds, reg = self.selections([deepcopy(self.proposal)])
        reg["proposals"][0]["generation"] = 2
        with self.assertRaises(Invalid):
            select_additions(ds, reg, self.bundle)

    def test_decision_cannot_change_fixed_endpoint(self):
        ds, reg = self.selections([self.proposal])
        ds[0]["target"] = ["a", "ap"]
        with self.assertRaises(Invalid):
            select_additions(ds, reg, self.bundle)

    def test_duplicate_or_incomplete_decisions_fail(self):
        ds, reg = self.selections([self.proposal])
        for values in (ds[:-1], ds + [ds[0]]):
            with self.assertRaises(Invalid):
                select_additions(values, reg, self.bundle)


class StagingAndEvaluation(Case):
    def test_blind_smoke_no_network_no_labels(self):
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("network forbidden")
        ):
            summary = smoke(self.path / "smoke")
        self.assertFalse(summary["labels_read"])
        self.assertEqual(summary["model_calls"], 0)
        root = self.path / "smoke"
        for stage in ("extract", "propose", "judge"):
            for req in read_jsonl(root / (stage + ".requests.jsonl")):
                self.assertNotIn('"gold"', json.dumps(req["messages"]))
                self.assertNotIn('"coref_clusters"', json.dumps(req["messages"]))
        self.assertTrue(read_json(root / "run/run.json")["complete"])

    def test_write_once_no_override(self):
        f = self.path / "frozen.json"
        write_once(f, {"a": 1})
        write_once(f, {"a": 1})
        with self.assertRaises(Invalid):
            write_once(f, {"a": 2})

    def test_plan_prompt_or_public_drift_detected(self):
        prepare(
            self.suite,
            self.controls,
            self.path / "cache",
            self.path / "plan",
            config=dict(DEFAULT_CONFIG, max_sources=1),
        )
        f = self.path / "plan/public.json"
        v = read_json(f)
        v["documents"][self.doc["id"]]["text"] += "changed"
        f.write_text(json.dumps(v))
        with self.assertRaises(Invalid):
            load_plan(self.path / "plan")

    def test_configuration_rules_not_swept(self):
        for key, value in (
            ("threshold", 0.4),
            ("max_additions_per_source", 2),
            ("recursive", True),
            ("temperature", 0),
            ("max_usd", 11),
            ("max_billed_tokens", 1500001),
        ):
            c = dict(DEFAULT_CONFIG)
            c[key] = value
            with self.subTest(key=key), self.assertRaises(Invalid):
                validate_config(c)

    def test_missing_or_duplicate_receipt_fails(self):
        prepare(
            self.suite,
            self.controls,
            self.path / "cache",
            self.path / "plan",
            config=dict(DEFAULT_CONFIG, max_sources=1),
        )
        req = extraction_requests(self.path / "plan")
        r = receipt(req[0], [self.extracted[self.root["id"]]])
        for values in ([], [r, r]):
            with self.assertRaises(Invalid):
                collect_items(req, values, "source_id")

    def test_synthetic_scoring_exact_but_not_real_model_gain(self):
        smoke(self.path / "smoke")
        root = self.path / "smoke"
        gold = {
            "documents": {
                self.doc["id"]: {
                    "positive_pairs": [["a", "b"], ["ap", "b"]],
                    "coreference": [["a", "ap"]],
                }
            }
        }
        result = evaluate(
            root / "plan", root / "registry", root / "run", gold, root / "evaluation"
        )
        self.assertAlmostEqual(result["baseline_full"]["f1"], 2 / 3)
        for v in VARIANTS:
            self.assertEqual(result["variants"][v]["tp_added"], 1)
            self.assertEqual(result["variants"][v]["fp_added"], 0)
            self.assertEqual(result["variants"][v]["after"]["f1"], 1.0)
            self.assertEqual(result["variants"][v]["status"], "SYNTHETIC_ONLY")
        self.assertEqual(result["total_new_usage"]["calls"], 0)
        self.assertTrue(result["not_confirmed_gain"])
        import csv

        with (root / "evaluation/comparison.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 4)

    def test_historical_parity_never_imports_gold_scoring_top_level(self):
        from proof_transport.validation import REFERENCE, historical_t2_parity

        historical = self.path / "archived.py"
        historical.write_text(
            'raise RuntimeError("Gold scoring must not execute")' + chr(10) + REFERENCE
        )
        result = historical_t2_parity(self.bundle, self.groups, historical)
        self.assertEqual(result["exact_T2_matches"], 1)
        self.assertFalse(result["gold_read"])
        historical.write_text(REFERENCE.replace("if ga != gb:", "if True:"))
        with self.assertRaises(Invalid):
            historical_t2_parity(self.bundle, self.groups, historical)

    def test_unsealed_run_cannot_score(self):
        smoke(self.path / "smoke")
        root = self.path / "smoke"
        f = root / "run/run.json"
        v = read_json(f)
        v["complete"] = False
        f.write_text(json.dumps(v))
        with self.assertRaises(Invalid):
            evaluate(
                root / "plan", root / "registry", root / "run", {}, root / "evaluation"
            )

    def test_witness_diagnostics_do_not_add_candidates_or_use_labels(self):
        from proof_transport.diagnostics import diagnostic_requests, inspect_diagnostics

        smoke(self.path / "smoke")
        root = self.path / "smoke"
        registry = read_json(root / "registry/registry.json")
        before = digest(registry)
        requests = diagnostic_requests(root / "plan", registry)
        self.assertEqual(len(requests), 1)
        body = json.loads(requests[0]["messages"][1]["content"])
        self.assertEqual(body["items"][0]["certificate"]["evidence"], [])
        self.assertNotIn('"gold"', json.dumps(body))
        raw = self.good("A2")
        raw["candidate_id"] = body["items"][0]["candidate_id"]
        summary, usage = inspect_diagnostics(
            root / "plan", registry, [receipt(requests[0], [raw])]
        )
        self.assertEqual(len(summary["claims_reuse_without_witness"]), 1)
        self.assertFalse(summary["adds_predictions"])
        self.assertEqual(digest(registry), before)
        self.assertEqual(usage[0]["mode"], "synthetic_fixture")

    def test_blind_packet_has_no_scores_labels_or_acceptance(self):
        smoke(self.path / "smoke")
        root = self.path / "smoke"
        blind_audit(
            root / "plan", read_json(root / "registry/registry.json"), root / "audit"
        )
        s = (root / "audit/blind-packets.jsonl").read_text()
        for bad in ('"gold"', '"score"', '"selected"', '"variant"', '"identity"'):
            self.assertNotIn(bad, s)


class BudgetSafety(Case):
    def pricing(self):
        raw = json.dumps(
            {
                "data": [
                    {
                        "id": DEFAULT_CONFIG["model"],
                        "canonical_slug": "openai/"
                        + DEFAULT_CONFIG["expected_version"],
                        "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
                        "context_length": 1050000,
                        "supported_parameters": [
                            "max_tokens",
                            "reasoning",
                            "response_format",
                        ],
                    }
                ]
            }
        )
        return client.tariff_snapshot(raw, DEFAULT_CONFIG["model"])

    def test_public_tariffs_and_worst_override(self):
        p = self.pricing()
        self.assertEqual(p["model_invocations"], 0)
        self.assertEqual(p["conservative_completion_usd_per_token"], "5E-7")

    def test_missing_user_approval_before_send(self):
        with self.assertRaises(Invalid):
            client.execute(
                [],
                {},
                DEFAULT_CONFIG,
                self.pricing(),
                {},
                self.path / "b.sqlite",
                self.path / "r.jsonl",
                send=lambda _: self.fail("No approval, no network"),
            )

    def test_token_and_dollar_caps(self):
        b = client.Budget(self.path / "b.sqlite", "s", 1.0, 100)
        self.addCleanup(b.close)
        b.reserve("x", 0.5, 50)
        for usd, tokens in ((0.6, 1), (0.1, 51)):
            with self.assertRaises(Invalid):
                b.reserve("y", usd, tokens)

    def test_unknown_request_never_retried(self):
        b = client.Budget(self.path / "b.sqlite", "s", 1.0, 100)
        self.addCleanup(b.close)
        b.reserve("x", 0.5, 50)
        with self.assertRaises(Invalid):
            b.reserve("x", 0.5, 50)

    def test_settle_cache_does_not_charge_twice(self):
        b = client.Budget(self.path / "b.sqlite", "s", 1.0, 100)
        self.addCleanup(b.close)
        b.reserve("x", 0.5, 50)
        r = {"usage": {"input_tokens": 10, "output_tokens": 10, "cost_usd": 0.1}}
        b.settle("x", r)
        self.assertEqual(b.reserve("x", 0.5, 50), r)
        self.assertEqual(
            b.db.execute("SELECT SUM(usd),SUM(tokens) FROM request").fetchone(),
            (0.1, 20),
        )

    def test_provider_bound_violation_recorded_and_scope_stopped(self):
        b = client.Budget(self.path / "b.sqlite", "s", 1.0, 100)
        self.addCleanup(b.close)
        b.reserve("x", 0.1, 10)
        with self.assertRaises(Invalid):
            b.settle(
                "x",
                {"usage": {"input_tokens": 10, "output_tokens": 10, "cost_usd": 0.2}},
            )
        self.assertEqual(
            b.db.execute("SELECT tokens,status FROM request").fetchone(),
            (20, "VIOLATION"),
        )
        with self.assertRaises(Invalid):
            b.reserve("y", 0.1, 10)

    def test_scope_cannot_be_reset_with_larger_budget(self):
        b = client.Budget(self.path / "b.sqlite", "s", 1.0, 100)
        b.close()
        with self.assertRaises(Invalid):
            client.Budget(self.path / "b.sqlite", "s", 2.0, 200)

    def approve(self, manifest, pricing, ledger):
        return {
            "approval": "APPROVED_BY_USER",
            "approved_by": "UNIT_TEST_ONLY_NOT_AN_OPERATOR_APPROVAL",
            "plan_hash": digest(manifest),
            "config_hash": digest(DEFAULT_CONFIG),
            "pricing_hash": digest(pricing),
            "model": DEFAULT_CONFIG["model"],
            "max_usd": 10.0,
            "max_billed_tokens": 1500000,
            "expires_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "ledger_path": str(ledger.resolve()),
        }

    def live_fixture(self):
        from proof_transport.pipeline import load_plan

        prepare(self.suite, self.controls, self.path / "cache", self.path / "plan")
        manifest, *_ = load_plan(self.path / "plan")
        requests = extraction_requests(self.path / "plan")
        pricing = self.pricing()
        ledger = self.path / "budget.sqlite"
        return (
            manifest,
            requests,
            pricing,
            ledger,
            self.approve(manifest, pricing, ledger),
        )

    def test_approval_is_bound_to_single_ledger(self):
        m, rs, p, _ledger, a = self.live_fixture()
        with self.assertRaises(Invalid):
            client.execute(
                rs,
                m,
                DEFAULT_CONFIG,
                p,
                a,
                self.path / "other.sqlite",
                self.path / "r.jsonl",
                send=lambda _: self.fail("Wrong ledger must not send"),
            )

    def test_live_fake_transport_accounts_reported_cost_and_no_second_call(self):
        m, rs, p, ledger, a = self.live_fixture()
        calls = []

        def fake(payload):
            calls.append(payload)
            return {
                "model": DEFAULT_CONFIG["model"],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "cost": 0.000001,
                },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"items": []})},
                    }
                ],
            }

        one = client.execute(
            rs, m, DEFAULT_CONFIG, p, a, ledger, self.path / "one.jsonl", send=fake
        )
        two = client.execute(
            rs, m, DEFAULT_CONFIG, p, a, ledger, self.path / "two.jsonl", send=fake
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(one, two)
        self.assertEqual(one[0]["usage"]["cost_usd"], 0.000001)
        self.assertFalse(one[0]["version_verified"])

    def test_provider_total_includes_all_billed_reasoning_tokens(self):
        m, rs, p, ledger, a = self.live_fixture()

        def fake(_):
            return {
                "model": DEFAULT_CONFIG["model"],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 45,
                    "cost": 0.00001,
                },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"items": []})},
                    }
                ],
            }

        rows = client.execute(
            rs, m, DEFAULT_CONFIG, p, a, ledger, self.path / "total.jsonl", send=fake
        )
        self.assertEqual(
            rows[0]["usage"]["input_tokens"] + rows[0]["usage"]["output_tokens"], 45
        )
        self.assertEqual(rows[0]["provider_completion_tokens"], 10)

    def test_failed_network_request_never_repaid(self):
        m, rs, p, ledger, a = self.live_fixture()

        def fail(_):
            raise ConnectionError("unit-test transport failure")

        with self.assertRaises(ConnectionError):
            client.execute(
                rs, m, DEFAULT_CONFIG, p, a, ledger, self.path / "r.jsonl", send=fail
            )
        with self.assertRaises(Invalid):
            client.execute(
                rs,
                m,
                DEFAULT_CONFIG,
                p,
                a,
                ledger,
                self.path / "r.jsonl",
                send=lambda _: self.fail("Never resend unknown request"),
            )

    def test_stale_tariff_refuses_before_send(self):
        m, rs, p, ledger, a = self.live_fixture()
        p["fetched_utc"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
        a["pricing_hash"] = digest(p)
        with self.assertRaises(Invalid):
            client.execute(
                rs,
                m,
                DEFAULT_CONFIG,
                p,
                a,
                ledger,
                self.path / "r.jsonl",
                send=lambda _: self.fail("Stale pricing must not send"),
            )

    def test_expired_or_stale_tariffs_refuse(self):
        p = self.pricing()
        m = {"x": 1}
        c = dict(DEFAULT_CONFIG)
        a = {
            "approval": "APPROVED_BY_USER",
            "approved_by": "test-fixture",
            "plan_hash": digest(m),
            "config_hash": digest(c),
            "pricing_hash": digest(p),
            "model": c["model"],
            "max_usd": 1.0,
            "max_billed_tokens": 1000,
            "expires_utc": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
            "ledger_path": str((self.path / "b.sqlite").resolve()),
        }
        with self.assertRaises(Invalid):
            client.validate_approval(a, m, c, p)


if __name__ == "__main__":
    unittest.main()
