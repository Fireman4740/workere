"""Adapters around the unchanged historical scorer and explicitly separate new tasks."""
from __future__ import annotations

from collections import defaultdict

from benchmark.evaluate import evaluate, evaluate_event_mentions
from benchmark.labels import BINARY_LABELS, is_norel
from benchmark.metrics import prf, set_metrics
from benchmark.schema import Document, Entity, PipelineInput, Prediction, PROTOCOLS
from benchmark.suite import documents

PROTOCOL_IDS = {"relations_gold": "workere-historical-relations/v1",
                "entity_extraction": "event-mentions-exact-untyped/v1",
                "end_to_end": "event-relations-exact-spans/v1"}


def public_inputs(suite, protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported protocol: {protocol}")
    for old in documents(suite):
        doc = Document(old.id, tuple(old.tokens), tuple(tuple(s) for s in old.sentences), old.lang)
        if protocol == "relations_gold":
            if len(old.mentions) != len(old.spans):
                raise ValueError("reference mentions and spans must be parallel")
            yield PipelineInput(doc, tuple(Entity(mid, tuple(span)) for mid, span in zip(old.mentions, old.spans)),
                                tuple(tuple(pair) for pair in old.pair_list))
        else:
            yield PipelineInput(doc)


def counts(gold, predicted):
    score = set_metrics(gold, predicted)
    return {"tp": score["tp"], "fp": score["predicted"] - score["tp"],
            "fn": score["support"] - score["tp"]}


def from_counts(count):
    p, r, f = prf(count["tp"], count["fp"], count["fn"])
    return {"precision": p, "recall": r, "f1": f,
            "support": count["tp"] + count["fn"], "predicted": count["tp"] + count["fp"]}


def historical_statistics(results, labels):
    """Exact sufficient counts for document bootstrap; never changes the reported scores."""
    grouped = defaultdict(list)
    for row in results["per_pair_predictions"]:
        grouped[row["id"]].append(row)
    out = []
    for doc in results["per_doc_metrics"]:
        rows = grouped[doc["id"]]
        stats = {key: {"tp": 0, "fp": 0, "fn": 0}
                 for key in ("primary", "binary", "binary_intra", "binary_inter")}
        positive_labels = {label for label in labels if not is_norel(label)}
        positive_labels |= {row[k] for row in rows for k in ("gold", "pred") if not is_norel(row[k])}
        by_label = {label: {"tp": 0, "fp": 0, "fn": 0} for label in positive_labels}
        for row in rows:
            gold, pred = row["gold"], row["pred"]
            gpos, ppos = not is_norel(gold), not is_norel(pred)
            for label in positive_labels:
                if gold == label and pred == label:
                    by_label[label]["tp"] += 1
                elif pred == label:
                    by_label[label]["fp"] += 1
                elif gold == label:
                    by_label[label]["fn"] += 1
            st = stats["primary"]
            st["tp"] += int(gold == pred and gpos)
            st["fp"] += int(gold != pred and ppos)
            st["fn"] += int(gold != pred and gpos)
            for key in ("binary", "binary_" + row.get("sentence_relation", "unknown")):
                if key in stats:
                    st = stats[key]
                    st["tp"] += int(gpos and ppos)
                    st["fp"] += int(not gpos and ppos)
                    st["fn"] += int(gpos and not ppos)
        stats["primary"]["by_label"] = by_label
        out.append({"id": doc["id"], "metrics": stats})
    return out


def legacy_summary(results):
    return {"primary": {"precision": results["micro_precision"], "recall": results["micro_recall"],
                         "f1": results["micro_f1"], "macro_f1": results["macro_f1"]},
            **{key: results[key] for key in ("binary", "binary_intra", "binary_inter")}}


def _end_to_end(suite, by_id):
    statistics = []
    unknown = 0
    binary = suite["config"]["data"]["binary_undirected"] or suite["config"]["data"]["vote_to_binary"]
    for doc in documents(suite):
        raw = Document(doc.id, tuple(doc.tokens), tuple(tuple(s) for s in doc.sentences), doc.lang)
        prediction = by_id[doc.id]
        g_entities = {mid: tuple(span) for mid, span in zip(doc.mentions, doc.spans)}
        p_entities = {e.id: e.token_indices for e in prediction.entities}
        g_relations, p_relations = set(), set()

        def endpoint(mid, entities):
            return ("span", entities[mid]) if mid in entities else ("unresolved", mid)

        def triple(source, label, target, entities):
            a, b = endpoint(source, entities), endpoint(target, entities)
            return (*sorted((a, b)), "causal") if binary else (a, b, label)

        for source, label, target in doc.gold:
            if not is_norel(label):
                g_relations.add(triple(source, label, target, g_entities))
        for relation in prediction.relations:
            if not is_norel(relation.label):
                p_relations.add(triple(relation.source, relation.label, relation.target, p_entities))

        def scope(key):
            a, b = key[:2]
            sa = raw.sentence_of(a[1]) if a[0] == "span" else None
            sb = raw.sentence_of(b[1]) if b[0] == "span" else None
            return "unknown" if sa is None or sb is None else "intra" if sa == sb else "inter"

        gs = {name: {key for key in g_relations if scope(key) == name} for name in ("intra", "inter")}
        ps = {name: {key for key in p_relations if scope(key) == name} for name in ("intra", "inter")}
        unknown += sum(scope(key) == "unknown" for key in g_relations | p_relations)
        stats = {"entities": counts(g_entities.values(), p_entities.values()),
                 "primary": counts(g_relations, p_relations),
                 "binary": counts({key[:2] for key in g_relations}, {key[:2] for key in p_relations}),
                 **{name: counts(gs[name], ps[name]) for name in ("intra", "inter")}}
        statistics.append({"id": doc.id, "metrics": stats})
    summary = aggregate_statistics(statistics)
    return {"summary": summary, "document_statistics": statistics,
            "results": {"entities": summary["entities"], "relations": summary["primary"],
                        "binary": summary["binary"], "intra": summary["intra"], "inter": summary["inter"],
                        "sentence_unknown_relations": unknown}}


def aggregate_statistics(statistics):
    sums, labels = {}, defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for record in statistics:
        for name, count in record["metrics"].items():
            target = sums.setdefault(name, {"tp": 0, "fp": 0, "fn": 0})
            for key in ("tp", "fp", "fn"):
                target[key] += count[key]
            for label, st in count.get("by_label", {}).items():
                for key in ("tp", "fp", "fn"):
                    labels[label][key] += st[key]
    result = {name: from_counts(st) for name, st in sums.items()}
    if labels and "primary" in result:
        result["primary"]["macro_f1"] = sum(from_counts(st)["f1"] for st in labels.values()) / len(labels)
    return result


def score_predictions(suite, rows, protocol, *, model="external"):
    inputs = list(public_inputs(suite, protocol))
    rows = list(rows)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate document prediction")
    raw = {row["id"]: row for row in rows}
    if set(raw) != {i.document.id for i in inputs}:
        raise ValueError("prediction coverage differs from the frozen suite")
    by_id = {i.document.id: Prediction.from_dict(raw[i.document.id], i.document) for i in inputs}
    if protocol == "relations_gold":
        for item in inputs:
            pred = by_id[item.document.id]
            expected = {e.id: e.token_indices for e in item.reference_entities}
            if {e.id: e.token_indices for e in pred.entities} != expected:
                raise ValueError("relations_gold must preserve all reference mention IDs and spans")
        triples = [{"id": p.id, "triples": [[r.source, r.label, r.target] for r in p.relations]}
                   for p in by_id.values()]
        old = evaluate(suite, triples, model=model)
        data = suite["config"]["data"]
        labels = BINARY_LABELS if data["binary_undirected"] or data["vote_to_binary"] else data["labels"]
        result = {"results": old["results"], "summary": legacy_summary(old["results"]),
                  "document_statistics": historical_statistics(old["results"], labels)}
    elif protocol == "entity_extraction":
        spans = [{"id": p.id, "spans": [list(e.token_indices) for e in p.entities]} for p in by_id.values()]
        old = evaluate_event_mentions(suite, spans, model=model)
        stats = [{"id": r["id"], "metrics": {"primary": {"tp": r["tp"],
                  "fp": r["predicted"] - r["tp"], "fn": r["support"] - r["tp"]}}}
                 for r in old["results"]["per_doc_metrics"]]
        result = {"results": old["results"], "summary": {"primary": {key: old["results"][key]
                  for key in ("precision", "recall", "f1", "support", "predicted")}},
                  "document_statistics": stats}
    else:
        result = _end_to_end(suite, by_id)
    return {"schema": "workere-evaluation/v2", "protocol": protocol,
            "protocol_id": PROTOCOL_IDS[protocol], "oracle_reference_mentions": protocol == "relations_gold",
            "suite_fingerprint": suite["fingerprint"], **result}
