#!/usr/bin/env python3
"""Agent 4 feasibility harness. Standard library only, NO model/API calls.

Commands:
  select: creates a gold-blind, deterministic pair manifest.
  prepare: validates edited documents and creates model-call task files.
  score: analyses post-parsed model outputs, without gold labels.
  evaluate: optional descriptive evaluation after scores are locked.

The caller must adapt their repo's output format to the documented JSONL schema.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("original", "target", "placebo_1", "placebo_2", "placebo_3", "rename_ids")
BANNED_KEYS = {"gold", "gold_label", "label", "labels", "is_positive", "ground_truth", "causal_gold", "y_true", "true_label", "tp", "fp", "fn"}
ALLOWED_REVIEW = {"approved", "rejected", "uncertain"}


def die(message):
    raise ValueError(message)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    die(f"{path}:{no}: JSON invalide: {e}")


def save_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def rank_hash(seed, *parts):
    return sha_bytes((str(seed) + "|" + "|".join(map(str, parts))).encode("utf-8"))


def check_no_gold(obj, context="source"):
    """Reject any gold-labelled keys at any depth (not the word in text)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in BANNED_KEYS:
                die(f"{context}: champ interdit avant évaluation: {k}")
            check_no_gold(v, context)
    elif isinstance(obj, list):
        for v in obj:
            check_no_gold(v, context)


def canon(a, b):
    a, b = str(a), str(b)
    if a == b:
        die(f"Paire réflexive invalide: {a}")
    return tuple(sorted((a, b)))


def parse_docs(path, allowed_splits=None):
    docs = {}
    for obj in load_jsonl(path):
        check_no_gold(obj, "baseline")
        doc_id = str(obj["doc_id"])
        if doc_id in docs:
            die(f"doc_id dupliqué: {doc_id}")
        if allowed_splits is not None and str(obj.get("split", "")) not in allowed_splits:
            die(f"{doc_id}: split {obj.get('split')!r} hors de {sorted(allowed_splits)}. Ne pas exposer réserve/test.")
        text = obj["text"]
        mentions = {str(m["id"]): m for m in obj["mentions"]}
        if len(mentions) != len(obj["mentions"]):
            die(f"{doc_id}: IDs de mentions non uniques")
        for mid, m in mentions.items():
            start, end = m["start"], m["end"]
            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
                die(f"{doc_id}/{mid}: offsets invalides")
            if text[start:end] != m["surface"]:
                die(f"{doc_id}/{mid}: surface non alignée : {text[start:end]!r} != {m['surface']!r}")
        seen_pairs = set()
        for pair in obj["pairs"]:
            p = canon(pair["a"], pair["b"])
            if p in seen_pairs:
                die(f"{doc_id}: paire dupliquée {p}")
            seen_pairs.add(p)
            if not set(p) <= set(mentions):
                die(f"{doc_id}: paire {p} hors mentions")
            s = pair["baseline_score"]
            if not isinstance(s, (int, float)) or not (0 <= s <= 1) or not math.isfinite(s):
                die(f"{doc_id}: score invalide {s}")
        docs[doc_id] = obj
    return docs


def select(args):
    allowed = set(args.allowed_splits.split(","))
    docs = parse_docs(args.baseline, allowed)
    candidates = defaultdict(list)
    for doc in docs.values():
        for p in doc["pairs"]:
            s = p["baseline_score"]
            cohort = "positive" if s >= args.positive_min else "negative" if args.negative_min <= s < args.negative_max else None
            if cohort is None:
                continue
            a, b = canon(p["a"], p["b"])
            candidates[cohort].append({"doc_id": str(doc["doc_id"]), "topic_id": str(doc["topic_id"]),
               "pair": [a, b], "cohort": cohort, "baseline_score": s,
               "rank": rank_hash(args.seed, doc["doc_id"], a, b)})
    choices = []
    for cohort, goal in [("positive", args.n_positive), ("negative", args.n_negative)]:
        group = sorted(candidates[cohort], key=lambda x: x["rank"])
        topic_seen, doc_counts, chosen = set(), defaultdict(int), []
        def add_one(item):
            if doc_counts[item["doc_id"]] >= args.max_per_doc:
                return False
            chosen.append(item)
            doc_counts[item["doc_id"]] += 1
            topic_seen.add(item["topic_id"])
            return True
        for item in group:
            if len(chosen) >= goal:
                break
            if item["topic_id"] not in topic_seen:
                add_one(item)
        for item in group:
            if len(chosen) >= goal:
                break
            if item not in chosen:
                add_one(item)
        if len(chosen) < goal:
            print(f"AVERTISSEMENT: cohorte {cohort}: seulement {len(chosen)}/{goal} candidats éligibles")
        choices.extend(chosen)
    for x in choices:
        x.pop("rank")
    output = {"kind": "blind_selection", "protocol": "agent4-feasibility-v0", "seed": args.seed,
              "baseline_sha256": sha_file(args.baseline), "selected": choices,
              "constraints": {"allowed_splits": sorted(allowed), "max_per_doc": args.max_per_doc,
                              "positive_min": args.positive_min,
                              "negative_band": [args.negative_min, args.negative_max]},
              "topics_n": len(set(x["topic_id"] for x in choices)), "docs_n": len(set(x["doc_id"] for x in choices))}
    save_json(args.out, output)
    print(f"Selection aveugle : {len(choices)} paires, {output['topics_n']} topics, {output['docs_n']} documents -> {args.out}")


def apply_edits(text, mentions, edits):
    """Offsets refer to the original document. Edits may not touch ANY mention."""
    prev = 0
    deltas = []
    result = []
    for e in sorted(edits, key=lambda x: x["start"]):
        start, end, after = e["start"], e["end"], e["after"]
        if not isinstance(after, str) or not (0 <= start < end <= len(text)) or start < prev:
            die("Édition invalide ou chevauchement; insertion pure non supportée en V0")
        if "before" not in e or text[start:end] != e["before"]:
            die(f"Édition: before obligatoire et doit correspondre aux offsets {start}:{end}")
        for m in mentions:
            if start < m["end"] and end > m["start"]:
                die(f"Édition touche une mention protégée: {m['id']}")
        result.extend([text[prev:start], after])
        prev = end
        deltas.append((end, len(after) - (end - start)))
    result.append(text[prev:])
    new_text = "".join(result)
    updated = []
    for m in mentions:
        shift = sum(delta for end, delta in deltas if end <= m["start"])
        start, end = m["start"] + shift, m["end"] + shift
        if new_text[start:end] != m["surface"]:
            die(f"Édition a décalé/corrompu la mention {m['id']}")
        updated.append({**m, "start": start, "end": end})
    return new_text, updated


def edit_weight(edits):
    # Token match proxy: this DOES NOT certify semantic placebo matching.
    return sum(max(len(re.findall(r"\S+", e.get("before", ""))),
                   len(re.findall(r"\S+", e["after"]))) for e in edits)


def keypair(d, p):
    return str(d) + "::" + "::".join(canon(*p))


def template(args):
    selection = read_json(args.selection)
    docs = parse_docs(args.baseline, set(selection["constraints"]["allowed_splits"]))
    if sha_file(args.baseline) != selection["baseline_sha256"]:
        die("Baseline différente du manifeste")
    entries = []
    for item in selection["selected"]:
        doc = docs[item["doc_id"]]
        existing_ids = {str(m["id"]) for m in doc["mentions"]}
        prefix = "__v0_"
        while any(prefix in x for x in existing_ids):
            prefix = "_" + prefix
        id_map = {str(m["id"]): prefix + str(i) for i, m in enumerate(doc["mentions"], 1)}
        for variant in CONDITIONS:
            if variant == "original":
                continue
            record = {"doc_id": item["doc_id"], "pair": item["pair"], "variant": variant,
                      "review_status": "uncertain", "reviewed_blind_to_gold": False,
                      "instructions": "RELECTURE NECESSAIRE: conserver tous les spans des mentions et les faits non visés"}
            if variant == "rename_ids":
                record["id_map"] = id_map
            else:
                record["edits"] = []  # original document offsets [start,end), copied verbatim in `before`
            entries.append(record)
    save_jsonl(args.out, entries)
    print(f"Templates sans édition sémantique : {len(entries)} lignes. Revue humaine indépendante requise -> {args.out}")


def prepare(args):
    selection = read_json(args.selection)
    docs = parse_docs(args.baseline, set(selection["constraints"]["allowed_splits"]))
    if sha_file(args.baseline) != selection["baseline_sha256"]:
        die("Le baseline ne correspond pas au manifeste gelé")
    expected = {keypair(x["doc_id"], x["pair"]): x for x in selection["selected"]}
    edits_by_key = defaultdict(dict)
    for entry in load_jsonl(args.edits):
        check_no_gold(entry, "edits")
        k = keypair(entry["doc_id"], entry["pair"])
        if k not in expected:
            die(f"Édition d'une paire non sélectionnée: {k}")
        var = entry["variant"]
        if var not in CONDITIONS:
            die(f"Variante inconnue {var}")
        if var in edits_by_key[k]:
            die(f"Variante dupliquée: {k}/{var}")
        edits_by_key[k][var] = entry
    tasks, review, errors = [], [], []
    for k, item in expected.items():
        doc = docs[item["doc_id"]]
        source_mentions = doc["mentions"]
        records = edits_by_key.get(k, {})
        mandatory = set(CONDITIONS) - {"original"}
        if not mandatory <= set(records):
            errors.append({"key": k, "reason": "missing_edits", "missing": sorted(mandatory - set(records))})
            continue
        if not all(records[v].get("review_status") == "approved" and records[v].get("reviewed_blind_to_gold") is True for v in mandatory):
            errors.append({"key": k, "reason": "not_blindly_approved"})
            continue
        edited = {}
        try:
            for v in CONDITIONS:
                if v in ("original", "rename_ids"):
                    dtext, dmentions = doc["text"], source_mentions
                else:
                    record = records[v]
                    dtext, dmentions = apply_edits(doc["text"], source_mentions, record["edits"])
                id_map = {str(m["id"]): str(m["id"]) for m in dmentions}
                if v == "rename_ids":
                    r = records[v]
                    id_map = {str(x): str(y) for x, y in r.get("id_map", {}).items()}
                    if set(id_map) != {str(m["id"]) for m in source_mentions} or len(set(id_map.values())) != len(id_map):
                        die("Permutation d'IDs non bijective sur toutes les mentions")
                    if all(x == y for x, y in id_map.items()):
                        die("La permutation d'IDs est identique : témoin inutile")
                edited[v] = (dtext, [{**m, "id": id_map[str(m["id"])]} for m in dmentions], id_map)
            wt = edit_weight(records["target"]["edits"])
            weights = [edit_weight(records[f"placebo_{j}"]["edits"]) for j in (1, 2, 3)]
            if wt == 0 or any(w == 0 or abs(w - wt) > max(1, math.ceil(0.2 * wt)) for w in weights):
                die(f"Placebos mal appariés en volume (proxy) : target={wt}, placebos={weights}")
            # Caller-declared semantic review is necessary but NEVER sufficient to certify semantics.
        except (ValueError, KeyError, TypeError) as err:
            errors.append({"key": k, "reason": "invalid_edit", "detail": str(err)})
            continue
        pair = item["pair"]
        record = {"key": k, "status": "eligible_structurally", "target_weight": wt,
                  "placebo_weights": weights, "review_declared": True}
        review.append(record)
        for variant in CONDITIONS:
            dtext, mentions, id_map = edited[variant]
            exposed_pair = [id_map[x] for x in pair]
            reads = args.original_reads if variant == "original" else args.reads
            for reading in range(reads):
                task_id = rank_hash("agent4-feasibility-task-v0", k, variant, reading, sha_bytes(dtext.encode()))[:22]
                tasks.append({"task_id": task_id, "key": k, "doc_id": item["doc_id"],
                              "topic_id": item["topic_id"], "cohort": item["cohort"],
                              "variant": variant, "reading": reading,
                              "document_text": dtext, "mentions": mentions,
                              "exposed_pair": exposed_pair, "id_map": id_map,
                              "candidate_pairs": [[id_map[a], id_map[b]] for p in doc["pairs"] for a, b in [canon(p["a"], p["b"])]],
                              "text_sha256": sha_bytes(dtext.encode("utf-8"))})
    save_jsonl(args.out_tasks, tasks)
    save_json(args.out_audit, {"selection_sha256": sha_file(args.selection), "baseline_sha256": sha_file(args.baseline),
                               "edits_sha256": sha_file(args.edits), "review": review, "rejected": errors,
                               "tasks_sha256": sha_file(args.out_tasks),
                               "call_budget_upper": len(tasks)})
    print(f"Validés structurellement : {len(review)}/{len(expected)}; rejetés : {len(errors)}; tâches modèle non exécutées : {len(tasks)}")


def score(args):
    tasks = list(load_jsonl(args.tasks))
    by_id = {x["task_id"]: x for x in tasks}
    if len(by_id) != len(tasks):
        die("task_id dupliqué")
    outputs = {}
    for rec in load_jsonl(args.predictions):
        tid = rec["task_id"]
        if tid not in by_id or tid in outputs:
            die(f"Prediction hors tâches ou doublon : {tid}")
        outputs[tid] = rec
    bucket = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        rec = outputs.get(task["task_id"])
        if not rec or rec.get("status") != "ok":
            bucket[task["key"]][task["variant"]].append(None)
            continue
        expected_mentions = {str(m["id"]) for m in task["mentions"]}
        votes = set()
        valid = True
        for pair in rec.get("predicted_ordered_pairs", []):
            if not isinstance(pair, list) or len(pair) != 2 or not set(map(str, pair)) <= expected_mentions:
                valid = False
                break
            if pair[0] == pair[1]:
                continue
            votes.add((str(pair[0]), str(pair[1])))
        if not valid:
            bucket[task["key"]][task["variant"]].append(None)
            continue
        a, b = task["exposed_pair"]
        bucket[task["key"]][task["variant"]].append((int((a,b) in votes) + int((b,a) in votes))/2)
    results = []
    for key, grouped in bucket.items():
        counts = {v: len(grouped.get(v, [])) for v in CONDITIONS}
        missing = {v: sum(x is None for x in grouped.get(v, [])) for v in CONDITIONS}
        if set(grouped) != set(CONDITIONS) or any(missing.values()):
            results.append({"key": key, "status": "abstain_incomplete_or_invalid_output", "counts": counts, "missing": missing})
            continue
        means = {v: statistics.mean(grouped[v]) for v in CONDITIONS}
        o, t, pl, rename = means["original"], means["target"], [means[f"placebo_{j}"] for j in (1, 2, 3)], means["rename_ids"]
        delta_target, delta_placebo = o-t, statistics.median([o-x for x in pl])
        contrast = delta_target-delta_placebo
        assert math.isclose(contrast, statistics.median(pl)-t, abs_tol=1e-9)
        results.append({"key": key, "status": "descriptive_only", "s_original": o, "s_target": t,
                        "s_placebos": pl, "s_rename": rename, "delta_target": delta_target,
                        "delta_placebo": delta_placebo, "contrast": contrast,
                        "rename_gap": abs(o-rename), "original_range": max(grouped["original"])-min(grouped["original"]),
                        "placebo_max_gap": max(abs(o-x) for x in pl),
                        "n_reads": counts})
    save_jsonl(args.out, results)
    save_json(str(args.out) + ".lock.json", {"scored_sha256": sha_file(args.out),
        "tasks_sha256": sha_file(args.tasks), "predictions_sha256": sha_file(args.predictions),
        "n_pairs": len(results), "n_scored": sum(x["status"] == "descriptive_only" for x in results),
        "status": "LOCK_BEFORE_GOLD_JOIN", "warning": "V0: descriptive only, NO F1 improvement claimed"})
    print(f"Empreintes calculées: {sum(x['status']=='descriptive_only' for x in results)}/{len(results)}, sorties scellées par hash; pas d'évaluation gold.")


def evaluate(args):
    lock = read_json(str(args.scores) + ".lock.json")
    if sha_file(args.scores) != lock["scored_sha256"]:
        die("Les scores ne correspondent plus au verrou avant gold")
    selected = read_json(args.selection)
    selection_keys = {keypair(x["doc_id"], x["pair"]): x for x in selected["selected"]}
    scored = {r["key"]: r for r in load_jsonl(args.scores)}
    gold = {}
    for row in load_jsonl(args.gold):
        key = keypair(row["doc_id"], row["pair"])
        if key in gold:
            die(f"Gold dupliqué {key}")
        gold[key] = bool(row["causal"])
    missing = [k for k in selection_keys if k not in gold]
    if missing:
        die(f"Gold manquant pour {len(missing)} paires sélectionnées")
    res = {"status": "diagnostic_on_selected_only", "notice": "Pas de F1 corpus: sélection stratifiée, aucun changement de la référence", "counts": {}, "cohorts": {}}
    for cohort in ("positive", "negative"):
        group = [k for k, x in selection_keys.items() if x["cohort"] == cohort]
        usable = [(scored[k]["contrast"], gold[k], k) for k in group if k in scored and scored[k]["status"] == "descriptive_only"]
        if not usable:
            res["cohorts"][cohort] = {"selected": len(group), "scored": 0}
            continue
        # Small-contrast tranche for baseline positive; large-contrast for baseline negative.
        usable.sort(key=lambda x: (x[0] if cohort == "positive" else -x[0], x[2]))
        n = max(1, math.ceil(len(usable) * args.tranche))
        chosen = usable[:n]
        success = sum((not y if cohort == "positive" else y) for _, y, _ in chosen)
        res["cohorts"][cohort] = {"selected": len(group), "scored": len(usable),
               "contrast_tranche_fraction": args.tranche, "tranche_n": n,
               "target_errors_or_correct_additions": success,
               "target_fraction": success/n,
               "warning": "Descriptif seulement: pas de contrôle à coût égal, pas d'intervalle, pas de F1"}
    save_json(args.out, res)
    print(f"Diagnostic descriptif de la sélection (aucun gain F1): {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Agent4 V0 — fichiers locaux, jamais d'appel API")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="Sélection de paires avant toute lecture gold")
    p.add_argument("--baseline", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allowed-splits", default="dev,pilot")
    p.add_argument("--seed", default="agent4-v0-fixed-20261027")
    p.add_argument("--n-positive", type=int, default=12)
    p.add_argument("--n-negative", type=int, default=0)
    p.add_argument("--max-per-doc", type=int, default=2)
    p.add_argument("--positive-min", type=float, default=.7)
    p.add_argument("--negative-min", type=float, default=.25)
    p.add_argument("--negative-max", type=float, default=.39)
    p.set_defaults(fn=select)
    p = sub.add_parser("template", help="Produit un gabarit pour révision des éditions (aucune édition automatique)")
    p.add_argument("--baseline", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=template)
    p = sub.add_parser("prepare", help="Vérifie les éditions et prépare les requêtes modèle (sans les appeler)")
    p.add_argument("--baseline", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--edits", required=True)
    p.add_argument("--out-tasks", required=True)
    p.add_argument("--out-audit", required=True)
    p.add_argument("--reads", type=int, default=1)
    p.add_argument("--original-reads", type=int, default=2)
    p.set_defaults(fn=prepare)
    p = sub.add_parser("score", help="Produit les contrastes à partir des sorties parsées")
    p.add_argument("--tasks", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=score)
    p = sub.add_parser("evaluate", help="Seulement APRES verrouillage des scores")
    p.add_argument("--scores", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tranche", type=float, default=.2)
    p.set_defaults(fn=evaluate)
    args = parser.parse_args()
    if args.command == "select" and (args.n_positive < 0 or args.n_negative < 0 or args.max_per_doc < 1):
        die("Effectifs de sélection invalides")
    if args.command == "prepare" and (args.reads < 1 or args.original_reads < 1):
        die("Nombre de lectures invalide")
    if args.command == "evaluate" and not (0 < args.tranche <= 1):
        die("Tranche doit être entre 0 et 1")
    args.fn(args)


if __name__ == "__main__":
    main()
