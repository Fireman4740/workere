"""J1: retain Luna coreference links only when a second, DeepSeek judge agrees.

Luna's already paid R3-T2 replies are read from SQLite; only missing replies are completed.
Gold is accessed only after judging; the 0.85 intrinsic precision gate precedes
any causal scoring. Connected-component closure is included in that evaluation.
"""
import json
import random
import sqlite3
from pathlib import Path

from pipelines.judges.common import (
    Dev44, Scorer, Spend, chain_links, chunks, client_from, compare, components,
    gold_chains, last_json, link_quality, mention_pattern, propagate, run_parallel,
)

JUDGE_SYSTEM = (
    "You verify event coreference in English news (ECB+ guidelines). Event mentions are marked inline "
    "as <ID text>. For each listed pair, decide whether both mentions denote the SAME real-world "
    "event instance (same action, time, place and participants). Answer false for different instances "
    "of the same event type, a sub-event and its larger event, a cause and its effect, a report about "
    "an event and that event, or merely related events. Be strict: true requires clear textual identity. "
    "When unsure answer false. Reason, then output the final answer as a JSON array with exactly one "
    'object per listed pair: [{"pair": "A,B", "same": true}, ...].'
)
JUDGE_USER = "Document {doc_id}:\n{doc_text}\n\nPairs to verify:\n{pair_lines}"


def parse_groups(content, body):
    ids = mention_pattern(body["messages"][-1]["content"])
    seen, out = set(), []
    for group in last_json(content, list):
        if not isinstance(group, list) or not all(type(m) in (str, int) for m in group):
            raise ValueError("groups must be arrays of mention ids")
        members = [str(m).strip() for m in group]
        if set(members) - ids:
            raise ValueError("unknown mention ids in group")
        if seen & set(members) or len(set(members)) != len(members):
            raise ValueError("a mention appears in multiple groups or twice")
        seen |= set(members)
        if len(members) >= 2:
            out.append(sorted(members))
    return out


def requested_pairs(body):
    text = body["messages"][-1]["content"].split("Pairs to verify:\n", 1)[1]
    return {",".join(sorted(line[2:].split(" ", 1)[0].split(",")))
            for line in text.splitlines() if line.startswith("- ")}


def parse_judgements(content, body):
    wanted, out = requested_pairs(body), {}
    for row in last_json(content, list):
        if (not isinstance(row, dict) or not isinstance(row.get("same"), bool)
                or not isinstance(row.get("pair"), str)):
            raise ValueError("each judgement needs a string pair and boolean same")
        key = ",".join(sorted(row["pair"].replace(" ", "").split(",")))
        if key not in wanted:
            raise ValueError(f"pair not requested: {row['pair']}")
        if key in out:
            raise ValueError(f"duplicate judgement for {key}")
        out[key] = row["same"]
    if set(out) != wanted:
        raise ValueError(f"missing judgements: {sorted(wanted - set(out))[:10]}")
    return out


def luna_groups(data, cache_dir, complete_missing=None):
    """Read validated Luna replies only; refuse missing or mismatched cached requests."""
    result = {}
    for doc in data.docs:
        path = Path(cache_dir).expanduser() / f"r3-t2-coref-{doc}.sqlite"
        def read():
            if not path.is_file():
                return []
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                return db.execute("SELECT request,response FROM decisions_v1").fetchall()
        rows = read()
        if not rows and complete_missing is not None:
            complete_missing(doc, path)
            rows = read()
        matches = []
        for request, response in rows:
            request, response = json.loads(request), json.loads(response)
            if request.get("model") != "openai/gpt-6-luna:batch":
                continue
            text = request["messages"][-1]["content"]
            if not text.startswith(f"Document {doc}:\n"):
                continue
            if mention_pattern(text) != set(data.mention_ids(doc)):
                raise ValueError(f"Luna cache mention inventory differs for {doc}")
            choice = (response.get("choices") or [{}])[0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"incomplete cached Luna reply for {doc}")
            if not response.get("model", "").startswith("openai/gpt-6-luna"):
                raise ValueError(f"wrong cached Luna served model for {doc}")
            matches.append(parse_groups(choice["message"]["content"], request))
        if len(matches) != 1:
            raise ValueError(f"expected one paid Luna coref reply for {doc}, got {len(matches)}")
        result[doc] = matches[0]
    return result


def jobs_for(data, luna, cfg):
    jobs = []
    for doc in data.docs:
        pairs = sorted(chain_links(luna[doc]))
        random.Random(f"J1::{doc}").shuffle(pairs)
        for index, part in enumerate(chunks(pairs, cfg["pairs_per_call"])):
            surface = data.suite[doc]["surfaces"]
            lines = "\n".join(f"- {a},{b} ({surface.get(a, '')!r} / {surface.get(b, '')!r})" for a, b in part)
            user = JUDGE_USER.format(doc_id=doc, doc_text=data.marked_text(doc), pair_lines=lines)
            jobs.append((doc, index, user))
    return jobs


def run(cfg):
    data = Dev44(cfg["controls"], cfg["suites_dir"])
    luna_spend = {"usd": 0.0, "completed_missing_documents": []}

    def complete_missing(doc, path):
        from pipelines.jev.coref import _client_class, SYSTEM, USER
        client = _client_class()(**{**cfg["luna_client"], "cache_path": str(path),
                                   "shared_budget": cfg["budget"]})
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.format(doc_id=doc, doc_text=data.marked_text(doc))}]
        def record(event):
            luna_spend["usd"] += event["cost_usd"] or 0
            print("J1_LUNA_COMPLETE " + json.dumps({"doc": doc, "cache_hit": event["cache_hit"],
                                                   "cost_usd": event["cost_usd"]}), flush=True)
        try:
            client.decide_many([messages], [record])
        finally:
            client.close()
        luna_spend["completed_missing_documents"].append(doc)

    luna = luna_groups(data, cfg["luna_cache_dir"], complete_missing=complete_missing)
    jobs = jobs_for(data, luna, cfg)
    print("J1_CANDIDATES " + json.dumps({"luna_documents": len(luna), "luna": luna_spend,
                                        "candidate_links": sum(len(chain_links(v)) for v in luna.values()),
                                        "deepseek_calls": len(jobs)}), flush=True)
    client, spend = client_from(cfg, parse_judgements), Spend()

    def job(doc, index, user):
        def call():
            reply = client.ask(JUDGE_SYSTEM, user, index + 1)
            spend.add(reply)
            answers = parse_judgements(reply["content"], {"messages": [{"content": user}]})
            print("J1_JUDGE " + json.dumps({"doc": doc, "chunk": index, "pairs": len(answers),
                                            "confirmed": sum(answers.values()),
                                            "cost_usd": reply["cost_usd"]}), flush=True)
            return doc, answers
        return call

    try:
        replies = run_parallel([job(*j) for j in jobs], cfg["workers"])
    finally:
        client.close()
    confirmed = {d: [] for d in data.docs}
    for doc, answers in replies:
        confirmed[doc].extend(tuple(k.split(",")) for k, same in answers.items() if same)
    agreed = {d: components(confirmed[d], data.mention_ids(d)) for d in data.docs}
    return score(cfg, data, luna, agreed, {**spend.value, "luna": luna_spend})


def score(cfg, data, luna, agreed, spend):
    """Gold enters here. Refuse causal evaluation if intrinsic precision fails."""
    gold = gold_chains(set(data.docs), cfg["dataset"]["repo_id"], cfg["dataset"]["revision"])
    intrinsic = {"luna": link_quality(luna, gold, data.docs),
                 "luna_and_deepseek": link_quality(agreed, gold, data.docs)}
    p = intrinsic["luna_and_deepseek"]["precision"]
    passed = p is not None and p >= cfg["gate_precision"]
    out = {"schema": "workere-judge-result/v1", "experiment": "J1 Luna x DeepSeek coref agreement",
           "documents": len(data.docs), "gate_precision": cfg["gate_precision"], "gate_passed": passed,
           "spend": spend, "chains": intrinsic, "agreed_groups": agreed,
           "causal_evaluation": "evaluated" if passed else "skipped_intrinsic_gate_failed"}
    if not passed:
        return out
    scorer = Scorer(data)
    base = scorer.fixed(lambda d: data.fused[d])
    scores = {d: propagate(data.fused[d], agreed[d]) for d in data.docs}
    fixed = scorer.fixed(lambda d: scores[d])
    lto, choices = scorer.lto({"luna_and_deepseek": lambda d: scores[d]})
    out.update(frozen_fusion=compare(base, base, data.docs),
               primary_fixed_0_39=compare(fixed, base, data.docs),
               secondary_threshold_lto=compare(lto, base, data.docs),
               threshold_choices_by_topic=choices,
               per_document={d: {"base": base[d], "primary": fixed[d], "secondary": lto[d]}
                             for d in data.docs})
    return out
