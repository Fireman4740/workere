"""Strict provenance registry and explicit historical parser adapters."""
from collections import defaultdict
from .contracts import STATUSES, digest, require

COMPATIBILITY = ("document_sha256", "model_identity", "template_sha256", "parameters",
                 "messages", "parser", "view_sha256", "auxiliary_sha256", "seed")


def compatibility_key(record):
    require(all(k in record for k in COMPATIBILITY), "incomplete cache provenance")
    return digest({k: record[k] for k in COMPATIBILITY})


def index(records):
    result, seen = [], set()
    for record in records:
        require(record["status"] in STATUSES, "unknown response status")
        realization = tuple(record[k] for k in ("logical_id", "calendar_id", "slot_id", "replicate_id"))
        require(realization not in seen, "duplicate realization; explicit reuse required")
        seen.add(realization)
        result.append(dict(record, compatibility_key=compatibility_key(record),
                           weights_identity_verified=bool(record["model_identity"].get("weights_sha256"))))
    return result


def reuse(source, target, *, mode):
    require(mode in ("resume_slot", "exact_control"), "explicit cache operation required")
    require(source["status"] == "valid" and compatibility_key(source) == compatibility_key(target), "incompatible cache")
    if mode == "resume_slot":
        require(all(source[k] == target[k] for k in ("calendar_id", "slot_id", "replicate_id", "logical_id")), "different realization")
    return dict(target, status="valid", relations=source["relations"], reused_from=source["logical_id"], reuse_mode=mode)


def historical_n15(rows, events, documents, spec):
    """Read N15 with its actual frozen parser, rejecting incomplete responses."""
    from pipelines.jev.n13 import verify_response
    by = {e["logical_id"]: e for e in events}
    require(len(by) == len(events), "duplicate historical event")
    output = []
    for row in rows:
        event = by.get(row["logical_id"])
        status, relations = "not_submitted", []
        if event:
            require(event["request"] == row["body"], "historical request mismatch")
            if event.get("error"):
                status = "transport_failed" if event.get("response") is None else "invalid_schema"
            else:
                try:
                    d = documents[row["doc"]]
                    relations = sorted(verify_response(event["response"], set(map(tuple, d["pair_list"]))))
                    status = "valid"
                except (ValueError, AssertionError, KeyError, TypeError):
                    status = "invalid_schema"
            choices = (event.get("response") or {}).get("choices", [])
            if choices and choices[0].get("finish_reason") == "length":
                status = "truncated"
        d = documents[row["doc"]]
        output.append(dict(logical_id=row["logical_id"], doc_id=row["doc"], family=row["condition"],
            calendar_id="n15", slot_id=row["slot"], replicate_id=0, status=status, relations=relations,
            document_sha256=digest(dict(revision=spec["revision"], tokens=d["tokens"], candidates=d["pair_list"])),
            model_identity=dict(served_name=spec["model"], weights_sha256=None),
            template_sha256=None, parameters={k:v for k,v in row["body"].items() if k != "messages"},
            messages=row["body"]["messages"], parser="n13.verify_response@a3242f7",
            view_sha256=digest(d["text"]), auxiliary_sha256=digest(None), seed=None))
    return index(output)


def paired_cohort(readers, documents):
    """Only identical document/candidate fingerprints enter a paired cohort."""
    valid = {}
    for name, rows in readers.items():
        rs = [r for r in rows if r["status"] == "valid"]
        require(len({r["doc_id"] for r in rs}) == len(rs), "name each reader/calendar/slot explicitly; duplicate document")
        valid[name] = {r["doc_id"]:r for r in rs}
    common = set(documents)
    for bank in valid.values():
        common &= set(bank)
    excluded = {}
    for d in list(common):
        if len({bank[d]["document_sha256"] for bank in valid.values()}) != 1:
            common.remove(d)
            excluded[d] = "revision/tokens/candidates mismatch"
    for d in set(documents) - common:
        excluded.setdefault(d, "missing or invalid reading")
    return dict(intersection=sorted(common), exclusions=excluded, corpus_complete=len(common) == len(documents))


def historical_n14(root):
    """Reconstruct full per-slot readings from the versioned N14 block parser.

    A missing/invalid block invalidates the whole reading, including when all
    positive links happen to appear in the other blocks.
    """
    import json
    from pathlib import Path
    from pipelines.jev.n14 import validate
    root = Path(root)
    read = lambda p:json.loads(p.read_text())
    rows = [json.loads(l) for l in (root/"requests.jsonl").read_text().splitlines()]
    docs = {d["id"]:d for d in read(root/"public.json")}
    spec = read(root/"protocol.json")
    grouped, receipts = defaultdict(list), {}
    for condition in ("M00","M10","M01","M11"):
        path = root/condition/"events.jsonl"
        events = [json.loads(l) for l in path.read_text().splitlines()]
        require(len({e["logical_id"] for e in events}) == len(events), "duplicate N14 receipt")
        receipts.update({e["logical_id"]:e for e in events})
    for row in rows:
        grouped[row["condition"],row["arm"],row["schedule"],row["slot"],row["doc"]].append(row)
    output = []
    for (condition,arm,calendar,slot,doc),requests in sorted(grouped.items()):
        coverage, positives, errors, evidence, response_ids = set(),set(),[],[],[]
        for row in requests:
            allowed = {tuple(sorted(p)) for p in row["allowed"]}
            require(not coverage.intersection(allowed), "overlapping N14 blocks")
            coverage.update(allowed)
            event = receipts.get(row["logical_id"])
            if not event:
                errors.append("not_submitted"); continue
            require(event["request"] == row["body"], "N14 request mismatch")
            evidence.append(dict(logical_id=row["logical_id"],request_sha256=digest(row["body"]),response_sha256=digest(event["response"])))
            try:
                require(not event.get("error"), "historical invalid response")
                positives.update(validate(event["response"],row))
                response_ids.append(event["response"]["id"])
            except (ValueError,AssertionError,KeyError,TypeError):
                errors.append("invalid_schema")
        require(coverage == {tuple(sorted(p)) for p in docs[doc]["pair_list"]}, "N14 candidate coverage mismatch")
        require(len(response_ids) == len(set(response_ids)), "duplicate N14 realization")
        output.append(dict(logical_id=f"n14:{condition}:{arm}:{calendar}:{slot}:{doc}",doc_id=doc,
            family=condition+":"+arm,calendar_id=str(calendar),slot_id=slot,replicate_id=0,
            status="valid" if not errors else errors[0], relations=sorted(positives) if not errors else None,
            document_sha256=digest(dict(revision=spec["revision"],tokens=docs[doc]["tokens"],candidates=sorted(coverage))),
            model_identity=dict(served_name=spec["model"],weights_sha256=None),parser="n14.validate@93b2a33",
            evidence=evidence, errors=errors))
    return dict(readings=output,public=list(docs.values()),revision=spec["revision"],source=str(root),source_run="208d3236-fa28-438c-be67-62e6552f1495")
