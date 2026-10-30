"""Label-free replay of exact document stopping; no model clients."""
import base64
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import zipfile

HERE = Path(__file__).resolve().parent


def bound(a, b, da, db, k):
    lo = (a / 7 + b / 7 + da + db) / 4
    hi = ((a + 7-k) / 7 + (b + 7-k) / 7 + da + db) / 4
    return True if lo >= .39 else False if hi < .39 else None


def replay(d):
    assert not ({"gold", "labels", "relations"} & set(d))
    allowed = set(map(tuple, d["pairs"]))
    assert allowed and len(allowed) == len(d["pairs"])
    assert all((b, a) in allowed and a != b for a, b in allowed)
    assert len(d["positive_pairs_by_slot"]) == len(d["response_ids"]) == 7
    assert len(set(d["response_ids"])) == 7
    ps = [set(map(tuple, p)) for p in d["positive_pairs_by_slot"]]
    assert all(p <= allowed for p in ps)
    scores = {(a, b): v for a, b, v in d["d1"]}
    assert set(scores) == allowed and all(0 <= v <= 1 for v in scores.values())
    counts = {p: sum(p in s for s in ps) for p in allowed}
    final = {p: (counts[p]/7 + counts[p[::-1]]/7 + scores[p] + scores[p[::-1]])/4 >= .39 for p in allowed}
    by_prefix = []
    stop = None
    for k in range(8):
        c = {p: sum(p in s for s in ps[:k]) for p in allowed}
        fixed = {p: bound(c[p], c[p[::-1]], scores[p], scores[p[::-1]], k) for p in allowed}
        assert all(v == final[p] for p, v in fixed.items() if v is not None)
        n = sum(v is not None for v in fixed.values())
        by_prefix.append(n)
        if n == len(allowed) and stop is None:
            stop = k
    assert stop is not None and stop <= 7
    total = sum(u["prompt_tokens"] + u["completion_tokens"] for u in d["usages"])
    used = sum(u["prompt_tokens"] + u["completion_tokens"] for u in d["usages"][:stop])
    return {"id": d["id"], "stop_after": stop, "ordered_candidates": len(allowed),
            "fixed_pairs_by_prefix": by_prefix, "decisions_changed": 0,
            "historical_tokens": total, "prefix_tokens": used}


def main():
    assert os.environ.get("ORX_RUN_ID"), "tracked execution required"
    socket.socket = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("N11 network disabled"))
    raw = (HERE/"inputs.json").read_bytes()
    protocol = json.loads((HERE/"protocol.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == protocol["input_sha256"]
    data = json.loads(raw)
    cohorts = {}
    details = {}
    for c in data["cohorts"]:
        rows = [replay(d) for d in c["documents"]]
        details[c["name"]] = rows
        n = len(rows)
        calls = sum(r["stop_after"] for r in rows)
        tokens = sum(r["historical_tokens"] for r in rows)
        prefix = sum(r["prefix_tokens"] for r in rows)
        saving = 1 - calls/(7*n)
        cohorts[c["name"]] = {
            "documents": n, "historical_calls": 7*n, "calls_used": calls,
            "calls_saved": 7*n-calls, "call_saving_fraction": saving,
            "stop_histogram": dict(sorted(collections.Counter(r["stop_after"] for r in rows).items())),
            "documents_fixed_by_prefix": [sum(r["stop_after"] <= k for r in rows) for k in range(8)],
            "historical_tokens": tokens, "prefix_tokens": prefix,
            "historical_token_saving_fraction": 1-prefix/tokens if tokens else 0,
            "decisions_changed": 0, "worth_live_validation_at_20percent_calls": saving >= .2}
    ids = {d["id"] for d in data["ESC233_metadata"]}
    summaries = {d["id"] for d in data["ordinary7_summary_inventory"]}
    result = {
        "study": "N11", "status": "COMPLETE_LABEL_FREE",
        "cohorts": cohorts, "ESC233_documents": len(ids),
        "ESC233_topics": len({d["topic"] for d in data["ESC233_metadata"]}),
        "ordinary7_summary_documents_all_cohorts": len(summaries),
        "ordinary7_summary_documents_in_ESC233": len(ids & summaries),
        "ESC233_without_ordinary7_summary": sorted(ids-summaries),
        "approved_reusable_requests_for_future_fresh_full": 0,
        "cache_reuse_status": "summary coverage only; exact future bodies and receipts not yet approved",
        "new_model_calls": 0, "new_API_usd": 0, "training": False,
        "human_annotations": 0, "target_gold_read": False,
        "limitations": ["cohorts are kept separate and are exposed",
                       "token prefixes are historical counterfactuals, not standalone invoices",
                       "d1 generation cost fixed and excluded from savings",
                       "small and medium document caches do not represent full long-document behavior"]}
    output = {"result.json": json.dumps(result, ensure_ascii=False, indent=2).encode(),
              "per-document.json": json.dumps(details, ensure_ascii=False, indent=2).encode(),
              "protocol.json": (HERE/"protocol.json").read_bytes(),
              "inputs.json": raw, "main.py": Path(__file__).read_bytes(),
              "test_n11.py": (HERE/"test_n11.py").read_bytes()}
    manifest = {"run": os.environ["ORX_RUN_ID"], "commit": os.environ["ORX_COMMIT"],
                "files_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in output.items()}}
    output["manifest.json"] = json.dumps(manifest, indent=2).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in output.items():
            z.writestr(k, v)
    blob = buf.getvalue()
    encoded = base64.b64encode(blob).decode()
    print("N11_RESULT "+json.dumps(result), flush=True)
    print("N11_EVIDENCE_BEGIN "+json.dumps({"zip_sha256": hashlib.sha256(blob).hexdigest(),
          "base64_chars": len(encoded), "chunks": (len(encoded)+5999)//6000}), flush=True)
    for i in range(0, len(encoded), 6000):
        print("N11_EVIDENCE_CHUNK "+json.dumps({"index": i//6000, "data": encoded[i:i+6000]}), flush=True)
    print("N11_EVIDENCE_END", flush=True)


if __name__ == "__main__":
    main()
