"""Reading trace files: pretty printing, and token spend per document.

A workere trace line is a self-describing record ({role, doc_id, pass, attempt, ok, error,
messages}); an agentere trace line is the bare message list. Both are read.
"""
from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Optional

COLORS = {"SystemMessage": "\033[35m", "HumanMessage": "\033[34m", "AIMessage": "\033[32m",
          "ToolMessage": "\033[33m"}
RESET = "\033[0m"


def records(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, list):
                yield {"role": "?", "doc_id": "?", "pass": None, "attempt": None, "ok": True,
                       "error": None, "messages": obj}
            else:
                yield obj


def usage(msg: dict) -> dict:
    kw = msg.get("kwargs", {})
    u = kw.get("usage_metadata") or {}
    cost = ((kw.get("response_metadata") or {}).get("token_usage") or {}).get("cost")
    return {"in": u.get("input_tokens", 0) or 0, "out": u.get("output_tokens", 0) or 0,
            "reasoning": ((u.get("output_token_details") or {}).get("reasoning", 0)) or 0,
            "cost": cost or 0.0}


def show(path: Path, doc: Optional[str] = None, role: Optional[str] = None, limit: int = 3,
         width: int = 110, full: bool = False, color: bool = True) -> str:
    out = []
    shown = 0
    for rec in records(path):
        if doc and rec["doc_id"] != doc or role and not str(rec["role"]).startswith(role):
            continue
        shown += 1
        if shown > limit:
            break
        head = (f"#{shown} role={rec['role']} doc={rec['doc_id']} pass={rec['pass']} "
                f"attempt={rec['attempt']} ok={rec['ok']}" + (f" error={rec['error']}" if rec["error"] else ""))
        out.append("=" * min(width, len(head)) + "\n" + head)
        for m in rec["messages"]:
            kind = m["id"][-1]
            content = m.get("kwargs", {}).get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            if not full and len(content) > 1500:
                content = content[:700] + f"\n  [... {len(content) - 1400} chars ...]\n" + content[-700:]
            u = usage(m)
            tag = f"{kind}" + (f"  (in {u['in']}, out {u['out']}, reasoning {u['reasoning']})" if kind == "AIMessage" else "")
            tag = f"{COLORS.get(kind, '')}{tag}{RESET}" if color else tag
            body = "\n".join(textwrap.fill(p, width) if len(p) > width else p for p in content.splitlines())
            out.append(f"--- {tag}\n{body}")
            for tc in m.get("kwargs", {}).get("tool_calls", []) or []:
                out.append(f"    tool call: {tc.get('name')}({json.dumps(tc.get('args'))[:300]})")
    return "\n".join(out)


def spend_per_doc(path: Path) -> Dict[str, dict]:
    """{doc_id: {calls, in, out, reasoning, cost}} over inference calls (workere traces)."""
    per = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "reasoning": 0, "cost": 0.0})
    for rec in records(path):
        if rec["role"] != "inference":
            continue
        for m in rec["messages"]:
            # demo turns are AIMessages too, but carry no usage: only real responses count
            if m["id"][-1] == "AIMessage" and m.get("kwargs", {}).get("usage_metadata"):
                u = usage(m)
                p = per[rec["doc_id"]]
                p["calls"] += 1
                for k in ("in", "out", "reasoning", "cost"):
                    p[k] += u[k]
    return dict(per)
