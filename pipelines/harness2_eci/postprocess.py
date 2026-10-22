"""Run a harness's optional `scripts/postprocess.py` over raw pair scores, in a subprocess.

Contract (stated to the harness editor): stdin is one JSON object
  {"document_id", "executor", "mentions": [{"id", "surface", "sentence"}],
   "pairs": [{"source", "target", "probability", "default_label", "votes"?}]}
and stdout must be one JSON object {"labels": {"<source>,<target>": "<label>"}} giving a label
for any pair whose default label should change. Allowed labels are the executor's label set.
A missing script keeps every default label; a failing script is an execution error that the
rollout reports in its trajectory, and the defaults are kept.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pipelines.harness2_eci.surface import POSTPROCESS

TIMEOUT_SECONDS = 60
LABELS = {"luna": {"causes", "causedby", "norel"}, "decision": {"causal", "norel"},
          "fusion": {"causal", "norel"}}


def apply(tree, executor, payload):
    """(labels by "a,b", log line or ""). Never raises for a faulty harness script."""
    defaults = {f"{p['source']},{p['target']}": p["default_label"] for p in payload["pairs"]}
    script = tree.get(POSTPROCESS)
    if script is None:
        return defaults, ""
    with tempfile.TemporaryDirectory(prefix="h2-post-") as tmp:
        path = Path(tmp) / "postprocess.py"
        path.write_text(script, encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, str(path)], input=json.dumps(payload), cwd=tmp,
                                  capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return defaults, f"[ERROR] {POSTPROCESS} timed out after {TIMEOUT_SECONDS}s; default labels kept"
    if done.returncode != 0:
        return defaults, (f"[ERROR] {POSTPROCESS} exited {done.returncode}; default labels kept\n"
                          f"{done.stderr[-2000:]}")
    try:
        changes = json.loads(done.stdout)["labels"]
        if not isinstance(changes, dict):
            raise TypeError("labels must be an object")
    except (ValueError, KeyError, TypeError) as exc:
        return defaults, f"[ERROR] {POSTPROCESS} stdout is not {{\"labels\": {{...}}}} ({exc}); defaults kept"
    allowed, out, rejected = LABELS[executor], dict(defaults), []
    for key, label in changes.items():
        if key in defaults and label in allowed:
            out[key] = label
        else:
            rejected.append(key)
    changed = sum(out[k] != defaults[k] for k in out)
    log = f"[OBSERVATION] {POSTPROCESS} changed {changed} of {len(out)} pair labels"
    if rejected:
        log += f"; ignored {len(rejected)} entries with unknown pairs or labels (e.g. {rejected[:3]})"
    return out, log
