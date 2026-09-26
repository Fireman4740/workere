"""Run directory, checkpoint, git state and MLflow.

One directory per run, `<runtime.logdir>/<stem>/`, stem = `run_<utc timestamp>_<8 hex>`:

  <stem>.json          the report payload (agentere schema, so agentere scripts read it)
  <stem>.config.yaml   the exact merged config — `--resume` reloads it from here
  <stem>.traces.jsonl  one line per LLM attempt (see llm.TraceWriter)
  <stem>.diff.patch    working-tree diff at launch, when the tree is dirty
  docs.jsonl           checkpoint: one line per finished document, appended as it finishes
  mlflow_run_id        the MLflow run to reopen on resume
  graphs/, figures/    causal graphs and histograms

Why the checkpoint: in agentere a run that died wrote nothing. Of the 113 agentere runs with a
trace file, 44 never produced a report — $21.37 of $135.34 spent (16%) and 7,684 of 38,137
calls — and all 46 of its FAILED or orphaned-RUNNING MLflow runs have zero params, because params
were logged only at the very end. Here params are logged when the run starts and every finished
document is on disk, so a crashed run is resumed, not repaid.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from eci.config import ROOT, flatten_params


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_stem() -> str:
    ts = now_iso().replace("+00:00", "Z").replace(":", "-").replace("+", "p")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"


def git_state(repo: Path = ROOT) -> Dict[str, Any]:
    def git(*args):
        try:
            p = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
            return p.returncode, p.stdout.strip()
        except FileNotFoundError:
            return 127, ""
    rc, head = git("rev-parse", "HEAD")
    if rc != 0:
        return {"is_git_repo": False, "error": "not a git repository"}
    _, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    _, porcelain = git("status", "--porcelain")
    _, diff = git("diff")
    return {"is_git_repo": True, "commit": head, "branch": branch, "dirty": bool(porcelain), "diff": diff}


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return json.loads(json.dumps(obj, default=str))


class RunDir:
    def __init__(self, path: Path):
        self.path = path
        self.stem = path.name
        self.path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, logdir: str) -> "RunDir":
        return cls(Path(logdir) / new_stem())

    def file(self, suffix: str) -> Path:
        return self.path / f"{self.stem}{suffix}"

    @property
    def checkpoint(self) -> Path:
        return self.path / "docs.jsonl"

    def write_config(self, cfg: dict) -> None:
        self.file(".config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                             encoding="utf-8")

    def read_config(self) -> dict:
        return yaml.safe_load(self.file(".config.yaml").read_text(encoding="utf-8"))

    # ── checkpoint ────────────────────────────────────────────────────────────

    def append_doc(self, result: dict) -> None:
        with open(self.checkpoint, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    def load_docs(self) -> Dict[str, dict]:
        """Finished documents, last line per id wins (a retried failure supersedes the failure).
        A torn last line — the process killed mid-write — is ignored."""
        out: Dict[str, dict] = {}
        if not self.checkpoint.exists():
            return out
        for line in self.checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in ("gold_triples", "pred_triples"):
                r[k] = [tuple(t) for t in r[k]]
            out[r["id"]] = r
        return out

    # ── report ────────────────────────────────────────────────────────────────

    def write_report(self, cfg: dict, results: dict, git: dict, status: str,
                     error: Optional[str]) -> Path:
        payload = {
            "run_id": self.stem.rsplit("_", 1)[-1],
            "timestamp_utc": now_iso(),
            "status": status,
            "framework": "workere",
            "cli_args": sys.argv,
            "config": _jsonable(cfg),
            "results": _jsonable(results),
            "git": _jsonable({k: v for k, v in git.items() if k != "diff"}),
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "executable": sys.executable, "cwd": str(Path.cwd())},
            "error": error,
        }
        path = self.file(".json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_git(self, git: dict) -> None:
        if git.get("is_git_repo") and git.get("diff"):
            self.file(".diff.patch").write_text(git["diff"], encoding="utf-8")


# ── MLflow ────────────────────────────────────────────────────────────────────

class Tracker:
    """MLflow for one run. A no-op when mlflow.enabled is false."""

    def __init__(self, cfg: dict, run_dir: RunDir):
        self.enabled = cfg["mlflow"]["enabled"]
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_id: Optional[str] = None
        if not self.enabled:
            return
        import mlflow
        import mlflow.utils.validation as val
        # Stored in full rather than silently truncated.
        val.MAX_TAG_VAL_LENGTH = val.MAX_TRACE_TAG_VAL_LENGTH = val.MAX_PARAM_VAL_LENGTH = 10_000_000
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    def start(self, git: dict) -> None:
        """Start (or, on resume, reopen) the MLflow run, params first."""
        if not self.enabled:
            return
        import mlflow
        id_file = self.run_dir.path / "mlflow_run_id"
        if id_file.exists():
            run = mlflow.start_run(run_id=id_file.read_text().strip())
        else:
            run = mlflow.start_run(run_name=self.run_dir.stem)
            id_file.write_text(run.info.run_id)
            mlflow.log_params(flatten_params(self.cfg))
            tags = {"framework": "workere", "run_dir": str(self.run_dir.path)}
            if git.get("is_git_repo"):
                tags.update({"git.commit": git["commit"], "git.branch": git["branch"],
                             "git.dirty": str(git["dirty"])})
            mlflow.set_tags(tags)
        self.run_id = run.info.run_id

    def progress(self, done: int) -> None:
        if self.enabled:
            import mlflow
            mlflow.log_metric("docs_done", done, step=done)

    def graph(self, svg_path: Path) -> None:
        if self.enabled and self.cfg["mlflow"]["log_causal_graphs"]:
            import mlflow
            mlflow.log_artifact(str(svg_path), artifact_path="causal_graphs")

    def finish(self, report: dict, usage: dict, cost: Optional[float], artifacts: List[Path],
               status: str) -> None:
        if not self.enabled:
            return
        import mlflow
        m = {k: report[k] for k in ("micro_f1", "macro_f1", "micro_precision", "micro_recall",
                                    "total_pairs", "skipped_docs", "retry_failed_docs",
                                    "sentence_unknown_pairs", "sentence_unknown_pct")}
        for scope in ("binary", "binary_intra", "binary_inter"):
            prefix = "binary" if scope == "binary" else scope
            for k in ("f1", "precision", "recall"):
                m[f"{prefix}_{k}"] = report[scope][k]
        for label, lm in report["per_label"].items():
            safe = label.lower().replace(" ", "_")
            for k in ("f1", "precision", "recall"):
                m[f"{safe}_{k}"] = lm[k]
        for lang, lm in report["per_lang_metrics"].items():
            safe = lang.lower().replace(" ", "_")
            mc = lm["multiclass"]
            for k in ("micro_f1", "macro_f1", "micro_precision", "micro_recall"):
                m[f"lang_{safe}_{k}"] = mc[k]
            m[f"lang_{safe}_total_pairs"] = lm["total_pairs"]
        t = usage["total"]
        m.update({"tokens_input": t["input_tokens"], "tokens_output": t["output_tokens"],
                  "tokens_cache_read": t["cache_read_tokens"], "tokens_reasoning": t["reasoning_tokens"],
                  "llm_calls": t["calls"], "llm_failed_attempts": t["failed_attempts"]})
        if cost is not None:
            m["cost_usd"] = cost
        if "llm_cache" in usage:
            m["llm_cache_hits"] = usage["llm_cache"]["hits"]
            m["llm_cache_misses"] = usage["llm_cache"]["misses"]
        mlflow.log_metrics({k: float(v) for k, v in m.items()})
        for path in artifacts:
            if path.exists():
                mlflow.log_artifact(str(path))
        mlflow.end_run(status=status)

    def abort(self) -> None:
        if self.enabled:
            import mlflow
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
