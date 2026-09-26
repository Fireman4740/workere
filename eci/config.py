"""Configuration: config.yaml, queue overrides, the active prompt file, and up-front validation.

Every setting is declared once, in config.yaml, and read by direct indexing
(`cfg["experiment"]["retries"]`) — never with a `.get(key, default)` that would be a second,
invisible declaration. Overrides (queue entries, `--set`) are deep-merged strictly: a key that
config.yaml does not declare is an error, so a typo fails at load time instead of silently
running with a value nobody chose.

Validation of enumerated values happens here too, before any document is loaded or any call is
made. agentere checked most of them lazily, so a typo'd `syntax.level` or `few_shot.selection`
surfaced minutes into a run, after the CoT pre-generation had already been paid for.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

SYNTAX_LEVELS = ("off", "mentions", "args", "paths")
DISCOURSE_COMPONENTS = ("skeleton", "participants", "pos")
SELECTIONS = ("random", "similarity", "mentions", "bert", "smallest")
WHITELIST_MODES = ("oracle", "file")
TIE_BREAKS = ("norel", "random")
BINARY_DEFAULTS = ("norel", "causal")
COREF_SOURCES = ("esc_ecb", "maven_span")
PROMPT_KEYS = {"system", "user_template", "user_suffix", "steps"}

# Sections that describe where and how a run is executed, never what it predicts. They are
# left out of the MLflow params so that two runs differing only in, say, a log directory still
# count as the same experiment when ablations are mined.
NON_PARAM_SECTIONS = {"mlflow", "rule_sets", "datasets", "prompts", "runtime"}


class ConfigError(ValueError):
    pass


def deep_merge(base: Dict, overrides: Dict, _path: str = "") -> Dict:
    """Recursively merge `overrides` into a copy of `base`. Unknown keys raise ConfigError."""
    result = dict(base)
    for k, v in overrides.items():
        full = f"{_path}.{k}" if _path else k
        if k not in result:
            raise ConfigError(f"override key '{full}' does not exist in config.yaml")
        if isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v, full)
        else:
            result[k] = v
    return result


@dataclass(frozen=True)
class PromptSpec:
    """The active prompt file, normalised. `steps` is None for single-call prompts."""
    name: str
    system: str
    user_template: str
    user_suffix: str
    steps: Optional[List[dict]]


def load_prompt(name: str, prompts_dir: Path) -> Dict[str, Any]:
    path = prompts_dir / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"prompt file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    unknown = set(raw) - PROMPT_KEYS
    if unknown:
        raise ConfigError(f"prompt {name}: unknown keys {sorted(unknown)} (allowed: {sorted(PROMPT_KEYS)})")
    for key in ("system", "user_template"):
        if key not in raw:
            raise ConfigError(f"prompt {name}: missing '{key}'")
    return raw


def prompt_spec(cfg: Dict[str, Any]) -> PromptSpec:
    """The prompt file is data, not configuration: `user_suffix` and `steps` are optional
    there, and their absence has a meaning (no suffix, single call)."""
    raw = cfg["prompt"]
    return PromptSpec(
        name=cfg["datasets"][cfg["active_dataset"]]["prompt"],
        system=raw["system"],
        user_template=raw["user_template"],
        user_suffix=(raw.get("user_suffix") or "").strip(),
        steps=raw.get("steps") or None,
    )


def load_config(path: Path | str = DEFAULT_CONFIG, overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """config.yaml, strictly merged with `overrides`, with cfg["prompt"] resolved and validated."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg = copy.deepcopy(cfg)
    active = cfg["active_dataset"]
    if active not in cfg["datasets"]:
        raise ConfigError(f"active_dataset '{active}' is not a key of datasets")
    cfg["prompt"] = load_prompt(cfg["datasets"][active]["prompt"], path.parent / "prompts")
    validate(cfg)
    return cfg


def overrides_from_agentere(snapshot: Dict[str, Any], base: Dict[str, Any]) -> tuple:
    """The part of an agentere `<run>.config.yaml` that workere declares, as overrides.

    Lets any past agentere run be re-run here from its own config snapshot. Skipped: the
    resolved `prompt` (reloaded from prompts/ by name), tracking (`mlflow`), `rule_sets`
    (rewritten here) and keys workere does not declare (e.g. `coref_injection` written by the
    agentere experiment scripts), which are returned so the caller can report them. Keys the
    snapshot lacks — features newer than the run — keep workere's value, which is their "off"
    setting, i.e. the behaviour the run had.
    """
    skip_top = {"prompt", "mlflow", "rule_sets"}
    dropped: List[str] = []

    def walk(snap: Dict, ref: Dict, path: str) -> Dict:
        out = {}
        for k, v in snap.items():
            full = f"{path}.{k}" if path else k
            if (not path and k in skip_top):
                continue
            if k not in ref:
                dropped.append(full)
                continue
            if isinstance(v, dict) and isinstance(ref[k], dict):
                out[k] = walk(v, ref[k], full)
            else:
                out[k] = v
        return out

    return walk(snapshot, base, ""), dropped


def _one_of(value, allowed, key):
    if value not in allowed:
        raise ConfigError(f"{key} = {value!r}; expected one of {list(allowed)}")


def validate(cfg: Dict[str, Any]) -> None:
    """Reject settings that could only fail later, after money has been spent."""
    ds = cfg["datasets"][cfg["active_dataset"]]
    _one_of(cfg["syntax"]["level"], SYNTAX_LEVELS, "syntax.level")
    discourse = cfg["syntax"]["discourse"] or []
    bad = [c for c in discourse if c not in DISCOURSE_COMPONENTS]
    if bad:
        raise ConfigError(f"syntax.discourse has unknown component(s) {bad}; "
                          f"expected any of {list(DISCOURSE_COMPONENTS)}")
    _one_of(cfg["few_shot"]["selection"], SELECTIONS, "few_shot.selection")
    _one_of(cfg["mention_whitelist"]["mode"], WHITELIST_MODES, "mention_whitelist.mode")
    if (cfg["mention_whitelist"]["enabled"] and cfg["mention_whitelist"]["mode"] == "file"
            and not Path(cfg["mention_whitelist"]["path"]).is_file()):
        raise ConfigError(f"mention_whitelist.path not found: {cfg['mention_whitelist']['path']!r}")
    _one_of(cfg["experiment"]["resampling"]["tie_breaking"], TIE_BREAKS,
            "experiment.resampling.tie_breaking")
    _one_of(cfg["data"]["binary_default"], BINARY_DEFAULTS, "data.binary_default")
    if cfg["data"]["binary_undirected"] and cfg["data"]["vote_to_binary"]:
        raise ConfigError("data.binary_undirected and data.vote_to_binary are alternatives; set one")
    if cfg["coref_hint"]["enabled"]:
        _one_of(cfg["coref_hint"]["source"], COREF_SOURCES, "coref_hint.source")
        root = Path(cfg["coref_hint"]["esl_root"])
        if cfg["coref_hint"]["source"] == "esc_ecb" and not (root / "annotated_data").is_dir():
            raise ConfigError(f"coref_hint.esl_root {str(root)!r} holds no ESC corpus "
                              f"(clone github.com/tommasoc80/EventStoryLine there)")
    hr = cfg["few_shot"]["human_rationales"]
    if cfg["few_shot"]["enabled"] and hr["only"] and not (hr["dir"] and Path(hr["dir"]).is_dir()):
        raise ConfigError(f"few_shot.human_rationales.only is set but dir {hr['dir']!r} is not a directory")
    if cfg["experiment"]["concurrency"] < 1 or cfg["experiment"]["doc_concurrency"] < 1:
        raise ConfigError("experiment.concurrency and experiment.doc_concurrency must be >= 1")
    if ds["kfold"]["enabled"] and ds["kfold"]["n_folds"] < 2:
        raise ConfigError(f"datasets.{cfg['active_dataset']}.kfold is enabled with n_folds < 2")


def n_passes(cfg: Dict[str, Any]) -> int:
    rs = cfg["experiment"]["resampling"]
    return rs["n_runs"] if rs["enabled"] else 1


def n_folds(cfg: Dict[str, Any]) -> int:
    """Folds used to exclude same-fold documents from a document's few-shot pool, or 1.

    `kfold` is NOT a train/test split: every document is evaluated and the headline metric is
    pooled over the whole split. The fold (doc_idx % n_folds) only scopes the demo pool.
    """
    kf = cfg["datasets"][cfg["active_dataset"]]["kfold"]
    return kf["n_folds"] if kf["enabled"] else 1


def flatten_params(cfg: Dict[str, Any]) -> Dict[str, str]:
    """MLflow params: the config flattened to dot keys, active dataset under `dataset.`.

    Byte-compatible with agentere's `_flatten_config` (values str()-ed and cut at 500 chars,
    None as ""), so runs from both frameworks can be diffed key by key when ablations are
    mined across the two tracking databases.
    """
    out: Dict[str, str] = {}

    def walk(d: Dict, prefix: str) -> None:
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(v, f"{key}.")
            else:
                out[key] = "" if v is None else str(v)[:500]

    for k, v in cfg.items():
        if k in NON_PARAM_SECTIONS:
            continue
        if isinstance(v, dict):
            walk(v, f"{k}.")
        else:
            out[k] = "" if v is None else str(v)[:500]
    walk(cfg["datasets"][cfg["active_dataset"]], "dataset.")
    return out
