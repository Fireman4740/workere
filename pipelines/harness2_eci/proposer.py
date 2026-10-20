"""Harness² improver sessions on the released OpenCode binary, routed to OpenRouter.

Upstream runs OpenCode from a pinned source checkout through bun and reaches Vertex AI. Here
the same `opencode run` session runs from the npm release, against `openrouter/<model>`,
and every session first reserves a conservative amount on the shared plan budget; the
reservation is settled with OpenCode's reported cost, or retained when no cost is reported.
"""
import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from harness2.core import proposer as upstream
from pipelines.jev.client import BudgetExceeded

OPENCODE_VERSION = "1.14.18"
INSTALL_ROOT = Path("~/.cache/workere-harness2").expanduser()


def opencode_binary(version=OPENCODE_VERSION):
    """The released OpenCode CLI, installed once per host into a versioned cache directory."""
    prefix = INSTALL_ROOT / f"opencode-{version}"
    binary = prefix / "node_modules" / ".bin" / "opencode"
    if not binary.is_file():
        prefix.mkdir(parents=True, exist_ok=True)
        subprocess.run(["npm", "install", "--no-fund", "--no-audit", "--prefix", str(prefix),
                        f"opencode-ai@{version}"], check=True, capture_output=True, text=True, timeout=900)
    found = subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if found != version:
        raise RuntimeError(f"OpenCode {version} expected at {binary}, found {found!r}")
    return binary


def openrouter_key():
    """The launcher may forward the key as OPENROUTER (the decision client accepts both)."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER")
    if not key:
        raise RuntimeError("missing credential: set OPENROUTER_API_KEY (or OPENROUTER)")
    return key


def session_cost(stdout, price_per_million):
    """Cost from the tokens of every completed step; 0 for a session refused before any step.

    None (retain the reservation) when no step completed for any reason other than an
    authentication refusal, since a partially streamed request may still have been charged.
    """
    total, steps, refused = 0.0, 0, False
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        if (part.get("type") or event.get("type")) in ("step-finish", "step_finish"):
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            prompt = sum(int(v or 0) for v in (tokens.get("input"), cache.get("read"), cache.get("write")))
            completion = sum(int(v or 0) for v in (tokens.get("output"), tokens.get("reasoning")))
            priced = (prompt * price_per_million[0] + completion * price_per_million[1]) / 1e6
            reported = part.get("cost") if isinstance(part.get("cost"), (int, float)) else 0.0
            total += max(priced, reported)
            steps += 1
        error = event.get("error") if isinstance(event.get("error"), dict) else {}
        if (error.get("data") or {}).get("statusCode") == 401:
            refused = True
    if steps:
        return round(total, 8)
    return 0.0 if refused else None


@dataclass
class OpenRouterProposer(upstream.Proposer):
    budget: object = None
    session_reserve_usd: float = .3
    price_per_million: tuple = (.1, .5)
    binary: Path = field(default_factory=opencode_binary)
    _route: threading.local = field(default_factory=threading.local, repr=False)

    def env(self):
        env = {k: v for k, v in os.environ.items() if k not in upstream._DROP_ENV}
        env["OPENROUTER_API_KEY"] = openrouter_key()
        return env

    def _session_model(self):
        """(provider-prefixed model, (input, output) USD per million tokens, reservation USD)."""
        return self.spec.model, self.price_per_million, self.session_reserve_usd

    def _config(self, full_model, prices):
        provider, model = full_model.split("/", 1)
        if provider != "openrouter":
            raise ValueError(f"expected an openrouter/<model> id, got {full_model!r}")
        # small_model pins title/summary calls to the same model: no other model is ever called.
        return {"$schema": "https://opencode.ai/config.json", "small_model": full_model,
                "agent": {"build": {"steps": self.spec.steps}},
                "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
                "provider": {"openrouter": {"options": {"apiKey": "{env:OPENROUTER_API_KEY}"}, "models": {model: {
                    "name": model, "tool_call": True, "reasoning": True,
                    "cost": {"input": prices[0], "output": prices[1]},
                    "limit": {"context": 1_050_000, "output": 65_536}}}}}}

    def _opencode(self, cwd, mission, system, wall):
        (cwd / "MISSION.md").write_text((f"{system}\n\n---\n\n" if system else "") + mission, encoding="utf-8")
        (cwd / "output").mkdir(exist_ok=True)
        model, prices, reserve = self._session_model()
        (cwd / "opencode.json").write_text(json.dumps(self._config(model, prices)), encoding="utf-8")
        env = self.env()
        env["OPENCODE_PERMISSION"] = json.dumps(
            {"external_directory": {"*": "deny", str(cwd): "allow", f"{cwd}/**": "allow"}})
        env["XDG_DATA_HOME"], env["XDG_STATE_HOME"] = str(cwd / ".ocdata"), str(cwd / ".ocstate")
        env["XDG_CONFIG_HOME"], env["XDG_CACHE_HOME"] = str(cwd / ".occonfig"), str(INSTALL_ROOT / "xdg-cache")
        env["OPENCODE_DISABLE_CLAUDE_CODE"] = env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"  # no host skills
        prompt = ("Read MISSION.md first. Do what it asks. Reply with the requested output as "
                  "your final message AND save it verbatim to output/answer.md.")
        cmd = [str(self.binary), "run", prompt, "--dir", str(cwd), "--model", model,
               "--format", "json"]
        if self.spec.effort:
            cmd += ["--variant", self.spec.effort]
        ticket = self.budget.reserve(reserve) if self.budget and reserve else None
        cost = None
        try:
            done = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=wall)
            cost = session_cost(done.stdout, prices)
        finally:
            if ticket:
                try:
                    self.budget.settle(ticket, cost)
                except BudgetExceeded as exc:
                    print(f"H2_BUDGET_WARNING session cost {cost} above reservation: {exc}", flush=True)
        print("H2_PROPOSER_SESSION " + json.dumps({"label": getattr(self._route, "label", None), "model": model,
              "rc": done.returncode, "cost_usd": cost,
              **(self.budget.status() if self.budget else {})}), flush=True)
        answer = cwd / "output" / "answer.md"
        if answer.is_file() and answer.read_text(errors="replace").strip():
            return answer.read_text(errors="replace"), cost or 0.0, ""
        text, err = upstream._opencode_stdout(done.stdout)
        if text.strip():
            return text, cost or 0.0, ""
        tail = upstream._strip_oc_banner(done.stderr or "").strip()[-600:]
        return "", cost or 0.0, f"rc={done.returncode}: {err or tail or 'empty reply'}"


ROLES = (("select.", "select"), ("experience.", "reflect"), ("agg", "compose"), ("inv", "propose"),
         ("dom", "propose"), ("c", "compose"))


def role_of(label):
    for prefix, role in ROLES:
        if label.startswith(prefix):
            return role
    raise ValueError(f"no proposer role for session label {label!r}")


@dataclass
class RoutedProposer(OpenRouterProposer):
    """One model chain per Harness² role; each transport attempt moves one step down the chain.

    `routes` maps propose / compose / select / reflect to provider-prefixed model ids;
    `models` maps each id to {"price": [input, output] USD per million, "reserve": USD}.
    """
    routes: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        missing = {r for _, r in ROLES} - set(self.routes)
        unpriced = {m for chain in self.routes.values() for m in chain} - set(self.models)
        if missing or unpriced or not all(self.routes.values()):
            raise ValueError(f"routes need every role {sorted(missing)} and prices for {sorted(unpriced)}")

    def call(self, mission, context_dir, label, **kwargs):
        self._route.label, self._route.attempt = label, 0
        return super().call(mission, context_dir, label, **kwargs)

    def _session_model(self):
        chain = self.routes[role_of(self._route.label)]
        model = chain[min(self._route.attempt, len(chain) - 1)]
        self._route.attempt += 1
        info = self.models[model]
        return model, tuple(info["price"]), info["reserve"]
