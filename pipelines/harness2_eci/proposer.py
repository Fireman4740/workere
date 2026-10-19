"""Harness² improver sessions on the released OpenCode binary, routed to OpenRouter.

Upstream runs OpenCode from a pinned source checkout through bun and reaches Vertex AI. Here
the same `opencode run` session runs from the npm release, against `openrouter/<model>`,
and every session first reserves a conservative amount on the shared plan budget; the
reservation is settled with OpenCode's reported cost, or retained when no cost is reported.
"""
import json
import os
import subprocess
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


@dataclass
class OpenRouterProposer(upstream.Proposer):
    budget: object = None
    session_reserve_usd: float = .3
    price_per_million: tuple = (.1, .5)
    binary: Path = field(default_factory=opencode_binary)

    def env(self):
        return {k: v for k, v in os.environ.items() if k not in upstream._DROP_ENV}

    def _config(self):
        provider, model = self.spec.model.split("/", 1)
        if provider != "openrouter":
            raise ValueError(f"expected an openrouter/<model> id, got {self.spec.model!r}")
        # small_model pins title/summary calls to the same model: no other model is ever called.
        return {"$schema": "https://opencode.ai/config.json", "small_model": self.spec.model,
                "agent": {"build": {"steps": self.spec.steps}},
                "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
                "provider": {"openrouter": {"options": {"apiKey": "{env:OPENROUTER_API_KEY}"}, "models": {model: {
                    "name": model, "tool_call": True, "reasoning": True,
                    "cost": {"input": self.price_per_million[0], "output": self.price_per_million[1]},
                    "limit": {"context": 1_050_000, "output": 65_536}}}}}}

    def _opencode(self, cwd, mission, system, wall):
        (cwd / "MISSION.md").write_text((f"{system}\n\n---\n\n" if system else "") + mission, encoding="utf-8")
        (cwd / "output").mkdir(exist_ok=True)
        (cwd / "opencode.json").write_text(json.dumps(self._config()), encoding="utf-8")
        env = self.env()
        env["OPENCODE_PERMISSION"] = json.dumps(
            {"external_directory": {"*": "deny", str(cwd): "allow", f"{cwd}/**": "allow"}})
        env["XDG_DATA_HOME"], env["XDG_STATE_HOME"] = str(cwd / ".ocdata"), str(cwd / ".ocstate")
        env["XDG_CONFIG_HOME"], env["XDG_CACHE_HOME"] = str(cwd / ".occonfig"), str(INSTALL_ROOT / "xdg-cache")
        env["OPENCODE_DISABLE_CLAUDE_CODE"] = env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"  # no host skills
        prompt = ("Read MISSION.md first. Do what it asks. Reply with the requested output as "
                  "your final message AND save it verbatim to output/answer.md.")
        cmd = [str(self.binary), "run", prompt, "--dir", str(cwd), "--model", self.spec.model,
               "--format", "json"]
        if self.spec.effort:
            cmd += ["--variant", self.spec.effort]
        ticket = self.budget.reserve(self.session_reserve_usd) if self.budget else None
        cost = None
        try:
            done = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=wall)
            cost = upstream._opencode_cost(done.stdout) or None  # zero means unpriced: keep the hold
        finally:
            if ticket:
                try:
                    self.budget.settle(ticket, cost)
                except BudgetExceeded as exc:
                    print(f"H2_BUDGET_WARNING session cost {cost} above reservation: {exc}", flush=True)
        print("H2_PROPOSER_SESSION " + json.dumps({"rc": done.returncode, "cost_usd": cost,
              **(self.budget.status() if self.budget else {})}), flush=True)
        answer = cwd / "output" / "answer.md"
        if answer.is_file() and answer.read_text(errors="replace").strip():
            return answer.read_text(errors="replace"), cost or 0.0, ""
        text, err = upstream._opencode_stdout(done.stdout)
        if text.strip():
            return text, cost or 0.0, ""
        tail = upstream._strip_oc_banner(done.stderr or "").strip()[-600:]
        return "", cost or 0.0, f"rc={done.returncode}: {err or tail or 'empty reply'}"
