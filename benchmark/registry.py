"""Explicit versioned plugin registry. Evaluation never imports model implementations."""
from __future__ import annotations

import importlib
import importlib.util
import hashlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path

KINDS = ("entity_extractor", "candidate_generator", "relation_predictor", "postprocessor", "pipeline")


@dataclass(frozen=True)
class Plugin:
    factory: object
    version: str
    requires_network: bool = False
    may_charge: bool = False
    mode: str = "inference"


class Registry:
    def __init__(self):
        self._plugins = {kind: {} for kind in KINDS}

    def register(self, kind, name, *, version="1", requires_network=False, may_charge=False,
                 mode="inference"):
        def decorate(factory):
            if kind not in self._plugins or not isinstance(name, str) or not name:
                raise ValueError("invalid plugin kind or name")
            if name in self._plugins[kind]:
                raise ValueError(f"duplicate plugin: {kind}/{name}")
            self._plugins[kind][name] = Plugin(factory, str(version), requires_network, may_charge, mode)
            return factory
        return decorate

    def describe(self):
        return [{"kind": kind, "name": name, "version": plugin.version,
                 "requires_network": plugin.requires_network, "may_charge": plugin.may_charge,
                 "execution_mode": plugin.mode,
                 "implementation": f"{plugin.factory.__module__}:{plugin.factory.__qualname__}"}
                for kind in KINDS for name, plugin in sorted(self._plugins[kind].items())]

    def resolve(self, kind, spec, policy):
        if not isinstance(spec, dict) or set(spec) - {"plugin", "version", "params"}:
            raise ValueError(f"invalid {kind} component specification")
        if "plugin" not in spec or "version" not in spec:
            raise ValueError("each component must pin plugin and version")
        name = spec["plugin"]
        if name not in self._plugins[kind]:
            raise ValueError(f"unknown {kind} plugin: {name}")
        plugin = self._plugins[kind][name]
        requirements = getattr(plugin.factory, "access_requirements", None)
        if requirements is not None:
            access = requirements(spec.get("params", {}))
            if set(access) != {"requires_network", "may_charge", "mode"}:
                raise ValueError("invalid dynamic plugin access declaration")
            plugin = replace(plugin, **access)
        if str(spec["version"]) != plugin.version:
            raise ValueError(f"plugin version mismatch: {kind}/{name}")
        if plugin.requires_network and not policy.get("allow_network", False):
            raise ValueError(f"network disabled for {kind}/{name}")
        if plugin.may_charge and not policy.get("allow_paid", False):
            raise ValueError(f"paid inference disabled for {kind}/{name}")
        if not isinstance(spec.get("params", {}), dict):
            raise ValueError("plugin params must be a mapping")
        return plugin

    def build(self, kind, spec, policy):
        return self.resolve(kind, spec, policy).factory(**spec.get("params", {}))


registry = Registry()
register = registry.register


def load_plugins(modules=()):
    importlib.import_module("benchmark.plugins.builtin")
    for module in modules:
        if not isinstance(module, str):
            raise ValueError("plugin module names must be strings")
        if module.endswith(".py"):
            path = Path(module).resolve()
            name = "_workere_plugin_" + hashlib.sha256(path.read_bytes()).hexdigest()
            if name not in sys.modules:
                spec = importlib.util.spec_from_file_location(name, path)
                loaded = importlib.util.module_from_spec(spec)
                sys.modules[name] = loaded
                try:
                    spec.loader.exec_module(loaded)
                except BaseException:
                    sys.modules.pop(name, None)
                    raise
        else:
            importlib.import_module(module)
    return registry
