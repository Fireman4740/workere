"""Graph-safe projection-removal hooks and position-level activation capture.

Projector (E3): on layers.L.mlp.down_proj, removes the positive component of the MLP
write along a unit direction d, scaled by a strength s:  y <- y - s * relu(y.d) d.
HeadProjector (E5): on layers.L.self_attn.o_proj, removes from the output the part of
selected heads' contributions that lies along d:  y <- y - (x.u) d,  u = W_o[:, heads]^T d
(zero outside the selected heads). This equals editing the selected heads' output weights,
without touching the int4 checkpoint.

Both read only device buffers, so they stay valid inside captured CUDA graphs provided
they are registered before the engine starts (install_interventions before build_llm).
"""
from __future__ import annotations

from typing import Any, Iterable

_INSTALLED: dict[str, Any] = {"heads": {}}


class Projector:
    def __init__(self, module, hidden_size: int, device="cuda"):
        import torch

        dtype = torch.get_default_dtype()
        self.direction = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.strength = torch.zeros(1, dtype=dtype, device=device)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        coeff = (tensor @ self.direction).clamp(min=0).unsqueeze(-1)
        edited = tensor - self.strength * coeff * self.direction
        return (edited, output[1]) if isinstance(output, tuple) else edited

    def set(self, direction, strength: float) -> None:
        import torch

        with torch.inference_mode():
            unit = direction / direction.norm()
            self.direction.copy_(unit.to(self.direction))
            self.strength.fill_(float(strength))

    def clear(self) -> None:
        import torch

        with torch.inference_mode():
            self.strength.zero_()


class HeadProjector:
    def __init__(self, module, input_size: int, hidden_size: int, device="cuda"):
        import torch

        dtype = torch.get_default_dtype()
        self.u = torch.zeros(input_size, dtype=dtype, device=device)
        self.direction = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        coeff = (inputs[0] @ self.u).unsqueeze(-1)
        edited = tensor - coeff * self.direction
        return (edited, output[1]) if isinstance(output, tuple) else edited

    def set(self, u, direction, strength: float = 1.0) -> None:
        import torch

        with torch.inference_mode():
            self.u.copy_((float(strength) * u).to(self.u))
            self.direction.copy_(direction.to(self.direction))

    def clear(self) -> None:
        import torch

        with torch.inference_mode():
            self.u.zero_()
            self.direction.zero_()


def install_interventions(projector_layer: int | None = None, head_layers: Iterable[int] = ()) -> None:
    """Patch RowParallelLinear so the hooks exist before CUDA-graph capture."""
    from vllm.model_executor.layers.linear import RowParallelLinear

    head_layers = set(head_layers)
    original = RowParallelLinear.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        if projector_layer is not None and prefix.endswith(f"layers.{projector_layer}.mlp.down_proj"):
            _INSTALLED["projector"] = Projector(self, self.output_size)
        for layer in head_layers:
            if prefix.endswith(f"layers.{layer}.self_attn.o_proj"):
                _INSTALLED["heads"][layer] = HeadProjector(self, self.input_size, self.output_size)

    RowParallelLinear.__init__ = patched


def installed_projector() -> Projector:
    if "projector" not in _INSTALLED:
        raise RuntimeError("no projector installed: call install_interventions(projector_layer=...)")
    return _INSTALLED["projector"]


def installed_heads() -> dict[int, HeadProjector]:
    return _INSTALLED["heads"]


def clear_all() -> None:
    if "projector" in _INSTALLED:
        _INSTALLED["projector"].clear()
    for head in _INSTALLED["heads"].values():
        head.clear()


def find_layer(model, layer: int):
    for name, module in model.named_modules():
        if name.split(".")[-2:] == ["layers", str(layer)]:
            return module
    raise KeyError(f"decoder layer not found: layers.{layer}")


def capture_rows(llm, text: str, taps: dict[str, tuple[Any, str]], positions: list[int]) -> dict[str, Any]:
    """One prefill of text; rows at the given token positions of each tap's input or output.

    taps maps a name to (module, "in" | "out" | "resid"); "resid" on a decoder layer sums
    its (hidden_states, residual) outputs. Only the first forward call is recorded,
    which is the whole prompt because chunked prefill is off.
    """
    import torch

    from offline import sampling_params

    store: dict[str, Any] = {}
    index = torch.tensor(positions, dtype=torch.long)

    def make_hook(name: str, side: str):
        def hook(module, inputs, output):
            if name in store:
                return None
            if side == "in":
                tensor = inputs[0]
            elif side == "resid":
                tensor = output[0] + output[1] if isinstance(output, tuple) and output[1] is not None else (
                    output[0] if isinstance(output, tuple) else output)
            else:
                tensor = output[0] if isinstance(output, tuple) else output
            store[name] = tensor.index_select(0, index.to(tensor.device)).detach().float().cpu()
            return None

        return hook

    handles = [module.register_forward_hook(make_hook(name, side)) for name, (module, side) in taps.items()]
    try:
        llm.generate([text], sampling_params(1), use_tqdm=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = [name for name in taps if name not in store]
    if missing:
        raise RuntimeError(f"capture missed taps {missing}")
    return store
