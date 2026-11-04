"""Offline vLLM helpers for the CTB50 steering pilot.

GPU-side scripts import only this module (plus numpy/torch/vLLM), so they run from
/app/local_extensions/steering inside the HyperQwen image. Imports of torch/vLLM stay
inside the functions so prompt construction and tokenizer dry runs work without a GPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

CHAT_KWARGS = {"enable_thinking": True, "preserve_thinking": True}
THINK_CLOSE = "\n</think>\n\n"
# Settings the serving config of this box needs for the hybrid (Mamba) Qwen3.8 on a 3090.
ENGINE_EXTRA = {
    "attention_backend": "FLASH_ATTN",
    "kv_cache_dtype": "bfloat16",
    "mamba_ssm_cache_dtype": "float16",
    "language_model_only": True,
}


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs=CHAT_KWARGS,
    )


def render_with_answer(tokenizer, messages: list[dict], answer: str) -> tuple[str, str]:
    """(prefix, full): the generation prompt, and prompt + empty thought + answer."""
    prefix = render(tokenizer, messages)
    return prefix, prefix + THINK_CLOSE + answer


def build_llm(
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int = 1,
    eager: bool = False,
    steer_layer: int | None = None,
    steer_positions: str = "last",
):
    """In-process engine with no chunked prefill.

    Chunked prefill would split a long prompt over several forward calls, and the
    capture hooks assume the whole prompt arrives in one call. Without eager, decode runs
    in full CUDA graphs and prefill stays eager, so capture hooks still see every prefill.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from vllm import LLM

    if steer_layer is not None:
        install_steerer(steer_layer, steer_positions)
    extra = dict(ENGINE_EXTRA)
    if not eager:
        extra["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    llm = LLM(
        model=model,
        tokenizer=model,
        enforce_eager=eager,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_model_len,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=True,
        **extra,
    )
    return llm, get_model(llm), llm.get_tokenizer()


def get_model(llm):
    """torch module of the decoder, across vLLM V0/V1 in-process layouts."""
    engine = llm.llm_engine
    candidates = (
        lambda: engine.model_executor.driver_worker.model_runner.model,
        lambda: engine.model_executor.driver_worker.worker.model_runner.model,
        lambda: engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
    )
    for candidate in candidates:
        try:
            return candidate()
        except AttributeError:
            continue
    raise RuntimeError("cannot reach the torch model; set VLLM_ENABLE_V1_MULTIPROCESSING=0")


def find_module(model, layer: int, kind: str):
    suffix = (
        f"layers.{layer}.mlp.down_proj"
        if kind == "down_proj"
        else f"layers.{layer}.self_attn.o_proj"
    )
    for name, module in model.named_modules():
        if name.endswith(suffix):
            return name, module
    raise KeyError(f"module not found: {suffix}")


def load_direction(path: str | Path, name: str = "v"):
    """(unit direction, reference norm of the module output) from a direction file."""
    import numpy as np
    import torch

    data = np.load(path)
    ref_norm = float(data["ref_norm"]) if "ref_norm" in data.files else None
    return torch.from_numpy(data[name]).float(), ref_norm


class Steerer:
    """One persistent hook whose dose is a device buffer, so conditions switch in place.

    The buffer is zero for the control, so control and steered runs share one code path
    (x + 0 is exact). Because the hook reads a buffer instead of Python scalars, it stays
    valid inside a captured CUDA graph, provided it is registered before capture.
    positions "last": the last row of a multi-token call (the last prompt token at
    prefill) and the single row of every decode call (max_num_seqs=1).
    """

    def __init__(self, module, hidden_size: int, positions: str = "last", dtype=None, device="cuda"):
        import torch

        if positions not in ("all", "last"):
            raise ValueError(f"positions must be all or last, got {positions!r}")
        self.positions = positions
        self.buffer = torch.zeros(hidden_size, dtype=dtype or torch.get_default_dtype(), device=device)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        rows = None if self.positions == "all" else _last_rows(tensor.shape[0])
        if rows is None:
            steered = tensor + self.buffer
        else:
            steered = tensor.clone()
            steered[rows] += self.buffer
        return (steered, output[1]) if isinstance(output, tuple) else steered

    def set(self, direction, alpha: float, ref_norm: float) -> None:
        import torch

        delta = (alpha * ref_norm) * direction
        with torch.inference_mode():
            self.buffer.copy_(delta.to(dtype=self.buffer.dtype, device=self.buffer.device))

    def clear(self) -> None:
        import torch

        with torch.inference_mode():
            self.buffer.zero_()

    def remove(self) -> None:
        self.handle.remove()


def _query_start_loc():
    """Per-request token offsets of the current vLLM batch, or None outside a vLLM forward."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None, True
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available
    except ImportError:
        return None, False
    if not is_forward_context_available():
        return None, False
    context = get_forward_context()
    if context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
        return None, True
    metadata = context.attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else None
    if not metadata:
        return None, False
    candidates = [m for m in metadata.values() if hasattr(m, "query_start_loc")]
    flash = [m for m in candidates if "FlashAttention" in type(m).__name__]
    chosen = (flash or candidates or [None])[0]
    return (None if chosen is None else chosen.query_start_loc), False


def _last_rows(num_rows: int):
    """Rows holding each request's last token, or None when that is every row.

    A captured decode graph only ever holds one new token per request, so every row is
    steered. In an eager batch (prefill, possibly mixed with decodes) the last row of each
    request is its last prompt token or its single decode token.
    """
    starts, graph = _query_start_loc()
    if graph or num_rows == 1:
        return None
    if starts is None:
        return -1
    last = starts[1:] - 1
    last = last[(last >= 0) & (last < num_rows)]
    if last.numel() >= num_rows:
        return None
    return last.long()


_INSTALLED: dict[str, Steerer] = {}


def install_steerer(layer: int, positions: str = "last") -> None:
    """Attach a Steerer to layers.<layer>.mlp.down_proj while the model is being built.

    CUDA graphs capture the module as it is at engine start, so the hook has to exist
    before the engine initialises; a hook added afterwards would silently not apply.
    """
    from vllm.model_executor.layers.linear import RowParallelLinear

    suffix = f"layers.{layer}.mlp.down_proj"
    original = RowParallelLinear.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if str(kwargs.get("prefix", "")).endswith(suffix):
            _INSTALLED["steerer"] = Steerer(self, self.output_size, positions)

    RowParallelLinear.__init__ = patched


def installed_steerer() -> Steerer:
    if "steerer" not in _INSTALLED:
        raise RuntimeError("no steerer installed: pass steer_layer to build_llm")
    return _INSTALLED["steerer"]


def last_token_hook(store: dict[str, Any], key: str):
    """Record the last-token output of the first call only (the whole prompt)."""

    def hook(module, inputs, output):
        if key in store:
            return None
        tensor = output[0] if isinstance(output, tuple) else output
        store[key] = tensor[-1].detach().float().cpu()
        return None

    return hook


def span_mean_hook(store: dict[str, Any], key: str, start: int, stop: int):
    """Record the mean output over tokens [start, stop) of the first call."""

    def hook(module, inputs, output):
        if key in store:
            return None
        tensor = output[0] if isinstance(output, tuple) else output
        store[key] = tensor[start:stop].detach().float().mean(dim=0).cpu()
        return None

    return hook


def sampling_params(max_tokens: int, temperature: float = 0.0, seed: int | None = None):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=temperature,
        seed=seed,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
    )


def _prefill(llm, text: str) -> None:
    llm.generate([text], sampling_params(1), use_tqdm=False)


def capture_last(llm, tokenizer, messages: list[dict], modules: dict[int, Any]) -> dict[int, Any]:
    """One prefill; last prompt-token output of every module, keyed by layer."""
    store: dict[str, Any] = {}
    handles = [m.register_forward_hook(last_token_hook(store, str(layer))) for layer, m in modules.items()]
    try:
        _prefill(llm, render(tokenizer, messages))
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in modules if str(layer) not in store]
    if missing:
        raise RuntimeError(f"hook did not capture layers {missing}")
    return {layer: store[str(layer)] for layer in modules}


def capture_answer_span(
    llm, tokenizer, messages: list[dict], answer: str, modules: dict[int, Any]
) -> dict[int, Any]:
    """One prefill of prompt + answer; mean output over the answer tokens, per layer."""
    prefix, full = render_with_answer(tokenizer, messages, answer)
    start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    stop = len(tokenizer(full, add_special_tokens=False)["input_ids"])
    if stop <= start:
        raise ValueError("answer span is empty after tokenisation")
    store: dict[str, Any] = {}
    handles = [
        m.register_forward_hook(span_mean_hook(store, str(layer), start, stop))
        for layer, m in modules.items()
    ]
    try:
        _prefill(llm, full)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in modules if str(layer) not in store]
    if missing:
        raise RuntimeError(f"hook did not capture layers {missing}")
    return {layer: store[str(layer)] for layer in modules}


def parse_layers(spec: str) -> Sequence[int]:
    return [int(value) for value in spec.split(",") if value.strip()]
