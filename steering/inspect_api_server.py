"""Read-only checkpoint/runtime metadata; deliberately excludes credentials."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root=Path('/app/models/Qwen3.8-27B-W4A16-AutoRound')
weights=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
         for p in sorted(root.glob('*.safetensors'))]
identity=dict(model=str(root),config_sha256=sha(root/'config.json'),
    tokenizer_config_sha256=sha(root/'tokenizer_config.json'),tokenizer_sha256=sha(root/'tokenizer.json'),
    vllm_version=version('vllm'),weights=weights)
safe=('MODEL','MAX_LEN','MAX_SEQS','GPU_UTIL','KV','KV_MEM','PREFIX_CACHE','INT8_ACT',
      'INT8_LAYERS','VISION','TOOLS','REQ_METRICS','QWEN_MAX_INPUT_TOKENS','EXTRA_ARGS','PREPARE')
print(json.dumps(dict(model_identity=identity,weights_signature=hashlib.sha256(
    json.dumps(weights,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
    settings={key:os.environ.get(key) for key in safe},
    numerical_protocol='W4A16 weights/activations, FP8 KV, FP16 SSM state, prefix cache, async batch',
    container='qwen38-27b-rtx3090-batch-1'),ensure_ascii=False,indent=2))
