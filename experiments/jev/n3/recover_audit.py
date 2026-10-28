"""Read-only recovery of completed N3 evidence. No model client or network."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

run_id = os.environ.get('ORX_RUN_ID')
assert run_id
source = Path('/home/toor/.orx/runs/ff35cbc2-b157-4564-87c0-271081d105ff/repo')
assert source.is_dir()
files = {}
directories = sorted((source / 'logs/jev-runs').glob('n3-*'))
assert len(directories) == 24
for directory in directories:
    for name in ['predictions.jsonl', 'decisions.jsonl', 'metadata.json', 'metrics.json']:
        path = directory / name
        assert path.is_file(), str(path)
        files['runs/' + directory.name + '/' + name] = path.read_bytes()
for name in ['protocol.json', 'preflight-frozen.json', 'controls.json']:
    files[name] = (source / 'experiments/jev/n3' / name).read_bytes()
manifest = {'run_id': run_id, 'source_run': 'ff35cbc2-b157-4564-87c0-271081d105ff',
            'source_commit': 'f2f58ed', 'new_API_calls': 0, 'human_annotation': False,
            'files_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
files['manifest.json'] = json.dumps(manifest, sort_keys=True).encode()
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, raw in files.items():
        archive.writestr(name, raw)
raw = buffer.getvalue()
encoded = base64.b64encode(raw).decode()
digest = hashlib.sha256(raw).hexdigest()
for index in range(0, len(encoded), 12000):
    print('N3_AUDIT_CHUNK ' + json.dumps({'sha256': digest, 'index': index // 12000,
                                       'data': encoded[index:index + 12000]}), flush=True)
print('N3_AUDIT_END ' + json.dumps({'sha256': digest, 'chunks': (len(encoded) + 11999) // 12000,
                                 'files': len(files), 'directories': len(directories), 'API_calls': 0}), flush=True)
