"""Decode the tracked N9 evidence stream; no remote/model calls."""
import argparse
import base64
import hashlib
import io
import json
import pathlib
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)
args = parser.parse_args()
meta, chunks, ended = None, {}, False
for line in args.log.read_text().splitlines():
    if "N9_EVIDENCE_BEGIN " in line:
        assert meta is None, "ambiguous repeated export"
        meta = json.loads(line.split("N9_EVIDENCE_BEGIN ", 1)[1])
    elif "N9_EVIDENCE_CHUNK " in line:
        row = json.loads(line.split("N9_EVIDENCE_CHUNK ", 1)[1])
        assert row["index"] not in chunks
        chunks[row["index"]] = row["data"]
    elif "N9_EVIDENCE_END" in line:
        ended = True
assert ended and meta and len(chunks) == meta["chunks"]
encoded = "".join(chunks[i] for i in range(meta["chunks"]))
assert len(encoded) == meta["base64_chars"]
raw = base64.b64decode(encoded, validate=True)
assert hashlib.sha256(raw).hexdigest() == meta["zip_sha256"]
args.output.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        target = args.output / name
        assert target.resolve().is_relative_to(args.output.resolve()), "unsafe archive path"
        assert not target.exists(), "refuse overwriting evidence"
    archive.extractall(args.output)
args.output.with_suffix(".zip").write_bytes(raw)
print(json.dumps({"collected": True, "zip_sha256": meta["zip_sha256"],
                  "output": str(args.output), "API_calls": 0}))
