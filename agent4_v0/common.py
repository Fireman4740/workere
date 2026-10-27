"""Pure format/offset helpers; no dataset loader and no API imports."""
import ast
import hashlib
import json
import math
from pathlib import Path


class PreflightError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PreflightError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    require(not p.exists(), f'refusing to overwrite sealed output: {p}')
    with p.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, values):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    require(not p.exists(), f'refusing to overwrite sealed output: {p}')
    with p.open('x') as stream:
        stream.write(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def probability(value):
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
            'invalid cached score (not a probability calibration claim)')
    return float(value)


def marked(tokens, mentions, spans):
    """Byte-for-byte rendering used by the frozen Luna document plugin, incl. discontinuity."""
    starts, ends = {}, {}
    require(len(mentions) == len(spans) and len(set(mentions)) == len(mentions), 'mention mismatch')
    for mid, span in zip(mentions, spans):
        require(bool(span) and len(set(span)) == len(span), 'empty/duplicate mention span')
        for i in span:
            require(type(i) is int and 0 <= i < len(tokens), 'span out of bounds')
            if i - 1 not in span:
                starts.setdefault(i, []).append(str(mid))
            if i + 1 not in span:
                ends.setdefault(i, []).append(str(mid))
    return ' '.join(''.join(f'<{m} ' for m in starts.get(i, [])) + tok
                    + '>' * len(ends.get(i, [])) for i, tok in enumerate(tokens))


def token_offsets(tokens):
    out, start = [], 0
    for token in tokens:
        out.append([start, start + len(token)])
        start += len(token) + 1
    return out


def validate_doc(doc):
    require(marked(doc['tokens'], doc['mentions'], doc['spans']) == doc['text'],
            'frozen marked text differs from token/span rendering')
    expected = {(a, b) for a in doc['mentions'] for b in doc['mentions'] if a != b}
    require(len(doc['candidates']) == len(expected) and
            {tuple(p) for p in doc['candidates']} == expected, 'candidate universe changed')
    require(doc['offsets'] == token_offsets(doc['tokens']), 'plain-text token offsets differ')
    return True
