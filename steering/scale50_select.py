"""Freeze additional ESC/MAVEN documents using only identifiers and length.

Run locally with cached corpus files and the previously frozen W5/W6 bundles.
The resulting gzip is committed, so compute runs never download or resample data.
"""
import argparse
import gzip
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as parquet

from eci.data import doc_from_row
from steering import wsp_combo as c


def document(row):
    doc = doc_from_row(row, 0, text_field='text', ann_field='annots',
                       valid_labels={'causes', 'causedby', 'norel'},
                       binary_undirected=False, shuffle_pair_list=False)
    gold = c.symmetric({tuple(sorted((a, b))) for a, label, b in doc.gold if label != 'norel'})
    return dict(id=doc.id, text=doc.text, pair_list=doc.pair_list, gold=sorted(','.join(p) for p in gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--esc-arrow', required=True)
    ap.add_argument('--maven-parquet', required=True)
    ap.add_argument('--full-bundles', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.full_bundles)
    esc_parent, esc_docs, _, _ = c.load_bundle(root / 'ESC')
    maven_parent, maven_docs, _, _ = c.load_bundle(root / 'MAVEN')
    prompt = json.loads((c.HERE/'data/esc/prompt.json').read_text())
    with pa.memory_map(args.esc_arrow, 'r') as stream:
        esc_rows = pa.ipc.open_stream(stream).read_all().to_pylist()
    bank = {d['id'] for d in prompt['demos']}
    eligible = sorted((r for r in esc_rows if str(r['id']) not in set(esc_docs) | bank),
                      key=lambda r: str(r['id']))
    esc_extra = [document(r) for r in eligible[:10]]
    if len(esc_extra) != 10:
        raise ValueError('need ten ESC documents outside both original test and demonstration bank')
    test_path = Path(args.maven_parquet)
    if c.sha(test_path) != maven_parent['data_sha256']['test']:
        raise ValueError('MAVEN test parquet fingerprint differs')
    maven_rows = parquet.read_table(test_path).to_pylist()
    if len(maven_rows) != 710:
        raise ValueError('MAVEN test count differs')
    order = sorted(maven_rows, key=lambda r: (len(r.get('tokens') or []), str(r['id'])))
    rng = random.Random(20261112)
    sampled = []
    for i, n in enumerate((7, 7, 6, 6)):
        bucket = [r for r in order[i*len(order)//4:(i+1)*len(order)//4]
                  if str(r['id']) not in maven_docs]
        sampled.extend(rng.sample(bucket, n))
    maven_extra = [document(r) for r in sampled]
    for label, docs, n in [('ESC', esc_extra, 10), ('MAVEN', maven_extra, 26)]:
        if len(docs) != n or len({d['id'] for d in docs}) != n:
            raise ValueError(f'{label} selection invalid')
    payload = dict(version='wsp-scale50-inputs-v1', seed=20261112,
        esc=dict(parent_manifest=esc_parent['id'], extra=esc_extra,
                 selection='first ten lexicographic IDs outside original test and all demonstration IDs',
                 source_sha256=c.sha(args.esc_arrow)),
        maven=dict(parent_manifest=maven_parent['id'], extra=maven_extra,
                   selection='seeded 7/7/6/6 sample by token-length quartile outside original24',
                   source_sha256=c.sha(test_path)),
        selection_uses_gold=False)
    Path(args.out).write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode(), mtime=0))
    print(json.dumps({k: [d['id'] for d in payload[k]['extra']] for k in ('esc','maven')}))


if __name__ == '__main__':
    main()
