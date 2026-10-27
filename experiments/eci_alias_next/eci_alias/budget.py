"""Fail-closed A/R usage gate, optionally bound to an expected document manifest."""
import argparse
import json
from pathlib import Path
from collections import defaultdict

REQUIRED = ('doc_id', 'arm', 'pass_index', 'input_tokens', 'completion_tokens',
            'model_version', 'demo_id', 'base_prompt_sha256', 'decode_sha256', 'truncated')
PROVENANCE = ('demo_sha256', 'full_prompt_sha256', 'token_count_source',
              'auxiliary_tokens_upper_bound', 'served_model', 'status')


def verify(records, max_input=16384, max_completion=4096, relative_tolerance=.01,
           expected_doc_ids=None, require_provenance=False):
    rows = {}
    issues = []
    if not records:
        issues.append('empty_usage')
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            issues.append('invalid_usage_record:' + str(i)); continue
        absent = [k for k in REQUIRED + (PROVENANCE if require_provenance else ()) if k not in r]
        if absent:
            issues.append('missing_usage_fields:' + ','.join(absent)); continue
        if (not isinstance(r['doc_id'], str) or not r['doc_id'] or
                type(r['pass_index']) is not int or r['pass_index'] not in range(7) or
                r['arm'] not in ('A', 'R')):
            issues.append('invalid_slot_identity:' + str(i)); continue
        key = (r['doc_id'], r['pass_index'], r['arm'])
        if key in rows:
            issues.append('duplicate_usage:' + str(key))
        rows[key] = r
        if r['truncated'] is not False:
            issues.append('truncation_or_unknown:' + str(key))
        for field in ('model_version', 'demo_id', 'base_prompt_sha256', 'decode_sha256'):
            if not isinstance(r[field], str) or not r[field]:
                issues.append('empty_provenance_' + field + ':' + str(key))
        for field, limit in (('input_tokens', max_input), ('completion_tokens', max_completion)):
            value = r[field]
            if type(value) is not int or value < 0 or value > limit or (field == 'input_tokens' and value == 0):
                issues.append('out_of_bounds_' + field + ':' + str(key))
        if require_provenance:
            if r['status'] != 'ok' or r['token_count_source'] != 'provider_usage':
                issues.append('failed_or_unverified_usage:' + str(key))
            for field in ('demo_sha256', 'full_prompt_sha256', 'base_prompt_sha256', 'decode_sha256'):
                value = r[field]
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    issues.append('invalid_digest_' + field + ':' + str(key))
            aux = r['auxiliary_tokens_upper_bound']
            if type(aux) is not int or not 0 <= aux <= 768:
                issues.append('auxiliary_limit_or_unknown:' + str(key))
            if not isinstance(r['served_model'], str) or not r['served_model']:
                issues.append('missing_served_model:' + str(key))
    diffs = []
    for did, p in sorted({(did, p) for did, p, arm in rows}):
        a, r = rows.get((did, p, 'A')), rows.get((did, p, 'R'))
        if a is None or r is None:
            issues.append('missing_AR_match:' + str((did, p))); continue
        fields = ('model_version', 'demo_id', 'base_prompt_sha256', 'decode_sha256')
        if require_provenance:
            fields += ('demo_sha256', 'served_model')
        for field in fields:
            if a[field] != r[field]:
                issues.append('mismatched_' + field + ':' + str((did, p)))
        ta, tr = a['input_tokens'], r['input_tokens']
        if type(ta) is int and type(tr) is int and ta > 0 and tr > 0:
            delta = abs(ta - tr) / max(ta, tr)
            diffs.append(delta)
            if delta > relative_tolerance:
                issues.append('input_token_mismatch:%s:%.4f' % ((did, p), delta))
    by_doc = defaultdict(set)
    for did, p, arm in rows:
        by_doc[did].add((p, arm))
    expected = {(p, arm) for p in range(7) for arm in ('A', 'R')}
    for did, seen in by_doc.items():
        if seen != expected:
            issues.append('not_seven_matched_passes:' + did)
    if expected_doc_ids is not None and set(by_doc) != set(expected_doc_ids):
        issues.append('document_manifest_mismatch')
    return {'status': 'PASS' if not issues else 'BLOCKED', 'matched_passes': len(diffs),
            'documents': len(by_doc), 'max_relative_input_token_difference': max(diffs) if diffs else None,
            'issues': issues, 'note': 'PASS does not establish identical hidden reasoning compute. Auxiliary bound is conservative, not an actual tokenizer measurement.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--usage', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected-doc-ids', help='one document ID per line')
    ap.add_argument('--require-provenance', action='store_true')
    ap.add_argument('--max-input', type=int, default=16384)
    ap.add_argument('--max-completion', type=int, default=4096,
                    help='explicitly select a registered budget; original default remains 4096')
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.usage).read_text().splitlines() if x.strip()]
    expected = Path(args.expected_doc_ids).read_text().splitlines() if args.expected_doc_ids else None
    result = verify(rows, max_input=args.max_input, max_completion=args.max_completion,
                    expected_doc_ids=expected, require_provenance=args.require_provenance)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print('BUDGET GATE', result['status'], result['matched_passes'], 'matched passes', len(result['issues']), 'issues')
    raise SystemExit(0 if result['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
