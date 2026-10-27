"""Local-only preparation. Does not approve edits, read evaluation gold, or contact models."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tarfile
from types import SimpleNamespace

from .common import digest, jsonl, load, require, rows, save, sha
from .kit import pilot

CONNECTIVES = re.compile(r'\b(?:as a result of|because of|due to|because|therefore|thus|consequently|so that)\b', re.I)


def audit_reserve(reference_root, register, h1_manifest, h1_log, out):
    previous = Path(register).parents[3] / 'empreinte-placebo-2026-10-27/audit-local.json'
    reserve = set(load(register)['remaining_unused_reserve'])
    manifest = load(h1_manifest)
    planned = set(manifest['selection']['document_ids'])
    # Completed per-document log entries, not status or aggregate score alone.
    log = Path(h1_log).read_text()
    completed = set(x + '.xml' for x in re.findall(r'esc-h1-haiku-([\d]+_[\d]+ecbplus)\s+seed=42 status=completed', log))
    result = {'reserve_count': len(reserve), 'h1_manifest_documents': len(planned),
              'reserve_in_h1_manifest': sorted(reserve & planned),
              'reserve_completed_h1_log': sorted(reserve & completed),
              'h1_completed_esc_documents': len(completed),
              'reserve_status': 'EXPOSED_H1_not_untouched',
              'labels_access_chronology_verified': False,
              'sources': {str(p): sha(p) for p in (register, h1_manifest, h1_log)},
              'previous_agent_audit_available': previous.exists()}
    save(Path(out) / 'reserve-audit.json', result)
    return result


def snapshots(archives, doc_ids, prompt_path, out):
    import yaml
    prompt = yaml.safe_load(Path(prompt_path).read_text())
    records, source_files = {}, {}
    for path in archives:
        source_files[str(path)] = sha(path)
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('/decisions.jsonl'):
                    continue
                for line in archive.extractfile(member):
                    event = json.loads(line)
                    did = event['document_id']
                    if did not in doc_ids or event.get('error'):
                        continue
                    request = event['request']
                    require(request['messages'][0] == {'role': 'system', 'content': prompt['system']},
                            'snapshot system prompt mismatch')
                    require(request['model'] == 'openai/gpt-6-luna:batch' and
                            request['reasoning'] == {'effort': 'high'} and request['max_tokens'] == 32768,
                            'snapshot model/settings mismatch')
                    demo = event['demonstration_id']
                    require(demo and demo.split('_')[0] != did.split('_')[0], 'demo is not outside topic')
                    key = (did, event['pass_index'])
                    item = {'doc_id': did, 'pass_index': event['pass_index'], 'demonstration_id': demo,
                            'request': request, 'request_sha256': digest(request),
                            'historical_served_model': event['response']['model'],
                            'source_archive': str(path), 'source_member': member.name}
                    if key in records:
                        require(records[key]['request_sha256'] == item['request_sha256'], 'conflicting snapshots')
                    records[key] = item
    complete = {did for did in doc_ids if {r['pass_index'] for r in records.values() if r['doc_id'] == did} == set(range(7))}
    jsonl(Path(out) / 'frozen-snapshots.jsonl', [r for (did, _), r in sorted(records.items()) if did in complete])
    result = {'complete_seven_pass_documents': sorted(complete), 'missing_documents': sorted(set(doc_ids) - complete),
              'snapshots_sha256': sha(Path(out) / 'frozen-snapshots.jsonl'),
              'archives_sha256': source_files, 'prompt_sha256': sha(prompt_path),
              'note': 'Demonstration answers may contain labels for other documents, never the pilot gold. V0 reuses pass 0 in all conditions and both originals.'}
    save(Path(out) / 'snapshots-manifest.json', result)
    return result


def proposals(baseline, selection, out):
    docs = pilot.parse_docs(baseline, {'dev'})
    selected = load(selection)['selected']
    require(sha(baseline) == load(selection)['baseline_sha256'], 'selection baseline hash mismatch')
    records, packets, evidence = [], [], []
    for item in selected:
        doc = docs[item['doc_id']]
        text, mentions = doc['text'], doc['mentions']
        pair_mentions = [m for m in mentions if m['id'] in item['pair']]
        contexts = []
        for m in pair_mentions:
            a, b = max(0,m['start']-180), min(len(text),m['end']+180)
            contexts.append({'mention_id':m['id'], 'start':a, 'end':b, 'quote':text[a:b]})
        alternatives = []
        for m in pair_mentions:
            for occurrence in re.finditer(re.escape(m['surface']),text,re.I):
                if occurrence.start() == m['start']:
                    continue
                a,b=max(0,occurrence.start()-120),min(len(text),occurrence.end()+120)
                alternatives.append({'surface':m['surface'],'start':a,'end':b,'quote':text[a:b],
                                     'identity_status':'lexical_repeat_not_verified_coreference'})
        evidence.append({'doc_id':doc['doc_id'],'pair':item['pair'], 'literal_pair_contexts':contexts,
                         'potential_alternative_contexts':alternatives,
                         'support_status':'proposals_only_not_a_causal_justification_certificate',
                         'distributed_and_redundant_support_audit':'pending_independent_human_review'})
        def distance(hit):
            return max(abs((hit.start()+hit.end())/2 - (m['start']+m['end'])/2) for m in pair_mentions)
        hits = [h for h in CONNECTIVES.finditer(text)
                if not any(h.start() < m['end'] and h.end() > m['start'] for m in mentions)]
        hits.sort(key=lambda h: (distance(h), h.start()))
        target = hits[0] if hits and distance(hits[0]) <= 500 else None
        # All are PROPOSALS: proximity and equal volume do NOT certify irrelevance.
        others = [h for h in hits if target is not None and h is not target and
                  abs(len(h.group().split())-len(target.group().split())) == 0]
        variants = [('target', target)] + [(f'placebo_{j}', others[j-1] if len(others) >= j else None) for j in (1,2,3)]
        for variant, hit in variants:
            edits = [] if hit is None else [{'start': hit.start(), 'end': hit.end(), 'before': hit.group(), 'after': ''}]
            # This automatic check is purely structural, not an approval.
            if edits:
                pilot.apply_edits(text, mentions, edits)
            record = {'doc_id': doc['doc_id'], 'pair': item['pair'], 'variant': variant,
                      'review_status': 'uncertain', 'reviewed_blind_to_gold': False,
                      'reviewed_blind_to_model_outputs': False, 'proposer': 'agent4-local-heuristic-v0',
                      'reviewer': None, 'edits': edits,
                      'proposal_status': 'requires_semantic_review' if hit else 'missing_editable_matched_span',
                      'alternative_support_status': 'not_independently_audited',
                      'support_quotes': [] if hit is None else [{'start': max(0,hit.start()-180),
                          'end': min(len(text),hit.end()+180),
                          'quote': text[max(0,hit.start()-180):min(len(text),hit.end()+180)]}],
                      'matching_distance_chars': None if hit is None else distance(hit),
                      'warning': 'A lexical cue proposal is not an exhaustive justification or a semantically valid ablation.'}
            record['proposal_sha256'] = digest(record)
            records.append(record)
        mids = [m['id'] for m in mentions]
        mapping = dict(zip(mids, mids[1:] + mids[:1]))
        rec = {'doc_id': doc['doc_id'], 'pair': item['pair'], 'variant': 'rename_ids',
               'id_map': mapping, 'review_status': 'uncertain', 'reviewed_blind_to_gold': False,
               'reviewed_blind_to_model_outputs': False, 'proposer': 'agent4-local-heuristic-v0', 'reviewer': None}
        rec['proposal_sha256'] = digest(rec)
        records.append(rec)
        packets.append({'doc_id': doc['doc_id'], 'pair': item['pair'], 'text': text, 'mentions': mentions,
                        'variants': records[-5:], 'instruction': 'Independent human review required. No labels or model scores are supplied. Audit alternative/distributed support, grammar and placebo matching before approval.'})
    jsonl(Path(out) / 'evidence-proposals.jsonl', evidence)
    jsonl(Path(out) / 'edit-proposals.jsonl', records)
    jsonl(Path(out) / 'human-review-packets.jsonl', packets)
    pilot.prepare(SimpleNamespace(baseline=str(baseline), selection=str(selection),
        edits=str(Path(out) / 'edit-proposals.jsonl'), out_tasks=str(Path(out) / 'tasks.jsonl'),
        out_audit=str(Path(out) / 'edit-audit.json'), reads=1, original_reads=2))
    return {'proposed_variants': len(records), 'approved_variants': 0,
            'missing_semantic_span_variants': sum(r.get('proposal_status') == 'missing_editable_matched_span' for r in records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--archive', action='append', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--reference-root', required=True)
    p.add_argument('--reserve-register', required=True)
    p.add_argument('--h1-manifest', required=True)
    p.add_argument('--h1-log', required=True)
    a = p.parse_args()
    out = Path(a.out)
    doc_ids = {x['doc_id'] for x in load(out / 'selection.json')['selected']}
    snapshot_info = snapshots(a.archive, doc_ids, a.prompt, out)
    edit_info = proposals(out / 'baseline.jsonl', out / 'selection.json', out)
    reserve_info = audit_reserve(a.reference_root, a.reserve_register, a.h1_manifest, a.h1_log, out)
    save(out / 'offline-preflight.json', {'status': 'INCONCLUSIVE', 'blockers': ['BLOCKED_HUMAN_REVIEW', 'BLOCKED_BUDGET', 'MODEL_AVAILABILITY_UNVERIFIED'],
         'model_calls': 0, 'model_cost_usd': 0, 'snapshots': snapshot_info, 'edits': edit_info,
         'reserve_completed_h1': len(reserve_info['reserve_completed_h1_log']),
         'gold_opened_for_selection_or_interventions': False})


if __name__ == '__main__':
    main()
