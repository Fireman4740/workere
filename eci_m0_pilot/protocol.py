"""Pure M0 protocol: rendering, strict validity guards, inverse remapping and E/V.
The supplied m0.py is imported unchanged. No labels are accepted by these functions.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eci.parse import answer_array, iter_json_arrays, parse_triples
from eci_m0_pilot.m0 import as_pairs, build, digest, pair, permutations, validate

ARMS = ('O', 'P', 'M', 'MP')
EMPTY_RULE = '\nAn empty final JSON array [] is valid even when event mentions are present. Do not invent a relation to avoid an empty output.\n'
MASK_RULE = '''\nOUTPUT AUTHORIZATION (the whole document remains evidence):
Extract the set of supported causal relations ONLY for the unordered mention pairs below.
These are possibilities, not suggested relations. Several relations or none are allowed.
Keep the same causal definitions, direction labels and symmetric output convention.
Do not emit any other pair. Mere co-occurrence or inclusion in this list is not evidence.
Authorized unordered pairs (either direction may be emitted):\n'''
TAG = re.compile(r'<([^>\s]+)\s')
PAIR_LITERAL = re.compile(r'"(\d+)\s*,\s*(\d+)"')
ID_LITERAL = re.compile(r'(ID is ")(\d+)(")')

def json_text(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False)

def remap_tags(text, mapping):
    return TAG.sub(lambda m: '<' + mapping.get(m[1], m[1]) + ' ', text)

def remap_template(text, mapping):
    # Only structured identifier sites, never arbitrary numbers in documentary text.
    text = remap_tags(text, mapping)
    text = PAIR_LITERAL.sub(lambda m: '"' + mapping.get(m[1], m[1]) + ',' + mapping.get(m[2], m[2]) + '"', text)
    return ID_LITERAL.sub(lambda m: m[1] + mapping.get(m[2], m[2]) + m[3], text)

def lexical_text(text):
    return TAG.sub('<ID ', text)

def render(doc, row, demo, template, seed=42):
    mapping = row['id_mapping']
    ids = {str(m['id']) for m in doc['mentions']}
    if set(mapping) != ids or set(mapping.values()) != ids:
        raise ValueError('mapping is not a bijection on document mention IDs')
    inverse = {v: k for k, v in mapping.items()}
    text = remap_tags(doc['text'], mapping)
    assert remap_tags(text, inverse) == doc['text']
    assert lexical_text(text) == lexical_text(doc['text'])
    assert set(TAG.findall(text)) == ids
    assert demo['topic'] != doc['topic'] and demo['doc_id'] != doc['doc_id']
    i = row['pass'] - 1
    dm = permutations(demo, 7, seed)[i] if row['arm'] in ('P', 'MP') else {str(x['id']): str(x['id']) for x in demo['mentions']}
    dt = remap_tags(demo['text'], dm)
    assert remap_tags(dt, {v: k for k, v in dm.items()}) == demo['text']
    da = []
    for item in demo['answer']:
        a, b = item['pair'].split(',')
        da.append({'pair': f'{dm[a]},{dm[b]}', 'label': item['label']})
    user_template = remap_template(template['user_template'], mapping)
    user = user_template.format(doc_text=text, pair_lines='', doc_id=doc['doc_id']) + EMPTY_RULE
    if row['arm'] in ('M', 'MP'):
        user += MASK_RULE + json.dumps(row['allowed_visible'])
    messages = [
        {'role': 'system', 'content': remap_template(template['system'], mapping)},
        {'role': 'user', 'content': remap_template(template['user_template'], dm).format(doc_text=dt, pair_lines='', doc_id=demo['doc_id']) + EMPTY_RULE},
        {'role': 'assistant', 'content': json.dumps(da)},
        {'role': 'user', 'content': user},
    ]
    # The unmasked control never serializes a candidate list.
    if row['arm'] in ('O', 'P'): assert MASK_RULE not in user
    return messages, dm

def requests_for(doc, demos, schedule, template, settings):
    rows = build(doc, 7, settings['seed']); validate(rows, doc, 7)
    bank = {d['doc_id']: d for d in demos}
    for row in rows:
        demo = bank[schedule[doc['doc_id']][row['pass'] - 1]]
        messages, dm = render(doc, row, demo, template, settings['seed'])
        row.update(demo_id=demo['doc_id'], demo_mapping=dm, messages=messages)
        body = {'messages': messages, 'reasoning': {'effort': settings['reasoning_effort']},
                'max_tokens': settings['max_output_tokens']}
        bound = len(json_text(body).encode()) + 1024
        row.update(prompt_utf8_upper_bound=bound, prompt_characters=sum(len(m['content']) for m in messages),
                   prompt_sha256=digest(json_text(messages)), body=body,
                   custom_id=f'{row["arm"]}-{row["pass"]}-{digest(doc["doc_id"])[:12]}')
        if bound + settings['max_output_tokens'] > settings['max_context_tokens']:
            raise ValueError('conservative context bound exceeded; no truncation')
        price_bound = (bound * settings['input_price_per_million'] + settings['max_output_tokens'] * settings['output_price_per_million']) / 1e6
        if price_bound > settings['reservation_per_request_usd']:
            raise ValueError('request exceeds conservative paid reservation')
    # Deterministic shuffle blocks condition/time effects; no previous answer is fed back.
    random.Random(int(digest(f'{settings["seed"]}:{doc["doc_id"]}:order')[:16], 16)).shuffle(rows)
    return rows

def decode_response(row, item, settings):
    out = {'state': 'DELIVERY_UNKNOWN', 'delivered': False, 'valid': False,
           'positive': [], 'explicit_reject': [], 'outside_mask': [], 'duplicates_removed': 0}
    response = (item.get('response') or {}).get('body')
    if not isinstance(response, dict): return out
    out['delivered'] = True
    out['served_model'] = response.get('model')
    out['usage'] = response.get('usage') or {}
    if response.get('model') not in settings['allowed_served_models']:
        out['state'] = 'MODEL_MISMATCH'; return out
    choice = (response.get('choices') or [{}])[0]
    if choice.get('finish_reason') != 'stop':
        out['state'] = 'INCOMPLETE'; return out
    text = (choice.get('message') or {}).get('content')
    if not isinstance(text, str):
        out['state'] = 'INVALID_FORMAT'; return out
    try:
        # Preserve historical parser, but quarantine malformed items it silently ignores
        # and a final empty array shadowed by an earlier populated reasoning array.
        last = next(json.loads(b) for b in iter_json_arrays(text) if _is_json(b))
        arr = answer_array(text)
        if arr != last: raise ValueError('ambiguous final array under historical parser')
        if any(not isinstance(x, dict) or not isinstance(x.get('pair'), str)
               or len(x['pair'].split(',')) != 2 or x.get('label') not in {'causes', 'causedby', 'norel'} for x in arr):
            raise ValueError('invalid relation schema')
        triples = parse_triples(text)
        inv = {v: k for k, v in row['id_mapping'].items()}
        allowed = as_pairs(row['allowed_original'])
        positive, reject, outside = [], set(), set()
        for a, label, b in triples:
            if a not in inv or b not in inv or a == b: raise ValueError('unknown/self identifier')
            p = pair(inv[a], inv[b])
            if p not in allowed: outside.add(p); continue
            if label == 'norel': reject.add(p)
            else: positive.append(p)
        if set(positive) & reject: raise ValueError('contradictory rejection')
        out.update(state='VALID_EMPTY' if not arr else 'VALID', valid=True,
                   positive=sorted(set(positive)), explicit_reject=sorted(reject),
                   outside_mask=sorted(outside), duplicates_removed=len(positive) - len(set(positive)))
    except (ValueError, KeyError, TypeError, StopIteration):
        out['state'] = 'INVALID_FORMAT'
    return out

def _is_json(s):
    try: json.loads(s); return True
    except ValueError: return False

def ledger_for(doc, rows, results, settings):
    universe = as_pairs(doc['pairs'])
    ledgers = {arm: {p: dict(E_planned=0, E_submitted=0, E_delivered=0, E_valid=0, V=0, R=0)
                      for p in sorted(universe)} for arm in ARMS}
    events = []
    for row in rows:
        item = results.get(row['custom_id'])
        event = decode_response(row, item, settings) if item else dict(
            state='NOT_SENT', delivered=False, valid=False, positive=[], explicit_reject=[], outside_mask=[], duplicates_removed=0)
        event.update(doc_id=doc['doc_id'], topic=doc['topic'], arm=row['arm'], pass_index=row['pass'],
                     custom_id=row['custom_id'], prompt_sha256=row['prompt_sha256'])
        events.append(event)
        for p in as_pairs(row['allowed_original']):
            s = ledgers[row['arm']][p]; s['E_planned'] += 1
            s['E_submitted'] += int(item is not None)
            s['E_delivered'] += int(event['delivered']); s['E_valid'] += int(event['valid'])
        for p in event['positive']: ledgers[row['arm']][tuple(p)]['V'] += 1
        for p in event['explicit_reject']: ledgers[row['arm']][tuple(p)]['R'] += 1
    return ({arm: [{'pair': list(p), **s} for p, s in values.items()] for arm, values in ledgers.items()}, events)

def coverage(doc, ledger):
    sentence = {str(m['id']): m.get('sentence') for m in doc['mentions']}
    strata = defaultdict(list); neighbors = defaultdict(list)
    for item in ledger:
        a, b = item['pair']; sa, sb = sentence[a], sentence[b]
        cat = 'unknown' if sa is None or sb is None else ('intra' if sa == sb else 'inter')
        strata[cat].append(item); neighbors[a].append(item); neighbors[b].append(item)
    def stats(items):
        return {'pairs': len(items), **{field: dict(total=sum(x[field] for x in items),
                 covered_once=sum(x[field] >= 1 for x in items), covered_twice=sum(x[field] >= 2 for x in items))
                 for field in ('E_planned', 'E_submitted', 'E_delivered', 'E_valid')}}
    return {'all': stats(ledger), 'strata': {s: stats(v) for s, v in strata.items()},
            'mentions': {m: {'candidate_degree': len(neighbors[m]),
                         'valid_neighbor_fraction': sum(x['E_valid'] > 0 for x in neighbors[m]) / len(neighbors[m]) if neighbors[m] else None,
                         'valid_incident_exposures': sum(x['E_valid'] for x in neighbors[m])} for m in sentence}}

def metrics(gold, pred):
    tp, fp, fn = len(gold & pred), len(pred - gold), len(gold - pred)
    den = 2 * tp + fp + fn
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2 * tp / den if den else 0,
            'precision': tp / (tp + fp) if tp + fp else None, 'recall': tp / (tp + fn) if tp + fn else None}

def evaluate(evaluation_rows, ledgers):
    """Gold is accepted only here, after predictions/ledgers have been frozen to disk."""
    keys = {(r['doc_id'], *r['pair']) for r in evaluation_rows}
    gold = {(r['doc_id'], *r['pair']) for r in evaluation_rows if r['gold']}
    base = {(r['doc_id'], *r['pair']) for r in evaluation_rows if r['baseline_positive']}
    zero = {(r['doc_id'], *r['pair']) for r in evaluation_rows if r['luna_zero'] and not r['baseline_positive']}
    out = {'baseline': metrics(gold, base), 'arms': {}}
    basef = out['baseline']['f1']
    for arm in ARMS:
        proposed = {(did, *r['pair']) for did, l in ledgers.items() for r in l[arm] if r['V'] > 0}
        assert proposed <= keys
        adds = proposed & zero
        score = metrics(gold, base | adds)
        t = len(adds & gold)
        score.update(additions=len(adds), true_additions=t, false_additions=len(adds) - t,
                     addition_precision=t / len(adds) if adds else None, delta_f1=score['f1'] - basef,
                     utility=2*t - basef * len(adds), removed_true=0, removed_false=0,
                     raw_proposed=len(proposed), zero_vote_new_tp=t)
        by_topic = {}
        for topic in sorted({r['doc_id'].split('_')[0] for r in evaluation_rows}):
            def subset(s): return {x for x in s if x[0].split('_')[0] == topic}
            by_topic[topic] = {'baseline': metrics(subset(gold), subset(base)),
                               'augmented': metrics(subset(gold), subset(base | adds)),
                               'true_additions': len(subset(adds & gold)), 'false_additions': len(subset(adds - gold))}
        out['arms'][arm] = {**score, 'by_topic': by_topic}
    out['primary_M_minus_O_f1'] = out['arms']['M']['f1'] - out['arms']['O']['f1']
    out['status'] = 'exploratory consumed development; not a confirmation or a cost-matched method-gain claim'
    return out
