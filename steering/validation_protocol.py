"""Frozen A--D binary policies; input planning never receives evaluation gold."""
from collections import defaultdict
import copy
import random

from steering import ap_campaign as ap
from steering import ap_protocol as p
from steering import campaign_adjudicate as a
from steering import campaign_api as api
from steering import campaign_blocks as c

VERSION = 'validation-combinations-20261109-v1'
SEED = 20261109
POLICIES = ('A_vote', 'B_filter', 'C_repair', 'D_filter_add')


def public_docs(docs):
    return {ds: {d: {k: row[k] for k in ('id', 'text', 'pair_list')}
                 for d, row in values.items()} for ds, values in docs.items()}


def screen_jobs(selected, docs, rules):
    """Exact C2 plain4096 text/order, including C1's original seeded shuffle."""
    jobs = []
    for ds, ids in selected.items():
        for doc_id in ids:
            pairs = c.pairs(docs[ds][doc_id])
            for block_index, start in enumerate(range(0, len(pairs), 12)):
                block = pairs[start:start+12]
                context = c.windows(docs[ds][doc_id]['text'], {e for pair in block for e in pair})
                for order in (0, 1):
                    indices = list(range(len(block)))
                    if order:
                        random.Random(f'{c.SEED}::{ds}::{doc_id}::{block_index}').shuffle(indices)
                    jobs.append(ap.job(c.make_messages(rules[ds], context, block, indices),
                        c.schema(len(block)), 4096, 384,
                        dict(dataset=ds, doc_id=doc_id, pairs=block, order=order,
                             layout='C2_original', block=block_index)))
    return jobs


def views_from(jobs, outputs):
    views = defaultdict(dict)
    for job in jobs:
        row, meta = outputs[job['key']], job['meta']
        obj = c.parse_answer(row.get('answer'), len(meta['pairs'])) if row['status'] == 'valid' else None
        if obj is None:
            continue
        view = views[(meta['dataset'], meta['doc_id'], meta['order'])]
        for i, pair in enumerate(meta['pairs']):
            pair = tuple(sorted(pair))
            assert pair not in view, 'duplicate pair in screen presentation'
            view[pair] = obj[str(i)] != 'N'
    return views


def routes_for(selected, docs, base, views):
    routes = []
    for ds, ids in selected.items():
        for doc_id in ids:
            routed = a.route_pairs(set(c.pairs(docs[ds][doc_id])), base[ds][doc_id],
                                  [views.get((ds, doc_id, order), {}) for order in (0, 1)])
            routes.extend(dict(dataset=ds, doc_id=doc_id, pair=list(pair),
                               baseline=pair in base[ds][doc_id]) for pair in sorted(routed))
    return routes


def combine(base, views, decisions):
    systems = {name: copy.deepcopy(base) for name in POLICIES}
    for ds, documents in base.items():
        for doc_id, positive in documents.items():
            for pair in positive:
                if all(views.get((ds, doc_id, o), {}).get(pair) is False for o in (0, 1)):
                    systems['B_filter'][ds][doc_id].discard(pair)
                    systems['D_filter_add'][ds][doc_id].discard(pair)
    systems['C_repair'] = p.apply(base, decisions)
    # Only original negatives are eligible for additions. This prevents repair
    # from re-adding an original positive rejected by the frozen filter.
    for (ds, doc_id, pair), label in decisions.items():
        if label in ('A', 'B') and pair not in base[ds][doc_id]:
            systems['D_filter_add'][ds][doc_id].add(pair)
    return systems


def fingerprint(job):
    """Only reuse identical scientific prompts, schema and token budgets."""
    return api.digest({k: job[k] for k in ('messages', 'schema', 'thinking', 'answer_tokens')})


def cache_identity(job):
    # Two presentation orders can coincidentally have identical text. Preserve
    # each recorded observation instead of replacing both by one lucky response.
    meta = job['meta']
    return (fingerprint(job), meta.get('dataset'), meta.get('doc_id'),
            meta.get('order'), meta.get('judgment'))


def document_schema():
    item = p.object_schema(dict(pair={'type': 'string'}, label=p.enum('causes', 'causedby')))
    return p.object_schema(dict(relations={'type': 'array', 'items': item}))


def document_parse(text):
    import json
    from steering.campaign_facts import unique_object
    try:
        obj = json.loads(text, object_pairs_hook=unique_object)
        if not isinstance(obj, dict) or set(obj) != {'relations'} or not isinstance(obj['relations'], list):
            return None
        for row in obj['relations']:
            if (not isinstance(row, dict) or set(row) != {'pair', 'label'} or
                    not isinstance(row['pair'], str) or row['pair'].count(',') != 1 or
                    row['label'] not in ('causes', 'causedby')):
                return None
        return obj
    except (TypeError, ValueError):
        return None


def document_vote(selected, docs, jobs, outputs):
    """Invalid reads count empty; a fixed five-read denominator prevents selection bias."""
    votes = defaultdict(list)
    quality = defaultdict(int)
    for job in jobs:
        row, meta = outputs[job['key']], job['meta']
        doc_id = meta['doc_id']
        obj = document_parse(row.get('answer')) if row['status'] == 'valid' else None
        chosen = set()
        if obj is None:
            quality['invalid_reads'] += 1
        else:
            universe = set(c.pairs(docs['MAVEN'][doc_id]))
            for item in obj['relations']:
                pair = tuple(sorted(x.strip() for x in item['pair'].split(',')))
                if pair in universe:
                    chosen.add(pair)
                else:
                    quality['out_of_universe_items'] += 1
        votes[doc_id].append(chosen)
    result = {'MAVEN': {}}
    for doc_id in selected['MAVEN']:
        reads = votes[doc_id]
        assert len(reads) == 5
        result['MAVEN'][doc_id] = {pair for pair in set().union(*reads)
                                  if sum(pair in reading for reading in reads) >= 3}
    return result, dict(quality)
