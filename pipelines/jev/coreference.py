"""Dedicated event-identity queries; no causal question or oracle chains in requests."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.io import write_json, write_jsonl
from benchmark.protocols import public_inputs
from pipelines.jev.client import DecisionClient
from pipelines.jev.state_builder import build_state

QUESTION = {'same_event': {'type': 'noul',
    'instructions': 'Do the two exact selected event mentions refer to the very same real-world event occurrence? Different descriptions may denote one occurrence. Answer true only for event identity. Distinct causes, consequences, enabling events, subevents, repeated similar events and events merely close in time are not the same occurrence.',
    'criteria': {'true': 'Both selected mentions denote the identical real-world event instance.',
                 'false': 'The mentions denote distinct events, including causally related events, subevents, repeated events, or mere temporal or topical association.'}}}


def unordered_pairs(entities, candidates):
    refs = {e.id: e for e in entities}
    return list(dict.fromkeys(tuple(sorted((a, b), key=lambda m: (refs[m].token_indices, m)))
                              for a, b in candidates))


def connected_components(mentions, edges, threshold=.5):
    parent = {m: m for m in mentions}
    def find(m):
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m
    for a, b, p in edges:
        if a not in parent or b not in parent:
            raise ValueError('unknown coreference endpoint')
        if p >= threshold:
            left, right = sorted((find(a), find(b)))
            parent[right] = left
    return {m: find(m) for m in mentions}


def collect(config, suite, output):
    """One canonical query per unordered pair, shared client and bounded concurrency."""
    output = Path(output)
    client = DecisionClient(**config['coreference_client'])
    lock, stop = threading.Lock(), threading.Event()
    stats = {'logical_queries': 0, 'api_calls': 0, 'cache_hits': 0,
             'known_api_cost_usd': 0.0, 'unknown_cost_events': 0}
    tasks, inputs = [], {}
    for item in public_inputs(suite, 'relations_gold'):
        refs = {e.id: e for e in item.reference_entities}
        inputs[item.document.id] = item
        for a, b in unordered_pairs(item.reference_entities, item.reference_candidates):
            tasks.append((item.document, refs[a], refs[b]))
    if len(tasks) != config['api_calls']:
        client.close()
        raise ValueError('coreference query count differs from the committed plan')
    try:
        with (output / 'coreference-decisions.jsonl').open('x') as journal:
            def one(task):
                document, a, b = task
                if stop.is_set():
                    raise RuntimeError('another coreference query failed; no further requests')
                state = build_state(document, a, b, 'document')
                def record(event):
                    with lock:
                        journal.write(json.dumps({'document_id': document.id, 'source': a.id, 'target': b.id, **event})+'\n')
                        journal.flush()
                        stats['api_calls'] += bool(event.get('api_call'))
                        stats['cache_hits'] += bool(event['cache_hit'])
                        if event['cost_usd'] is None:
                            stats['unknown_cost_events'] += 1
                        else:
                            stats['known_api_cost_usd'] += event['cost_usd']
                try:
                    packet = client.decide(state, QUESTION, on_attempt=record)
                    p = packet['response']['answers']['same_event']['noul']
                    with lock:
                        stats['logical_queries'] += 1
                        if stats['logical_queries'] % 100 == 0 or stats['logical_queries'] == len(tasks):
                            print('COREF_PROGRESS '+json.dumps(stats),flush=True)
                    return {'id': document.id, 'source': a.id, 'target': b.id,
                            'positive_probability': p, 'cache_key': packet['cache_key']}
                except BaseException:
                    stop.set()
                    raise
            with ThreadPoolExecutor(max_workers=config['max_workers']) as pool:
                responses = list(pool.map(one, tasks))
        write_jsonl(output / 'coreference-responses.jsonl', responses)
        by_doc = {d: [] for d in inputs}
        for row in responses:
            by_doc[row['id']].append((row['source'], row['target'], row['positive_probability']))
        clusters = {d: connected_components([e.id for e in inputs[d].reference_entities], edges, .5)
                    for d, edges in by_doc.items()}
        stats['api_cost_usd'] = None if stats['unknown_cost_events'] else stats['known_api_cost_usd']
        stats['coreference_threshold'] = .5
        stats['question'] = QUESTION
        write_json(output / 'coreference-clusters.json', clusters)
        write_json(output / 'coreference-resources.json', stats)
        return clusters, stats
    finally:
        client.close()
