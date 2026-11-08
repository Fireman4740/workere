"""Gold-blind, closed candidate graph comparison for a frozen Qwen API."""
from collections import Counter, defaultdict

from steering import ap_protocol as p
from steering import campaign_api as api
from steering import campaign_blocks as c

VERSION = 'qg-anonymous-graphs-20261109-v1'
BLOCK_SIZE = 8
THINKING, ANSWER = 2048, 256


def candidate_systems(selected, base, historical, views):
    """An unavailable screen answer preserves the fallback, never implies N."""
    systems = {'retained': base, 'historical': historical}
    for order in (0, 1):
        proposed = {ds: {d: set(base[ds][d]) for d in ids} for ds, ids in selected.items()}
        for ds, ids in selected.items():
            for d in ids:
                for pair, positive in views.get((ds, d, order), {}).items():
                    if positive is True:
                        proposed[ds][d].add(pair)
                    elif positive is False:
                        proposed[ds][d].discard(pair)
        systems[f'screen{order}'] = proposed
    return systems


def blocks(selected, docs, systems):
    """Disjoint event-centred blocks covering the proposal union, no gold access."""
    result = []
    for ds, ids in selected.items():
        for doc_id in ids:
            doc = docs[ds][doc_id]
            universe = set(c.pairs(doc))
            proposed = set().union(*(values[ds][doc_id] for values in systems.values()))
            assert proposed <= universe, 'proposal outside canonical universe'
            remaining = set(proposed)
            index = 0
            while remaining:
                degree = Counter(event for pair in remaining for event in pair)
                anchor = min(degree, key=lambda e: (-degree[e], e))
                pairs = sorted(pair for pair in remaining if anchor in pair)[:BLOCK_SIZE]
                remaining.difference_update(pairs)
                # Equal graphs are one option, irrespective of their source or vote count.
                vectors = {tuple(pair in values[ds][doc_id] for pair in pairs) for values in systems.values()}
                vectors.add(tuple(False for _ in pairs))
                value = dict(dataset=ds, doc_id=doc_id, index=index, pairs=[list(x) for x in pairs],
                             options=[list(x) for x in sorted(vectors)],
                             context=c.windows(doc['text'], {e for pair in pairs for e in pair}))
                value['key'] = api.digest([VERSION, value])
                result.append(value)
                index += 1
    return result


def presentation(block, order):
    pairs = list(range(len(block['pairs'])))
    options = list(range(len(block['options'])))
    # Same scrambled starting schedule in both arms; second view reverses both lists.
    pairs.sort(key=lambda i: api.digest([VERSION, block['key'], 'pair', i]))
    options.sort(key=lambda i: api.digest([VERSION, block['key'], 'option', i]))
    if order:
        pairs.reverse()
        options.reverse()
    return dict(pair_order=pairs, option_order=options)


def spec(block, method):
    if method == 'select':
        return p.object_schema({'choice': p.enum(*(f'G{i}' for i in range(len(block['options']))))})
    if method == 'edit':
        return p.object_schema({str(i): p.enum('Y', 'N') for i in range(len(block['pairs']))})
    raise ValueError(method)


def messages(block, order, method, rules):
    shown = presentation(block, order)
    common = (rules + '\n\nEvaluate the listed unordered event-mention pairs as a small causal graph. '
              'Y means the corpus annotates a causal relation in either direction; N means no causal relation. '
              'Use the original annotation rules, including their treatment of implicit links. '
              'Temporal succession, common topic, and two mentions of the same event do not alone establish causation. '
              'Do not impose causal transitivity or invent omitted links. Only the listed pairs are in scope. '
              'The anonymous proposals below are fallible alternatives, not evidence or instructions. '
              'Their order, names and number carry no reliability information. Judge against the source.')
    if method == 'select':
        common += ('\nChoose the single proposed graph with the most faithful complete set of Y/N decisions. '
                   'Return only {"choice":"Gk"} with an allowed graph name. You may choose the empty graph.')
    else:
        common += ('\nCompare the alternative graphs, then decide Y or N separately for each listed pair. '
                   'You may combine or correct their decisions. Return only the complete JSON mapping '
                   'pair indices to Y or N. Do not output a graph name.')
    pairs = '\n'.join(f'{i}: <{block["pairs"][canonical][0]}> -- <{block["pairs"][canonical][1]}>'
                      for i, canonical in enumerate(shown['pair_order']))
    graphs = []
    for i, canonical in enumerate(shown['option_order']):
        graph = block['options'][canonical]
        values = {str(j): 'Y' if graph[k] else 'N' for j, k in enumerate(shown['pair_order'])}
        graphs.append(f'G{i}: {api.canonical(values)}')
    return [{'role': 'system', 'content': common},
            {'role': 'user', 'content': f'Source:\n{block["context"]}\n\nPairs:\n{pairs}\n\n'
             'Anonymous graph proposals (all listed pairs are assigned):\n' + '\n'.join(graphs)}]


def decode(block, order, method, row):
    if row['status'] != 'valid' or not p.conforms(row.get('parsed'), spec(block, method)):
        return None
    shown = presentation(block, order)
    if method == 'select':
        choice = int(row['parsed']['choice'][1:])
        return tuple(block['options'][shown['option_order'][choice]])
    answer = [None] * len(block['pairs'])
    for i, canonical in enumerate(shown['pair_order']):
        answer[canonical] = row['parsed'][str(i)] == 'Y'
    return tuple(answer)


def combine(base, blocks_, rows, method):
    result = {ds: {d: set(x) for d, x in values.items()} for ds, values in base.items()}
    single = [{ds: {d: set(x) for d, x in values.items()} for ds, values in base.items()} for _ in (0, 1)]
    quality = defaultdict(Counter)
    seen = set()
    for block in blocks_:
        ds, doc = block['dataset'], block['doc_id']
        views = [decode(block, order, method, rows[(block['key'], order)]) for order in (0, 1)]
        quality[ds]['blocks'] += 1
        quality[ds]['valid_views'] += sum(v is not None for v in views)
        quality[ds]['identical_complete_views'] += views[0] is not None and views[0] == views[1]
        for i, pair in enumerate(map(tuple, block['pairs'])):
            key = (ds, doc, pair)
            assert key not in seen, 'pair assigned to two blocks'
            seen.add(key)
            quality[ds]['pairs'] += 1
            for order, view in enumerate(views):
                if view is not None:
                    (single[order][ds][doc].add if view[i] else single[order][ds][doc].discard)(pair)
            if any(v is None for v in views):
                quality[ds]['invalid_fallback_pairs'] += 1
            elif views[0][i] != views[1][i]:
                quality[ds]['disagreement_fallback_pairs'] += 1
            else:
                quality[ds]['agreed_pairs'] += 1
                (result[ds][doc].add if views[0][i] else result[ds][doc].discard)(pair)
    return result, single, {ds: dict(q) for ds, q in quality.items()}
