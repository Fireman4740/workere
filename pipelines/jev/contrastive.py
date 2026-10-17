"""Lightweight structural demo retrieval and explicit training-bank negatives.

Only public query tokens, sentence boundaries and reference mention spans enter
retrieval. Training-bank labels select valid contrasts; query labels never enter.
This is SERE-inspired guidance, not its ConceptNet/syntax-tree implementation.
"""
import hashlib
import math
from collections import Counter, defaultdict

from benchmark.data import parse_annotations


def profile(tokens, sentences, spans):
    words = [t.lower() for t in tokens]
    n = len(spans)
    counts = Counter(next((i for i, (a, b) in enumerate(sentences) if a <= sp[0] < b), -1) for sp in spans)
    intra = sum(c * (c - 1) for c in counts.values()) / max(n * (n - 1), 1)
    names = [' '.join(tokens[i].lower() for i in sp) for sp in spans]
    cause = {'because', 'caused', 'causing', 'therefore', 'triggered', 'prevent', 'prevents', 'prevented', 'due', 'led', 'resulted'}
    temporal = {'after', 'before', 'while', 'when', 'following', 'then'}
    modal = {'not', 'never', 'no', 'would', 'could', 'might', 'can', 'should', "n't"}
    return (min(n / 60, 1), intra, min(n / max(len(sentences), 1) / 5, 1),
            1 - len(set(names)) / max(n, 1),
            min(30 * sum(w in cause for w in words) / max(len(words), 1), 1),
            min(20 * sum(w in temporal for w in words) / max(len(words), 1), 1),
            min(20 * sum(w in modal for w in words) / max(len(words), 1), 1))


def negative_pairs(demo, limit=4):
    gold = {frozenset((a, b)) for a, lab, b in parse_annotations(demo['annots']) if lab != 'norel'}
    neighbors = defaultdict(set)
    for pair in gold:
        if len(pair) == 2:
            a, b = pair
            neighbors[a].add(b); neighbors[b].add(a)
    chain = {demo['mentions'][i]: k for k, cl in enumerate(demo.get('coref_clusters', [])) for i in cl}
    sent = {m: next((i for i, (a, b) in enumerate(demo['sentences']) if a <= sp[0] < b), -1)
            for m, sp in zip(demo['mentions'], demo['spans'])}
    candidates = []
    mentions = demo['mentions']
    for i, a in enumerate(mentions):
        for b in mentions[i + 1:]:
            if frozenset((a, b)) in gold:
                continue
            same = a in chain and b in chain and chain[a] == chain[b]
            shared = bool(neighbors[a] & neighbors[b])
            local = sent[a] == sent[b]
            category = 'same_occurrence' if same else 'shared_neighbor' if shared else 'same_sentence' if local else 'other'
            candidates.append((-(3 * same + 2 * shared + local), int(a), int(b), a, b, category))
    candidates.sort()
    selected = []
    for cat in ['same_occurrence', 'shared_neighbor', 'same_sentence', 'other']:
        found = next((r for r in candidates if r[-1] == cat), None)
        if found is not None:
            selected.append(found)
    for candidate in candidates:
        if len(selected) >= limit:
            break
        if candidate not in selected:
            selected.append(candidate)
    return [(r[3], r[4]) for r in selected[:limit]]


def select_demonstrations(document, entities, bank, passes=7):
    query = profile(document.tokens, document.sentences, [e.token_indices for e in entities])
    def score(demo):
        vector = profile(demo['tokens'], demo['sentences'], demo['spans'])
        dot = sum(a * b for a, b in zip(query, vector))
        scale = math.sqrt(sum(a * a for a in query) * sum(a * a for a in vector))
        return dot / scale if scale else 0.
    eligible = [d for d in bank if d['id'].split('_')[0] != document.id.split('_')[0]
                and any(lab != 'norel' for _, lab, _ in parse_annotations(d['annots'])) and negative_pairs(d)]
    if len(eligible) < passes:
        raise ValueError('insufficient contrasting demonstration documents outside query topic')
    return sorted(eligible, key=lambda d: (-score(d), hashlib.sha256(('42::' + document.id + '::' + d['id']).encode()).hexdigest()))[:passes]
