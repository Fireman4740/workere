"""Exploratory DEV44 logistic fusion. No model calls or reserved-data reads."""
import json
import os
from pathlib import Path
import re
import socket
from unittest.mock import patch
import numpy as np
from benchmark.configuration import read_yaml
from benchmark.io import fingerprint, write_json
from benchmark.metrics import set_metrics
from pipelines.judges.common import Dev44, GRID, f1_arr

SCHEMA = 'workere-offline-fusion/v1'
SCORE_FEATURES = ['luna_logit', 'd1_logit']
STRUCTURE_FEATURES = SCORE_FEATURES + ['model_gap', 'luna_order_gap', 'd1_order_gap',
    'sentence_distance_log1p', 'same_sentence', 'token_distance_log1p', 'same_surface']


def load(path):
    cfg = read_yaml(path)
    if cfg.get('schema') != SCHEMA or cfg.get('feature_set') not in {'scores', 'structure'}:
        raise ValueError('unsupported offline fusion configuration')
    if cfg.get('l2_penalty') != 1 or cfg.get('api_budget_usd') != 0:
        raise ValueError('fixed L2=1 and zero API budget required')
    for key in ('controls', 'suites_dir'):
        cfg[key] = str((Path(path).resolve().parent / cfg[key]).resolve())
    return cfg


def features(data, doc, a, b, feature_set):
    """Only public metadata and model scores enter the design matrix."""
    p, q = data.by_doc[doc][(a, b)], data.by_doc[doc][(b, a)]
    luna, d1 = (p['luna'] + q['luna']) / 2, (p['d1'] + q['d1']) / 2
    values = np.clip([luna, d1], 1e-4, 1 - 1e-4)
    out = list(np.log(values / (1 - values)))
    if feature_set == 'scores':
        return out
    s = data.suite[doc]
    gap = abs(s['sentence_of'][a] - s['sentence_of'][b])
    starts = {str(m): min(span) for m, span in zip(s['mentions'], s['spans'])}
    clean = lambda text: re.sub(r'\W+', ' ', text.casefold()).strip()
    left, right = clean(s['surfaces'][a]), clean(s['surfaces'][b])
    return out + [abs(luna - d1), abs(p['luna'] - q['luna']), abs(p['d1'] - q['d1']),
                  np.log1p(gap), float(gap == 0), np.log1p(abs(starts[a] - starts[b])),
                  float(bool(left) and left == right)]


def population(cfg):
    data = Dev44(cfg['controls'], cfg['suites_dir'])
    rows = []
    for doc in data.docs:
        for a, b in data.unordered(doc):
            p, q = data.by_doc[doc][(a, b)], data.by_doc[doc][(b, a)]
            if bool(p['gold']) != bool(q['gold']):
                raise ValueError('DEV44 labels must respect undirected scoring')
            rows.append(dict(doc=doc, topic=data.topic[doc], pair=[a, b],
                x=features(data, doc, a, b, cfg['feature_set']),
                fused=data.fused[doc][(a, b)], gold=bool(p['gold'])))
    if len(rows) * 2 != sum(len(v) for v in data.keys.values()):
        raise ValueError('candidate orders are not paired exactly')
    return data, rows


def plan(cfg):
    data, rows = population(cfg)
    return dict(schema=SCHEMA, feature_set=cfg['feature_set'], documents=len(data.docs),
        topics=len(set(data.topic.values())), unordered_pairs=len(rows),
        feature_names=SCORE_FEATURES if cfg['feature_set'] == 'scores' else STRUCTURE_FEATURES,
        l2_penalty=cfg['l2_penalty'], api_calls=0, api_cost_usd=0,
        unused_reserve_documents_read=0, res30_documents_read=0,
        protocol='exploratory DEV44 leave-one-topic-out; scaling/fit/threshold on training topics only',
        controls_fingerprint=fingerprint(json.loads(Path(cfg['controls']).read_text())),
        row_fingerprint=fingerprint(rows))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def fit(x, y, penalty=1.):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or not len(y) or not np.isfinite(x).all():
        raise ValueError('invalid training matrix')
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    reg = np.r_[0., np.full(x.shape[1], penalty)]
    beta = np.r_[np.log((y.sum() + .5) / (len(y) - y.sum() + .5)), np.zeros(x.shape[1])]
    objective = lambda b: float(np.logaddexp(0, z @ b).sum() - y @ (z @ b) + .5 * (reg * b * b).sum())
    for iteration in range(100):
        p = sigmoid(z @ beta)
        grad = z.T @ (p - y) + reg * beta
        hessian = (z.T * (p * (1 - p))) @ z + np.diag(reg + 1e-9)
        step = np.linalg.solve(hessian, grad)
        size = 1.
        while size > 1e-8 and objective(beta - size * step) > objective(beta):
            size *= .5
        beta -= size * step
        if np.max(np.abs(size * step)) < 1e-8:
            break
    else:
        raise ValueError('logistic fit did not converge')
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=beta.tolist(),
                l2_penalty=penalty, iterations=iteration + 1)


def predict(model, x):
    z = (np.asarray(x, float) - model['mean']) / model['scale']
    return sigmoid(np.column_stack((np.ones(len(z)), z)) @ np.array(model['coefficients']))


def counts(pred, gold):
    return np.array([(pred & gold).sum(), (pred & ~gold).sum(), (~pred & gold).sum()], int)


def threshold(scores, gold):
    table = np.array([counts(scores >= t, gold) for t in GRID])
    return float(GRID[int(np.argmax(f1_arr(table)))])


def ap(scores, gold):
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], gold[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]) + 1, len(s)]
    hits = np.cumsum(y)[ends - 1]
    return float(np.sum(hits / ends * np.diff(np.r_[0, hits])) / gold.sum()) if gold.sum() else 0.


def summary(per):
    total = np.sum(list(per.values()), axis=0)
    return dict(f1=float(100 * f1_arr(total)), **dict(zip(['tp', 'fp', 'fn'], map(int, total))))


def compare(candidate, base, topic, draws=4000):
    out = dict(candidate=summary(candidate), control=summary(base))
    out['delta_points'] = out['candidate']['f1'] - out['control']['f1']
    docs = sorted(base)
    for unit in ['document', 'topic']:
        groups = [[d] for d in docs] if unit == 'document' else [
            [d for d in docs if topic[d] == t] for t in sorted(set(topic.values()), key=int)]
        c = np.array([np.sum([candidate[d] for d in g], axis=0) for g in groups])
        b = np.array([np.sum([base[d] for d in g], axis=0) for g in groups])
        ids = np.random.default_rng(42).integers(0, len(groups), size=(draws, len(groups)))
        delta = np.sort(100 * (f1_arr(c[ids].sum(1)) - f1_arr(b[ids].sum(1))))
        out[unit + '_bootstrap'] = dict(samples=draws,
            ci95=[float(delta[int(.025 * draws)]), float(delta[int(.975 * draws) - 1])],
            ci97_5=[float(delta[int(.0125 * draws)]), float(delta[int(.9875 * draws) - 1])])
    return out


def evaluate(cfg, data, rows):
    x, y = np.array([r['x'] for r in rows]), np.array([r['gold'] for r in rows], bool)
    f, topics = np.array([r['fused'] for r in rows]), np.array([r['topic'] for r in rows])
    oof = np.full(len(rows), np.nan)
    decisions, tuned = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
    folds = {}
    for held in sorted(set(topics), key=int):
        train, test = topics != held, topics == held
        model = fit(x[train], y[train], cfg['l2_penalty'])
        t = threshold(predict(model, x[train]), y[train])
        control_t = threshold(f[train], y[train])
        oof[test] = predict(model, x[test])
        decisions[test], tuned[test] = oof[test] >= t, f[test] >= control_t
        train_docs = sorted({r['doc'] for r, keep in zip(rows, train) if keep})
        test_docs = sorted({r['doc'] for r, keep in zip(rows, test) if keep})
        assert all(data.topic[d] != held for d in train_docs)
        assert not set(train_docs) & set(test_docs)
        folds[str(held)] = dict(training_documents=train_docs, test_documents=test_docs,
            model=model, threshold=t, threshold_only_control=control_t)
        print('OFFLINE_FUSION_FOLD ' + json.dumps(dict(topic=str(held), training_pairs=int(train.sum()),
            test_pairs=int(test.sum()), threshold=t, iterations=model['iterations'])), flush=True)
    assert np.isfinite(oof).all()
    variants = dict(base=f >= .39, candidate=decisions, threshold_only=tuned)
    per = {name: {} for name in variants}
    for doc in data.docs:
        mask = np.array([r['doc'] == doc for r in rows])
        keys = [tuple(r['pair']) for r, keep in zip(rows, mask) if keep]
        gold = {k for k, keep in zip(keys, y[mask]) if keep}
        for name, decision in variants.items():
            per[name][doc] = counts(decision[mask], y[mask]).tolist()
            chosen = {k for k, keep in zip(keys, decision[mask]) if keep}
            m = set_metrics(gold, chosen)
            assert [m['tp'], m['predicted'] - m['tp'], m['support'] - m['tp']] == per[name][doc]
    full = fit(x, y, cfg['l2_penalty'])
    full['threshold'] = threshold(predict(full, x), y)
    return {**plan(cfg), 'schema': 'workere-offline-fusion-result/v1', 'effective_configuration': cfg,
        'source_run_id': os.environ.get('ORX_RUN_ID'), 'source_commit': os.environ.get('ORX_COMMIT'),
        'primary_vs_frozen': compare(per['candidate'], per['base'], data.topic),
        'candidate_vs_threshold_only': compare(per['candidate'], per['threshold_only'], data.topic),
        'threshold_only_vs_frozen': compare(per['threshold_only'], per['base'], data.topic),
        'average_precision': dict(base=ap(f, y), candidate=ap(oof, y)),
        'folds': folds, 'per_document_counts': per, 'full_dev44_model_for_future_only': full,
        'inference_limit': 'Repeatedly used DEV44; exploratory. Bootstrap holds fitted fold choices fixed.',
        'candidate_score_fingerprint': fingerprint(oof.tolist()), 'network_blocked': True}


def run(config_path, output):
    cfg = load(config_path)
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('network forbidden in offline experiment')), \
            patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('network forbidden in offline experiment')):
        data, rows = population(cfg)
        print('OFFLINE_FUSION_PLAN ' + json.dumps(plan(cfg)), flush=True)
        result = evaluate(cfg, data, rows)
        write_json(output, result)
        print('OFFLINE_FUSION_RESULT ' + json.dumps(result), flush=True)
    return 0
