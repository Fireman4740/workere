"""Train-only screen of conclusion/causal-recall weight-space directions.

Stages run in separate processes, releasing the GPU between extraction and decode.
No test-panel labels, gradients, training optimizer, or external model calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from steering import ctb
from steering.weight_space import (LAYERS, SEED, condition_plan, counts, digest,
                                  final_pairs, install_writers, make_split, metrics,
                                  orthogonal_control, paired_interval, right_factor, unit)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def unique_rows(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError(f'duplicate record {k}')
        result[k] = row
    return result


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suite, prompt = ctb.load_suite(), ctb.load_prompt()
    split = make_split(suite, prompt, ctb.eval_ids())
    traces = unique_rows(read_rows(args.traces), lambda r: r['doc_id'])
    prompts = unique_rows(read_rows(args.prompts), lambda r: r['doc_id'])
    for doc_id in split['fit'] + split['screen']:
        if doc_id not in traces or doc_id not in prompts:
            raise ValueError(f'missing fit trace/prompt: {doc_id}')
        if prompts[doc_id]['messages'] != ctb.messages_for(suite[doc_id], prompt):
            raise ValueError(f'cached training prompt differs: {doc_id}')
    # Exact cached prompt identity was checked above. Use its recorded token count
    # with a conservative allowance for the forced-closing instruction.
    screen_context_bound = max(int(traces[d]['prompt_tokens']) for d in split['screen']) + 8192 + 4096 + 256
    if screen_context_bound >= args.max_model_len:
        raise ValueError(f'screen context needs {screen_context_bound} tokens, limit={args.max_model_len}')
    manifest = {
        'protocol': 'wsp-causal-v1', 'split': split,
        'source_sha256': {name: sha(path) for name, path in (
            ('traces', args.traces), ('prompts', args.prompts),
            ('suite', ctb.DATA / 'ctb-suite.json'), ('prompt', ctb.DATA / 'ctb-prompt.json'))},
        'implementation_sha256': {name: sha(HERE / name) for name in
                                  ('weight_space.py', 'wsp_pilot.py', 'run_vllm.py', 'offline.py', 'ctb.py')},
        'conditions': condition_plan(), 'model': args.model,
        'max_model_len': args.max_model_len, 'batch': args.batch,
        'screen_context_upper_bound': screen_context_bound,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'representation': 'quantized base + FP32 rank-one sidecar; no merged/requantized checkpoint',
        'selection': 'screen only; confirmation and historical panels are not opened for scoring',
        'precision_floor_delta_pp': -2.0,
        'primary': 'micro binary F1 on ordered candidate pairs, all screen documents',
        'inference_limit': 'Exploratory split of an already studied corpus; not a new independent benchmark.',
    }
    manifest['id'] = digest(manifest)
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('manifest drift: use a fresh output directory')
    write_json(path, manifest)
    rows = [{'doc_id': d, 'pass': 0, 'messages': ctb.messages_for(suite[d], prompt)}
            for d in split['screen']]
    (out / 'screen_prompts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    census = {group: {'documents': len(ids),
                      'candidate_pairs_unordered': sum(len(ctb.unordered_pairs(suite[d])) for d in ids),
                      'gold_pairs_unordered': sum(len(ctb.causal_pairs(suite[d])) for d in ids)}
              for group, ids in split.items() if group in ('fit', 'screen')}
    fit_valid = [traces[d] for d in split['fit'] if final_pairs(traces[d]) is not None]
    census['fit']['valid_cached_traces'] = len(fit_valid)
    write_json(out / 'census.json', census)
    print('WSP_PREFLIGHT ' + json.dumps({'manifest_id': manifest['id'], 'census': census}), flush=True)
    return manifest


def span_positions(tokenizer, text, start, end):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > a and a >= start and b <= end]
    return encoded['input_ids'], positions


def capture_trace(llm, model, tokenizer, head, thinking, max_model_len):
    """Token-weighted writer moments and document-weighted cc contrasts.

    The 48-token tail is compared to DISJOINT earlier thinking positions; neither
    prompt tokens nor the end-of-thinking marker enter either mean.
    """
    import torch
    from interventions import find_layer
    from offline import find_module, sampling_params

    text = head + thinking
    tokens, positions = span_positions(tokenizer, text, len(head), len(text))
    if len(tokens) >= max_model_len - 2 or len(positions) < 96:
        return None, {'reason': 'overlength_or_short', 'tokens': len(tokens), 'thinking': len(positions)}
    store, handles = {}, []

    def writer_hook(layer):
        def hook(module, inputs, output):
            key = f'writer_{layer}'
            if key in store:
                return
            x, y = inputs[0], output[0] if isinstance(output, tuple) else output
            # Bound temporary FP32 memory: never materialize all token activations.
            total = torch.zeros(x.shape[-1], dtype=torch.float32, device=x.device)
            norm_total = torch.zeros((), dtype=torch.float32, device=x.device)
            for offset in range(0, len(positions), 128):
                ix = torch.tensor(positions[offset:offset + 128], device=x.device)
                total += x.index_select(0, ix).float().sum(0)
                norm_total += y.index_select(0, ix).float().norm(dim=-1).sum()
            store[key] = (total.cpu().numpy(), float(norm_total.cpu()), len(positions))
        return hook

    def residual_hook(layer):
        def hook(module, inputs, output):
            key = f'cc_{layer}'
            if key in store:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            residual = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            def average(selected):
                total = torch.zeros(hidden.shape[-1], device=hidden.device, dtype=torch.float32)
                for offset in range(0, len(selected), 128):
                    ix = torch.tensor(selected[offset:offset + 128], device=hidden.device)
                    rows = hidden.index_select(0, ix).float()
                    if residual is not None:
                        rows += residual.index_select(0, ix).float()
                    total += rows.sum(0)
                return total / len(selected)
            store[key] = (average(positions[-48:]) - average(positions[:-48])).cpu().numpy()
        return hook

    for layer in LAYERS:
        handles.append(find_module(model, layer, 'down_proj')[1].register_forward_hook(writer_hook(layer)))
        handles.append(find_layer(model, layer).register_forward_hook(residual_hook(layer)))
    try:
        llm.generate([{'prompt_token_ids': tokens}], sampling_params(1), use_tqdm=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(store) != 2 * len(LAYERS):
        raise RuntimeError('activation capture incomplete')
    return store, {'tokens': len(tokens), 'thinking': len(positions)}


def extract(args):
    from interventions import capture_rows, find_layer
    from offline import build_llm, render
    from probe import end_positions
    import vllm
    if vllm.__version__ != '0.32.0':
        raise RuntimeError(f'verified runtime is vLLM 0.32.0, found {vllm.__version__}')

    manifest = prepare(args)
    out = Path(args.out)
    target = out / 'directions.npz'
    if target.exists():
        meta = json.loads((out / 'directions.json').read_text())
        if meta['manifest_id'] != manifest['id'] or meta['npz_sha256'] != sha(target):
            raise ValueError('direction cache mismatch')
        print('WSP_REUSE_DIRECTIONS ' + meta['npz_sha256'], flush=True)
        return
    suite = ctb.load_suite()
    traces = unique_rows(read_rows(args.traces), lambda r: r['doc_id'])
    prompts = unique_rows(read_rows(args.prompts), lambda r: r['doc_id'])
    llm, model, tokenizer = build_llm(args.model, args.max_model_len,
                                     args.gpu_memory_utilization, eager=True)
    stats, tp_vectors, fn_vectors = {}, {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
    details = []
    taps = {f'L{l}': (find_layer(model, l), 'resid') for l in LAYERS}
    for number, doc_id in enumerate(manifest['split']['fit'], 1):
        row, doc = traces[doc_id], suite[doc_id]
        predicted = final_pairs(row)
        if predicted is None:
            details.append({'doc': doc_id, 'skip': 'unfinished_or_invalid_cached_trace'})
            continue
        head = render(tokenizer, prompts[doc_id]['messages'])
        thinking = row['text'].split('</think>', 1)[0]
        store, detail = capture_trace(llm, model, tokenizer, head, thinking, args.max_model_len)
        if store is None:
            details.append({'doc': doc_id, 'skip': detail})
            continue
        for layer in LAYERS:
            u_sum, norm_sum, n = store[f'writer_{layer}']
            st = stats.setdefault(layer, {'u': np.zeros_like(u_sum, dtype=np.float64),
                                         'norm': 0.0, 'n': 0, 'cc': []})
            st['u'] += u_sum
            st['norm'] += norm_sum
            st['n'] += n
            st['cc'].append(store[f'cc_{layer}'])
        gold = ctb.causal_pairs(doc) & set(ctb.unordered_pairs(doc))
        predicted_unordered = {tuple(sorted(p)) for p in predicted}
        pairs = sorted(gold)
        detail.update({'doc': doc_id, 'tp_pairs': len(gold & predicted_unordered),
                       'fn_pairs': len(gold - predicted_unordered)})
        # Common neutral pair query for BOTH classes: no gold label in the text.
        if pairs:
            surface = ctb.surfaces(doc)
            full = head + thinking + '\n\nPair check: is there a causal relation between these events?\n'
            ends = []
            for a, b in pairs:
                full += f'{a} ({surface[a]}) / {b} ({surface[b]}):'
                ends.append(len(full))
                full += '\n'
            positions = end_positions(tokenizer, full, ends)
            if len(tokenizer(full, add_special_tokens=False)['input_ids']) < args.max_model_len - 2:
                captured = capture_rows(llm, full, taps, positions)
                for layer in LAYERS:
                    x = captured[f'L{layer}'].numpy()
                    good = np.array([p in predicted_unordered for p in pairs])
                    if good.any():
                        tp_vectors[layer].append(x[good].mean(0))
                    if (~good).any():
                        fn_vectors[layer].append(x[~good].mean(0))
                detail['recall_captured'] = True
            else:
                detail['recall_captured'] = False
        details.append(detail)
        print('WSP_EXTRACT ' + json.dumps({'index': number, **detail}), flush=True)
    if not stats or min(len(st['cc']) for st in stats.values()) < 8:
        raise RuntimeError('fewer than 8 usable fit traces; no screen launched')
    if min(len(tp_vectors[l]) for l in LAYERS) < 3 or min(len(fn_vectors[l]) for l in LAYERS) < 3:
        raise RuntimeError('recall contrast needs at least 3 documents per class; no fallback direction')
    payload, layer_report = {}, {}
    for layer in LAYERS:
        st = stats[layer]
        mean_input, mean_norm = st['u'] / st['n'], st['norm'] / st['n']
        right = right_factor(mean_input, mean_norm)
        cc = unit(np.mean(st['cc'], axis=0))
        recall = unit(np.mean(tp_vectors[layer], axis=0) - np.mean(fn_vectors[layer], axis=0))
        payload[f'right_{layer}'] = right
        report = {'thinking_tokens': st['n'], 'fit_documents': len(st['cc']),
                  'true_positive_documents': len(tp_vectors[layer]),
                  'false_negative_documents': len(fn_vectors[layer]),
                  'mean_output_norm': mean_norm, 'mean_input_norm': float(np.linalg.norm(mean_input)),
                  'delta_frobenius_at_0.10': 0.1 * float(np.linalg.norm(right)),
                  'cc_recall_cosine': float(cc @ recall)}
        for family, d in (('cc', cc), ('recall', recall)):
            payload[f'{family}_{layer}'] = d
            for seed in (0, 1):
                r = orthogonal_control(d, SEED + 1000 * seed + layer + (100 if family == 'recall' else 0))
                payload[f'{family}_random{seed}_{layer}'] = r
                report[f'{family}_random{seed}_cosine'] = float(d @ r)
        layer_report[str(layer)] = report
    np.savez(target, **payload)
    summary = {'manifest_id': manifest['id'], 'npz_sha256': sha(target), 'layers': layer_report,
               'documents': details, 'recall': 'document-balanced mean(TP) minus mean(FN), gold causal pairs only',
               'cc': 'document-balanced residual last 48 think tokens minus other think tokens',
               'writer_moments': 'token-weighted across completed fit traces only',
               'limitation': 'Directions refitted on CTB; different data, precision, engine and decoding from paper.'}
    write_json(out / 'directions.json', summary)
    print('WSP_DIRECTIONS ' + json.dumps({k: v for k, v in summary.items() if k != 'documents'}), flush=True)


def apply_arm(writers, data, arm):
    for writer in writers.values():
        writer.clear()
    if arm['family']:
        family = arm['family']
        if 'random_seed' in arm:
            family += f"_random{arm['random_seed']}"
        for layer, writer in writers.items():
            writer.set(data[f'{family}_{layer}'], data[f'right_{layer}'], arm['alpha'])


def generate(args):
    from offline import build_llm
    from run_vllm import generate_condition
    import vllm
    if vllm.__version__ != '0.32.0':
        raise RuntimeError(f'verified runtime is vLLM 0.32.0, found {vllm.__version__}')

    manifest = prepare(args)
    out = Path(args.out)
    extraction = json.loads((out / 'directions.json').read_text())
    if extraction['manifest_id'] != manifest['id'] or extraction['npz_sha256'] != sha(out / 'directions.npz'):
        raise ValueError('extraction provenance mismatch')
    data = np.load(out / 'directions.npz')
    rows = read_rows(out / 'screen_prompts.jsonl')
    conditions = manifest['conditions']
    output = out / 'screen.jsonl'
    done = unique_rows(read_rows(output), lambda r: (r['condition'], r['doc_id'])) if output.exists() else {}
    allowed = {(c['name'], r['doc_id']) for c in conditions for r in rows}
    if not set(done) <= allowed:
        raise ValueError('foreign rows in generation cache')
    for row in done.values():
        if row.get('manifest_id') != manifest['id'] or row.get('directions_sha256') != extraction['npz_sha256']:
            raise ValueError('generation cache provenance mismatch')
    if len(done) == len(allowed):
        print('WSP_GENERATIONS_COMPLETE', flush=True)
        return
    writers = install_writers()
    llm, _, tokenizer = build_llm(args.model, args.max_model_len, args.gpu_memory_utilization,
                                 max_num_seqs=args.batch)
    if set(writers) != set(LAYERS):
        raise RuntimeError(f'not all nine MLP sites installed: {sorted(writers)}')
    # Same hook path for all arms. This catches a missing device-buffer update.
    first = [rows[0]]
    from offline import render
    from vllm import SamplingParams
    probe_text = render(tokenizer, first[0]['messages'])
    probe_params = SamplingParams(max_tokens=2, temperature=0.0, logprobs=20)
    base_probe = llm.generate([probe_text], probe_params, use_tqdm=False)[0]
    apply_arm(writers, data, conditions[1])
    edit_probe = llm.generate([probe_text], probe_params, use_tqdm=False)[0]
    apply_arm(writers, data, conditions[0])
    reset_probe = llm.generate([probe_text], probe_params, use_tqdm=False)[0]
    if list(base_probe.outputs[0].token_ids) != list(reset_probe.outputs[0].token_ids):
        raise RuntimeError('baseline did not restore after clearing edit')
    base_probs = base_probe.outputs[0].logprobs[0]
    edit_probs = edit_probe.outputs[0].logprobs[0]
    reset_probs = reset_probe.outputs[0].logprobs[0]
    common = base_probs.keys() & edit_probs.keys()
    delta = max((abs(base_probs[k].logprob - edit_probs[k].logprob) for k in common), default=0.)
    reset_delta = max((abs(base_probs[k].logprob - reset_probs[k].logprob)
                       for k in base_probs.keys() & reset_probs.keys()), default=float('inf'))
    if delta <= 1e-5 or reset_delta > 1e-5:
        raise RuntimeError(f'edit not visible or reset failed: edit delta={delta}, reset={reset_delta}')
    print('WSP_HOOK_SMOKE ' + json.dumps({'sites': sorted(writers), 'reset_equal': True,
          'first_tokens': [list(x.outputs[0].token_ids) for x in (base_probe, edit_probe, reset_probe)],
          'max_first_token_logprob_change': delta, 'reset_logprob_delta': reset_delta}), flush=True)

    class ProvenanceSink:
        def __init__(self, stream):
            self.stream = stream
        def write(self, line):
            row = json.loads(line)
            row.update(manifest_id=manifest['id'], directions_sha256=extraction['npz_sha256'])
            self.stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        def flush(self):
            self.stream.flush()

    for arm in conditions:
        pending = [r for r in rows if (arm['name'], r['doc_id']) not in done]
        if not pending:
            continue
        apply_arm(writers, data, arm)
        print('WSP_ARM ' + json.dumps(arm), flush=True)
        with output.open('a') as stream:
            generate_condition(llm, tokenizer, pending, arm, 12288, ProvenanceSink(stream))
        score(args, partial=True)


def purpose_candidates(doc):
    """Broad lexical diagnostic only; this is NOT a gold purpose/enablement label."""
    pattern = re.compile(r'\b(to|allow\w*|enabl\w*|help\w*|prevent\w*|block\w*|so that|in order)\b', re.I)
    # Sentence structure varies by dataset version; tagged document spans are stable.
    texts = re.split(r'(?<=[.!?])\s+', doc['text'])
    flagged = set()
    for a, b in ctb.unordered_pairs(doc):
        markers = (f'<{a} ', f'<{b} ')
        if any(all(m in text for m in markers) and pattern.search(text) for text in texts):
            flagged |= {(a, b), (b, a)}
    return flagged


def score(args, partial=False):
    out = Path(args.out)
    manifest = json.loads((out / 'manifest.json').read_text())
    direction_meta = json.loads((out / 'directions.json').read_text())
    if direction_meta['manifest_id'] != manifest['id'] or direction_meta['npz_sha256'] != sha(out / 'directions.npz'):
        raise ValueError('scoring direction provenance mismatch')
    suite = ctb.load_suite()
    ids = manifest['split']['screen']
    expected = {(c['name'], d) for c in manifest['conditions'] for d in ids}
    rows = unique_rows(read_rows(out / 'screen.jsonl'), lambda r: (r['condition'], r['doc_id']))
    if not set(rows) <= expected:
        raise ValueError('unexpected condition/document in screen outputs')
    if not partial and set(rows) != expected:
        raise ValueError(f'incomplete screen: {len(rows)}/{len(expected)} records')
    predictions, results, per_doc = {}, {}, {}
    for arm in manifest['conditions']:
        name = arm['name']
        if any((name, d) not in rows for d in ids):
            continue
        predictions[name], per_doc[name] = {}, {}
        invalid, tokens, thinking_tokens, forced, outside = 0, [], [], 0, 0
        purpose_counts = {}
        for d in ids:
            row = rows[(name, d)]
            if (row.get('manifest_id') != manifest['id'] or
                    row.get('directions_sha256') != direction_meta['npz_sha256']):
                raise ValueError('scoring manifest mismatch')
            pred = final_pairs(row)
            invalid += pred is None
            pred = pred or set()
            candidates = {tuple(p) for p in suite[d]['pair_list']}
            outside += len(pred - candidates)
            gold = {(a, b) for a, label, b in suite[d]['gold'] if label != 'norel'}
            # Retain every document, including invalid answers, with all its FN.
            per_doc[name][d] = counts(pred, gold)
            predictions[name][d] = pred
            proxy = purpose_candidates(suite[d])
            purpose_counts[d] = counts(pred & proxy, gold & proxy)
            tokens.append(row['num_tokens'])
            thinking_tokens.append(row.get('thinking_tokens', row['num_tokens']))
            forced += row['budget_forced']
        results[name] = {**metrics(per_doc[name]), 'documents': len(ids), 'invalid_answers': invalid,
                         'outside_candidate_pairs': outside, 'mean_generated_tokens': float(np.mean(tokens)),
                         'mean_thinking_tokens': float(np.mean(thinking_tokens)),
                         'forced_closures': forced, 'lexical_purpose_proxy': metrics(purpose_counts)}
    if 'base' in per_doc:
        for name in results:
            if name == 'base':
                continue
            results[name]['vs_base_f1_pp'] = paired_interval(per_doc['base'], per_doc[name])
            added, lost, fp_added = 0, 0, 0
            for d in ids:
                gold = {(a, b) for a, label, b in suite[d]['gold'] if label != 'norel'}
                add = predictions[name][d] - predictions['base'][d]
                remove = predictions['base'][d] - predictions[name][d]
                added += len(add & gold)
                lost += len(remove & gold)
                fp_added += len(add - gold)
            results[name]['changes_vs_base'] = {'recovered_gold_ordered': added,
                                               'lost_gold_ordered': lost, 'added_fp_ordered': fp_added}
            family = next(c['family'] for c in manifest['conditions'] if c['name'] == name)
            if name == f'{family}_0.10':
                for seed in (0, 1):
                    ctrl = f'{family}_random{seed}_0.10'
                    if ctrl in per_doc:
                        results[name][f'vs_random{seed}_f1_pp'] = paired_interval(per_doc[ctrl], per_doc[name])
    result = {'manifest_id': manifest['id'], 'complete': set(rows) == expected,
              'status': 'exploratory_train_screen', 'conditions': results, 'per_document': per_doc,
              'caveat': 'Unadjusted paired intervals describe a multi-arm screen, not confirmation.'}
    write_json(out / 'screen_scores.json', result)
    print('WSP_SCORES ' + json.dumps({k: v for k, v in result.items() if k != 'per_document'}), flush=True)
    if result['complete']:
        # No test run is launched by this script. Freeze one candidate before confirmation.
        eligible = []
        for family in ('cc', 'recall'):
            chosen = results[f'{family}_0.10']
            controls = [results[f'{family}_random{s}_0.10'] for s in (0, 1)]
            if (chosen['f1'] > max(results['base']['f1'], *(x['f1'] for x in controls))
                    and chosen['precision'] >= results['base']['precision'] - 0.02
                    and chosen['invalid_answers'] <= results['base']['invalid_answers']):
                eligible.append(family)
        write_json(out / 'screen_decision.json', {'eligible_for_confirmation': eligible,
                   'automatic_confirmation': False, 'no_success_claim': True,
                   'next': 'Freeze one direction before opening reserved confirmation; stop if neither qualifies.'})
        print('WSP_COMPLETE ' + json.dumps({'eligible_for_confirmation': eligible}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('prepare', 'extract', 'generate', 'score'))
    p.add_argument('--out', required=True)
    p.add_argument('--traces', default='/app/local_extensions/steering/mech/out/T0/train_traces.jsonl')
    p.add_argument('--prompts', default='/app/local_extensions/steering/mech/out/T0/train_prompts.jsonl')
    p.add_argument('--model', default='/app/models/Qwen3.8-27B-W4A16-AutoRound')
    p.add_argument('--max-model-len', type=int, default=24576)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    args = p.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
