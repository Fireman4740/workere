"""Confirm one frozen W1 candidate on the previously reserved CTB documents.

No extraction, refitting, parameter selection, or historical-panel evaluation.
The W1 inference helpers stay byte-identical; source hashes are checked at entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from steering import ctb
from steering.weight_space import (LAYERS, counts, digest, final_pairs,
                                  install_writers, make_split, metrics)
from steering.wsp_pilot import (apply_arm, purpose_candidates, read_rows, sha,
                                unique_rows, write_json)


def verify_source(source, plan):
    """The plan pins the entire extraction artifact and its selection evidence."""
    source = Path(source)
    for name, expected in plan['source_files_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError(f'frozen source changed: {name}')
    parent = json.loads((source / 'manifest.json').read_text())
    if (parent['id'] != plan['source_manifest_id'] or
            parent['id'] != digest({k: v for k, v in parent.items() if k != 'id'})):
        raise ValueError('parent manifest identity mismatch')
    directions = json.loads((source / 'directions.json').read_text())
    if (directions['manifest_id'] != parent['id'] or
            directions['npz_sha256'] != plan['source_files_sha256']['directions.npz']):
        raise ValueError('direction extraction identity mismatch')
    decision = json.loads((source / 'screen_decision.json').read_text())
    scores = json.loads((source / 'screen_scores.json').read_text())
    if (not scores['complete'] or scores['manifest_id'] != parent['id'] or
            'cc' not in decision['eligible_for_confirmation']):
        raise ValueError('source did not establish an eligible cc candidate')
    if plan['selected_condition'] != 'cc_0.10':
        raise ValueError('confirmation candidate must stay cc at 0.10')
    expected = [c for c in parent['conditions']
                if c['name'] in ('base', 'cc_0.10') or 'random_seed' in c]
    if plan['conditions'] != expected or len(expected) != 6:
        raise ValueError('must retain base, frozen cc, and all four random controls')
    ids = plan['confirmation_documents']
    if ids != parent['split']['confirmation'] or len(ids) != 20:
        raise ValueError('confirmation documents differ from the reserved split')
    if set(ids) & set(parent['split']['fit'] + parent['split']['screen'] +
                      parent['split']['excluded_historical'] + parent['split']['demonstrations']):
        raise ValueError('confirmation overlap/leakage')
    return parent


def prepare(args):
    source, out = Path(args.source_dir), Path(args.out)
    if source.resolve() == out.resolve():
        raise ValueError('confirmation must not overwrite the source screen')
    plan = json.loads(Path(args.plan).read_text())
    parent = verify_source(source, plan)
    for name, value in plan['frozen_generation'].items():
        if getattr(args, name) != value or parent[name] != value:
            raise ValueError(f'generation setting differs from W1: {name}')
    for name, expected in parent['implementation_sha256'].items():
        if sha(HERE / name) != expected:
            raise ValueError(f'frozen W1 implementation changed: {name}')
    for name, path in (('traces', args.traces), ('prompts', args.prompts),
                       ('suite', ctb.DATA / 'ctb-suite.json'), ('prompt', ctb.DATA / 'ctb-prompt.json')):
        if sha(path) != parent['source_sha256'][name]:
            raise ValueError(f'input source differs from W1: {name}')
    suite, prompt = ctb.load_suite(), ctb.load_prompt()
    if make_split(suite, prompt, ctb.eval_ids()) != parent['split']:
        raise ValueError('reserved split changed')
    ids = plan['confirmation_documents']
    prompts = unique_rows(read_rows(args.prompts), lambda r: r['doc_id'])
    rows = []
    for d in ids:
        messages = ctb.messages_for(suite[d], prompt)
        if prompts[d]['messages'] != messages:
            raise ValueError(f'confirmation prompt drift: {d}')
        rows.append({'doc_id': d, 'pass': 0, 'messages': messages})
    # Only the recorded input lengths are used from historical traces. No cached
    # confirmation prediction, final-answer parse, or gold-dependent choice.
    lengths = {r['doc_id']: int(r['prompt_tokens']) for r in read_rows(args.traces)}
    bound = max(lengths[d] for d in ids) + 8192 + 4096 + 256
    if bound >= args.max_model_len:
        raise ValueError(f'confirmation context needs {bound}, limit={args.max_model_len}')
    manifest = {'protocol': plan['protocol'], 'plan': plan, 'plan_sha256': sha(args.plan),
                'conditions': plan['conditions'], 'documents': ids,
                'source_manifest_id': parent['id'],
                'directions_sha256': plan['source_files_sha256']['directions.npz'],
                'source_sha256': parent['source_sha256'],
                'frozen_implementation_sha256': parent['implementation_sha256'],
                'confirmation_implementation_sha256': {
                    n: sha(HERE / n) for n in ('wsp_confirmation.py', 'run_wsp.sh', 'config.env')},
                'generation': plan['frozen_generation'], 'context_upper_bound': bound,
                'scope': plan['scope']}
    manifest['id'] = digest(manifest)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('confirmation manifest drift; refuse mixed results')
    for name in ('directions.npz', 'directions.json'):
        target = out / name
        if target.exists() and sha(target) != plan['source_files_sha256'][name]:
            raise ValueError(f'confirmation source copy changed: {name}')
        if not target.exists():
            shutil.copyfile(source / name, target)
    write_json(path, manifest)
    write_json(out / 'frozen_plan.json', plan)
    (out / 'confirmation_prompts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print('WSP_CONFIRM_PREFLIGHT ' + json.dumps({'manifest_id': manifest['id'],
          'source_manifest_id': parent['id'], 'directions_sha256': manifest['directions_sha256'],
          'documents': len(ids), 'conditions': plan['conditions'], 'context_upper_bound': bound,
          'decision': plan['decision'], 'bootstrap': plan['bootstrap']}), flush=True)
    return manifest


def load_records(out, manifest, partial=False):
    out = Path(out)
    if manifest['id'] != digest({k: v for k, v in manifest.items() if k != 'id'}):
        raise ValueError('confirmation manifest fingerprint mismatch')
    if sha(out / 'directions.npz') != manifest['directions_sha256']:
        raise ValueError('confirmation direction fingerprint mismatch')
    meta = json.loads((out / 'directions.json').read_text())
    if (meta['manifest_id'] != manifest['source_manifest_id'] or
            meta['npz_sha256'] != manifest['directions_sha256']):
        raise ValueError('confirmation directions do not match source extraction')
    path = out / 'confirmation.jsonl'
    rows = unique_rows(read_rows(path), lambda r: (r['condition'], r['doc_id'])) if path.exists() else {}
    conditions = {c['name']: c for c in manifest['conditions']}
    expected = {(name, d) for name in conditions for d in manifest['documents']}
    if not set(rows) <= expected:
        raise ValueError('foreign condition/document in confirmation cache')
    if not partial and set(rows) != expected:
        raise ValueError(f'incomplete confirmation: {len(rows)}/{len(expected)}')
    for row in rows.values():
        c = conditions[row['condition']]
        if (row.get('manifest_id') != manifest['id'] or
                row.get('directions_sha256') != manifest['directions_sha256'] or
                row.get('condition_sha256') != digest(c)):
            raise ValueError('confirmation cache provenance mismatch')
        if any(row.get(k) != c[k] for k in ('alpha', 'budget', 'temperature')) or row.get('pass') != 0:
            raise ValueError('confirmation row generation settings differ')
    return rows, expected


def generate(args):
    from offline import build_llm, render
    from run_vllm import generate_condition
    import vllm
    from vllm import SamplingParams

    if vllm.__version__ != '0.32.0':
        raise RuntimeError(f'verified runtime is vLLM 0.32.0, found {vllm.__version__}')
    manifest = prepare(args)
    out = Path(args.out)
    done, expected = load_records(out, manifest, partial=True)
    if set(done) == expected:
        print('WSP_CONFIRM_GENERATIONS_COMPLETE', flush=True)
        return
    rows = read_rows(out / 'confirmation_prompts.jsonl')
    data = np.load(out / 'directions.npz', allow_pickle=False)
    conditions = manifest['conditions']
    writers = install_writers()
    llm, _, tokenizer = build_llm(args.model, args.max_model_len,
                                  args.gpu_memory_utilization, max_num_seqs=args.batch)
    if set(writers) != set(LAYERS):
        raise RuntimeError(f'not all nine MLP sites installed: {sorted(writers)}')
    probe_text = render(tokenizer, rows[0]['messages'])
    probe_params = SamplingParams(max_tokens=2, temperature=0.0, logprobs=20)
    base_probe = llm.generate([probe_text], probe_params, use_tqdm=False)[0]
    apply_arm(writers, data, conditions[1])
    edit_probe = llm.generate([probe_text], probe_params, use_tqdm=False)[0]
    apply_arm(writers, data, conditions[0])
    reset_probe = llm.generate([probe_text], probe_params, use_tqdm=False)[0]
    if list(base_probe.outputs[0].token_ids) != list(reset_probe.outputs[0].token_ids):
        raise RuntimeError('baseline did not restore after clearing edit')
    base_probs, edit_probs, reset_probs = (x.outputs[0].logprobs[0]
                                         for x in (base_probe, edit_probe, reset_probe))
    delta = max((abs(base_probs[k].logprob - edit_probs[k].logprob)
                 for k in base_probs.keys() & edit_probs.keys()), default=0.)
    reset_delta = max((abs(base_probs[k].logprob - reset_probs[k].logprob)
                       for k in base_probs.keys() & reset_probs.keys()), default=float('inf'))
    if delta <= 1e-5 or reset_delta > 1e-5:
        raise RuntimeError(f'edit not visible or reset failed: edit delta={delta}, reset={reset_delta}')
    smoke = {'sites': sorted(writers), 'reset_equal': True,
             'max_first_token_logprob_change': delta, 'reset_logprob_delta': reset_delta,
             'vllm': vllm.__version__, 'directions_sha256': manifest['directions_sha256']}
    write_json(out / 'runtime_smoke.json', smoke)
    print('WSP_CONFIRM_HOOK_SMOKE ' + json.dumps(smoke), flush=True)

    class ProvenanceSink:
        def __init__(self, stream, arm):
            self.stream, self.arm = stream, arm
        def write(self, line):
            row = json.loads(line)
            row.update(manifest_id=manifest['id'], directions_sha256=manifest['directions_sha256'],
                       condition_sha256=digest(self.arm))
            self.stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        def flush(self):
            self.stream.flush()

    for arm in conditions:
        pending = [r for r in rows if (arm['name'], r['doc_id']) not in done]
        if not pending:
            continue
        apply_arm(writers, data, arm)
        print('WSP_CONFIRM_ARM ' + json.dumps(arm), flush=True)
        with (out / 'confirmation.jsonl').open('a') as stream:
            generate_condition(llm, tokenizer, pending, arm, 12288, ProvenanceSink(stream, arm))
        score(args, partial=True)


def paired_comparison(reference, candidate, plan):
    """Paired document bootstrap, fixed primary and five-comparison intervals."""
    if not reference or set(reference) != set(candidate):
        raise ValueError('paired comparisons need identical nonempty document sets')
    ids = sorted(reference)
    indices = np.random.default_rng(plan['seed']).integers(0, len(ids), (plan['draws'], len(ids)))
    values = []
    for per_doc in (reference, candidate):
        array = np.array([[per_doc[d][k] for k in ('tp', 'fp', 'fn')] for d in ids])
        totals = array[indices].sum(axis=1)
        den = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        values.append(np.divide(200 * totals[:, 0], den, out=np.zeros(plan['draws']), where=den > 0))
    diff = values[1] - values[0]
    point = 100 * (metrics(candidate)['f1'] - metrics(reference)['f1'])
    return {'delta_f1_pp': point,
            'ci95_pp': list(map(float, np.quantile(diff, plan['primary_quantiles']))),
            'ci99_bonferroni_pp': list(map(float, np.quantile(diff, plan['familywise_quantiles'])))}


def confirmation_decision(results, comparisons, plan):
    chosen, base = results[plan['selected_condition']], results['base']
    precision_ok = chosen['precision'] >= base['precision'] + plan['decision']['precision_floor_delta_pp'] / 100
    invalid_ok = chosen['invalid_answers'] <= base['invalid_answers']
    guardrails = precision_ok and invalid_ok
    primary = guardrails and comparisons['base']['ci95_pp'][0] > 0
    specific = (guardrails and len(comparisons) == plan['bootstrap']['familywise_comparisons']
                and all(c['ci99_bonferroni_pp'][0] > 0 for c in comparisons.values()))
    return {'precision_guardrail': precision_ok, 'invalid_answer_guardrail': invalid_ok,
            'primary_task_gain_confirmed': bool(primary),
            'direction_specific_gain_confirmed': bool(specific),
            'next': ('Consider a separately frozen matched historical-panel evaluation; no automatic promotion.'
                     if primary else 'Stop this series; no dose, prompt, or direction tuning on confirmation.'),
            'scope': plan['scope']}


def score(args, partial=False):
    out = Path(args.out)
    manifest = json.loads((out / 'manifest.json').read_text())
    if sha(ctb.DATA / 'ctb-suite.json') != manifest['source_sha256']['suite']:
        raise ValueError('scoring corpus differs from frozen source')
    rows, expected = load_records(out, manifest, partial=partial)
    suite, ids = ctb.load_suite(), manifest['documents']
    per_doc, predictions, results = {}, {}, {}
    for arm in manifest['conditions']:
        name = arm['name']
        if any((name, d) not in rows for d in ids):
            continue
        per_doc[name], predictions[name], proxy_counts = {}, {}, {}
        invalid, forced, outside, total_tokens, thinking_tokens = 0, 0, 0, [], []
        for d in ids:
            row = rows[(name, d)]
            parsed = final_pairs(row)
            invalid += parsed is None
            pred = parsed or set()
            gold = {(a, b) for a, label, b in suite[d]['gold'] if label != 'norel'}
            candidates = {tuple(p) for p in suite[d]['pair_list']}
            outside += len(pred - candidates)
            predictions[name][d], per_doc[name][d] = pred, counts(pred, gold)
            proxy = purpose_candidates(suite[d])
            proxy_counts[d] = counts(pred & proxy, gold & proxy)
            total_tokens.append(row['num_tokens'])
            thinking_tokens.append(row['thinking_tokens'])
            forced += row['budget_forced']
        results[name] = {**metrics(per_doc[name]), 'documents': len(ids), 'invalid_answers': invalid,
                         'outside_candidate_pairs': outside, 'forced_closures': forced,
                         'mean_generated_tokens': float(np.mean(total_tokens)),
                         'mean_thinking_tokens': float(np.mean(thinking_tokens)),
                         'lexical_purpose_proxy': metrics(proxy_counts)}
    if 'base' in results:
        for name in results:
            if name == 'base':
                continue
            added, lost, added_fp, removed_fp = 0, 0, 0, 0
            for d in ids:
                gold = {(a, b) for a, label, b in suite[d]['gold'] if label != 'norel'}
                new = predictions[name][d] - predictions['base'][d]
                removed = predictions['base'][d] - predictions[name][d]
                added += len(new & gold)
                lost += len(removed & gold)
                added_fp += len(new - gold)
                removed_fp += len(removed - gold)
            results[name]['changes_vs_base'] = {'recovered_gold_ordered': added,
                     'lost_gold_ordered': lost, 'added_fp_ordered': added_fp, 'removed_fp_ordered': removed_fp}
    plan = manifest['plan']
    selected = plan['selected_condition']
    comparisons = {name: paired_comparison(values, per_doc[selected], plan['bootstrap'])
                   for name, values in per_doc.items() if name != selected} if selected in per_doc else {}
    complete = set(rows) == expected
    decision = confirmation_decision(results, comparisons, plan) if complete else None
    report = {'manifest_id': manifest['id'], 'complete': complete, 'status': 'reserved_confirmation',
              'conditions': results, 'selected_minus_comparators': comparisons,
              'per_document': per_doc, 'bootstrap': plan['bootstrap'], 'decision': decision,
              'caveat': plan['scope']}
    write_json(out / 'confirmation_scores.json', report)
    print('WSP_CONFIRM_SCORES ' + json.dumps({k: v for k, v in report.items() if k != 'per_document'}), flush=True)
    if complete:
        write_json(out / 'confirmation_decision.json', decision)
        print('WSP_CONFIRM_COMPLETE ' + json.dumps(decision), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('prepare', 'generate', 'score'))
    p.add_argument('--out', required=True)
    p.add_argument('--source-dir', required=True)
    p.add_argument('--plan', default=str(HERE / 'wsp_confirmation_plan.json'))
    p.add_argument('--traces', default='/app/local_extensions/steering/mech/out/T0/train_traces.jsonl')
    p.add_argument('--prompts', default='/app/local_extensions/steering/mech/out/T0/train_prompts.jsonl')
    p.add_argument('--model', default='/app/models/Qwen3.8-27B-W4A16-AutoRound')
    p.add_argument('--max-model-len', type=int, default=24576)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--gpu-memory-utilization', type=float, default=.90)
    args = p.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
