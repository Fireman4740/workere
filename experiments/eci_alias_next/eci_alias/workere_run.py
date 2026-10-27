"""Tracked workere adapter: offline rendering, then at most one 14-request smoke.

Uses the inherited Luna batch transport, without sync fallback or automatic
resubmission. Run only through the project's tracked experiment launcher.
"""
from pathlib import Path
import base64
import io
import json
import math
import os
import zipfile
import yaml
from .core import Document, compact, digest, choose_mentions, prepare_view, validate_pair_view
from .cli import read_jsonl, write_jsonl
from .budget import verify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUTPUT_CONTRACT = (
    '\nEXPERIMENT OUTPUT ADAPTER (same in every condition): The historical demonstration '
    'answers retain their original serialization. For this query only, final output must '
    'be a JSON list of [source_id, "causes" or "causedby", target_id], or []. '
    'This changes serialization, not causal criteria. Registered auxiliary handles are '
    'valid mention identifiers in addition to original inline numeric IDs. They point '
    'to existing mentions, not additional occurrences. Never infer identity between '
    'different original mentions from the auxiliary records. Every pair of distinct '
    'original mentions is a candidate. Do not output labels other than causes/causedby.\n'
)


def parse_response(content):
    if not isinstance(content, str):
        raise ValueError('missing final content')
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError('final response is not a JSON triple list')
    return value


def byte_bound(messages):
    # Conservative byte-level GPT-family bound; not a measured tokenizer count.
    # 1024 covers the fixed four-message chat framing, with short role strings.
    return len(compact(messages).encode('utf-8')) + 1024


def render(d, view, demo_id, demos, prompt):
    base_user = prompt['user_template'].format(doc_text=d.text, pair_lines='', doc_id=d.doc_id)
    system = prompt['system'] + OUTPUT_CONTRACT
    base_messages = [{'role': 'system', 'content': system}, *demos['messages'][demo_id],
                     {'role': 'user', 'content': base_user + OUTPUT_CONTRACT}]
    # Public model payload: exact source text, frozen demo, explicit pointer records only.
    # C is complete for this adapter; no quadratic alias/candidate expansion in prompt.
    auxiliary = '\n\nAUXILIARY POINTER INDEX\n' + view.common_instruction + '\n' + compact(view.auxiliary_records)
    messages = base_messages[:-1] + [{'role': 'user', 'content': base_messages[-1]['content'] + auxiliary}]
    return messages, {
        'base_prompt_sha256': digest(compact(base_messages)),
        'full_prompt_sha256': digest(compact(messages)),
        'demo_sha256': digest(compact(demos['messages'][demo_id])),
        'input_tokens_upper_bound': byte_bound(messages),
        'auxiliary_tokens_upper_bound': len(auxiliary.encode('utf-8')),
        'auxiliary_bound_source': 'utf8_bytes_for_byte_level_GPT_tokenizer',
    }


def prepare(out):
    manifest = json.loads((DATA / 'input-manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        import hashlib
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen input hash mismatch: ' + name)
    prompt_bytes = Path('prompts/esl_standard_eci_lean_coref.yaml').read_bytes()
    import hashlib
    if hashlib.sha256(prompt_bytes).hexdigest() != manifest['historical_prompt_sha256']:
        raise ValueError('historical prompt changed')
    prompt = yaml.safe_load(prompt_bytes)
    demos = json.loads((DATA / 'frozen_demos.json').read_text())
    documents = [Document.from_public_json(x) for x in read_jsonl(DATA / 'public_docs_DEV44.jsonl')]
    jobs, views, checks = [], [], []
    for d in documents:
        if len(d.candidates) != len(d.mentions) * (len(d.mentions) - 1) // 2:
            raise ValueError('this adapter requires complete original candidate universe')
        selected = choose_mentions(d)
        while True:
            av, rv = prepare_view(d, 'A', selected), prepare_view(d, 'R', selected)
            rendered = []
            for p, demo_id in enumerate(demos['per_document'][d.doc_id]):
                if demo_id.split('_')[0] == d.doc_id.split('_')[0]:
                    raise ValueError('same-topic demonstration')
                for v in (av, rv):
                    messages, meta = render(d, v, demo_id, demos, prompt)
                    rendered.append({'doc_id': d.doc_id, 'arm': v.arm, 'pass_index': p,
                                     'demo_id': demo_id, 'messages': messages, **meta})
            if max(j['auxiliary_tokens_upper_bound'] for j in rendered) <= 768 or not selected:
                break
            selected = selected[:-1]
        if not selected:
            # Never pay for a purported alias intervention with zero aliases.
            raise ValueError('no alias fits auxiliary bound on ' + d.doc_id)
        check = validate_pair_view(d, av, rv)
        check['original_selection_count'] = len(choose_mentions(d))
        check['selected_after_conservative_bound'] = len(selected)
        check['max_input_tokens_upper_bound'] = max(j['input_tokens_upper_bound'] for j in rendered)
        check['input_preflight_within_cap'] = check['max_input_tokens_upper_bound'] <= 16384
        check['bound_not_actual_tokenizer'] = True
        check['max_auxiliary_tokens_upper_bound'] = max(j['auxiliary_tokens_upper_bound'] for j in rendered)
        checks.append(check)
        jobs += rendered
        views += [av.to_json(), rv.to_json()]
    write_jsonl(out / 'rendered_jobs_DEV44.jsonl', jobs)
    write_jsonl(out / 'private_views_DEV44.jsonl', views)
    write_jsonl(out / 'render_checks_DEV44.jsonl', checks)
    return documents, jobs, views, checks


def emit_bundle(out):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, str(p.relative_to(out)))
    print('ALIAS_ARTIFACT_BUNDLE ' + base64.b64encode(buffer.getvalue()).decode(), flush=True)


def main():
    config = json.loads((ROOT / 'workere-run.json').read_text())
    out = Path('logs/alias-followup-tracked')
    out.mkdir(parents=True, exist_ok=True)
    status = {'kind': 'alias_AR_development_smoke', 'stage': 'preparing', 'real_model_calls_attempted': 0,
              'synthetic': False, 'full_pilot_launched': False}
    client = None
    try:
        documents, jobs, views, checks = prepare(out)
        did = documents[0].doc_id  # fixed DEV44 order; no gold/score involved
        smoke = [j for j in jobs if j['doc_id'] == did]
        smoke_views = [v for v in views if v['doc_id'] == did]
        write_jsonl(out / 'smoke_jobs.jsonl', smoke)
        write_jsonl(out / 'smoke_private_views.jsonl', smoke_views)
        status.update(smoke_document=did, planned_smoke_requests=len(smoke),
                      prepared_documents=len(documents), prepared_jobs=len(jobs),
                      auxiliary_bound='conservative_UTF8_bytes_not_exact_tokenizer',
                      offline_input_bound_pass_documents=sum(c['input_preflight_within_cap'] for c in checks))
        print('ALIAS_PREFLIGHT ' + compact(status), flush=True)
        if config['stage'] == 'offline':
            status.update(stage='offline_complete', launch_blocker='stage is offline; zero API calls')
            return
        if config['stage'] != 'smoke' or config['max_requests'] != 14 or len(smoke) != 14:
            raise ValueError('only the preregistered 14-request smoke is enabled')
        if max(j['input_tokens_upper_bound'] for j in smoke) > 16384:
            raise ValueError('conservative full-input bound exceeds 16384; no truncation/no calls')
        if not (os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER')):
            raise ValueError('missing remote OpenRouter credential; no model calls')
        from pipelines.jev.batch import BatchChatClient
        from pipelines.jev.client import DecisionError
        class AliasBatchClient(BatchChatClient):
            def _validate_response(self, body, response):
                if not isinstance(response, dict) or response.get('model') not in {
                    'openai/gpt-6-luna', 'openai/gpt-6-luna:batch', 'openai/gpt-6-luna-20260922'}:
                    raise DecisionError('unexpected served model')
                choice = response.get('choices', [{}])[0]
                if choice.get('finish_reason') != 'stop':
                    raise DecisionError('incomplete final generation')
                parse_response(choice.get('message', {}).get('content'))
            def validate_request(self, messages, questions):
                body = {'model': self.identity['model'], 'messages': messages,
                        'reasoning': {'effort': 'high'}, 'max_tokens': 4096}
                bound = byte_bound(messages)
                if bound > 16384 or bound + 4096 > 20480:
                    raise DecisionError('input bound exceeds frozen cap')
                return body, bound
        cache = Path.home() / '.local/share/openresearch/workere-decisions/alias-smoke-20261027.sqlite'
        client = AliasBatchClient(model='openai/gpt-6-luna:batch', model_version='gpt-6-luna-20260922',
             endpoint='https://openrouter.ai/api/v1/batches', mode='live', cache_path=str(cache),
             max_calls=14, budget_usd=config['max_budget_usd'], max_cost_per_call_usd=config['reservation_per_request_usd'],
             reasoning_effort='high', max_output_tokens=4096, max_context_tokens=20480,
             retries=0, timeout=600,
             shared_budget={'path': str(cache.parent / 'alias-smoke-budget.sqlite'),
                            'scope': 'alias-smoke-20261027-14requests', 'limit_usd': config['max_budget_usd'],
                            'initial_spend_usd': 0})
        # Unpaid metadata request; do not assume historical prices still apply.
        catalog = client._http('GET', 'https://openrouter.ai/api/v1/models')
        model = next((m for m in catalog.get('data', []) if m.get('id') == 'openai/gpt-6-luna'), None)
        if model is None:
            raise ValueError('frozen Luna route absent from current provider catalog')
        (out / 'provider_model_metadata.json').write_text(json.dumps(model, indent=2) + '\n')
        tokenizer = (model.get('architecture') or {}).get('tokenizer', '')
        if not isinstance(tokenizer, str) or not any(x in tokenizer.lower() for x in ('gpt', 'o200k', 'cl100k')):
            raise ValueError('provider tokenizer not verified as GPT byte-level; conservative bound cannot authorize launch')
        pricing = model.get('pricing') or {}
        ip, op = float(pricing['prompt']), float(pricing['completion'])
        worst = 16384 * ip + 4096 * op
        if not all(math.isfinite(x) and x >= 0 for x in (ip, op, worst)) or worst > config['reservation_per_request_usd']:
            raise ValueError('current provider prices exceed per-request reservation')
        if 14 * config['reservation_per_request_usd'] > config['max_budget_usd']:
            raise ValueError('smoke reservations exceed hard scope budget')
        status.update(stage='submitting_or_resuming_smoke', paid_submission_status='receipt_is_authoritative',
                      tokenizer_metadata=tokenizer, max_price_bound_usd_per_request=worst)
        responses, usage, callbacks = [], [], []
        decode_hash = digest(compact({'reasoning': {'effort': 'high'}, 'max_tokens': 4096, 'temperature': 'provider_default_as_historical'}))
        def make_callback(job):
            def record(event):
                raw = event.get('response') or {}
                choice = raw.get('choices', [{}])[0]
                error = event.get('error')
                try:
                    parsed = parse_response(choice.get('message', {}).get('content')) if not error else None
                except (ValueError, KeyError, TypeError):
                    parsed = None
                ok = parsed is not None and not error
                res = {k: job[k] for k in ('doc_id', 'arm', 'pass_index')}
                res.update(status='ok' if ok else 'error', response=parsed,
                           error=error, batch_id=event.get('batch_id'), cache_hit=event.get('cache_hit'))
                actual = raw.get('usage') or {}
                u = {k: job[k] for k in ('doc_id', 'arm', 'pass_index', 'demo_id', 'base_prompt_sha256',
                                        'full_prompt_sha256', 'demo_sha256', 'auxiliary_tokens_upper_bound')}
                u.update(input_tokens=actual.get('prompt_tokens'), completion_tokens=actual.get('completion_tokens'),
                         model_version='gpt-6-luna-20260922', served_model=raw.get('model'),
                         decode_sha256=decode_hash, truncated=choice.get('finish_reason') != 'stop',
                         status='ok' if ok else 'error', token_count_source='provider_usage',
                         cost_usd=event.get('cost_usd'), api_call=event.get('api_call'),
                         reasoning_tokens=(actual.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                         raw_usage=actual)
                responses.append(res); usage.append(u)
                if event.get('api_call'):
                    status['real_model_calls_attempted'] += 1
                write_jsonl(out / 'responses.jsonl', responses)
                write_jsonl(out / 'usage.jsonl', usage)
                with (out / 'attempts.jsonl').open('a') as f:
                    f.write(compact({'job': {k: job[k] for k in ('doc_id','arm','pass_index')}, **event}) + '\n')
                print('ALIAS_SLOT ' + compact(u), flush=True)
            return record
        callbacks = [make_callback(j) for j in smoke]
        try:
            client.decide_many([j['messages'] for j in smoke], callbacks)
        except Exception as exc:
            status['transport_error'] = type(exc).__name__ + ': ' + str(exc)
        status['real_model_calls_attempted'] = client.calls if client.calls or not status.get('transport_error') else None
        gate = verify(usage, expected_doc_ids=[did], require_provenance=True)
        (out / 'budget_gate.json').write_text(json.dumps(gate, indent=2) + '\n')
        status.update(stage='smoke_complete' if gate['status'] == 'PASS' else 'smoke_blocked', budget_gate=gate['status'])
        print('ALIAS_BUDGET_GATE ' + compact(gate), flush=True)
        if len(responses) != 14:
            raise ValueError('missing attempted slots; no synthetic errors substituted')
        from argparse import Namespace
        from .cli import do_aggregate, do_evaluate
        do_aggregate(Namespace(views=str(out/'smoke_private_views.jsonl'), responses=str(out/'responses.jsonl'),
                               passes=7, threshold_votes=4, out=str(out/'predictions.jsonl')))
        # Gold is read only after predictions have been written and fixed.
        gold = [r for r in read_jsonl(DATA/'evaluator_only/gold_DEV44.jsonl') if r['doc_id'] == did]
        write_jsonl(out/'smoke_gold_evaluator_only.jsonl', gold)
        do_evaluate(Namespace(views=str(out/'smoke_private_views.jsonl'), results=str(out/'predictions.jsonl'),
                              gold=str(out/'smoke_gold_evaluator_only.jsonl'), out=str(out/'evaluation.json')))
        result = json.loads((out/'evaluation.json').read_text())
        result['interpretation'] = 'One-document development smoke only. Not a DEV44 result or confirmation; gate PASS is required for cost-matched interpretation.'
        (out/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
        print('ALIAS_REAL_SMOKE_RESULT ' + compact(result), flush=True)
    except Exception as exc:
        status.update(stage='blocked', launch_blocker=type(exc).__name__ + ': ' + str(exc))
        print('ALIAS_BLOCKER ' + compact(status), flush=True)
    finally:
        if client is not None:
            client.close()
        (out/'status.json').write_text(json.dumps(status, indent=2) + '\n')
        print('ALIAS_STATUS ' + compact(status), flush=True)
        emit_bundle(out)
    if status['stage'] not in {'offline_complete', 'smoke_complete'}:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
