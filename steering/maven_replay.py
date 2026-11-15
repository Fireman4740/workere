"""Frozen transfers of N1/N2/VC1/W4-filter/MV2 to the same MAVEN200 panel.

Gold is kept out of planners and model jobs. Every missing response prohibits a
final score; immutable per-stage records allow planned GPU leases to resume.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import time

from steering import campaign_adjudicate as adjudicate
from steering import campaign_api as api
from steering import campaign_blocks as blocks
from steering import maven_budget as mb
from steering import maven_pool as mp
from steering import validation_protocol as vp
from steering.compare import paired_delta

VERSION = 'maven-replay-20261115-v1'
ARMS = ('N1', 'N2', 'VC1', 'W4', 'MV2')
SOURCE_HASH = '9098b43d14473cc4f6f44cedd43351f590aeb6e21e87aaa1abb88ef0f6b89366'
VC1_ID = '08c1351e-6ac9-4471-aca9-1d608aeb7527'


def freeze(path, value):
    """Compare serialized JSON, including tuples read back as lists; never rewrite."""
    path = Path(path)
    if path.exists():
        if api.canonical(mb.read(path)) != api.canonical(value):
            raise ValueError(f'immutable input changed: {path}')
    else:
        mb.atomic(path, value)


def config(path):
    cfg = mb.read(path)
    if (cfg['protocol'] != VERSION or cfg['arm'] not in ARMS
            or cfg['documents'] != 200 or cfg['passes'] != 5
            or cfg['engine']['max_model_len'] != 65536
            or cfg['reference_completion_sha256'] != SOURCE_HASH):
        raise ValueError('frozen transfer contract changed')
    return cfg


def job(phase, doc, messages, schema, thinking, answer_tokens, **meta):
    row = dict(phase=phase, doc_id=doc, messages=messages, schema=schema,
               thinking=thinking, answer_tokens=answer_tokens, **meta)
    # JSON-native before hashing/writing prevents tuple/list restart discrepancies.
    row = json.loads(api.canonical(row))
    row['key'] = api.digest(row)
    return row


def document_jobs(bases, sampled=False):
    by_doc = defaultdict(list)
    for base in bases:
        by_doc[base['doc_id']].append(base)
    result = []
    for doc, values in sorted(by_doc.items()):
        values.sort(key=lambda b: b['pass_id'])
        for p, base in enumerate(values):
            source = values[0] if sampled else base
            result.append(job('sampled' if sampled else 'diverse', doc,
                source['messages'], vp.document_schema(), 8192, 4096,
                pass_id=p, demo_id=source['demo_id'], temperature=.6 if sampled else 0.,
                seed=int(api.digest([VERSION, doc, p])[:8], 16)))
    return result


def screen_jobs(public, rules):
    result = []
    for doc, data in sorted(public.items()):
        pairs = blocks.pairs(data)
        for index, start in enumerate(range(0, len(pairs), 12)):
            block = pairs[start:start+12]
            context = blocks.windows(data['text'], {e for p in block for e in p})
            for order in (0, 1):
                indices = list(range(len(block)))
                if order:
                    random.Random(f'{blocks.SEED}::MAVEN::{doc}::{index}').shuffle(indices)
                result.append(job('screen', doc,
                    blocks.make_messages(rules, context, block, indices), blocks.schema(len(block)),
                    4096, 384, pairs=block, order=order, block=index))
    return result


def verification_jobs(bases, rows, public):
    """N2: retain MAVEN's implicit causation rule; no CTB lexical-anchor restriction."""
    result = []
    for base in bases:
        if base['pass_id'] != 0:
            continue
        doc = base['doc_id']
        chosen = [document_pairs(row, public[doc]) for row in rows if row['doc_id'] == doc]
        if len(chosen) != 5:
            raise ValueError('N2 requires all five proposal readings, including invalid empty reads')
        counts = Counter(p for reading in chosen for p in reading)
        lines = [f'- {a},{b}: proposed by {n}/5 readings'
                 for (a, b), n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
        note = ('\n\nPrevious readings proposed the following causal pairs and support counts. '
                'They may be wrong in either direction.\n' + ('\n'.join(lines) or '- none') +
                '\nRe-examine every proposal against the unchanged annotation rules. '
                'Keep only document-supported relations; implicit causation is allowed. '
                'You may add a missed pair only when the document supports it. '
                'Return the same final JSON format.')
        messages = copy.deepcopy(base['messages'])
        messages[-1]['content'] += note
        result.append(job('verification', doc, messages, vp.document_schema(), 8192, 4096))
    return result


def w4_jobs(public, rules, baseline, supports):
    """Selected unsteered filter only; freeze its MAVEN/D routing adaptation."""
    result, routes = [], {}
    for doc, data in sorted(public.items()):
        selected = sorted(baseline[doc], key=lambda p: (
            abs(supports[doc][p]/5-.5), api.digest([VERSION, doc, p])))[:24]
        routes[doc] = dict(available=len(baseline[doc]), selected=[list(p) for p in selected])
        for start in range(0, len(selected), 8):
            block = selected[start:start+8]
            for reverse in (False, True):
                shown = [p[::-1] if reverse else p for p in block]
                pairs = '\n'.join(f'{i}: A=<{a}>, B=<{b}>' for i, (a, b) in enumerate(shown))
                messages = [dict(role='system', content=rules +
                    '\nClassify only the listed pairs. A means event A causes event B; '
                    'B means event B causes event A; N means no annotated causal relation. '
                    'Check evidence for each pair and distinguish causation from temporal order, '
                    'association and coreference. Do not relax the annotation rules. '
                    'Return one JSON object mapping every index to A, B or N.'),
                    dict(role='user', content=f'Document:\n{data["text"]}\n\nPairs:\n{pairs}')]
                result.append(job('filter', doc, messages, blocks.schema(len(block)),
                                  4096, 512, pairs=block, reverse=reverse))
    return result, routes


def views(jobs, rows):
    result = defaultdict(dict)
    for req, row in zip(jobs, rows, strict=True):
        obj = row['object']
        if obj is not None:
            for i, pair in enumerate(req['pairs']):
                p = tuple(pair)
                slot = (req['doc_id'], req['order'])
                if p in result[slot]:
                    raise ValueError('duplicate screen pair')
                result[slot][p] = obj[str(i)] != 'N'
    return result


def local_filter(baseline, screen):
    return {d: {p for p in ps if not all(screen.get((d, o), {}).get(p) is False
                                         for o in (0, 1))} for d, ps in baseline.items()}


def direct_jobs(public, rules, baseline, screen):
    jobs, routes = [], []
    for doc, data in sorted(public.items()):
        for pair in blocks.pairs(data):
            if not any(pair in screen.get((doc, o), {}) and
                       screen[doc, o][pair] != (pair in baseline[doc]) for o in (0, 1)):
                continue
            routes.append([doc, list(pair)])
            context = blocks.windows(data['text'], set(pair))
            for reverse in (False, True):
                presented = pair[::-1] if reverse else pair
                jobs.append(job('direct', doc, adjudicate.messages(rules, context, presented, 'direct'),
                    adjudicate.answer_schema('direct'), 2560, 512, pair=pair, reverse=reverse))
    return jobs, routes


def asymmetric(original, checks):
    if len(checks) != 4 or None in checks:
        return original
    return sum(checks) > 1 if original else all(checks)


def document_pairs(row, public):
    if row['object'] is None:
        return set()
    universe = set(blocks.pairs(public))
    return {tuple(sorted(x.strip() for x in item['pair'].split(',')))
            for item in row['object']['relations']} & universe


def source(cfg):
    root = Path(cfg['reference_root'])
    if mb.sha(root/'complete.json') != cfg['reference_completion_sha256']:
        raise ValueError('D source completion changed')
    complete, manifest = mb.read(root/'complete.json'), mb.read(root/'manifest.json')
    if (complete['reads'] != 1000 or mb.sha(root/'manifest.json') != complete['manifest_sha256']
            or mb.sha(root/'report.json') != complete['report_sha256']):
        raise ValueError('D source is incomplete or modified')
    docs, jobs = mb.read(root/'documents.json'), mb.read(root/'jobs.json')
    if api.digest(jobs) != manifest['jobs_sha256'] or len(docs) != 200:
        raise ValueError('D source panel changed')
    rows = {}
    for req in jobs:
        key = req['key']
        path = root/'answers'/(key+'.json')
        if mb.sha(path) != complete['answers'][key]:
            raise ValueError('D answer changed')
        rows[key] = mb.read(path)
    structure = mb.read(root/'doc_structure.json')
    if (docs != mb.read(Path(cfg['historical_root'])/'documents.json')
            or structure != mb.read(Path(cfg['historical_root'])/'doc_structure.json')):
        raise ValueError('source evaluation panel differs from pinned history')
    public, supports, readings = {}, defaultdict(Counter), defaultdict(list)
    for req in jobs:
        d = req['doc_id']
        text = req['messages'][-1]['content']
        if not text.startswith('Document:\n'):
            raise ValueError('unexpected target text wrapper')
        value = dict(id=d, text=text[len('Document:\n'):], pair_list=docs[d]['pair_list'])
        if d in public and public[d] != value:
            raise ValueError('document text depends on demonstration')
        public[d] = value
        reading = mp.reading_pairs(rows[req['key']], set(blocks.pairs(value)), Counter())
        supports[d].update(reading)
        readings[d].append(reading)
    if any(len(readings[d]) != 5 for d in public):
        raise ValueError('reference requires five fixed readings per document')
    baseline = {d:mp.vote(readings[d],3) for d in public}
    return manifest, jobs, docs, public, structure, baseline, supports


def checked_rows(out, manifest, jobs):
    result = []
    for req in jobs:
        record = mb.read(out/'responses'/(req['key']+'.json'))
        if record['contract'] != api.digest([manifest, req]) or record['sha256'] != api.digest(record['data']):
            raise ValueError('response contract/content mismatch')
        result.append(record['data'])
    return result


def producer(cfg):
    out = Path(cfg['campaign_root'])/'VC1'
    complete, manifest = mb.read(out/'complete.json'), mb.read(out/'manifest.json')
    if (manifest['configuration']['experiment_id'] != VC1_ID
            or manifest['configuration']['reference_completion_sha256'] != SOURCE_HASH
            or mb.sha(out/'manifest.json') != complete['manifest_sha256']
            or mb.sha(out/'report.json') != complete['report_sha256']):
        raise ValueError('incompatible VC1 screen producer')
    jobs = mb.read(out/'screen_jobs.json')
    if api.digest(jobs) != manifest['initial_jobs_sha256']:
        raise ValueError('VC1 screen job list changed')
    if set(complete['responses']) != {j['key'] for j in jobs}:
        raise ValueError('VC1 screen panel incomplete')
    for key, expected in complete['responses'].items():
        if mb.sha(out/'responses'/(key+'.json')) != expected:
            raise ValueError('VC1 screen response changed')
    return jobs, checked_rows(out, manifest, jobs), mb.sha(out/'complete.json')


def ssh_engine(cfg):
    from importlib.metadata import version
    previous = mb.read(Path(cfg['reference_root'])/'engine_identity.json')
    model = Path(cfg['model_path'])
    files = {name:mb.sha(model/name) for name in previous['files']}
    if files != previous['files']:
        raise ValueError('3090 model/tokenizer bytes differ from the frozen D checkpoint')
    identity = dict(files=files, engine=cfg['engine'], served_model=cfg['model'],
        container_image=cfg['container_image'],vllm_version=version('vllm'),
        torch_version=version('torch'),batch_launcher_sha256=mb.sha('/app/batch/start_qwen.sh'))
    freeze(Path(cfg['campaign_root'])/'engine_identity.json',identity)
    print('MC_SSH_ENGINE',api.canonical(dict(sha256=api.digest(identity),files=len(files),
          reference_weights_identical=True,runtime_matches_L40S=False)),flush=True)
    return identity


def prepare(cfg, hash_engine=True):
    out = mb.output_dir(cfg)
    if (out/'complete.json').exists():
        raise ValueError('experiment already complete; no additional GPU allocation')
    for name, expected in cfg['historical_sha256'].items():
        if mb.sha(Path(cfg['historical_root'])/name) != expected:
            raise ValueError('historical input changed')
    original, bases, docs, public, structure, baseline, supports = source(cfg)
    if set(public) != set(original['selected']):
        raise ValueError('document set changed')
    rules = bases[0]['messages'][0]['content'].split('\n\nReturn only a JSON object')[0]
    arm, routes, upstream = cfg['arm'], None, None
    if arm in ('N1', 'N2'):
        jobs = document_jobs(bases, sampled=arm == 'N2')
    elif arm == 'VC1':
        jobs = screen_jobs(public, rules)
    elif arm == 'W4':
        jobs, routes = w4_jobs(public, rules, baseline, supports)
    else:
        source_jobs, source_rows, upstream = producer(cfg)
        jobs, routes = direct_jobs(public, rules, baseline, views(source_jobs, source_rows))
    engine = (ssh_engine(cfg) if cfg.get('backend')=='ssh3090' else mb.freeze_engine(cfg)) if hash_engine else mb.read(Path(cfg['reference_root'])/'engine_identity.json')
    if cfg.get('backend')!='ssh3090' and api.digest(engine) != original['engine_sha256']:
        raise ValueError('reference/current engine differs')
    manifest = dict(protocol=VERSION, configuration=cfg, selected=original['selected'],
        demo_ids=original['demo_ids'], engine_sha256=api.digest(engine),
        reference_engine_sha256=original['engine_sha256'],
        reference_manifest_sha256=mb.sha(Path(cfg['reference_root'])/'manifest.json'),
        initial_jobs_sha256=api.digest(jobs), upstream_vc1_completion_sha256=upstream,
        expected_initial=len(jobs), source_code=cfg['source_commits'])
    name = {'N1':'diverse', 'N2':'sampled', 'VC1':'screen', 'W4':'filter', 'MV2':'direct'}[arm]
    for filename, value in [('manifest',manifest),(name+'_jobs',jobs),('public',public),
                            ('documents',docs),('doc_structure',structure),('base_jobs',bases),
                            ('baseline',{d:sorted(ps) for d,ps in baseline.items()}),
                            ('engine_identity',engine),('routing',routes)]:
        freeze(out/(filename+'.json'), value)
    print('MC_PLAN',api.canonical(dict(arm=arm, jobs=len(jobs), configuration=cfg,
        manifest_sha256=mb.sha(out/'manifest.json'),
        candidate_pairs=sum(len(blocks.pairs(x)) for x in public.values()))),flush=True)


class Reader:
    smoke = mb.Reader.smoke

    def __init__(self, cfg, manifest, url):
        from transformers import AutoTokenizer
        self.cfg, self.manifest, self.out = cfg, manifest, mb.output_dir(cfg)
        self.tok = AutoTokenizer.from_pretrained(cfg['model_path'], trust_remote_code=True)
        self.client = api.Client(url, os.environ['MB_API_KEY'], self.out/'http-cache',
                                 api.digest(manifest), timeout=21600)
        entry = next(x for x in self.client.call('/models')[0]['data'] if x['id']==cfg['model'])
        if entry.get('max_model_len') != cfg['engine']['max_model_len']:
            raise ValueError('server context mismatch')
        text, ids = mp.prompt_tokens(self.tok,[dict(role='user',content='Return JSON.')])
        tokenized = self.client.call('/../tokenize',dict(model=cfg['model'],prompt=text,add_special_tokens=False))[0]
        if tokenized['tokens'] != ids:
            raise ValueError('tokenizer mismatch')

    def check(self, jobs):
        for req in jobs:
            _, ids = mp.prompt_tokens(self.tok,req['messages'])
            if len(ids)+req['thinking']+req['answer_tokens']+32 > self.cfg['engine']['max_model_len']:
                raise ValueError('request exceeds context; refusing truncation/exclusion')

    def one(self, req):
        key, contract = req['key'], api.digest([self.manifest,req])
        target = self.out/'responses'/(key+'.json')
        if target.exists():
            return checked_rows(self.out,self.manifest,[req])[0]
        trace_path = self.out/'traces'/(key+'.json')
        if trace_path.exists():
            record = mb.read(trace_path)
            if record['contract'] != contract or record['sha256'] != api.digest(record['data']):
                raise ValueError('trace changed')
            trace = record['data']
        else:
            text, ids = mp.prompt_tokens(self.tok,req['messages'])
            body = api.payload(self.cfg['model'],[ids],req['thinking'])
            body.update(temperature=req.get('temperature',0.),seed=req.get('seed',20261115))
            result, _, elapsed = self.client.call('/completions',body)
            choice = api.ordered_choices(result,1)[0]
            if choice['finish_reason'] not in ('length','stop'):
                raise ValueError('unexpected thinking termination')
            after = self.tok.encode(text+choice['text']+'\n</think>\n\n',add_special_tokens=False)
            if len(after)+req['answer_tokens'] > self.cfg['engine']['max_model_len']:
                raise ValueError('final continuation exceeds context')
            trace = dict(thought=choice['text'], final_prompt_ids=after,
                think_tokens=len(choice.get('token_ids') or self.tok.encode(choice['text'],add_special_tokens=False)),
                forced=choice.get('stop_reason')!='</think>',seconds=elapsed,
                run=os.environ['ORX_RUN_ID'],commit=os.environ['ORX_COMMIT'])
            freeze(trace_path,dict(contract=contract,sha256=api.digest(trace),data=trace))
        body = api.payload(self.cfg['model'],[trace['final_prompt_ids']],req['answer_tokens'],'direct')
        body['structured_outputs']['json'] = req['schema']
        body.update(temperature=req.get('temperature',0.),seed=req.get('seed',20261115))
        result, _, elapsed = self.client.call('/completions',body)
        final = api.ordered_choices(result,1)[0]
        obj = None
        if final['finish_reason']=='stop':
            if req['phase'] in ('diverse','sampled','verification'):
                obj = vp.document_parse(final['text'])
            elif req['phase']=='direct':
                obj = adjudicate.parse_final(final['text'],'direct')
            else:
                obj = blocks.parse_answer(final['text'],len(req['pairs']))
        row = dict(key=key,doc_id=req['doc_id'],phase=req['phase'],object=obj,answer=final['text'],
            forced=trace['forced'],truncated=final['finish_reason']=='length',
            think_tokens=trace['think_tokens'],answer_tokens=len(final.get('token_ids') or self.tok.encode(final['text'],add_special_tokens=False)),
            request_seconds=trace['seconds']+elapsed,run=os.environ['ORX_RUN_ID'],
            commit=os.environ['ORX_COMMIT'],trace_run=trace['run'])
        freeze(target,dict(contract=contract,sha256=api.digest(row),data=row))
        return row


def phase(reader, name, jobs, deadline):
    out = reader.out
    freeze(out/(name+'_jobs.json'),jobs)
    done = [j for j in jobs if (out/'responses'/(j['key']+'.json')).exists()]
    checked_rows(out,reader.manifest,done)
    missing = [j for j in jobs if not (out/'responses'/(j['key']+'.json')).exists()]
    print('MC_PHASE',api.canonical(dict(phase=name,expected=len(jobs),cached=len(done))),flush=True)
    if not missing:
        return True
    reader.check(missing)
    todo, count, errors = iter(missing), len(done), []
    with ThreadPoolExecutor(max_workers=reader.cfg['engine']['concurrency']) as pool:
        pending = {}
        while pending or time.monotonic()<deadline:
            while not errors and time.monotonic()<deadline and len(pending)<reader.cfg['engine']['concurrency']:
                req = next(todo,None)
                if req is None:
                    break
                pending[pool.submit(reader.one,req)] = req
            if not pending:
                break
            finished,_ = wait(pending,return_when=FIRST_COMPLETED)
            for future in finished:
                req = pending.pop(future)
                try:
                    future.result()
                    count += 1
                except Exception as exc:
                    errors.append(dict(key=req['key'],error=repr(exc)[:1200]))
                    print('MC_ERROR',api.canonical(errors[-1]),flush=True)
            print('MC_PROGRESS',api.canonical(dict(arm=reader.cfg['arm'],phase=name,done=count,
                expected=len(jobs),errors=len(errors))),flush=True)
    if errors:
        raise RuntimeError('transport/contract failures; preserve missing work, no score')
    return count==len(jobs)


def paired_systems(cfg, out):
    public = mb.read(out/'public.json')
    baseline = {d:{tuple(p) for p in ps} for d,ps in mb.read(out/'baseline.json').items()}
    manifest = mb.read(out/'manifest.json')
    systems, all_jobs, stats = dict(D=baseline), [], {}
    if cfg['arm'] in ('N1','N2'):
        name = 'diverse' if cfg['arm']=='N1' else 'sampled'
        jobs = mb.read(out/(name+'_jobs.json'))
        rows = checked_rows(out,manifest,jobs)
        all_jobs += jobs
        reads = defaultdict(list)
        for row in rows:
            reads[row['doc_id']].append(document_pairs(row,public[row['doc_id']]))
        if any(len(reads[d])!=5 for d in public):
            raise ValueError('incomplete document-reading panel')
        systems[name+'_vote5'] = {d:mp.vote(reads[d],3) for d in public}
        if cfg['arm']=='N2':
            jobs = mb.read(out/'verification_jobs.json')
            rows = checked_rows(out,manifest,jobs)
            all_jobs += jobs
            systems['N2'] = {r['doc_id']:document_pairs(r,public[r['doc_id']]) for r in rows}
            if set(systems['N2']) != set(public):
                raise ValueError('incomplete N2 final panel')
        else:
            systems['N1'] = systems['diverse_vote5']
    elif cfg['arm']=='VC1':
        jobs = mb.read(out/'screen_jobs.json')
        all_jobs += jobs
        systems['VC1'] = local_filter(baseline,views(jobs,checked_rows(out,manifest,jobs)))
    elif cfg['arm']=='W4':
        jobs = mb.read(out/'filter_jobs.json')
        rows = checked_rows(out,manifest,jobs)
        all_jobs += jobs
        checks = defaultdict(dict)
        for req,row in zip(jobs,rows,strict=True):
            for i,p in enumerate(req['pairs']):
                label = row['object'][str(i)] if row['object'] is not None else None
                checks[req['doc_id'],tuple(p)][req['reverse']] = label
        systems['W4'] = {d:{p for p in ps if not all(checks[d,p].get(o)=='N' for o in (False,True))}
                         for d,ps in baseline.items()}
        stats['routed_pairs'] = len(checks)
    else:
        sj,sr,pin = producer(cfg)
        if pin != manifest['upstream_vc1_completion_sha256']:
            raise ValueError('VC1 producer changed')
        screen = views(sj,sr)
        jobs = mb.read(out/'direct_jobs.json')
        rows = checked_rows(out,manifest,jobs)
        all_jobs += jobs
        judges = defaultdict(dict)
        for req,row in zip(jobs,rows,strict=True):
            judges[req['doc_id'],tuple(req['pair'])][req['reverse']] = (
                None if row['object'] is None else row['object']['label']!='N')
        for name in ('MV2','five_view_majority','four_view_consensus','screen_majority_control'):
            systems[name] = {d:set(ps) for d,ps in baseline.items()}
        coverage = Counter()
        for doc,data in public.items():
            for p in blocks.pairs(data):
                original = p in baseline[doc]
                local = [screen.get((doc,o),{}).get(p) for o in (0,1)]
                checks = local + [judges[doc,p].get(o) for o in (False,True)]
                coverage['candidate_pairs'] += 1
                coverage['complete_four_checks'] += None not in checks
                decisions = dict(MV2=asymmetric(original,checks))
                if None not in local:
                    decisions['screen_majority_control'] = sum(local)+original>=2
                if None not in checks:
                    n = sum(checks)
                    decisions['five_view_majority'] = n+original>=3
                    decisions['four_view_consensus'] = n!=0 if original else n==4
                for name,present in decisions.items():
                    if present: systems[name][doc].add(p)
                    else: systems[name][doc].discard(p)
        stats['coverage'] = dict(coverage)
    return systems, all_jobs, stats


def score(cfg):
    out = mb.output_dir(cfg)
    manifest, docs, structure = (mb.read(out/(name+'.json')) for name in ('manifest','documents','doc_structure'))
    systems,jobs,stats = paired_systems(cfg,out)
    rows = checked_rows(out,manifest,jobs)
    per_doc, metrics = {}, {}
    for name,pred in systems.items():
        fake = [dict(key=d,doc_id=d) for d in docs]
        answers = {d:dict(status='valid',answer=api.canonical(dict(relations=[dict(pair=','.join(p),label='causes') for p in sorted(pred[d])]))) for d in docs}
        counts = mb.pairs_and_counts(fake,answers,docs,structure,1)[1]
        per_doc[name] = counts
        metrics[name] = {scope:mb.metric(values) for scope,values in counts.items()}
    contrasts = {}
    comparisons = [(cfg['arm'],'D')]
    if cfg['arm']=='N2': comparisons.append(('N2','sampled_vote5'))
    for name,ref in comparisons:
        contrasts[name+' - '+ref] = dict(delta_ci95=paired_delta(per_doc[name]['all'],per_doc[ref]['all'],
            draws=5000,seed=20261112),primary=ref=='D',unit='document')
    report = dict(protocol=VERSION,configuration=cfg,manifest_sha256=mb.sha(out/'manifest.json'),
        jobs_done=len(rows),jobs_expected=len(jobs),metrics=metrics,contrasts=contrasts,per_document=per_doc,
        invalid=sum(r['object'] is None for r in rows),truncated=sum(r['truncated'] for r in rows),
        forced=sum(r['forced'] for r in rows),stats=stats,
        cost=dict(think_tokens=sum(r['think_tokens'] for r in rows),answer_tokens=sum(r['answer_tokens'] for r in rows),
                  source_D_cost='Retain full C+D reference recipe cost; new calls are incremental.',
                  upstream_VC1_cost_required=cfg['arm']=='MV2'),
        runs=sorted({r['run'] for r in rows}),scope='Previously exposed200docs. Recipe transfers, not equal-budget ablations. Five exploratory comparisons; nominal unadjusted intervals.')
    mb.atomic(out/'report.json',report)
    mb.atomic(out/'predictions.json',{name:{d:sorted(ps) for d,ps in pred.items()} for name,pred in systems.items()})
    freeze(out/'complete.json',dict(arm=cfg['arm'],experiment_id=cfg['experiment_id'],
        manifest_sha256=mb.sha(out/'manifest.json'),report_sha256=mb.sha(out/'report.json'),
        phase_jobs_sha256={p.name:mb.sha(p) for p in out.glob('*_jobs.json')},
        responses={j['key']:mb.sha(out/'responses'/(j['key']+'.json')) for j in jobs},runs=report['runs']))
    print('MC_RESULT',api.canonical({k:v for k,v in report.items() if k!='per_document'}),flush=True)
    print('MC_COMPLETE',api.canonical(dict(path=str(out/'complete.json'),sha256=mb.sha(out/'complete.json'))),flush=True)


def run(cfg,url):
    out = mb.output_dir(cfg)
    manifest = mb.read(out/'manifest.json')
    if manifest['configuration'] != cfg or (out/'complete.json').exists():
        raise ValueError('configuration changed or experiment already complete')
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reader = Reader(cfg,manifest,url)
        reader.smoke()
        tick = time.monotonic()
        deadline = tick+cfg['lease_seconds']
        attempt = out/'attempts'/(os.environ['ORX_RUN_ID']+'-'+os.environ.get('SLURM_JOB_ID',os.environ.get('REPLAY_LEASE_ID','gpu'))+'.json')
        record = dict(run=os.environ['ORX_RUN_ID'],commit=os.environ['ORX_COMMIT'],started_unix=time.time())
        mb.atomic(attempt,record)
        try:
            name = {'N1':'diverse','N2':'sampled','VC1':'screen','W4':'filter','MV2':'direct'}[cfg['arm']]
            jobs = mb.read(out/(name+'_jobs.json'))
            if api.digest(jobs)!=manifest['initial_jobs_sha256']:
                raise ValueError('initial job list changed')
            if not phase(reader,name,jobs,deadline): return 75
            if cfg['arm']=='N2':
                jobs = verification_jobs(mb.read(out/'base_jobs.json'),checked_rows(out,manifest,jobs),mb.read(out/'public.json'))
                if not phase(reader,'verification',jobs,deadline): return 75
            score(cfg)
            return 0
        finally:
            record.update(finished_unix=time.time(),inference_seconds=time.monotonic()-tick)
            mb.atomic(attempt,record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cmd',choices=('wait-source','prepare','run','score'))
    parser.add_argument('--config',default='steering/maven_replay_config.json')
    parser.add_argument('--url')
    args = parser.parse_args()
    cfg = config(args.config)
    if args.cmd=='wait-source':
        predecessor = cfg.get('predecessor') if cfg.get('backend')=='ssh3090' else ('VC1' if cfg['arm']=='MV2' else None)
        if predecessor:
            deadline=time.monotonic()+14*86400
            while not (Path(cfg['campaign_root'])/predecessor/'complete.json').exists():
                if time.monotonic()>deadline: raise TimeoutError(predecessor+' producer not complete')
                print('MC_WAIT '+predecessor+' complete; no GPU reserved',flush=True)
                time.sleep(60)
    elif args.cmd=='prepare': prepare(cfg)
    elif args.cmd=='score': score(cfg)
    else: raise SystemExit(run(cfg,args.url))


if __name__=='__main__': main()
