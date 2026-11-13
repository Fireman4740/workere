"""Paired MAVEN200 event-coreference and inter-sentence recall experiments.

Public inference jobs are constructed before scoring and never contain test gold.
Each node runs its treatment and control on one L40S, with fixed five-read votes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import fcntl
import itertools
import json
import os
from pathlib import Path
import re
import time

from steering import campaign_api as api
from steering import maven_budget as mb
from steering import maven_pool as mp
from steering.compare import paired_delta

VERSION = 'maven-recall-20261113-v1'
COMMON = ('Identify document-supported CAUSE and PRECONDITION relations between marked event mentions. '
          'Include explicit and implicit causation, enablement and prevention. Reject temporal order, '
          'correlation and mere plausibility. Distinct subevents and repeated occurrences are not identical events. '
          'A mention that only expresses a causal connective is not itself a causal argument. ')


def obj_schema(fields):
    return dict(type='object', properties=fields, required=list(fields), additionalProperties=False)


def enum(values):
    return dict(type='string', enum=list(values))


def graph_schema(ids, mode):
    cluster = dict(type='array', items=enum(ids), minItems=1)
    if mode == 'singleton':
        cluster['maxItems'] = 1
    return obj_schema(dict(clusters=dict(type='array', items=cluster),
                           edges=dict(type='array', items=obj_schema(dict(cause=enum(ids), effect=enum(ids))))))


def groups_of(clusters, ids):
    ids, seen, groups = set(ids), set(), []
    for group in clusters:
        if not isinstance(group, list) or not group or any(not isinstance(e, str) or e not in ids for e in group):
            raise ValueError('unknown/empty cluster')
        if len(set(group)) != len(group) or seen.intersection(group):
            raise ValueError('clusters must be disjoint')
        seen.update(group)
        groups.append(sorted(group))
    groups.extend([[e] for e in sorted(ids-seen)])
    return {e: group for group in groups for e in group}


def parse_graph(value, ids, mode):
    if not isinstance(value, dict) or set(value) != {'clusters', 'edges'}:
        raise ValueError('graph keys')
    if not isinstance(value['clusters'], list) or not isinstance(value['edges'], list):
        raise ValueError('graph arrays')
    if mode == 'singleton' and any(len(g) != 1 for g in value['clusters']):
        raise ValueError('singleton control merged mentions')
    groups = groups_of(value['clusters'], ids)
    raw, expanded = set(), set()
    for edge in value['edges']:
        if not isinstance(edge, dict) or set(edge) != {'cause', 'effect'}:
            raise ValueError('edge keys')
        x, y = edge['cause'], edge['effect']
        if not isinstance(x, str) or not isinstance(y, str) or x not in groups or y not in groups or groups[x] == groups[y]:
            raise ValueError('unknown endpoint or within-event causal edge')
        raw.add(tuple(sorted((x, y))))
        expanded.update(tuple(sorted((a, b))) for a in groups[x] for b in groups[y])
    return raw, expanded


def propagate(pairs, clusters, ids):
    groups = groups_of(clusters, ids)
    return {tuple(sorted((a, b))) for x, y in pairs if groups[x] != groups[y]
            for a in groups[x] for b in groups[y]}


def historical_views(compact, docs):
    readings, keys = defaultdict(list), set()
    for row in compact['reads']:
        if row['budget'] != 'long':
            continue
        key = (row['doc'], row['pass'])
        if key in keys or row['doc'] not in docs or row['pass'] not in range(5):
            raise ValueError('invalid historical reading panel')
        keys.add(key)
        readings[row['doc']].append({tuple(sorted(p.split(','))) for p in row['pairs']}
                                    if row['status'] == 'valid' else set())
    if keys != {(d, p) for d in docs for p in range(5)}:
        raise ValueError('historical five-read panel incomplete')
    return {d: mp.vote(readings[d], 3) for d in docs}


def candidates(doc_id, text, structure, universe, baseline, cap, seed):
    """Gold-blind lexical ranking, equal quotas by distance; no evaluation labels."""
    tagged = {m.group(1): m for m in mp.c.TAG.finditer(text)}
    stop = set('the a an and or to of in on for with by from as is was were be at that this it its'.split())
    contexts = {e: set(re.findall(r'[a-z]{3,}', text[max(0, m.start()-100):m.end()+100].lower()))-stop
                for e, m in tagged.items()}
    sentence = structure['sentence_of']
    buckets = defaultdict(list)
    for x, y in universe-baseline:
        distance = abs(sentence[x]-sentence[y])
        if distance == 0:
            continue
        bucket = 1 if distance == 1 else 2 if distance <= 3 else 4
        score = len(contexts[x] & contexts[y]) / max(1, len(contexts[x] | contexts[y]))
        buckets[bucket].append((-score, api.digest([seed, doc_id, x, y]), (x, y)))
    ordered = {b: [r[2] for r in sorted(buckets[b])] for b in (1, 2, 4)}
    # Round robin guarantees representation of each available distance stratum.
    selected = []
    for row in itertools.zip_longest(*(ordered[b] for b in (1, 2, 4))):
        selected.extend(pair for pair in row if pair is not None)
        if len(selected) >= cap:
            break
    return selected[:cap]


def graph_messages(base, mode, ids):
    messages = copy.deepcopy(base)
    instruction = ('Group mentions only when they denote the identical event occurrence. Leave uncertain '
                   'identity as separate singletons. Clusters must be disjoint; omitted IDs are singletons. '
                   'Output one representative mention ID for each endpoint of each causal edge. The program '
                   'will expand that edge over every cross-cluster mention pair. Do not link within a cluster. '
                   if mode == 'coref' else
                   'Treat every mention as a separate singleton node. Do not merge mentions or expand '
                   'a causal edge to other mentions. Clusters may be omitted or contain singleton IDs only. ')
    messages[0]['content'] = COMMON + instruction + 'Return exactly a JSON object with clusters and edges (cause, effect).'
    demo = json.loads(messages[2]['content'])
    edges = set()
    for r in demo['relations']:
        x, y = r['pair'].split(',')
        edges.add((x, y) if r['label'] == 'causes' else (y, x))
    messages[2]['content'] = api.canonical(dict(clusters=[], edges=[dict(cause=x, effect=y) for x, y in sorted(edges)]))
    messages[-1]['content'] += '\nAllowed mention IDs: '+api.canonical(ids)
    return messages


def build_jobs(cfg, bases, docs, structures, baseline):
    jobs, candidate_map = [], {}
    if cfg['kind'] == 'search':
        for base in bases:
            doc = base['doc_id']
            if doc not in candidate_map:
                candidate_map[doc] = candidates(doc, base['messages'][-1]['content'], structures[doc],
                    set(mp.c.pairs(docs[doc])), baseline[doc], cfg['candidate_cap'], cfg['seed'])
    for base in bases:
        doc = base['doc_id']
        ids = sorted(structures[doc]['sentence_of'])
        modes = ('singleton', 'coref') if cfg['kind'] == 'coref' else ('direct', 'structured')
        blocks = [None] if cfg['kind'] == 'coref' else [candidate_map[doc][i:i+cfg['block_size']]
                    for i in range(0, len(candidate_map[doc]), cfg['block_size'])]
        for i, block in enumerate(blocks):
            for mode in modes:
                messages = graph_messages(base['messages'], mode, ids) if cfg['kind'] == 'coref' else base['messages']
                job = dict(base_key=base['key'], doc_id=doc, pass_id=base['pass_id'], demo_id=base['demo_id'],
                           mode=mode, block=i, pairs=block, ids=ids, messages=messages)
                job['key'] = api.digest(job)
                jobs.append(job)
    # Interleave controls/treatments; balanced order alternates by document-demo.
    jobs.sort(key=lambda j: (j['doc_id'], j['pass_id'], j['block'],
                            (j['mode'] == ('coref' if cfg['kind']=='coref' else 'structured')) ^ (j['pass_id'] % 2)))
    return jobs, candidate_map


def config(path):
    cfg = mb.read(path)
    if cfg['protocol'] != VERSION or cfg['kind'] not in ('coref', 'search'):
        raise ValueError('unknown recall protocol')
    if cfg['documents'] != 200 or cfg['passes'] != 5 or cfg['engine']['max_model_len'] != 65536:
        raise ValueError('frozen panel/context changed')
    return cfg


def prepare(cfg):
    out, history = mb.output_dir(cfg), Path(cfg['historical_root'])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('completed experiment; refusing GPU allocation')
    for name, expected in cfg['historical_sha256'].items():
        if mb.sha(history/name) != expected:
            raise ValueError('historical file changed: '+name)
    old, docs, structure = (mb.read(history/n) for n in ('manifest.json', 'documents.json', 'doc_structure.json'))
    if old['revision'] != mp.MAVEN_REVISION or old['prompt_sha256'] != mb.sha(mp.PROMPT):
        raise ValueError('historical revision/prompt changed')
    bases = mb.import_jobs(cfg, old, docs)
    baseline = historical_views(mb.read(history/'reads_compact.json'), docs)
    jobs, selected = build_jobs(cfg, bases, docs, structure, baseline)
    engine = mb.freeze_engine(cfg)
    manifest = dict(protocol=VERSION, configuration=cfg, engine_sha256=api.digest(engine),
                    jobs_sha256=api.digest(jobs), expected_reads=len(jobs), selected=old['selected'],
                    demo_ids=old['demo_ids'], historical_sha256=cfg['historical_sha256'])
    values = [('manifest', manifest), ('jobs', jobs), ('documents', docs), ('doc_structure', structure),
              ('baseline', {d:sorted(p) for d,p in baseline.items()}), ('candidates', selected),
              ('engine_identity', engine)]
    if cfg['kind'] == 'coref':
        # Offline diagnostic only: NEVER inserted in a public job or model request.
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        path = hf_hub_download(mp.MAVEN_REPO, 'data/test-00000-of-00001.parquet',
                              repo_type='dataset', revision=mp.MAVEN_REVISION)
        if mb.sha(path) != old['sources']['test']['sha256']:
            raise ValueError('oracle diagnostic dataset bytes changed')
        rows = pq.read_table(path, columns=['id','mentions','coref_clusters']).to_pylist()
        oracle = {r['id']: [[r['mentions'][i] for i in group] for group in r['coref_clusters']]
                  for r in rows if r['id'] in docs}
        if set(oracle) != set(docs):
            raise ValueError('oracle diagnostic panel incomplete')
        for d, groups in oracle.items():
            groups_of(groups, structure[d]['sentence_of'])
        values.append(('oracle_diagnostic_only', oracle))
    for name, value in values:
        mb.atomic(out/(name+'.json'), value, immutable=True)
    print('MR_PLAN', api.canonical(dict(arm=cfg['arm'], jobs=len(jobs), configuration=cfg,
          manifest_sha256=mb.sha(out/'manifest.json'),
          candidate_pairs=sum(len(p) for p in selected.values()))), flush=True)


class Reader(mb.Reader):
    def smoke(self):
        super().smoke()
        probes = [
            (graph_schema(['e1','e2'],'coref'), 'Return {"clusters":[],"edges":[]}.',
             lambda value: parse_graph(value,['e1','e2'],'coref')),
            (graph_schema(['e1','e2'],'singleton'), 'Return {"clusters":[],"edges":[]}.',
             lambda value: parse_graph(value,['e1','e2'],'singleton')),
            (obj_schema(dict(labels=obj_schema({'0':enum('NAB')}))), 'Return {"labels":{"0":"N"}}.',
             lambda value: value['labels']['0'] in ('N','A','B')),
            (obj_schema(dict(frames=dict(type='array',items=obj_schema(dict(id=enum(['e1']),
                 participant=dict(type='string'),quote=dict(type='string')))))),
             'Return {"frames":[]}.', lambda value: grounded_frames(value,'',{'e1'}))]
        for schema,instruction,validate in probes:
            text=self.tok.apply_chat_template([dict(role='user',content=instruction)],tokenize=False,
                                              add_generation_prompt=True,enable_thinking=False)
            ids=self.tok.encode(text,add_special_tokens=False)
            body=api.payload(self.cfg['model'],[ids],128,'direct')
            body['structured_outputs']['json']=schema
            response,_,_=self.client.call('/completions',body)
            choice=api.ordered_choices(response,1)[0]
            if choice['finish_reason']!='stop':raise ValueError('custom schema smoke truncated')
            validate(json.loads(choice['text']))
        print('MR_SCHEMA_SMOKE',api.canonical(dict(schemas=len(probes),status='passed')),flush=True)

    def stage(self, job, name, messages, schema, thinking, answer):
        identity = api.digest(dict(protocol=VERSION, engine=self.manifest['engine_sha256'],
            job=job['key'], name=name, messages=messages, schema=schema, thinking=thinking, answer=answer))
        target = self.out/'stages'/(identity+'.json')
        if target.exists():
            row = mb.read(target)
            if row['contract'] != identity or row['sha256'] != api.digest(row['data']):
                raise ValueError('stage identity/content mismatch')
            return row['data']
        text, ids = mp.prompt_tokens(self.tok, messages)
        if len(ids)+thinking+answer+32 > self.cfg['engine']['max_model_len']:
            raise ValueError('stage exceeds context; refusing truncation')
        first, _, seconds1 = self.client.call('/completions', api.payload(self.cfg['model'], [ids], thinking), cache=True)
        thought = api.ordered_choices(first, 1)[0]
        if thought['finish_reason'] not in ('stop','length'):
            raise ValueError('thinking transport finish')
        final_ids = self.tok.encode(text+thought['text']+'\n</think>\n\n', add_special_tokens=False)
        if len(final_ids)+answer > self.cfg['engine']['max_model_len']:
            raise ValueError('final prompt exceeds context')
        body = api.payload(self.cfg['model'], [final_ids], answer, 'direct')
        body['structured_outputs']['json'] = schema
        result, _, seconds2 = self.client.call('/completions', body, cache=True)
        final = api.ordered_choices(result, 1)[0]
        parsed = None
        if final['finish_reason']=='stop':
            from steering.campaign_facts import unique_object
            try:
                parsed = json.loads(final['text'], object_pairs_hook=unique_object)
            except ValueError:
                pass
        data = dict(answer=final['text'], parsed=parsed, answer_finish=final['finish_reason'],
                    thought=thought['text'], thinking_forced=thought.get('stop_reason')!='</think>',
                    think_tokens=len(thought.get('token_ids') or self.tok.encode(thought['text'], add_special_tokens=False)),
                    answer_tokens=len(final.get('token_ids') or self.tok.encode(final['text'], add_special_tokens=False)),
                    prompt_tokens=len(ids), final_input_tokens=len(final_ids), seconds=seconds1+seconds2,
                    run=os.environ['ORX_RUN_ID'], commit=os.environ['ORX_COMMIT'])
        mb.atomic(target, dict(contract=identity, sha256=api.digest(data), data=data), immutable=True)
        return data

    def one(self, job):
        target = self.out/'responses'/(job['key']+'.json')
        contract = api.digest([self.manifest, job])
        if target.exists():
            row = mb.read(target)
            if row['contract']!=contract or row['sha256']!=api.digest(row['data']):
                raise ValueError('response identity/content mismatch')
            return row['data'], True
        stages, valid, raw, expanded = [], True, set(), set()
        evidence_invalid = False
        if self.cfg['kind']=='coref':
            result = self.stage(job, 'graph', job['messages'], graph_schema(job['ids'], job['mode']), 20480, 4096)
            stages.append(result)
            try:
                raw, expanded = parse_graph(result['parsed'], job['ids'], job['mode'])
            except (ValueError, TypeError, KeyError):
                valid = False
        else:
            messages = copy.deepcopy(job['messages'])
            table = '\nCandidate pairs (fixed indices): '+api.canonical({str(i):p for i,p in enumerate(job['pairs'])})
            source_text = messages[-1]['content']
            frames = []
            if job['mode']=='structured':
                extraction = copy.deepcopy(messages)
                extraction[0]['content'] = ('Extract participants of the listed marked events from the document. '
                    'Return frames with id, participant and quote. Quote must be a verbatim substring of the '
                    'document including the <eventID surface> tag and the participant string. '
                    'Do not invent arguments, infer causality or resolve uncertain entities. Omit unsupported frames.')
                extraction[-1]['content'] += table
                endpoints = sorted({e for pair in job['pairs'] for e in pair})
                schema = obj_schema(dict(frames=dict(type='array', items=obj_schema(dict(id=enum(endpoints),
                                     participant=dict(type='string'), quote=dict(type='string'))))))
                result = self.stage(job, 'participants', extraction, schema, 1024, 2048)
                stages.append(result)
                frames = grounded_frames(result['parsed'], source_text, set(endpoints))
                if frames is None:
                    frames, evidence_invalid = [], True
            messages[0]['content'] = COMMON + ('For EACH listed pair answer A if first causes/enables second, '
                'B if second causes/enables first, N otherwise. Consider the full document. '
                'Return labels for exactly the supplied indices. Do not assume a proposed pair is causal. ')
            messages[-1]['content'] += table
            if job['mode']=='structured':
                messages[-1]['content'] += '\nPredicted, verbatim-grounded participant frames (fallible; verify): '+api.canonical(frames)
            labels = obj_schema({str(i):enum('NAB') for i in range(len(job['pairs']))})
            schema = obj_schema(dict(labels=labels))
            result = self.stage(job, 'judge', messages, schema, 3072 if job['mode']=='structured' else 6144, 512)
            stages.append(result)
            parsed = result['parsed']
            if (not isinstance(parsed,dict) or set(parsed)!={'labels'} or not isinstance(parsed['labels'],dict)
                    or set(parsed['labels'])!={str(i) for i in range(len(job['pairs']))}
                    or any(v not in ('N','A','B') for v in parsed['labels'].values())):
                valid = False
            else:
                expanded = {tuple(sorted(pair)) for i,pair in enumerate(job['pairs']) if parsed['labels'][str(i)]!='N'}
                raw = expanded
        data = dict(key=job['key'], doc_id=job['doc_id'], pass_id=job['pass_id'], mode=job['mode'],
                    valid=valid, evidence_invalid=evidence_invalid, raw=sorted(raw) if valid else [],
                    pairs=sorted(expanded) if valid else [],
                    cost={k:sum(s[k] for s in stages) for k in
                          ('think_tokens','answer_tokens','prompt_tokens','final_input_tokens','seconds')},
                    forced=sum(s['thinking_forced'] for s in stages),
                    truncated=sum(s['answer_finish']=='length' for s in stages), run=os.environ['ORX_RUN_ID'])
        mb.atomic(target, dict(contract=contract, sha256=api.digest(data), data=data), immutable=True)
        return data, False


def grounded_frames(value, text, allowed):
    if not isinstance(value,dict) or set(value)!={'frames'} or not isinstance(value['frames'],list):
        return None
    for frame in value['frames']:
        if (not isinstance(frame,dict) or set(frame)!={'id','participant','quote'}
                or not isinstance(frame['id'],str) or frame['id'] not in allowed or not isinstance(frame['participant'],str)
                or not isinstance(frame['quote'],str) or not frame['participant']
                or frame['quote'] not in text or frame['participant'] not in frame['quote']
                or not re.search(r'<'+re.escape(frame['id'])+r'\s',frame['quote'])):
            return None
    return value['frames']


def run(cfg, url):
    out = mb.output_dir(cfg)
    manifest, jobs = mb.read(out/'manifest.json'), mb.read(out/'jobs.json')
    if manifest['configuration']!=cfg or api.digest(jobs)!=manifest['jobs_sha256']:
        raise ValueError('changed public jobs/configuration')
    if (out/'complete.json').exists():
        raise ValueError('experiment already complete')
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reader = Reader(cfg, manifest, jobs, url)
        reader.smoke()
        todo, stats, errors = iter(jobs), Counter(), []
        tick = time.monotonic()
        with ThreadPoolExecutor(max_workers=cfg['engine']['concurrency']) as pool:
            pending = {pool.submit(reader.one,j):j for j in itertools.islice(todo,cfg['engine']['concurrency'])}
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = pending.pop(future)
                    try:
                        row, hit = future.result()
                        stats['done']+=1
                        stats['invalid']+=not row['valid']
                        stats['cache_hits']+=hit
                        stats['tokens']+=row['cost']['think_tokens']+row['cost']['answer_tokens']
                    except Exception as exc:
                        errors.append(dict(key=job['key'],error=repr(exc)[:1000]))
                        print('MR_ERROR',api.canonical(errors[-1]),flush=True)
                if not errors:
                    for job in itertools.islice(todo,len(finished)):
                        pending[pool.submit(reader.one,job)]=job
                print('MR_PROGRESS',api.canonical(dict(arm=cfg['arm'],expected=len(jobs),**stats,
                      minutes=round((time.monotonic()-tick)/60,2),errors=len(errors))),flush=True)
        mb.atomic(out/'attempts'/(os.environ['ORX_RUN_ID']+'.json'),
                  dict(seconds=time.monotonic()-tick,errors=errors,stats=stats))
        if errors:
            raise RuntimeError('incomplete inference; resume only missing jobs')
        score(cfg)


def counts_for(predictions, docs, structure):
    jobs, rows = [], {}
    for doc, pred in predictions.items():
        jobs.append(dict(doc_id=doc,key=doc))
        rows[doc] = dict(status='valid',answer=api.canonical(dict(relations=[dict(pair=','.join(p),label='causes') for p in pred])))
    return mb.pairs_and_counts(jobs,rows,docs,structure,1)[1]


def changes(pred, reference, docs):
    result = Counter()
    for doc in docs:
        gold = {tuple(sorted(p.split(','))) for p in docs[doc]['gold']}
        added, removed = pred[doc]-reference[doc], reference[doc]-pred[doc]
        result.update(added_tp=len(added & gold), added_fp=len(added-gold),
                      removed_tp=len(removed & gold), removed_fp=len(removed-gold))
    return dict(result)


def score(cfg):
    out = mb.output_dir(cfg)
    manifest,jobs,docs,structure = (mb.read(out/(n+'.json')) for n in ('manifest','jobs','documents','doc_structure'))
    if manifest['configuration']!=cfg or api.digest(jobs)!=manifest['jobs_sha256']:
        raise ValueError('changed scoring contract')
    rows, hashes = [], {}
    for job in jobs:
        path = out/'responses'/(job['key']+'.json')
        record = mb.read(path)  # any missing response prohibits a final report
        if record['contract']!=api.digest([manifest,job]) or record['sha256']!=api.digest(record['data']):
            raise ValueError('scoring response mismatch')
        rows.append(record['data'])
        hashes[job['key']] = mb.sha(path)
    baseline = {d:{tuple(p) for p in ps} for d,ps in mb.read(out/'baseline.json').items()}
    by_pass = defaultdict(set)
    for row in rows:
        by_pass[row['mode'],row['doc_id'],row['pass_id']].update(tuple(p) for p in row['pairs'])
        if cfg['kind']=='coref' and row['mode']=='coref':
            by_pass['coref_no_expansion',row['doc_id'],row['pass_id']].update(tuple(p) for p in row['raw'])
    modes = ['singleton','coref','coref_no_expansion'] if cfg['kind']=='coref' else ['direct','structured']
    systems, predictions, per_doc = {}, {}, {}
    for mode in modes:
        for threshold in range(1,6):
            name = mode+f'_ge{threshold}'
            pred = {d:mp.vote([by_pass[mode,d,p] for p in range(5)],threshold) for d in docs}
            if cfg['kind']=='search':
                pred = {d:pred[d] | baseline[d] for d in docs}
            predictions[name]=pred
    predictions['historical_long_vote5']=baseline
    if cfg['kind']=='coref':
        oracle = mb.read(out/'oracle_diagnostic_only.json')
        predictions['oracle_propagation_diagnostic']={d:propagate(baseline[d],oracle[d],structure[d]['sentence_of']) for d in docs}
    for name,pred in predictions.items():
        counts = counts_for(pred,docs,structure)
        per_doc[name]=counts
        systems[name] = dict(scopes={scope:mb.metric(values) for scope,values in counts.items()},
                             changes_vs_history=changes(pred,baseline,docs))
    treatment, control = ('coref_ge3','singleton_ge3') if cfg['kind']=='coref' else ('structured_ge3','direct_ge3')
    contrasts={}
    comparisons = [(treatment,control),(treatment,'historical_long_vote5'),(control,'historical_long_vote5')]
    if cfg['kind']=='coref':
        comparisons.append((treatment,'coref_no_expansion_ge3'))
    for name,reference in comparisons:
        contrasts[name+' - '+reference] = dict(delta_ci95=paired_delta(per_doc[name]['all'],per_doc[reference]['all'],
            draws=cfg['bootstrap_draws'],seed=cfg['seed']),changes=changes(predictions[name],predictions[reference],docs),
            primary=reference==control,unit='document')
    coverage = None
    if cfg['kind']=='search':
        selected=mb.read(out/'candidates.json')
        gold_missed = {d:{tuple(sorted(p.split(','))) for p in docs[d]['gold']}-baseline[d] for d in docs}
        coverage=dict(selected_pairs=sum(len(x) for x in selected.values()),
            missed_gold=sum(len(g) for g in gold_missed.values()),
            candidate_missed_gold=sum(len({tuple(p) for p in selected[d]} & gold_missed[d]) for d in docs),
            note='Post-hoc evaluation only; selection did not receive gold.')
    costs={mode:{k:sum(r['cost'][k] for r in rows if r['mode']==mode) for k in rows[0]['cost']}
           for mode in {r['mode'] for r in rows}} if rows else {}
    report=dict(protocol=VERSION,configuration=cfg,manifest_sha256=mb.sha(out/'manifest.json'),
        jobs_done=len(rows),jobs_expected=len(jobs),metrics=systems,contrasts=contrasts,per_document=per_doc,
        candidate_coverage=coverage,cost=costs,invalid=sum(not r['valid'] for r in rows),
        invalid_evidence=sum(r['evidence_invalid'] for r in rows),forced=sum(r['forced'] for r in rows),
        truncated=sum(r['truncated'] for r in rows),runs=sorted({r['run'] for r in rows}),
        scope='Previously exposed MAVEN200; nominal unadjusted intervals. Oracle diagnostic excluded from selection.')
    mb.atomic(out/'predictions.json',{n:{d:sorted(ps) for d,ps in p.items()} for n,p in predictions.items()})
    mb.atomic(out/'report.json',report)
    mb.atomic(out/'complete.json',dict(manifest_sha256=mb.sha(out/'manifest.json'),responses=hashes,
              report_sha256=mb.sha(out/'report.json'),runs=report['runs']),immutable=True)
    print('MR_RESULT',api.canonical({k:v for k,v in report.items() if k!='per_document'}),flush=True)
    print('MR_COMPLETE',api.canonical(dict(path=str(out/'complete.json'),sha256=mb.sha(out/'complete.json'))),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cmd',choices=('prepare','run','score'))
    parser.add_argument('--config',default='steering/maven_recall_config.json')
    parser.add_argument('--url',default='http://127.0.0.1:18100/v1')
    args=parser.parse_args()
    cfg=config(args.config)
    if args.cmd=='prepare': prepare(cfg)
    elif args.cmd=='run': run(cfg,args.url)
    else: score(cfg)


if __name__=='__main__':
    main()
