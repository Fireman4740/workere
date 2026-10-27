"""Prepare M1 label-free protocol and exact prompts; never make model calls."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eci_m0_pilot.m0 import build, as_pairs, digest, pair, validate
from eci_m0_pilot.protocol import json_text, render, remap_tags, lexical_text

M0 = '4811a1b9663e55e957d4587f6a59b90d2c678f76'
SOURCE = 'orx/kb-event-centric-cause-proposer-dev10'
EXCLUDE = {'23_1ecbplus.xml', '32_5ecbplus.xml'}
SETTINGS = dict(seed=20261027, model='openai/gpt-6-luna', reasoning_effort='high',
    allowed_served_models=['openai/gpt-6-luna','openai/gpt-6-luna:batch','openai/gpt-6-luna-20260922'],
    max_output_tokens=10000, max_context_tokens=1050000, input_price_per_million=.1,
    output_price_per_million=.5, budget_usd=3., max_logical_requests=528,
    poll_deadline_seconds=90000)

def readgit(commit, path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{path}'])

def sha(data): return hashlib.sha256(data.read_bytes() if isinstance(data,Path) else data).hexdigest()
def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json_text(value)+'\n')

def inputs():
    docs = {d['doc_id']:d for d in map(json.loads,readgit(M0,'eci_m0_pilot/data/public_train233.jsonl').splitlines())}
    selection = json.loads(readgit(SOURCE,'experiments/jev/round4/selection.json'))
    # Historical file contains labels. Whitelist only scores and identifiers here;
    # no label value enters selection, requests, admission, or this return value.
    raw = readgit(SOURCE,'experiments/jev/round4/controls.json')
    records = [{k:r[k] for k in ('doc','a','b','d1','luna')} for r in json.loads(raw)]
    groups=defaultdict(list)
    for r in records: groups[(r['doc'],pair(r['a'],r['b']))].append(r)
    reference={}; complete=[]
    for did in selection['document_ids']:
        if did not in docs: continue
        rows=[]; ok=True
        for p in sorted(as_pairs(docs[did]['pairs'])):
            rr=groups.get((did,p),[])
            if len(rr)!=2 or {(r['a'],r['b']) for r in rr}!={p,p[::-1]}:
                ok=False;break
            score=sum(float(r['d1'])+float(r['luna']) for r in rr)/4
            rows.append(dict(pair=p, baseline_positive=score>=.39,
                luna_zero=all(float(r['luna'])==0 for r in rr), baseline_score=score,
                orientations=sorted(rr,key=lambda r:(r['a'],r['b']))))
        if ok and len(rows)==len(as_pairs(docs[did]['pairs'])):
            reference[did]=rows;complete.append(did)
    return docs,selection,reference,complete,sha(raw)

def select(docs, complete, n=20):
    # Quantile strata on candidate count, then topic-balanced deterministic
    # round robin. Labels, historical error counts and baseline decisions unused.
    pool=sorted((docs[d] for d in complete if d not in EXCLUDE),key=lambda d:(len(d['pairs']),d['doc_id']))
    if len(pool)<12 or len({d['topic'] for d in pool})<5: raise ValueError('fewer than12 complete docs or5 topics')
    strata=[pool[i*len(pool)//4:(i+1)*len(pool)//4] for i in range(4)]
    remaining=[list(s) for s in strata]; used=Counter(); chosen=[]
    for turn in range(min(n,len(pool))):
        s=turn%4
        if not remaining[s]:s=next(i for i in range(4) if remaining[i])
        d=min(remaining[s],key=lambda d:(used[d['topic']],digest(f'20261027:select:{d["doc_id"]}')))
        remaining[s].remove(d);chosen.append(d['doc_id']);used[d['topic']]+=1
    substitutes=[]
    for s in range(4):
        if remaining[s]:substitutes.append(min(remaining[s],key=lambda d:digest(f'20261027:sub:{d["doc_id"]}'))['doc_id'])
    return chosen,substitutes,{d['doc_id']:i for i,s in enumerate(strata) for d in s}

def requests(doc,demos,template,cfg):
    original=build(doc,7,cfg['seed']);validate(original,doc,7)
    bank=[d for d in demos if d['topic']!=doc['topic']]
    # Historical calendar rule: seeded sample of7 outside-topic demonstrations.
    schedule=random.Random(f'42::{doc["doc_id"]}').sample(bank,7)
    out=[]
    for channel,count in [('O_shared',7),('O_extra',8),('MP',7)]:
        for i in range(count):
            arm='MP' if channel=='MP' else 'O'
            row=dict(next(r for r in original if r['arm']==arm and r['pass']==i%7+1))
            demo=schedule[i%7] # explicit cyclic schedule; same slot across channels
            messages,dm=render(doc,row,demo,template,cfg['seed'])
            body=dict(messages=messages,reasoning={'effort':cfg['reasoning_effort']},max_tokens=cfg['max_output_tokens'])
            bound=len(json_text(body).encode())+1024
            row.update(channel=channel,channel_pass=i+1,demo_id=demo['doc_id'],demo_mapping=dm,
                messages=messages,body=body,prompt_sha256=digest(json_text(messages)),
                prompt_utf8_upper_bound=bound,custom_id=f'M1-{channel}-{i+1}-{digest(doc["doc_id"])[:12]}')
            assert bound+cfg['max_output_tokens']<=cfg['max_context_tokens']
            out.append(row)
    return out,[d['doc_id'] for d in schedule]

def prepare(dest):
    docs,selection,reference,complete,control_sha=inputs()
    chosen,substitutes,strata=select(docs,complete)
    demos=json.loads(readgit(M0,'eci_m0_pilot/data/demonstrations.json'))
    template=json.loads(readgit(M0,'eci_m0_pilot/data/prompt.json'))
    rows=[];schedules={};metadata=[]
    for did in chosen:
        rr,schedule=requests(docs[did],demos,template,SETTINGS);rows+=rr;schedules[did]=schedule
        metadata.append(dict(doc_id=did,topic=docs[did]['topic'],mentions=len(docs[did]['mentions']),
            candidates=len(docs[did]['pairs']),text_characters=len(docs[did]['text']),size_stratum=strata[did]))
    assert len(rows)==22*len(chosen) and len({r['custom_id'] for r in rows})==len(rows)
    # No labels in these snapshots. Reference is deliberately separate from gold.
    save(dest/'m1-public-documents.json',{d:docs[d] for d in chosen})
    save(dest/'m1-reference-without-gold.json',{d:reference[d] for d in chosen})
    encoded=''.join(json_text(r)+'\n' for r in rows)
    (dest/'m1-requests.jsonl').write_text(encoded)
    monetary_bound=sum((r['prompt_utf8_upper_bound']*.1+SETTINGS['max_output_tokens']*.5)/1e6 for r in rows)
    protocol=dict(schema='workere-m1/v1',hypothesis='H7O+7MP exceeds C15O by useful complementary gold links at observable comparable weighted token cost',
        null='No useful complementarity versus8 ordinary additional reads',settings=SETTINGS,
        sample='exploratory consumed DEV44 excluding2 M0 docs; not blind confirmation',
        selection_uses_labels=False,selection_rule='candidate-count quartiles, round-robin strata, fewest selected docs per topic then SHA256 tie-break',
        eligible_complete_documents=complete,document_ids=chosen,technical_substitutes=substitutes,
        substitute_policy='pre-generation only, technical unavailability; never after results; none automatically used',
        metadata=metadata,demonstration_schedule=schedules,demonstration_rule='historical Random(42::doc_id) sample7 outside-topic; O_extra cycles slots1..7,1; MP uses same7 slots',
        admission='P_C=P0 union ((shared union extra) intersect Z0); P_H=P0 union ((shared union MP) intersect Z0); Z0=historical zero vote and P0 negative; no removals',
        historical_threshold=.39,bootstrap=dict(unit='topic',replicates=20000,seed=20261027,interval='percentile paired95%'),
        cost_match='abs(weighted_H/weighted_C-1)<=.05; report total and incremental; no adaptive allocation',
        weighted_cost='prompt_tokens*.1/1e6 + completion_tokens*.5/1e6; reasoning already included in completion; cache separately',
        technical_retry_policy='no paid retries, no sync fallback; GET retrieval may retry; persist before POST; unknown POST is never repeated',
        output_limit_note='10000 inclusive completion/reasoning tokens in every channel, frozen for3USD ceiling; differs from M0 32768; truncations quarantined, no rescue',
        total_unique_requests=len(rows),conservative_api_ceiling_usd=monetary_bound,
        go='deltaF1>=.01, cost within5%, positive conditional MP utility, gain survives removal of best topic, no integrity confound',
        no_go='C>=H, nonpositive MP conditional utility, gain entirely from one topic, or no useful MP complementarity',
        inconclusive='budget/provider/integrity failure, incomplete or noncomparable resources, weak or unstable positive signal',
        no_training=True,no_automatic_M2=True,
        provenance=dict(m0_commit=M0,controls_commit=subprocess.check_output(['git','rev-parse',SOURCE]).decode().strip(),
            controls_sha256=control_sha,requests_sha256=sha(encoded.encode()),
            files_sha256={name:sha((dest/name).read_bytes()) for name in ['m1-public-documents.json','m1-reference-without-gold.json','m1-requests.jsonl']}),
        traceability_limitation='Before preparation, an inspection command accidentally displayed a portion of historical controls including gold. No gold aggregate, error or label used for deterministic selection. This is not a blind study.')
    assert monetary_bound<=3 and len(chosen)>=12 and len({m['topic'] for m in metadata})>=5
    save(dest/'m1-protocol-frozen.json',protocol)
    print(json_text(dict(documents=len(chosen),topics=len({m['topic'] for m in metadata}),pairs=sum(m['candidates'] for m in metadata),
        unique_requests=len(rows),conservative_ceiling_usd=monetary_bound,protocol_sha256=sha((dest/'m1-protocol-frozen.json').read_bytes()))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'m1-protocol-frozen.json').exists():raise SystemExit('refuse to overwrite frozen protocol')
    prepare(a.out)
