#!/usr/bin/env python3
"""M0 experimental planner/evaluator. No model calls. Standard library only.
Input JSONL each line: {doc_id, topic?, mentions:[{id,sentence?,span?}],
 pairs:[[id1,id2],...], text?, gold_pairs?:[[id1,id2]],
 baseline_positive?:[[id1,id2]], luna_zero?:[[id1,id2]]}.
Never use gold to construct a plan. Outputs for arm M/MP only are allowed subsets.
"""
import argparse, hashlib, json, random, statistics
from collections import defaultdict
from pathlib import Path

def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def pair(a,b):
    a,b=str(a),str(b)
    if a==b: raise ValueError('self pair')
    return tuple(sorted((a,b)))
def as_pairs(values): return {pair(*x) for x in values}
def plan(doc, k=7, seed=42):
    did=str(doc['doc_id']); ids=[str(m['id']) for m in doc['mentions']]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate mention id')
    lookup={str(m['id']):m.get('sentence') for m in doc['mentions']}
    pairs=as_pairs(doc['pairs'])
    for a,b in pairs:
        if a not in lookup or b not in lookup: raise ValueError('unknown mention in pairs')
    blocks=[[] for _ in range(k)]; counts=[0]*k; deg=[defaultdict(int) for _ in range(k)]
    strata=defaultdict(list)
    for a,b in pairs:
        s1,s2=lookup[a],lookup[b]
        cat='unknown' if s1 is None or s2 is None else ('intra' if s1==s2 else 'inter')
        strata[cat].append((a,b))
    # Assign each stratum separately to evenly-sized partitions, balancing vertex incidence.
    for cat in ['intra','inter','unknown']:
        edges=strata[cat]
        r=random.Random(int(digest(f'{seed}:{did}:{cat}')[:16],16))
        edges.sort(); r.shuffle(edges)
        quota=[len(edges)//k + (1 if i<len(edges)%k else 0) for i in range(k)]
        # rotate quotas by hashed offset to avoid always filling early blocks
        off=int(digest(f'{seed}:{did}:{cat}:quota')[:8],16)%k
        quota=quota[-off:]+quota[:-off] if off else quota
        used=[0]*k
        for a,b in edges:
            choices=[i for i in range(k) if used[i]<quota[i]]
            i=min(choices,key=lambda j:(deg[j][a]+deg[j][b], counts[j],digest(f'{seed}:{did}:{cat}:{a}:{b}:{j}')))
            blocks[i].append([a,b]); counts[i]+=1; used[i]+=1; deg[i][a]+=1; deg[i][b]+=1
    got=[pair(*p) for block in blocks for p in block]
    if len(got)!=len(pairs) or set(got)!=pairs: raise AssertionError('partition invalid')
    return blocks

def permutations(doc,k=7,seed=42):
    ids=sorted(str(m['id']) for m in doc['mentions']); out=[]
    for i in range(k):
        shuffled=list(ids); random.Random(int(digest(f'{seed}:{doc["doc_id"]}:pi:{i}')[:16],16)).shuffle(shuffled)
        out.append(dict(zip(ids,shuffled)))
    return out

def build(doc,k=7,seed=42):
    blocks=plan(doc,k,seed); perms=permutations(doc,k,seed)
    full=[list(p) for p in sorted(as_pairs(doc['pairs']))]
    rows=[]
    for arm in ('O','P','M','MP'):
        for i in range(k):
            allow=blocks[i] if arm in ('M','MP') else full
            mapping=perms[i] if arm in ('P','MP') else {str(m['id']):str(m['id']) for m in doc['mentions']}
            rows.append({'doc_id':doc['doc_id'],'topic':doc.get('topic'),'arm':arm,'pass':i+1,
                'allowed_original':allow,'id_mapping':mapping,
                'allowed_visible':[[mapping[a],mapping[b]] for a,b in allow],
                'demo_id':f'DEMO_SLOT_{i+1}', 'status':'NOT_SENT'})
    return rows

def validate(rows,doc,k):
    universe=as_pairs(doc['pairs'])
    for arm in ('O','P','M','MP'):
        current=[r for r in rows if r['arm']==arm]
        assert len(current)==k
        if arm in ('M','MP'):
            counts=defaultdict(int)
            for r in current:
                for x in r['allowed_original']: counts[pair(*x)]+=1
            assert set(counts)==universe and all(n==1 for n in counts.values())
        else:
            assert all(as_pairs(r['allowed_original'])==universe for r in current)
        for r in current:
            mapping=r['id_mapping']; assert len(set(mapping.values()))==len(mapping)
            assert as_pairs(r['allowed_visible'])=={pair(mapping[a],mapping[b]) for a,b in r['allowed_original']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--passes',type=int,default=7)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    allrows=[]; summary=[]
    for raw in args.input.read_text().splitlines():
        if not raw.strip():continue
        doc=json.loads(raw); rows=build(doc,args.passes,args.seed); validate(rows,doc,args.passes)
        allrows.extend(rows)
        b=[r for r in rows if r['arm']=='M']
        summary.append({'doc_id':doc['doc_id'],'topic':doc.get('topic'),'pairs':len(as_pairs(doc['pairs'])),
                        'block_sizes':[len(r['allowed_original']) for r in b],
                        'max_block':max(len(r['allowed_original']) for r in b)})
    with (args.out/'plan.jsonl').open('w') as f:
        for r in allrows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    (args.out/'summary.json').write_text(json.dumps({'documents':len(summary),'passages':len(allrows),'per_document':summary},ensure_ascii=False,indent=2))
    print(f'OK {len(summary)} docs, {len(allrows)} planned arm-pass entries. NO MODEL CALLS.')
if __name__=='__main__': main()
