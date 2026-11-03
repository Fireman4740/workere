"""Freeze N14's four previously attempted CTB documents and a 2x2 intervention.

No target selection by labels. Existing gold is copied for later evaluation;
labels are used only to format authorized few-shot demonstrations.
"""
import collections
import copy
import hashlib
import json
import pathlib
from pipelines.jev.n12 import views

P = pathlib.Path('experiments/jev/n14')
OLD = pathlib.Path('experiments/jev/n13/ctb50')
CONDITIONS = ('M00', 'M10', 'M01', 'M11')
ARMS = ('original', 'local', 'table')
MAX_PAIRS = 32
FORMAT_FULL = '''Return only a JSON array of {"pair":"ID1,ID2","label":"causes|causedby|norel"}.
Use the original mention IDs. For positive relations include both orientations with inverse labels.
Unlisted pairs mean norel. Do not output explanations or infer transitive links.'''
FORMAT_BOUNDED = '''Return only a JSON array of {"pair":"ID1,ID2","label":"causes|causedby|norel"}.
Classify exactly the requested pairs, each exactly once, using their supplied orientation.
Include norel explicitly. Do not add reverse pairs or other pairs. Do not output explanations.'''
CLEAN = '''Identify textually expressed causal relations between marked event mentions.
causes means the first event causes, enables, or prevents the absence of the second;
causedby means the reverse relation; norel means neither is supported by the text.
The wording expressing the connection can be a connective, a predicate or a construction;
there is no closed list of trigger words. Require evidence connecting these particular events.
Do not add a causal relation merely from temporal order, co-occurrence, reporting or world knowledge.
Only intra-sentence event pairs are eligible. Mentions use their original inline IDs.'''

def read(p):
    return json.loads(p.read_text())

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()

def save(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def marked_tokens(d):
    starts, ends = collections.defaultdict(list), collections.defaultdict(list)
    for mid, span in zip(d['mentions'], d['spans']):
        for i in span:
            if i-1 not in span: starts[i].append(mid)
            if i+1 not in span: ends[i].append(mid)
    result = [''.join('<'+m+' ' for m in starts[i])+token+'>'*len(ends[i]) for i,token in enumerate(d['tokens'])]
    assert ' '.join(result) == d['text']
    return result

def batches(d):
    by_sentence = collections.defaultdict(list)
    for a,b in sorted({tuple(sorted(x)) for x in d['pair_list']}):
        assert d['sentence_of'][a] == d['sentence_of'][b]
        by_sentence[d['sentence_of'][a]].append((a,b))
    # Deterministic nearby-sentence packing, <=32 unordered pairs and <=3 target sentences.
    output, current, sids = [], [], []
    for sid, pairs in sorted(by_sentence.items()):
        for pair in pairs:
            if current and (len(current) == MAX_PAIRS or sid-sids[0] > 2):
                output.append(current); current=[]; sids=[]
            current.append(pair)
            if sid not in sids: sids.append(sid)
    if current: output.append(current)
    flattened = [p for batch in output for p in batch]
    assert len(flattened) == len(set(flattened)) == len(d['pair_list'])//2
    return output

def unit_views(d, pairs):
    tokens=marked_tokens(d)
    target=sorted({d['sentence_of'][a] for a,b in pairs})
    context=sorted({s for sid in target for s in range(max(0,sid-1),min(len(d['sentences']),sid+2))})
    def text(sids, headers=False):
        return '\n\n'.join(('[Sentence '+str(s)+']\n' if headers else '')+' '.join(tokens[slice(*d['sentences'][s])]) for s in sids)
    original=text(context)
    local=text(target, True)
    inventory=[]
    for mid,sp in zip(d['mentions'],d['spans']):
        if d['sentence_of'][mid] in context:
            inventory.append({'id':mid,'sentence_zero_based':d['sentence_of'][mid], 'surface':' '.join(d['tokens'][i] for i in sp)})
    for a,b in pairs:
        assert '<'+a+' ' in original and '<'+b+' ' in original
        assert '<'+a+' ' in local and '<'+b+' ' in local
    return {'original':original,'local':local,'table':original+'\n\nOccurrence inventory (original IDs; no identity inferred):\n'+json.dumps(inventory)},context,target

def build():
    P.mkdir(exist_ok=True)
    docs=read(OLD/'smoke-public.json')
    save('public.json',docs); save('gold.json',read(OLD/'smoke-gold.json'))
    save('selection.json',read(OLD/'selection.json'))
    bank=read(OLD/'demo-bank.json')['documents']
    demo={d['id']:d for d in bank}
    # Resolve example metadata only; targets and their requests are inherited exactly.
    project=pathlib.Path.home()/'.local/share/openresearch/worktrees/75e891b6-a1c1-41a2-a06f-99092098f31e'
    if (P/'demo-public.json').exists():
        demo_public={d['id']:d for d in read(P/'demo-public.json')}
    else:
        suite=read(project/'chat_dde16780-d6aa-4731-afeb-737127fd99e3/experiments/jev/haiku/data/ctb-suite.json')
        demo_public={d['id']:d for d in suite['documents'] if d['id'] in demo}
    save('demo-bank.json',{'documents':bank})
    save('demo-public.json',[{k:v for k,v in d.items() if k not in ('gold','annots')} for d in demo_public.values()])
    old=[json.loads(x) for x in (OLD/'requests.jsonl').read_text().splitlines() if json.loads(x)['stage']=='smoke']
    docmap={d['id']:d for d in docs};rows=[]
    for condition in CONDITIONS:
        clean, bounded = condition[1]=='1', condition[2]=='1'
        for r in old:
            d=docmap[r['doc']];prefix=copy.deepcopy(r['body']['messages'][:3])
            old_rules=prefix[0]['content'].split('# Output Format')[0]
            groups=batches(d) if bounded else [sorted({tuple(sorted(x)) for x in d['pair_list']})]
            if bounded:
                example=demo_public[r['demo']];example_groups=batches(example)
                labs={tuple(item['pair'].split(',')):item['label'] for item in demo[r['demo']]['answer']}
                # Existing example labels may select an illustrative positive example batch;
                # this never selects target batches, documents or prompts by target gold.
                eg=next((g for g in example_groups if any(p in labs or p[::-1] in labs for p in g)),example_groups[0])
                etext=unit_views(example,eg)[0]['original']
                answer=[]
                for a,b in eg:
                    lab=labs.get((a,b))
                    if lab is None:
                        reverse=labs.get((b,a),'norel');lab={'causes':'causedby','causedby':'causes','norel':'norel'}[reverse]
                    answer.append({'pair':a+','+b,'label':lab})
                prefix[0]['content']=(CLEAN if clean else old_rules)+'\n'+FORMAT_BOUNDED
                if not clean:prefix[0]['content']+='\nIMPORTANT: Reason step by step, then output the final JSON.'
                prefix[1]['content']='Text:\n'+etext+'\nRequested pairs:\n'+json.dumps([a+','+b for a,b in eg])
                prefix[2]['content']=json.dumps(answer)
            elif clean:
                prefix[0]['content']=CLEAN+'\n'+FORMAT_FULL
                prefix[1]['content']='Text:\n'+demo[r['demo']]['text']+'\nClassify all eligible intra-sentence pairs.'
            for index, group in enumerate(groups):
                if bounded:
                    text=unit_views(d,group)[0][r['arm']]
                    user='Text:\n'+text+'\nRequested pairs:\n'+json.dumps([a+','+b for a,b in group])
                    if not clean:user+='\nIMPORTANT: Reason step by step, then output the final JSON.'
                elif clean:
                    # Keep the old representation text byte-for-byte.
                    old_user=r['body']['messages'][-1]['content']
                    text=old_user.split('Text:\n',1)[1].split('\n\n Classify all the intra-sentence mention pairs',1)[0]
                    user='Text:\n'+text+'\nClassify all eligible intra-sentence pairs.'
                else:user=r['body']['messages'][-1]['content']
                body=copy.deepcopy(r['body']);body['reasoning_effort']='medium'
                body['chat_template_kwargs']['reasoning_effort']='medium'
                body['messages']=prefix+[{'role':'user','content':user}]
                logical='n14:'+condition+':'+r['logical_id']+':b'+str(index)
                allowed=group if bounded else d['pair_list']
                rows.append({'logical_id':logical,'condition':condition,'bounded':bounded,'batch':index,
                             'schedule':r['schedule'],'slot':r['slot'],'arm':r['arm'],'doc':r['doc'],'demo':r['demo'],
                             'allowed':[list(p) for p in allowed],'body':body,'body_sha256':hashlib.sha256(canonical(body)).hexdigest()})
    raw=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode()
    (P/'requests.jsonl').write_bytes(raw)
    counts=dict(collections.Counter(r['condition'] for r in rows))
    spec=read(OLD/'protocol.json')
    spec.update(study='N14-CTB4-2x2',user_authorization='2026-11-03 medium plus other recommendations in parallel on same documents',
                documents=4,conditions={c:{'clean_prompt':c[1]=='1','bounded_execution':c[2]=='1'} for c in CONDITIONS},
                reasoning_effort='medium',request_counts=counts,maximum_generation_calls=len(rows),
                input_sha256=hashlib.sha256(raw).hexdigest(),concurrency=8,condition_concurrency=2,
                state_root='~/.local/share/openresearch/workere-n14-medium',generation_retries=0,
                stages=['same4only'],primary_score='all4complete per condition, unordered2of3majority separately per calendar and representation',
                exposure='exploratory previously exposed development docs; not blind confirmation; prompts frozen before this run',
                factorial_interpretation='prompt bundle(clean instructions+generic causal definition) x execution bundle(explicit pairs+strict coverage+bounded context+adapted examples), not isolated effects of each component',
                coverage='all original candidates once per reading; invalid or missing responses never negative; complete-document diagnostic and common complete subset reported separately',
                cost_comparison='resource vectors and raw scores; no equal-cost superiority claim',
                full50='not automatically launched in this CTB4 factorial',
                template_gate='actual tokenizer medium vs xhigh must differ before any generation',
                source_run='09873773-2908-459d-bced-dc436272fb32',no_service_changes=True)
    for key in ('folds','stages_policy','active_baseline','relaunch_policy','previous_run','uncertainty','demos','comparison_scope','local_view'):
        spec.pop(key,None)
    save('protocol.json',spec)
    print(json.dumps({'requests':counts,'maximum_calls':len(rows),'same_documents':[d['id'] for d in docs],'unordered_pairs':sum(len(d['pair_list'])//2 for d in docs)}))

if __name__=='__main__':build()
