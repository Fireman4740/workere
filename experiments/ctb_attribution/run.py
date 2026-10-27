"""One bounded offline automatic CTB diagnostic. NEVER calls an LLM.
Pure standard library; all 20/40 inputs and gold are frozen snapshots.
Gold is opened only after all provisional supports and predictions are saved.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import random
import time
from core import extract,decode,matched,PATTERNS,REPORTING,COORD,CLAUSAL
from test_core import run_tests

HERE=Path(__file__).parent
OUT=Path('logs/ctb_attribution')

def jsonable(x):
    if isinstance(x,set):return sorted(jsonable(v) for v in x)
    if isinstance(x,tuple):return [jsonable(v) for v in x]
    if isinstance(x,list):return [jsonable(v) for v in x]
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    return x

def metric(p,g):
    tp=len(p&g);fp=len(p-g);fn=len(g-p)
    return {'tp':tp,'fp':fp,'fn':fn,'predicted_positive':len(p),
            'precision':tp/len(p) if p else 0,'recall':tp/len(g) if g else 0,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}

def changes(old,new,g):
    adds=new-old;removes=old-new
    return {'additions':len(adds),'correct_additions':len(adds&g),'false_additions':len(adds-g),
            'removals':len(removes),'TP_removed':len(removes&g),'FP_removed':len(removes-g)}

def paired_ci(rows,left,right,g):
    counts=[]
    for r in rows:
        d=r['document_id'];mask={(d,*p) for p in r['candidates']}
        local_g=g&mask
        counts.append((metric({(d,*p) for p in r['outputs'][left]},local_g),
                       metric({(d,*p) for p in r['outputs'][right]},local_g)))
    def f1(vals):
        tp=sum(x['tp'] for x in vals);fp=sum(x['fp'] for x in vals);fn=sum(x['fn'] for x in vals)
        return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    rng=random.Random(741);draw=[]
    for _ in range(2000):
        idx=[rng.randrange(len(counts)) for _ in counts]
        draw.append(100*(f1([counts[i][0] for i in idx])-f1([counts[i][1] for i in idx])))
    draw.sort()
    return [draw[49],draw[1949]]

def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((HERE/'rules-freeze.json').read_text())
    assert hashlib.sha256((HERE/'core.py').read_bytes()).hexdigest()==freeze['core_sha256']
    print('CONFIG '+json.dumps({'freeze':freeze,'technical_tests':run_tests(),
        'new_model_calls':0,'API_modules_imported':False,'policies':['conservative_primary','support_only_sensitivity']}),flush=True)
    ledgers=[]
    for split in ['development','locked_test']:
        payload=json.loads((HERE/'data'/f'{split}.json').read_text())
        assert hashlib.sha256((HERE/'data'/f'{split}.json').read_bytes()).hexdigest()==freeze['inputs_sha256'][split]
        for u in payload['units']:
            t=time.perf_counter_ns();ev=extract(u);f=decode(ev,False);a=decode(ev,True)
            noedge=decode(ev,False);novote=decode(ev,True,False)
            assert f==noedge
            base={tuple(p) for p in u['baseline_positive_pairs']}
            control=matched(f,a,base,u['unit_id'])
            outputs={'R':base,'F':base|f['supported'],'A':base|a['supported'],
                'F_matched':base|control,'A_no_edges':base|noedge['supported'],
                'A_no_vote_atoms':base|novote['supported'],
                'F_support_only':f['supported'],'A_support_only':a['supported']}
            assert len(control)==len(a['supported'])
            assert len(outputs['F_matched'])==len(outputs['A'])
            for k in ['F','A','F_matched','A_no_edges','A_no_vote_atoms']:assert base<=outputs[k]
            for witness in ev:
                x,y=witness['cue_span'];assert u['text'][x:y]==witness['cue']
            ledgers.append({'unit_id':u['unit_id'],'document_id':u['document_id'],'split':split,
                'candidates':u['candidate_pairs'],'baseline':base,'evidence':ev,
                'F_supported':f['supported'],'A_supported':a['supported'],
                'F_bindings':f['bindings'],'A_bindings':a['bindings'],
                'F_matched_supported':control,'outputs':outputs,
                'automatic_competition_slot':any(e['functional_cause_slot'] or e['functional_effect_slot'] for e in ev),
                'support_decision_changed':f['supported']!=a['supported'],
                'bindings_changed':f['bindings']!=a['bindings'],
                'latency_ns':time.perf_counter_ns()-t})
    # Persist inference before opening labels. No attribution truth is synthesized.
    raw=json.dumps(jsonable(ledgers),ensure_ascii=False,indent=2)+'\n'
    (OUT/'supports-and-predictions-before-gold.json').write_text(raw)
    inference_hash=hashlib.sha256(raw.encode()).hexdigest()
    evaluation=json.loads((HERE/'data/evaluation.json').read_text())
    gold={tuple(x) for x in evaluation['gold']};global_base={tuple(x) for x in evaluation['baseline']}
    assert (len(gold),len(global_base))==(298,271)
    results={'scope':'AUTOMATIC_PROVISIONAL_BINDINGS_NOT_HUMAN_CERTIFIED_K',
        'objective':'unique role-binding atoms + cached pass-vote atoms, no learned parameters',
        'independent_factual_verifier':'limited lexical/argument-bracketing F_auto; participant/incident/date/place/granularity mostly UNKNOWN',
        'inference_sha256_before_gold':inference_hash,'new_model_calls':0,'new_model_tokens':0,
        'baseline_global':metric(global_base,gold),'splits':{}}
    for split in ['development','locked_test']:
        rows=[r for r in ledgers if r['split']==split]
        mask={(r['document_id'],*p) for r in rows for p in r['candidates']}
        variants={k:{(r['document_id'],*p) for r in rows for p in r['outputs'][k]} for k in rows[0]['outputs']}
        local_gold=gold&mask;base=global_base&mask
        summary={}
        for k,ps in variants.items():
            final=(global_base-mask)|ps
            summary[k]={'sample':metric(ps,local_gold),'global_with_other_predictions_frozen':metric(final,gold),
                        'changes_vs_R':changes(base,ps,local_gold)}
        data={'units':len(rows),'candidate_pairs':len(mask),
            'units_with_provisional_evidence':sum(bool(r['evidence']) for r in rows),
            'evidence_segments':sum(len(r['evidence']) for r in rows),
            'automatic_competition_units':sum(r['automatic_competition_slot'] for r in rows),
            'support_decision_changed_units':sum(r['support_decision_changed'] for r in rows),
            'binding_changed_units':sum(r['bindings_changed'] for r in rows),
            'F_supported_pairs':sum(len(r['F_supported']) for r in rows),
            'A_supported_pairs':sum(len(r['A_supported']) for r in rows),
            'human_certified_K':None,'attributions_corrected_or_degraded':None,
            'variants':summary,
            'A_minus_F_sample_f1_pp':100*(summary['A']['sample']['f1']-summary['F']['sample']['f1']),
            'A_minus_F_ci95_document_bootstrap_pp':paired_ci(rows,'A','F',gold),
            'A_minus_F_matched_sample_f1_pp':100*(summary['A']['sample']['f1']-summary['F_matched']['sample']['f1']),
            'A_minus_F_pair_changes':changes(variants['F'],variants['A'],local_gold),
            'latency_ms':sum(r['latency_ns'] for r in rows)/1e6}
        # Gold-blind matched sparsification controls, fixed seeds, never select best.
        controls=[]
        for seed in range(32):
            ps=set()
            for r in rows:
                b=r['baseline'];d=r['document_id']
                picked=matched({'supported':r['F_supported']},{'supported':r['A_supported']},b,r['unit_id'],f'placebo-{seed}')
                ps|={(d,*p) for p in b|picked}
            controls.append(metric(ps,local_gold)['f1'])
        data['matched_controls_32_seeds']={'min_f1':min(controls),'max_f1':max(controls),'mean_f1':sum(controls)/32}
        results['splits'][split]=data
    results['elapsed_seconds']=time.perf_counter()-started
    (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('CTB_ATTRIBUTION_RESULT '+json.dumps(results,ensure_ascii=False),flush=True)
    print('ARTIFACTS '+str(OUT.resolve()),flush=True)

if __name__=='__main__':main()
