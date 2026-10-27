"""Reproducible, paired-topic M1 evaluation. Gold opened only after frozen outputs."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

def canonical(a,b):
    a,b=str(a),str(b)
    if a==b:raise ValueError('self pair')
    return tuple(sorted((a,b)))

def metric_counts(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)

def metrics(gold,pred):return metric_counts(len(gold&pred),len(pred-gold),len(gold-pred))
def save(p,x):p.write_text(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def bootstrap(bytopic,reps=20000,seed=20261027):
    topics=sorted(bytopic);rng=random.Random(seed);values=[]
    for _ in range(reps):
        sample=rng.choices(topics,k=len(topics))
        ms={s:metric_counts(*(sum(bytopic[t][s][k] for t in sample) for k in ('tp','fp','fn'))) for s in ('C','H')}
        values.append(ms['H']['f1']-ms['C']['f1'])
    values.sort()
    def q(p):
        j=(len(values)-1)*p;i=int(j)
        return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(j-i)
    return dict(unit='topic',replicates=reps,seed=seed,topics=len(topics),ci95=[q(.025),q(.975)],
        caution='percentile exploratory cluster bootstrap; one stochastic realization; not independent pair resampling')

def evaluate(gold,base,channels,universe,doc_topics):
    channels={c:set(v) for c,v in channels.items()}
    shared,extra,mp=(channels[c] for c in ('O_shared','O_extra','MP'))
    adds={'C':shared|extra,'H':shared|mp,'shared':shared}
    basef=metrics(gold,base)['f1']
    def system(g,b,a):
        score=metrics(g,b|a);t=len(a&g);n=len(a)
        assert score['f1']==(2*(len(b&g)+t)/(len(b)+len(g)+n) if len(b)+len(g)+n else 0.)
        return dict(**score,additions=n,true_additions=t,false_additions=n-t,addition_precision=t/n if n else None,
            utility_vs_P0=2*t-metrics(g,b)['f1']*n)
    def subset(ids):
        f=lambda s:{k for k in s if k[0] in ids}
        return dict(baseline=metrics(f(gold),f(base)),**{s:system(f(gold),f(base),f(a)) for s,a in adds.items()})
    bydoc={d:subset({d}) for d in doc_topics}
    bytopic={t:subset({d for d in doc_topics if doc_topics[d]==t}) for t in sorted(set(doc_topics.values()))}
    categories={}
    for bits in ('100','010','001','110','101','011','111'):
        items={k for k in shared|extra|mp if ''.join(str(int(k in c)) for c in (shared,extra,mp))==bits}
        categories[bits]=dict(order=['O_shared','O_extra','MP'],pairs=len(items),tp=len(items&gold),fp=len(items-gold))
    conditional=mp-shared;tp=len(conditional&gold);n=len(conditional)
    # Utility relative to reference PLUS common reads, not historical reference alone.
    sharedf=metrics(gold,base|shared)['f1']
    conditional_score=dict(additions=n,tp=tp,fp=n-tp,precision=tp/n if n else None,
        utility=2*tp-sharedf*n,f1_delta=metrics(gold,base|shared|mp)['f1']-sharedf,
        ordinary_extra_missed_tp=len((mp-shared-extra)&gold),ordinary_extra_missed_fp=len(mp-shared-extra-gold))
    loto={}
    for t in bytopic:
        s=subset({d for d in doc_topics if doc_topics[d]!=t})
        loto[t]=dict(C=s['C'],H=s['H'],delta_f1=s['H']['f1']-s['C']['f1'])
    total_delta=metrics(gold,base|adds['H'])['f1']-metrics(gold,base|adds['C'])['f1']
    best=max(bytopic,key=lambda t:(total_delta-loto[t]['delta_f1'],t))
    return dict(baseline=metrics(gold,base),systems={s:system(gold,base,a) for s,a in adds.items()},
        delta_f1=metrics(gold,base|adds['H'])['f1']-metrics(gold,base|adds['C'])['f1'],
        candidates=len(universe),gold_positive=len(gold),marginal_precision_threshold_P0=basef/2,
        categories=categories,conditional_MP=conditional_score,by_document=bydoc,by_topic=bytopic,
        leave_one_topic_out=loto,best_topic=best,without_best_topic=loto[best],
        informative_topics=sum(bytopic[t]['C']!=bytopic[t]['H'] for t in bytopic),
        topics_H_over_C=sum(bytopic[t]['H']['f1']>bytopic[t]['C']['f1'] for t in bytopic),
        documents_H_over_C=sum(bydoc[d]['H']['f1']>bydoc[d]['C']['f1'] for d in bydoc),
        bootstrap=bootstrap(bytopic))

def main(dest,repo):
    protocol=json.loads((dest/'m1-protocol-frozen.json').read_text())
    outputs=json.loads((dest/'m1-outputs-before-gold.json').read_text())
    freeze=json.loads((dest/'m1-output-freeze-receipt.json').read_text())
    assert sha(dest/'m1-outputs-before-gold.json')==freeze['outputs_sha256']
    assert sha(dest/'m1-protocol-frozen.json')==freeze['protocol_sha256']
    assert outputs['gold_used'] is False
    ref=json.loads((dest/'m1-reference-without-gold.json').read_text())
    ids=protocol['document_ids'];assert set(ref)==set(ids)
    universe={(did,*canonical(*r['pair'])) for did in ids for r in ref[did]}
    base={(did,*canonical(*r['pair'])) for did in ids for r in ref[did] if r['baseline_positive']}
    zero={(did,*canonical(*r['pair'])) for did in ids for r in ref[did] if r['luna_zero'] and not r['baseline_positive']}
    channels={c:{(r['doc_id'],*canonical(*r['pair'])) for r in outputs['admitted_channels'][c]} for c in ('O_shared','O_extra','MP')}
    assert all(v<=zero for v in channels.values())
    for s,cc in [('C',('O_shared','O_extra')),('H',('O_shared','MP'))]:
        assert {(r['doc_id'],*canonical(*r['pair'])) for r in outputs['final_predictions'][s]}==base|channels[cc[0]]|channels[cc[1]]
    # First deliberate evaluation-gold access in this script, strictly below freeze validation.
    save(dest/'m1-gold-access-receipt.json',dict(time_unix=time.time(),outputs_sha256=freeze['outputs_sha256'],
        note=protocol['traceability_limitation']))
    bundled=repo/'m1-diversity/evaluation-only/controls.json.gz'
    raw=(gzip.decompress(bundled.read_bytes()) if bundled.exists() else
        subprocess.check_output(['git','-C',str(repo),'show',protocol['provenance']['controls_commit']+':experiments/jev/round4/controls.json']))
    assert hashlib.sha256(raw).hexdigest()==protocol['provenance']['controls_sha256']
    seen=Counter();gold=set()
    for r in json.loads(raw):
        k=(r['doc'],*canonical(r['a'],r['b']))
        if k in universe:
            seen[k]+=1
            if r['gold']:gold.add(k)
    assert set(seen)==universe and set(seen.values())=={2}
    dt={m['doc_id']:m['topic'] for m in protocol['metadata']}
    results=evaluate(gold,base,channels,universe,dt)
    costs=json.loads((dest/'m1-costs.json').read_text());results['costs']=costs
    valid=outputs['integrity']['all_planned_responses_valid']
    matched=costs.get('cost_match_total_within5pct') is True
    if not valid or not matched:decision='INCONCLUSIF'
    elif results['delta_f1']<=0 or results['conditional_MP']['utility']<=0 or results['without_best_topic']['delta_f1']<=0:decision='NO-GO'
    elif results['delta_f1']>=.01:decision='GO exploratoire'
    else:decision='INCONCLUSIF'
    results.update(decision=decision,outputs_sha256=freeze['outputs_sha256'],protocol_sha256=freeze['protocol_sha256'],
        status='exploratory consumed development; no confirmed gain; no automatic M2')
    save(dest/'m1-results.json',results)
    events=outputs['events'];lookup={}
    for row in outputs['ledgers']:
        lookup[(row['doc_id'],*row['pair'])]=row
    with (dest/'m1-pair-audit.csv').open('w',newline='') as f:
        fields=['doc_id','topic','a','b','baseline_positive','eligible_Z0','O_shared','O_extra','MP','C','H','E_MP_planned','E_MP_delivered','E_MP_valid','V_MP','R_MP','gold_evaluation_only']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for k in sorted(universe):
            l=lookup[k]
            w.writerow(dict(doc_id=k[0],topic=dt[k[0]],a=k[1],b=k[2],baseline_positive=int(k in base),eligible_Z0=int(k in zero),
                **{c:int(k in channels[c]) for c in channels},C=int(k in base|channels['O_shared']|channels['O_extra']),H=int(k in base|channels['O_shared']|channels['MP']),
                E_MP_planned=l['MP']['E_planned'],E_MP_delivered=l['MP']['E_delivered'],E_MP_valid=l['MP']['E_valid'],V_MP=l['MP']['V'],R_MP=l['MP']['R'],gold_evaluation_only=int(k in gold)))
    report(dest,protocol,results,outputs)
    print(json.dumps(dict(decision=decision,delta_f1=results['delta_f1'],C=results['systems']['C'],H=results['systems']['H']),sort_keys=True))

def report(dest,p,r,o):
    s=['# M1 — diversité complémentaire à budget préenregistré','',
       '**Décision : '+r['decision']+'.** Exploration sur DEV44 consommé, indépendante des deux documents M0. Aucun gain confirmé.','',
       f"Exécution : {len(p['document_ids'])} documents, {len(r['by_topic'])} topics, {r['candidates']} paires originales ; {o['integrity']['logical_requests']} requêtes uniques. Sept lectures ordinaires communes, huit ordinaires supplémentaires, sept MP. P0 et son seuil 0,39 restent inchangés. Les seuls ajouts admis appartiennent à Z0.", '',
       'M0 avait motivé cette hypothèse par une union post hoc. M1 mesure sa complémentarité face à huit nouvelles lectures ordinaires, avec partage effectif des sept premières réponses. Aucun rectangle, transport de preuve, apprentissage ou retrait de P0.','',
       '## Tableau final','',
       '| Système | TP | FP | FN | Précision % | Rappel % | F1 % | Ajouts TP/FP | Précision marginale % |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{100*v:.2f}'
    for name,m in [('P0',r['baseline']),('C',r['systems']['C']),('H',r['systems']['H'])]:
        s.append(f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {str(m.get('true_additions','—'))+'/'+str(m.get('false_additions','—'))} | {fmt(m.get('addition_precision'))} |")
    s+=['',f"ΔF1 H−C : **{100*r['delta_f1']:+.3f} points**. Seuil marginal de P0 sur cet échantillon : {fmt(r['marginal_precision_threshold_P0'])} %. Ce seuil n'est pas celui de TEST-S40.", '',
       '## Complémentarité','', '| O_shared | O_extra | MP | TP | FP |','|---:|---:|---:|---:|---:|']
    for bits,m in r['categories'].items():s.append('| '+' | '.join(bits)+f" | {m['tp']} | {m['fp']} |")
    c=r['conditional_MP'];b=r['bootstrap']
    s+=['',f"Après O_shared, MP apporte {c['tp']} TP et {c['fp']} FP ; utilité marginale {c['utility']:.4f}, ΔF1 {100*c['f1_delta']:+.3f} points. Parmi ses ajouts non partagés, {c['ordinary_extra_missed_tp']} TP et {c['ordinary_extra_missed_fp']} FP échappent aussi à O_extra.", '',
       '## Répartition par topic','', '| Topic | C TP/FP/FN | H TP/FP/FN | F1 C % | F1 H % | Δ points | LOTO Δ points |','|---|---|---|---:|---:|---:|---:|']
    for t,m in r['by_topic'].items():
        C,H=m['C'],m['H'];counts=lambda x:f"{x['tp']}/{x['fp']}/{x['fn']}"
        s.append(f"| {t} | {counts(C)} | {counts(H)} | {fmt(C['f1'])} | {fmt(H['f1'])} | {100*(H['f1']-C['f1']):+.3f} | {100*r['leave_one_topic_out'][t]['delta_f1']:+.3f} |")
    s+=['','## Documents et variabilité','', '| Document | Topic | Paires | F1 P0 % | F1 C % | F1 H % | Δ H−C points |','|---|---|---:|---:|---:|---:|---:|']
    for d in p['metadata']:
        m=r['by_document'][d['doc_id']]
        s.append(f"| {d['doc_id']} | {d['topic']} | {d['candidates']} | {fmt(m['baseline']['f1'])} | {fmt(m['C']['f1'])} | {fmt(m['H']['f1'])} | {100*(m['H']['f1']-m['C']['f1']):+.3f} |")
    s+=['',f"Bootstrap apparié par topic ({b['replicates']} tirages) : IC95% [{100*b['ci95'][0]:+.3f}, {100*b['ci95'][1]:+.3f}] points. Topics informatifs : {r['informative_topics']}; H dépasse C sur {r['topics_H_over_C']} topics et {r['documents_H_over_C']} documents. Sans le topic {r['best_topic']} ayant le meilleur écart local : {100*r['without_best_topic']['delta_f1']:+.3f} points. Les paires ne sont jamais rééchantillonnées indépendamment. Une seule réalisation stochastique ; l'IC ne mesure pas toute la variabilité des générations.", '',
       '## Ressources et validité','', '```json',json.dumps(r['costs'],ensure_ascii=False,indent=2),'```','',
       'Le coût token pondéré observable n’est pas une mesure de FLOPs. Les sept réponses partagées entrent dans les deux coûts contrefactuels de systèmes, mais ne sont payées qu’une fois dans la dépense réelle. Les tokens de raisonnement inclus dans la sortie ne sont pas additionnés une seconde fois. Une ventilation fournisseur absente ne devient pas un coût facturé mesuré par bras.','',
       'Les états livraison absente, format invalide, troncature, vide valide, rejet explicite et absence d’exposition restent distincts dans les événements et compteurs E/V/R. Le rapport ne traite pas une omission comme un rejet explicite.','',
       '## Limites et traçabilité','',p['traceability_limitation'],'',p['output_limit_note'],'',
       'Le checkpoint est celui déclaré par le fournisseur ; un alias et une empreinte système absente ne prouvent pas l’immuabilité des poids. Les labels ESC mesurent une concordance d’annotation, pas une adjudication causale indépendante. TEST-S40 ne choisit aucune variante ; les 88 documents exposés dans H1 ne sont pas une réserve aveugle.','',
       'Aucun ajustement de seuil, d’allocation ou de documents après évaluation. Aucune relance pour sauver une sortie vide ou un résultat négatif. Aucun M2 automatique.','',
       '## Confirmation minimale si GO','',
       'Seulement en cas de GO exploratoire : geler ce protocole et les tarifs, répéter C/H avec plusieurs réalisations préenregistrées sur des topics indépendants de toute mise au point ; dimensionner et autoriser séparément le budget. Une réserve déjà exposée ne suffit pas à une confirmation aveugle.','',
       '## Reproduction','',
       '`uv run --isolated --no-project python m1-diversity/m1-analysis.py --dir <dossier-artifacts> --repo <worktree>`','',
       'Sources internes : protocole et requêtes gelés, réponses brutes avant gold, reçu de gel, registre des ressources, audit de paires et manifeste SHA-256 dans ce dossier.']
    (dest/'m1-final-report.md').write_text('\n'.join(s)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir',type=Path,required=True);ap.add_argument('--repo',type=Path,required=True)
    a=ap.parse_args();main(a.dir,a.repo)
