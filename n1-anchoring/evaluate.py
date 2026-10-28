"""Evaluation-only labels opened strictly after all model outputs are frozen."""
import argparse,gzip,hashlib,json,sys,time,importlib.util
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m1analysis',ROOT/'m1-diversity/m1-analysis.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def evaluate(out):
 p=load(out/'protocol.json');o=load(out/'outputs-before-gold.json');freeze=load(out/'freeze.json')
 assert o['gold_used'] is False and sha(out/'outputs-before-gold.json')==freeze['outputs_sha256'] and sha(out/'protocol.json')==freeze['protocol_sha256']
 docs=load(out/'documents.json');ref=load(out/'reference.json');original=load(out/'channels.json')
 dec=lambda rr:{(r['doc_id'],*m.canonical(*r['pair'])) for r in rr}
 universe={(did,*m.canonical(*r['pair'])) for did,rr in ref.items() for r in rr};base={(did,*m.canonical(*r['pair'])) for did,rr in ref.items() for r in rr if r['baseline_positive']}
 accepted=dec(o['accepted']);channels={c:dec(rr) for c,rr in original.items()};assert accepted<=set.union(*channels.values()) and not accepted&base
 m.save(out/'gold-access-receipt.json',dict(time_unix=time.time(),frozen_outputs_sha256=freeze['outputs_sha256']))
 source=ROOT/'m1-diversity/evaluation-only/controls.json.gz';raw=gzip.decompress(source.read_bytes())
 hist=load(ROOT/'m1-diversity/frozen/m1-protocol-frozen.json');assert hashlib.sha256(raw).hexdigest()==hist['provenance']['controls_sha256']
 gold=set();seen=Counter()
 for r in json.loads(raw):
  k=(r['doc'],*m.canonical(r['a'],r['b']))
  if k in universe:
   seen[k]+=1
   if r['gold']:gold.add(k)
 assert set(seen)==universe and set(seen.values())=={2}
 filtered={c:v&accepted for c,v in channels.items()};topics={d:docs[d]['topic'] for d in docs}
 result=m.evaluate(gold,base,filtered,universe,topics);result['unfiltered']=m.evaluate(gold,base,channels,universe,topics)
 union=set.union(*filtered.values());exclusive=filtered['MP']-(filtered['O_shared']|filtered['O_extra']);allordinary=base|filtered['O_shared']|filtered['O_extra'];n=len(exclusive);tp=len(exclusive&gold)
 result['verified_union']=m.metrics(gold,base|union);result['exclusive_MP_after_all_ordinary']=dict(tp=tp,fp=n-tp,precision=tp/n if n else None,utility=2*tp-m.metrics(gold,allordinary)['f1']*n,delta_f1=m.metrics(gold,allordinary|exclusive)['f1']-m.metrics(gold,allordinary)['f1'])
 result['acceptance']=dict(proposed=len(set.union(*channels.values())),accepted=len(accepted),states=o['integrity'],validation_reasons=dict(Counter(i['validation'] for e in o['events'] for i in e['items'])))
 pp=completion=reason=cache=0
 for e in o['events']:
  u=e.get('usage') or {};pp+=u.get('prompt_tokens',0);completion+=u.get('completion_tokens',0);reason+=(u.get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0;cache+=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0
 result['resources']=dict(provider_batch=load(out/'costs.json'),input_tokens=pp,completion_tokens_including_reasoning=completion,reasoning_tokens=reason,cached_input_tokens=cache,reserved_ceiling_usd=p['conservative_api_ceiling_usd'])
 h=result['systems']['H'];delta=h['f1']-result['baseline']['f1'];valid=o['integrity']['all_valid'];result['H_minus_P0']=delta
 gains=[v['H']['f1']-v['baseline']['f1'] for v in result['by_topic'].values()]
 # Verify global gains after leaving out each topic, rather than requiring all local topics win.
 loto=[]
 for t in set(topics.values()):
  keep=lambda s:{k for k in s if topics[k[0]]!=t}
  loto.append(m.metrics(keep(gold),keep(base|filtered['O_shared']|filtered['MP']))['f1']-m.metrics(keep(gold),keep(base))['f1'])
 result['H_minus_P0_loto']=loto
 if not valid:decision='INCONCLUSIF_INTEGRITY'
 elif delta>=.01 and h['utility_vs_P0']>0 and min(loto)>0:decision='GO_EXPLORATORY'
 elif delta<=0 or h['utility_vs_P0']<=0:decision='NO_GO'
 else:decision='INCONCLUSIVE_SMALL_POSITIVE'
 result.update(decision=decision,limitations=p['limitations'],run_id=freeze['run_id'],commit=freeze['commit'])
 m.save(out/'results.json',result)
 lines=['# N1 — vérification automatique des preuves et mentions','',f'Décision : **{decision}**. Sans entraînement ni annotation humaine. Données DEV20 déjà exposées : résultat exploratoire.','', '| Système | TP | FP | FN | F1 % |','|---|---:|---:|---:|---:|']
 for name,v in [('P0',result['baseline']),('C filtré',result['systems']['C']),('H filtré',h),('Union filtrée',result['verified_union'])]:lines.append(f"| {name} | {v['tp']} | {v['fp']} | {v['fn']} | {100*v['f1']:.3f} |")
 lines+=['',f"H−P0 : {100*delta:+.3f} points. H−C : {100*result['delta_f1']:+.3f} points.",'','```json',json.dumps(dict(acceptance=result['acceptance'],resources=result['resources'],exclusive_MP_after_all_ordinary=result['exclusive_MP_after_all_ordinary']),ensure_ascii=False,indent=2),'```','','Les sorties invalides/tronquées sont mises en quarantaine. Elles ne valent ni rejet explicite ni preuve négative. P0 reste intégralement conservé. Un certificat exact est une condition structurelle ; la justesse sémantique reste celle du jugement automatique et de son accord avec les annotations existantes. Aucun entraînement, aucun choix de seuil après résultats, aucune promotion automatique.']
 (out/'rapport-n1.md').write_text('\n'.join(lines)+'\n');print('N1_EVALUATED '+json.dumps(dict(decision=decision,H_minus_P0=delta,H=h,C=result['systems']['C'])))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dir',type=Path,required=True);evaluate(ap.parse_args().dir)
