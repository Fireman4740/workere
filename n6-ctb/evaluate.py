"""Separate N6 evaluator; all question decisions frozen before existing gold access."""
import argparse,json,time,hashlib
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def key(r):return(r['doc_id'],*sorted(r['pair']))
def metric(g,p):
 tp=len(g&p);fp=len(p-g);fn=len(g-p)
 return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def main(out):
 freeze=load(out/'freeze.json');responses=load(out/'responses-before-gold.json');protocol=load(out/'protocol.json');public=load(out/'public.json')
 assert sha(out/'responses-before-gold.json')==freeze['responses_sha256'] and sha(out/'protocol.json')==freeze['protocol_sha256'] and sha(out/'public.json')==freeze['public_sha256'] and responses['gold_used'] is False
 rr=responses['results'];expected={r['custom_id'] for r in public['requests']};assert len(rr)==len(expected) and {r['custom_id'] for r in rr}==expected
 valid=all(r['valid'] for r in rr);states=dict(Counter(r['state'] for r in rr));costs=[r.get('cost_usd') for r in rr];known_cost=sum(c for c in costs if c is not None)
 result=dict(decision='INCONCLUSIVE_INTEGRITY' if not valid else 'PENDING_EVALUATION',integrity=dict(all_valid=valid,states=states,valid_responses=sum(r['valid'] for r in rr),planned_requests=len(rr),canonical_pairs=protocol['canonical_pairs']),cost=dict(reported_total_usd=known_cost if all(c is not None for c in costs) else None,reported_known_usd=known_cost,unknown_requests=sum(c is None for c in costs),conservative_upper_bound_usd=sum(c if c is not None else protocol['per_request_reservation_usd'] for c in costs),reservation_usd=protocol['api_ceiling_usd'],cap_usd=protocol['cap_usd'],paid_POSTs=sum(r.get('post_attempts',0) for r in rr),paid_retries=0),threshold=.5,threshold_comparison='>=',run_id=freeze['run_id'],commit=freeze['commit'],scope=protocol['scope'],training=False,human_annotation=False,primary=None)
 if valid:
  save(out/'gold-access-receipt.json',dict(time_unix=time.time(),frozen_responses_sha256=freeze['responses_sha256']))
  goldrows=load(HERE/'frozen/evaluation-only.json');universe={key(r) for r in goldrows};gold={key(r) for r in goldrows if r['gold']};assert len(universe)==protocol['canonical_pairs']
  predictions={arm:{key(r) for r in rr if r['arm']==arm and r['probability']>=.5} for arm in ['generic','lexical']}
  assert all({key(r) for r in rr if r['arm']==arm}==universe for arm in predictions)
  haiku={(d['doc_id'],*sorted([r['source'],r['target']])) for d in public['haiku_reference'] for r in d['row']['relations'] if r['label']!='norel'} & universe
  measures={arm:metric(gold,pred) for arm,pred in predictions.items()};measures['haiku_frozen']=metric(gold,haiku)
  bydoc={d:{arm:metric({k for k in gold if k[0]==d},{k for k in pred if k[0]==d}) for arm,pred in predictions.items()} for d in protocol['documents']}
  delta=measures['lexical']['f1']-measures['generic']['f1'];result.update(primary=dict(systems=measures,delta_f1_lexical_minus_generic=delta,by_document=bydoc,added=metric(gold,predictions['lexical']-predictions['generic']),removed=metric(gold,predictions['generic']-predictions['lexical'])),decision='POSITIVE_EXPLORATORY' if delta>0 else 'NO_GAIN')
 save(out/'results.json',result)
 lines=['# N6 — question lexicale CTB contre question causale générique','',f"Décision : **{result['decision']}**. {protocol['scope']}. Aucun entraînement et aucune annotation humaine.",'',f"Réponses valides :{result['integrity']['valid_responses']}/{len(rr)}. Coût fournisseur connu :{known_cost:.8f}USD. Réservation :{protocol['api_ceiling_usd']:.3f}USD, plafond{protocol['cap_usd']:.2f}USD."]
 if result['primary']:
  lines+=['','| Système | TP | FP | FN | F1 % |','|---|---:|---:|---:|---:|']
  for name,v in result['primary']['systems'].items():lines.append(f"| {name} | {v['tp']} | {v['fp']} | {v['fn']} | {100*v['f1']:.3f} |")
  lines+=['',f"Lexical−générique :{100*result['primary']['delta_f1_lexical_minus_generic']:+.3f} points."]
 else:lines+=['','Le résultat primaire est bloqué. Aucune réponse manquante ou invalide n’est assimilée à une décision négative.']
 lines+=['','Une orientation canonique par paire, contexte identique, seuil0,5 fixé dans les deux bras. Le lecteur Haiku sauvegardé sert de contexte ; aucun seuil de fusion0,39 n’est transporté. Cohorte réduite et déjà exposée, sans confirmation indépendante. Une inspection accidentelle du schéma des labels après gel du protocole a affiché12 valeurs ; aucune sélection ou question n’a été changée.']
 (out/'rapport-n6.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dir',type=Path,required=True);main(ap.parse_args().dir)
