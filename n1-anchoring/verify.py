"""Deterministic output validity and exact anchoring checks; no gold access."""
import json,re
VERDICTS={'SUPPORTED','CONTRADICTED','UNKNOWN'}
def quote_valid(text,quote,mention=None):
 if not isinstance(quote,str) or not quote.strip() or text.count(quote)!=1:return False
 return mention is None or re.search(r'<'+re.escape(str(mention))+r'\s',quote) is not None
def anchored(item,doc,pair):
 if item.get('verdict')!='SUPPORTED':return False,'not_supported'
 if {str(item.get('cause_id')),str(item.get('effect_id'))}!=set(pair):return False,'wrong_direction_ids'
 if not quote_valid(doc['text'],item.get('evidence_quote')):return False,'evidence_not_unique_exact'
 bb=item.get('bindings');known={m['id'] for m in doc['mentions']}
 if not isinstance(bb,list) or len(bb)!=2:return False,'binding_schema'
 if {str(b.get('mention_id')) for b in bb if isinstance(b,dict)}!=set(pair):return False,'binding_ids'
 for b in bb:
  mid=str(b['mention_id'])
  if mid not in known or not quote_valid(doc['text'],b.get('occurrence_quote'),mid):return False,'occurrence_not_exact_tagged'
  if b.get('scope_consistent') is not True:return False,'scope_not_consistent'
  if b.get('modality') not in {'realized','hypothetical','negated','reported','unknown'}:return False,'modality_schema'
  if any(not isinstance(b.get(k),str) or not b[k].strip() for k in ['participant','incident','time']):return False,'binding_description_missing'
 return True,'accepted'
def decode(row,raw,doc,cfg):
 body=(raw.get('response') or {}).get('body') or {};choices=body.get('choices') or []
 event=dict(custom_id=row['custom_id'],doc_id=row['doc_id'],valid=False,state='missing_or_error',served_model=body.get('model'),usage=body.get('usage'),items=[],system_fingerprint=body.get('system_fingerprint'))
 if raw.get('error') or (raw.get('response') or {}).get('status_code',200)!=200:return event
 if body.get('model') not in cfg['allowed_served_models']:event['state']='model_mismatch';return event
 if len(choices)!=1:return event
 choice=choices[0];event['finish_reason']=choice.get('finish_reason')
 if choice.get('finish_reason')=='length':event['state']='truncated';return event
 if choice.get('finish_reason')!='stop':event['state']='nonstop';return event
 content=(choice.get('message') or {}).get('content')
 try:obj=json.loads(content)
 except (ValueError,TypeError):event['state']='json_invalid';return event
 items=obj.get('items') if isinstance(obj,dict) else None;expected={c['candidate_id']:c for c in row['candidates']}
 if not isinstance(items,list) or len(items)!=len(expected) or any(not isinstance(i,dict) for i in items):event['state']='coverage_invalid';return event
 if {i.get('candidate_id') for i in items}!=set(expected) or any(i.get('verdict') not in VERDICTS for i in items):event['state']='coverage_invalid';return event
 event.update(valid=True,state='valid')
 for item in items:
  pair=expected[item['candidate_id']]['pair'];ok,reason=anchored(item,doc,pair)
  event['items'].append(dict(candidate_id=item['candidate_id'],pair=pair,accepted=ok,validation=reason,model_item=item))
 return event
