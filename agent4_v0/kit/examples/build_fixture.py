#!/usr/bin/env python3
"""Creates a fully fictional fixture. Not ESC data; never use as research evidence."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
text = ('Le feu a causé une fuite. La pluie a retardé le match. '
        'Le vent a endommagé le toit. L\'orage a provoqué une panne.')
ids = [('e1','feu'),('e2','fuite'),('e3','pluie'),('e4','match'),('e5','vent'),('e6','toit'),('e7',"orage"),('e8','panne')]
mentions = []
for e,s in ids:
    pos = text.find(s)
    assert pos >= 0
    mentions.append({'id':e,'start':pos,'end':pos+len(s),'surface':s})
base = {'doc_id':'SYNTHETIC-01', 'topic_id':'SYNTHETIC-topic', 'split':'dev', 'text':text,
        'mentions':mentions, 'pairs':[{'a':'e1','b':'e2','baseline_score':.84},
                 {'a':'e3','b':'e4','baseline_score':.34}]}
with (HERE/'baseline_synthetic.jsonl').open('w',encoding='utf8') as f: f.write(json.dumps(base,ensure_ascii=False)+'\n')
replacements=[('target','a causé','a précédé'),('placebo_1','a retardé','a précédé'),
              ('placebo_2','a endommagé','a précédé'),('placebo_3','a provoqué','a précédé')]
rows=[]
for variant,before,after in replacements:
    start=text.index(before)
    rows.append({'doc_id':base['doc_id'], 'pair':['e1','e2'],'variant':variant,
                 'review_status':'approved','reviewed_blind_to_gold':True,
                 'edits':[{'start':start,'end':start+len(before),'before':before,'after':after}]})
rows.append({'doc_id':base['doc_id'], 'pair':['e1','e2'],'variant':'rename_ids',
             'review_status':'approved','reviewed_blind_to_gold':True,
             'id_map':{e:'v'+str(k+1) for k,(e,_) in enumerate(ids)}})
with (HERE/'edits_synthetic.jsonl').open('w',encoding='utf8') as f:
    for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
with (HERE/'gold_synthetic.jsonl').open('w',encoding='utf8') as f:
    f.write(json.dumps({'doc_id':base['doc_id'],'pair':['e1','e2'],'causal':True})+'\n')
print('Données synthétiques générées')
