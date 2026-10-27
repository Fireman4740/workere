"""Only fake predictions for validating file I/O and math, never real models."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
pred=[]
for line in (ROOT/'out/tasks.jsonl').read_text(encoding='utf8').splitlines():
    task=json.loads(line)
    a,b=task['exposed_pair']
    pairs=[] if task['variant']=='target' else [[a,b],[b,a]]
    pred.append({'task_id':task['task_id'], 'status':'ok', 'predicted_ordered_pairs':pairs,
                 'origin':'MOCK_NOT_A_MODEL'})
with (ROOT/'out/predictions_MOCK.jsonl').open('w',encoding='utf8') as f:
    for p in pred: f.write(json.dumps(p,ensure_ascii=False)+'\n')
print(f'{len(pred)} fausses sorties générées à des fins de test logiciel')
