from pathlib import Path
import os
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1'
import json,subprocess
from dataclasses import asdict
from datasets import Dataset, concatenate_datasets
from benchmark.data import doc_from_row
from benchmark.io import fingerprint
from pipelines.judges.calibration import load_freeze

root=Path('experiments/jev/confirmation')
subprocess.run(['git','ls-files','--error-unmatch',str(root/'frozen-rules.json')],check=True,capture_output=True)
assert not subprocess.check_output(['git','diff','HEAD','--',str(root/'frozen-rules.json')],text=True)
rules=load_freeze(root/'frozen-rules.json')
selection=json.loads((root/'selection.json').read_text())
assert selection['frozen_rules_fingerprint']==rules['fingerprint']
ids={r['id'] for r in selection['documents']}
prior=json.loads(Path('experiments/jev/round3/round3-selection.json').read_text())
assert ids<={r['id'] for r in prior['holdout_rest']}
assert not ids&set(rules['training_documents'])
bank=json.loads((root/'demo-bank.json').read_text())['documents']
assert not ids&{r['id'] for r in bank}
revision='e6418e8747ac1c7fecf66d56ae41fbde7fd5967a'
cache=Path.home()/'.cache/huggingface/datasets/Nofing___event_story_line-0.9-standard-eci-workere/default/0.0.0'/revision
files=sorted(cache.glob('*train*.arrow'))
assert files, 'pinned train Arrow files are not available'
rows=concatenate_datasets([Dataset.from_file(str(p)) for p in files])
public=[];gold={}
for i,row in enumerate(rows):
    if row['id'] not in ids:continue
    doc=doc_from_row(row,i,text_field='text',ann_field='annots',valid_labels=None,
                     binary_undirected=True,shuffle_pair_list=False)
    d=asdict(doc)
    gold[doc.id]=[list(k) for k in sorted({(a,b) for a,lab,b in d.pop('gold') if lab!='norel'})]
    public.append({k:d[k] for k in ['id','tokens','sentences','lang','mentions','spans','surfaces','sentence_of','pair_list']})
public.sort(key=lambda d:d['id'])
assert {d['id'] for d in public}==ids
assert sum(len(d['pair_list']) for d in public)==selection['ordered_pairs']
bundle={'schema':'workere-confirmation-public/v1','revision':revision,'documents':public}
bundle['fingerprint']=fingerprint(bundle)
(root/'public-inputs.json').write_text(json.dumps(bundle,separators=(',',':'))+'\n')
evaluation={'schema':'workere-confirmation-gold/v1','public_fingerprint':bundle['fingerprint'],'positive_pairs':gold}
(root/'evaluation-only.json').write_text(json.dumps(evaluation,separators=(',',':'))+'\n')
print(json.dumps({'exported_documents':len(public),'ordered_pairs':sum(len(d['pair_list']) for d in public),
                   'public_fingerprint':bundle['fingerprint'],'rules_already_frozen':rules['fingerprint']}))
