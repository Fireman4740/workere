"""Generate deliberately simulated, not model-produced response records.

For pipeline functionality tests only. Do NOT quote resulting F1 as experiments.
"""
import json
from pathlib import Path
import sys
viewpath=Path(sys.argv[1]);out=Path(sys.argv[2])
views=[json.loads(s) for s in viewpath.read_text().splitlines() if s.strip()]
with out.open('w') as f:
    for v in views:
        id0,id1=v['mentions'][:2]
        for p in range(7):
            # This deliberately makes A outperform R on synthetic fixtures.
            # It is NOT a simulated model or measurement.
            resp=[[id0,'causes',id1]] if p < (4 if v['arm']=='A' else 3) else []
            f.write(json.dumps({'doc_id':v['doc_id'],'arm':v['arm'],
                                'pass_index':p,'status':'ok','response':resp})+'\n')
print('WROTE artificial response fixtures, not model predictions')
