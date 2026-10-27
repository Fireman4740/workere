"""Read-only historical M0 audit. Writes only the new M1 audit file."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import sha,save,M0,readgit
from eci_m0_pilot.protocol import json_text,ledger_for

def main(source,dest):
    before=json.loads((source/'smoke/outputs-before-gold.json').read_text())
    requests=json.loads((source/'smoke/requests.json').read_text())
    config=json.loads(readgit(M0,'eci_m0_pilot/config.json'))
    docs={d['doc_id']:d for d in map(json.loads,readgit(M0,'eci_m0_pilot/data/public_train233.jsonl').splitlines())}
    actual=sha((json_text(before)+'\n').encode())
    assert actual=='dd625f9cf60b3a13363780c1fe95dc0589ebd45a754fb3085d4e8896e331cd7b'
    for did,l in before['ledgers'].items():
        rebuilt,events=ledger_for(docs[did],requests[did],before['raw_results'],config)
        assert rebuilt==l
        assert json.loads(json.dumps(events))==[e for e in before['events'] if e['doc_id']==did]
    result=dict(status='PASS',source_absolute=str(source.resolve()),m0_commit=M0,
        run_id='08eaa19b-af21-400d-94c9-20e96d0dbb93',outputs_semantic_sha256=actual,
        raw_parser_and_remapping_reproduced=True,requests=len(before['raw_results']),
        historical_artifacts_unchanged=True,new_model_calls=0,
        files_sha256={str(p.relative_to(source)):sha(p) for p in sorted((source/'smoke').iterdir()) if p.is_file()},
        source_modules_sha256={name:sha(readgit(M0,'eci_m0_pilot/'+name)) for name in ['m0.py','protocol.py','config.json','run_pilot.py']})
    save(dest,result);print(json_text(result))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();main(a.source,a.out)
