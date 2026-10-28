"""Tracked N1 controller with one durable batch submission and separate gold evaluator."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from eci_m0_pilot.run_pilot import catalog_gate,batch_run,save,emit,export
from verify import decode

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 runid=os.environ.get('ORX_RUN_ID');assert runid,'only tracked experiment execution'
 frozen=HERE/'frozen';protocol=json.loads((frozen/'protocol.json').read_text());cfg=protocol['settings']
 for name,h in protocol['files_sha256'].items():assert sha(frozen/name)==h
 rows=json.loads((frozen/'requests.json').read_text());docs=json.loads((frozen/'documents.json').read_text())
 out=ROOT/'logs'/('n1-'+runid);out.mkdir(parents=True,exist_ok=True)
 for p in frozen.iterdir():shutil.copyfile(p,out/p.name)
 emit('N1_CONFIG',dict(protocol=protocol,run_id=runid,commit=os.environ.get('ORX_COMMIT')))
 gate=catalog_gate(cfg,max(r['prompt_utf8_upper_bound'] for r in rows)+cfg['max_output_tokens'])
 reserved=len(rows)*cfg['reservation_per_request_usd'];assert reserved==protocol['conservative_api_ceiling_usd'] and reserved<=2
 state=Path('~/.local/share/openresearch/workere-decisions/n1-anchoring-v1').expanduser()/sha(frozen/'protocol.json');state.mkdir(parents=True,exist_ok=True)
 with (state/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  reservation=state/'reservation.json'
  if not reservation.exists():save(reservation,dict(api_ceiling_usd=reserved,fresh_hard_cap_usd=2,protocol_sha256=sha(frozen/'protocol.json'),first_run_id=runid))
  assert json.loads(reservation.read_text())['api_ceiling_usd']==reserved
  save(out/'preflight.json',dict(status='PASS',catalog=gate,reservation=json.loads(reservation.read_text()),requests=len(rows),no_human_annotation=True,no_training=True))
  raw,cost=batch_run(rows,cfg,state)
  save(out/'raw.json',raw);save(out/'costs.json',cost)
  export({p.name:p.read_text() for p in out.iterdir() if p.is_file()})
  events=[decode(r,raw[r['custom_id']],docs[r['doc_id']],cfg) for r in rows]
  outputs=dict(gold_used=False,events=events,accepted=[dict(doc_id=e['doc_id'],pair=i['pair'],candidate_id=i['candidate_id']) for e in events for i in e['items'] if i['accepted']],integrity=dict(all_valid=all(e['valid'] for e in events),states=dict(Counter(e['state'] for e in events))))
  save(out/'outputs-before-gold.json',outputs)
  save(out/'freeze.json',dict(outputs_sha256=sha(out/'outputs-before-gold.json'),protocol_sha256=sha(out/'protocol.json'),frozen_unix=time.time(),run_id=runid,commit=os.environ.get('ORX_COMMIT')))
  emit('N1_FROZEN_BEFORE_GOLD',json.loads((out/'freeze.json').read_text()))
  subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--dir',str(out)],check=True)
  for name in ['prepare.py','verify.py','run.py','evaluate.py','test_verify.py']:shutil.copyfile(HERE/name,out/name)
  save(out/'manifest.json',dict(run_id=runid,commit=os.environ.get('ORX_COMMIT'),files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
  export({p.name:p.read_text() for p in out.iterdir() if p.is_file()})
  emit('N1_RESULT',json.loads((out/'results.json').read_text()))
if __name__=='__main__':main()
