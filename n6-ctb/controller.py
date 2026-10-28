"""N6 synchronous Decisions with durable before-POST receipts and no paid retries."""
import base64,fcntl,gzip,hashlib,json,math,os,sys,threading,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ENDPOINT='https://openrouter.ai/api/alpha/decisions'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(canonical(v)+'\n');temp.replace(p)
def emit(name,v):print(name+' '+canonical(v),flush=True)
def export(out):
 files={p.name:p.read_text() for p in out.iterdir() if p.is_file()};packed=gzip.compress(canonical(files).encode(),mtime=0);h=hashlib.sha256(packed).hexdigest();encoded=base64.b64encode(packed).decode()
 for i in range(0,len(encoded),12000):emit('N6_EXPORT_CHUNK',dict(sha256=h,index=i//12000,data=encoded[i:i+12000]))
 emit('N6_EXPORT_END',dict(sha256=h,chunks=math.ceil(len(encoded)/12000),files=sorted(files)))
def key():
 value=os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER')
 if not value:raise RuntimeError('missing provider credentials')
 return value
def http(method,url,body=None):
 req=urllib.request.Request(url,method=method,data=canonical(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+key(),'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)
def validate(response,expected):
 if not isinstance(response,dict) or response.get('model')!=expected:raise ValueError('served_model_mismatch')
 answer=(response.get('answers') or {}).get('causal') or {};p=answer.get('noul')
 if answer.get('type')!='noul' or type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('invalid_noul')
 cost=(response.get('usage') or {}).get('cost')
 if cost is not None and (type(cost) not in (int,float) or not math.isfinite(cost) or cost<0):raise ValueError('invalid_cost')
 return p,cost

def execute(row,state,p,transport=http):
 h=digest(row['body']);path=state/(row['custom_id']+'.json')
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if path.exists():
   old=json.loads(path.read_text());assert old['request_sha256']==h
   if old['state']=='SUBMITTING':return dict(**old,valid=False,error='ambiguous_prior_POST_never_repeated')
   return old
  record=dict(custom_id=row['custom_id'],doc_id=row['doc_id'],pair=row['pair'],arm=row['arm'],request_sha256=h,request=row['body'],reserved_usd=p['per_request_reservation_usd'],state='SUBMITTING',valid=False,post_attempts=1,first_run_id=os.environ.get('ORX_RUN_ID'),started_unix=time.time())
  save(path,record) # atomic durable journal BEFORE the only paid POST
  try:
   response=transport('POST',ENDPOINT,row['body']);record['response']=response
   probability,cost=validate(response,p['expected_served']);record.update(probability=probability,cost_usd=cost,valid=True,state='COMPLETED')
   if cost is not None and cost>p['per_request_reservation_usd']:record.update(valid=False,state='COST_BOUND_EXCEEDED',error='provider_exceeded_per_request_reservation')
  except BaseException as exc:
   if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
   record.update(state='ERROR',error=('HTTP'+str(exc.code)) if isinstance(exc,urllib.error.HTTPError) else type(exc).__name__+':'+str(exc)[:200])
  record['ended_unix']=time.time();save(path,record);return record

def catalog_gate(p,rows):
 catalog=http('GET','https://openrouter.ai/api/v1/models');models=[m for m in catalog.get('data',[]) if m.get('id')==p['model']]
 if len(models)!=1:raise RuntimeError('catalog model absent or ambiguous')
 m=models[0];price=float((m.get('pricing') or {})['prompt']);assert math.isfinite(price) and price>=0
 for row in rows:
  assert row['body']['model']==p['model']
  assert row['input_utf8_bound']>=len(canonical(row['body']).encode())
  assert row['input_utf8_bound']*price<=p['per_request_reservation_usd']
  assert row['input_utf8_bound']<=m['context_length']
 return dict(model=m['id'],canonical_slug=m.get('canonical_slug'),context_length=m.get('context_length'),prompt_price_per_token=price)

def main():
 runid=os.environ.get('ORX_RUN_ID');assert runid,'tracked run required'
 frozen=HERE/'frozen';p=json.loads((frozen/'protocol.json').read_text());assert sha(frozen/'public.json')==p['public_sha256']
 public=json.loads((frozen/'public.json').read_text());rows=public['requests'];assert len(rows)==p['request_count'] and len({r['custom_id'] for r in rows})==len(rows)
 reserved=len(rows)*p['per_request_reservation_usd'];assert abs(reserved-p['api_ceiling_usd'])<1e-12 and reserved<=p['cap_usd']
 # Never open evaluation-only.json before the complete response-freeze receipt.
 out=ROOT/'logs'/('n6-'+runid);out.mkdir(parents=True,exist_ok=True)
 for name in ['public.json','protocol.json']:(out/name).write_bytes((frozen/name).read_bytes())
 emit('N6_CONFIG',dict(protocol=p,run_id=runid,commit=os.environ.get('ORX_COMMIT'),parallelism=8,threshold=.5,threshold_comparison='>=',endpoint=ENDPOINT))
 gate=catalog_gate(p,rows);save(out/'preflight.json',dict(status='PASS',catalog=gate,reserved_usd=reserved,cap_usd=p['cap_usd'],paid_retries=0,no_human_annotation=True,no_training=True))
 state=Path('~/.local/share/openresearch/workere-decisions/n6-ctb-v1').expanduser()/sha(frozen/'protocol.json');state.mkdir(parents=True,exist_ok=True)
 with (state/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  reservation=state/'reservation.json'
  if not reservation.exists():save(reservation,dict(reserved_usd=reserved,cap_usd=p['cap_usd'],first_run_id=runid))
  assert json.loads(reservation.read_text())['reserved_usd']==reserved
  stop=threading.Event()
  def work(row):
   if stop.is_set():return dict(custom_id=row['custom_id'],doc_id=row['doc_id'],pair=row['pair'],arm=row['arm'],valid=False,state='NOT_SUBMITTED_AFTER_COST_BREACH',post_attempts=0)
   result=execute(row,state,p)
   if result['state']=='COST_BOUND_EXCEEDED':stop.set()
   emit('N6_RESPONSE',dict(custom_id=row['custom_id'],arm=row['arm'],valid=result['valid'],state=result['state'],cost_usd=result.get('cost_usd')))
   return result
  with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(work,rows))
  save(out/'responses-before-gold.json',dict(gold_used=False,results=results));save(out/'freeze.json',dict(responses_sha256=sha(out/'responses-before-gold.json'),protocol_sha256=sha(out/'protocol.json'),public_sha256=sha(out/'public.json'),frozen_unix=time.time(),run_id=runid,commit=os.environ.get('ORX_COMMIT')))
  emit('N6_FROZEN_BEFORE_GOLD',json.loads((out/'freeze.json').read_text()));export(out)
  import subprocess
  subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--dir',str(out)],check=True)
  for name in ['controller.py','evaluate.py','test_controller.py']:(out/name).write_bytes((HERE/name).read_bytes())
  save(out/'manifest.json',dict(run_id=runid,commit=os.environ.get('ORX_COMMIT'),files_sha256={x.name:sha(x) for x in out.iterdir() if x.is_file()}));export(out)
  emit('N6_RESULT',json.loads((out/'results.json').read_text()))
if __name__=='__main__':main()
