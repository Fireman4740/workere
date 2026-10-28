"""Frozen all-subset pass ablation; no model or network client imported."""
import hashlib,itertools,json,random,statistics
from pathlib import Path

def f1(c):
 t,p,n=c;return 200*t/(2*t+p+n) if t else 0.
def main(command,output):
 root=Path(__file__).resolve().parents[2];p=root/'experiments/jev/n5/inputs.json';bundle=json.loads(p.read_text());docs=bundle['documents'];result={'study':'N5','documents':len(docs),'topics':len({d['id'].split('_')[0] for d in docs}),'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'new_api_calls':0,'threshold':.39,'role':'exploratory exposed historical CONF40; not independent confirmation'}
 if command=='dry-run':
  result['subsets']={str(k):len(list(itertools.combinations(range(7),k))) for k in [1,3,7]};Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2));print('N5_PREFLIGHT '+json.dumps(result),flush=True);return 0
 rows={};by_topic={}
 for n in [1,3,7]:
  subsets=list(itertools.combinations(range(7),n));all_counts=[];all_cost=[];topic_sums={}
  for subset in subsets:
   total=[0,0,0];cost=0
   for d in docs:
    allowed=set(map(tuple,d['pairs']));gold={(a,b) for a,lab,b in d['gold'] if lab!='norel'};d1={(a,b):v for a,b,v in d['d1']};assert allowed<=set(d1)
    votes={k:0 for k in allowed}
    for i in subset:
     labels={(a,b):lab for a,lab,b in d['passes'][i]}
     for pair,lab in labels.items():
      if pair in allowed and lab!='norel':votes[pair]+=1
    counts=[0,0,0]
    for a,b in allowed:
     assert (b,a) in allowed
     pred=(votes[(a,b)]/n+votes[(b,a)]/n+d1[(a,b)]+d1[(b,a)])/4>=.39;y=(a,b) in gold
     counts[0]+=pred and y;counts[1]+=pred and not y;counts[2]+=(not pred) and y
    total=[a+b for a,b in zip(total,counts)];topic=d['id'].split('_')[0];topic_sums.setdefault(topic,[0.,0.,0.]);topic_sums[topic]=[a+b/len(subsets) for a,b in zip(topic_sums[topic],counts)];cost+=sum(d['costs'][i] for i in subset)
   all_counts.append(total);all_cost.append(cost)
  fs=[f1(c) for c in all_counts];mean_counts=[statistics.mean(c[i] for c in all_counts) for i in range(3)]
  rows[str(n)]={'subsets':len(subsets),'mean_F1':statistics.mean(fs),'min_F1':min(fs),'max_F1':max(fs),'sd_across_subsets':statistics.pstdev(fs),'mean_TP_FP_FN':mean_counts,'F1_of_expected_counts':f1(mean_counts),'mean_historical_allocated_API_cost_USD':statistics.mean(all_cost),'relative_Luna_calls':n/7,'all_subset_F1':fs};by_topic[str(n)]=topic_sums
 topics=sorted(by_topic['7']);contrasts={}
 for n in [1,3]:
  rng=random.Random(20261028);deltas=[]
  for _ in range(2000):
   draw=rng.choices(topics,k=len(topics));c=[[sum(by_topic[str(k)][t][i] for t in draw) for i in range(3)] for k in [n,7]];deltas.append(f1(c[0])-f1(c[1]))
  deltas.sort();contrasts[f'{n}_minus_7']={'delta_F1_of_expected_counts':rows[str(n)]['F1_of_expected_counts']-rows['7']['F1_of_expected_counts'],'topic_bootstrap_CI95':[deltas[50],deltas[1949]],'resamples':2000,'seed':20261028}
 result.update(results=rows,contrasts=contrasts,per_topic_expected_counts=by_topic,limits=['subset spread is schedule sensitivity within historical generations, not full generation uncertainty','cost is historical equal-per-batch allocation, excludes fixed d1 and infrastructure','topic CI targets F1 of expected subset counts, not expectation of nonlinear F1','no best subset selected; no threshold refit'])
 Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2));print('N5_RESULT '+json.dumps(result),flush=True);return 0
