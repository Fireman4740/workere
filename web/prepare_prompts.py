"""Read prompt templates/builders at each experiment's frozen Git commit.

No execution of research code. Dynamic document bodies and demonstration records
are represented by the template variables; no request is regenerated or sent.
"""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


OFFLINE={14:7,18:8,29:27,30:27,31:18,32:18,37:33,42:8,46:39,52:17,56:53,60:57,61:52,68:65,69:65,73:67,105:102,109:102,112:110}
MODULES={
    15:['pipelines/jev/coreference.py'],
    20:['pipelines/jev/llm.py','pipelines/jev/coreference.py','prompts/esl_standard_eci_lean_coref.yaml'],
    24:['pipelines/harness2_eci/predictors.py','prompts/esl_standard_eci_lean_coref.yaml'],
    25:['pipelines/harness2_eci/predictors.py','prompts/esl_standard_eci_lean_coref.yaml'],
    40:['experiments/jev/qwen/run.py'],49:['experiments/jev/qwen/run.py'],64:['experiments/jev/qwen/run.py'],
    27:['pipelines/judges/band_verifier.py'],28:['pipelines/judges/coref_agreement.py'],
    33:['prompts/esl_luna_round_k.yaml'],34:['prompts/esl_luna_round_k.yaml'],35:['prompts/esl_luna_round_k.yaml'],36:['prompts/esl_luna_round_k.yaml'],38:['prompts/esl_luna_round_k.yaml'],
    41:['experiments/eci_alias_next/eci_alias/workere_run.py'],
    42:['eci_m0_pilot/data/prompt.json'],43:['experiments/jev/proof_transport_launch/plan/prompts.json'],
    44:['eci_m0_pilot/data/prompt.json','eci_m0_pilot/run.py'],45:['agent4_v0/automated_run.py'],
    48:['m1-diversity/prepare.py','eci_m0_pilot/data/prompt.json'],50:['n1-anchoring/prepare.py','n1-anchoring/verify.py'],
    51:['experiments/jev/n2_transport/n2.py'],53:['pipelines/jev/n3.py'],54:['n4-readers/run.py'],55:['n6-ctb/controller.py'],
    57:['pipelines/jev/n7.py'],58:['pipelines/jev/n8.py'],59:['pipelines/jev/n9.py'],62:['pipelines/jev/n12.py'],
    63:['experiments/jev/n13/ctb50/prompt.json','pipelines/jev/n13.py'],64:['pipelines/jev/llm.py'],
    65:['pipelines/jev/n14.py','experiments/jev/n14/prepare.py'],67:['pipelines/jev/n15.py'],
    88:['steering/pair_verifier.py'],89:['steering/pair_verifier.py'],90:['steering/campaign_blocks.py'],
    91:['steering/campaign_facts.py'],92:['steering/campaign_adjudicate.py'],93:['steering/campaign_api.py'],
    94:['steering/campaign_api.py','steering/ap_campaign.py'],95:['steering/campaign_api.py'],96:['steering/campaign_api.py'],
    97:['steering/ap_campaign.py','steering/campaign_api.py'],98:['steering/ap_campaign.py','steering/campaign_api.py'],99:['steering/ap_campaign.py','steering/campaign_api.py'],
    100:['steering/qg_campaign.py'],101:['steering/qg_campaign.py'],102:['steering/validation_campaign.py'],103:['steering/validation_campaign.py'],104:['steering/validation_campaign.py'],
    105:['steering/method_vote.py'],109:['steering/method_vote.py'],
    110:['steering/wsp_pilot.py'],111:['steering/wsp_confirmation.py','steering/wsp_pilot.py'],
    113:['steering/wsp_combo.py'],114:['steering/wsp_combo.py'],115:['steering/wsp_combo.py'],
    116:['steering/wsp_scale50.py','steering/wsp_combo.py'],117:['steering/wsp_scale50.py','steering/wsp_combo.py'],
}


def enrich(data):
    cache={};bank={};found=0
    def blob(commit,path):
        key=(commit,path)
        if key not in cache:
            p=subprocess.run(['git','show',commit+':'+path],capture_output=True,text=True)
            cache[key]=p.stdout if p.returncode==0 else ''
        return cache[key]
    for e in data['experiments']:
        n=e['number'];commit=e['code']['commit'];paths=[];e['prompt_refs']=[]
        e['prompt_status']='Gabarits et fonctions de construction relus au commit de référence. Les modules peuvent définir plusieurs branches : les remplacements de consigne et paramètres déclarés précisent la condition. Les champs variables (document, mentions, exemples) ne sont pas une nouvelle requête exécutée.'
        if n in OFFLINE:
            e['prompt_status']=f"Aucun nouveau prompt d’inférence dans ce calcul hors ligne. Les sorties du lecteur amont sont réutilisées ; voir la fiche {OFFLINE[n]} pour leur construction."
            e['prompt_upstream']=OFFLINE[n]
        if e['state']=='planifie':
            e['prompt_status']='Aucune exécution propre à ce plan : aucun prompt ne peut être présenté comme effectivement envoyé. Les sources de préparation disponibles sont indiquées ci-dessous.'
        if n<=17 and n!=8: paths+=['pipelines/jev/state_builder.py']
        paths+=MODULES.get(n,[])
        if n in [20,28]: paths += [p for p in e['code']['changed_files'] if p.endswith('.py') and re.search('round3|coref',p) and not p.startswith('tests/')]
        if n in [8,17,18,19,20,21,22,23,24,25,28,39,40,49,64]:paths+=['pipelines/jev/llm.py']
        # Resolve declared prompt files from active and per-document configs.
        configs=['experiments/jev/active.yaml','experiments/jev/qwen/active.yaml']+[p for p in e['code']['changed_files'] if p.endswith('.yaml')]
        for p in configs:
            for name in re.findall(r'prompts/([\w-]+\.yaml)',blob(commit,p)):paths.append('prompts/'+name)
        # Keep prompt overrides from the active configuration; displaying a
        # shared default alone would misrepresent R1a's actual instructions.
        if n<=25:
            paths=configs[:1]+[p for p in configs[2:5] if p!=configs[0]]+paths
        if n in [40,49,64]:
            queue=blob(commit,'queues/baseline_coref.yaml')
            for name in re.findall(r'prompt:\s*["\']?([\w-]+)',queue):paths.append('prompts/'+name+'.yaml')
        if n==55:paths+=['n6-ctb/frozen/public.json']
        if n in range(66,89) and n not in [67,68,69,73]:paths+=['steering/data/ctb-prompt.json','steering/extract_prompts.py']
        if n in [74,75,76,77,78]:paths+=['steering/run_vllm.py']
        if n in [82,83,84,85,86,87]:paths+=['steering/usc.py']
        if n in [80,86]:paths+=['steering/data/esc/prompt.json']
        if n>=90:
            if 'CTB' in e['datasets']:paths+=['steering/data/ctb-prompt.json']
            if 'ESC' in e['datasets']:paths+=['steering/data/esc/prompt.json']
            if 'MAVEN' in e['datasets']:paths+=['prompts/maven_ere_standard_eci_lean_coref.yaml']
        if n==21:paths+=['prompts/esl_standard_eci_event_semantics.yaml']
        if n==19:paths+=['prompts/esl_standard_eci_lean_why.yaml']
        if n==22:paths+=['prompts/esl_standard_eci_lean_coref.yaml']
        if n in [53,54,57,58,59,62,63,65,67]:paths+=['prompts/esl_standard_eci_lean_coref.yaml']
        seen=set()
        for path in dict.fromkeys(paths):
            content=blob(commit,path)
            if not content:continue
            snippets=[]
            if path.endswith('.yaml') and path.startswith('prompts/'):
                snippets=[('Fichier de prompt YAML',content,1,len(content.splitlines()),'template')]
            elif path.endswith('.yaml'):
                lines=content.splitlines();pieces=[];starts=[]
                for i,line in enumerate(lines):
                    match=re.match(r'^(\s+)(instructions|criteria|question\w*|state_\w+|demonstrations|passes|threshold|reasoning_effort|max_output_tokens):',line)
                    if not match:continue
                    indent=len(match[1]);j=i+1
                    while j<len(lines) and (not lines[j].strip() or len(lines[j])-len(lines[j].lstrip())>indent):j+=1
                    pieces.extend(lines[i:j]);starts.append(i+1)
                if pieces:snippets=[('Paramètres déclarés et remplacements de consigne','\n'.join(pieces),min(starts),None,'template')]
            elif path.endswith('.json'):
                try: obj=json.loads(content)
                except ValueError:continue
                def visit(value,pointer=''):
                    if isinstance(value,dict):
                        for k,v in value.items():
                            if k in ['system','user_template','user','instructions','context'] and isinstance(v,str):
                                snippets.append((pointer+'/'+k,v,1,None,'template'))
                            elif k not in ['demos','demonstrations','documents','examples'] and isinstance(v,dict):visit(v,pointer+'/'+k)
                visit(obj)
                if n==55 and path.endswith('/public.json'):
                    for request in obj.get('requests',[]):
                        body=request.get('body',{})
                        if body.get('questions'):
                            snippets.append(('Questions Noul · bras '+str(request.get('arm','')),json.dumps(body['questions'],ensure_ascii=False,indent=2),1,None,'template'))
                if path.endswith('/prompts.json') and isinstance(obj,dict):
                    for key,value in obj.items():
                        if isinstance(value,str) and len(value)>35:snippets.append((key,value,1,None,'template'))
            elif path.endswith('.py'):
                try: tree=ast.parse(content)
                except SyntaxError:continue
                lines=content.splitlines()
                for item in ast.walk(tree):
                    if isinstance(item,(ast.Assign,ast.AnnAssign)):
                        name=','.join(t.id for t in getattr(item,'targets',[]) if isinstance(t,ast.Name)) or getattr(getattr(item,'target',None),'id','')
                        if re.search(r'SYSTEM|PROMPT|INSTRUCTION|CRITERIA|QUESTION|CONTEXT|RULES|FORCE_CLOSE',name):
                            try: val=ast.literal_eval(item.value)
                            except (ValueError,TypeError):continue
                            if isinstance(val,str) and len(val)>35:snippets.append((name,val,item.lineno,item.end_lineno,'template'))
                    if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        segment='\n'.join(lines[item.lineno-1:item.end_lineno])
                        if (re.search(r'prompt|messages|question|build_state|make_reading_jobs|verification_jobs',item.name,re.I) or (re.search(r"\b(?:system|user_template|messages|instructions)\b",segment) and re.search(r"content|template|messages|criteria",segment))) and len(segment)<24000:
                            snippets.append((item.name+'(…) · assemblage exact',segment,item.lineno,item.end_lineno,'builder'))
            for label,text,start,end,kind in snippets:
                digest=hashlib.sha256((path+'\n'+label+'\n'+text).encode()).hexdigest()[:20]
                if digest in seen:continue
                seen.add(digest)
                bank.setdefault(digest,{'id':digest,'label':label,'content':text,'kind':kind,'path':path})
                e['prompt_refs'].append({'id':digest,'commit':commit,'path':path,'line':start,'end':end,'blob_sha256':hashlib.sha256(content.encode()).hexdigest()})
        if e['prompt_refs']:found+=1
        elif n not in OFFLINE and e['state']!='planifie':
            e['prompt_status']='Le prompt exact n’a pas été retrouvé dans les sources ciblées du commit de référence. La fiche conserve le protocole, mais aucun texte reconstitué n’est présenté comme le prompt envoyé.'
    data['prompt_bank']=bank
    data['provenance']['prompt_experiments_with_source']=found
    data['provenance']['prompt_templates_and_builders']=len(bank)
    return data


if __name__=='__main__':
    p=Path(__file__).with_name('data.json');d=enrich(json.loads(p.read_text()));p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in d['provenance'].items() if k.startswith('prompt_')}))
