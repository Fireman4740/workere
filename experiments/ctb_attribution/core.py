"""Small automatic CTB diagnostic. Not a complete ECI pipeline.
Inputs are frozen text/mentions/candidates and cached Haiku votes; no gold.
The single-head scope assumption is PROVISIONAL, not human-certified identity.
"""
import hashlib
import itertools
import re

# Longest-match lexical constructions, declared from CTB guidelines and DEV text.
PATTERNS = [
    ('passive', r'\b(?:caused|hampered|bedeviled|constrained|affected|triggered|prevented)\s+by\b'),
    ('reverse', r'\bas\s+a\s+result\s+of\b|\bbecause\s+of\b|\bdue\s+to\b|\bthanks\s+(?:in\s+part\s+)?to\b|\bas\s+a\s+consequence\s+of\b|\bresult(?:s|ed)?\s+from\b'),
    ('reverse', r'\bbecause\b'),
    ('forward', r'\b(?:lead|leads|led|leading)\s+to\b|\bresult(?:s|ed|ing)?\s+in\b|\b(?:touch(?:es|ed)?|set(?:s)?)\s+off\b'),
    ('forward', r'\b(?:cause|causes|caused|causing|enable|enables|enabled|enabling|allow|allows|allowed|allowing|trigger|triggers|triggered|triggering|prevent|prevents|prevented|preventing|eliminate|eliminates|eliminated|eliminating|make|makes|made|making|force|forces|forced|forcing|prompt|prompts|prompted|prompting)\b')]
REPORTING = {'said','say','says','reported','report','reports','noted','notes','announced','declared','declares','cited','agreed','considered','attributed'}
COORD = {'and','or','nor','both',','}
CLAUSAL = {'that','which','when','because','if','unless','to'}
MODALS = {'may','might','could','would','should','if','unless'}


def cues(text):
    found=[]
    for kind,pattern in PATTERNS:
        for m in re.finditer(pattern,text,re.I):found.append((m.start(),m.end(),kind))
    selected=[]
    for a,b,k in sorted(found,key=lambda v:(v[0],-(v[1]-v[0]))):
        if not any(a<e and s<b for s,e,_ in selected):selected.append((a,b,k))
    return selected


def scope_options(role_ids, tokens):
    """Never impose a capacity on an evidence passage or coordinated argument.
    Only an uncoordinated, non-clausal provisional argument slot is functional.
    This is an automatic structural hypothesis, not a certified coreference fact.
    """
    words={x.lower() for x in tokens}
    functional=len(role_ids)>1 and not (words & (COORD|CLAUSAL))
    return ([{x} for x in sorted(role_ids)] if functional else [set(role_ids)]),functional


def extract(unit):
    """Produce identical provisional supports/facts for F and A, with offsets."""
    start,end=unit['sentence_token_ranges'][unit['target_sentence_index']]
    tokens=unit['tokens'];sentence=' '.join(tokens[start:end]);starts=[];pos=0
    for tok in tokens:starts.append(pos);pos+=len(tok)+1
    sent_start=starts[start]
    events=[]
    for m in unit['mentions']:
        if m['in_target_component']:
            sp=m['token_indices']
            events.append({'id':m['id'],'a':starts[sp[0]]-sent_start,
                           'b':starts[sp[-1]]+len(tokens[sp[-1]])-sent_start,
                           'surface':m['surface'],'tokens':sp})
    votes={tuple(p['pair']):int(p['votes']) for p in unit['candidate_votes']}
    evidences=[]
    for number,(a,b,kind) in enumerate(cues(sentence)):
        # Clause ranges are lexical approximations. Missing support never refutes a pair.
        left=0;right=len(sentence)
        for m in re.finditer(r'[;!?]',sentence):
            if m.end()<=a:left=m.end()
            elif m.start()>=b:right=min(right,m.start());break
        prefix=sentence[max(left,a-20):a].lower().split()
        negated=any(w in {'not','never'} for w in prefix[-3:]) and 'only' not in prefix[-3:]
        if kind=='forward':source_range=(left,a);effect_range=(b,right)
        elif kind=='passive':source_range=(b,right);effect_range=(left,b)
        else:
            comma=sentence.find(',',b,right)
            if not sentence[left:a].strip() and comma>=0:
                source_range=(b,comma);effect_range=(comma+1,right)
            else:source_range=(b,right);effect_range=(left,a)
        def members(bounds,role):
            out=[]
            for e in events:
                if e['surface'].lower() in REPORTING:continue
                if not (bounds[0]<=e['a'] and e['b']<=bounds[1]):continue
                if kind!='passive' and e['a']<b and a<e['b']:continue
                out.append(e)
            return out
        ss=members(source_range,'cause');ee=members(effect_range,'effect')
        hypotheses=[]
        for s in ss:
            for e in ee:
                if s['id']==e['id']:continue
                pair=tuple(sorted((s['id'],e['id'])))
                if votes.get(pair,0)==0:continue  # cached proposal exists; not a probability threshold fit
                hypotheses.append({'cause':s['id'],'effect':e['id'],'pair':pair,
                    'votes':votes[pair], 'admissible':not negated})
        active=[h for h in hypotheses if h['admissible']]
        if not hypotheses:continue
        sid={h['cause'] for h in active};eid={h['effect'] for h in active}
        stokens=sentence[source_range[0]:source_range[1]].split()
        etokens=sentence[effect_range[0]:effect_range[1]].split()
        so,sfunctional=scope_options(sid,stokens);eo,efunctional=scope_options(eid,etokens)
        evidences.append({'id':unit['unit_id']+'::cue'+str(number),
            'cue_span':[sent_start+a,sent_start+b], 'cue':sentence[a:b], 'kind':kind,
            'sentence_span':[sent_start,sent_start+len(sentence)],'sentence_quote':sentence,
            'cause_slot_span':[sent_start+x for x in source_range],
            'effect_slot_span':[sent_start+x for x in effect_range],
            'cause_quote':sentence[slice(*source_range)],'effect_quote':sentence[slice(*effect_range)],
            'facts':{'lexical_anchor':'SAT','role_bracketing':'SAT_PROVISIONAL',
                     'negation':'CONFLICT' if negated else 'NO_EXPLICIT_CONFLICT',
                     'participants':'UNKNOWN','incident':'UNKNOWN','date':'UNKNOWN',
                     'location':'UNKNOWN','granularity':'UNKNOWN',
                     'modal_operators':[w for w in sentence.lower().split() if w in MODALS],
                     'modal_scope':'UNKNOWN_NOT_REJECTED'},
            'hypotheses':hypotheses,'functional_cause_slot':sfunctional,
            'functional_effect_slot':efunctional,
            'competition_certificate':'AUTOMATIC_UNCOORDINATED_NONCLAUSAL_SLOT_ASSUMPTION_NOT_HUMAN_CERTIFIED',
            'source_options':[sorted(x) for x in so], 'effect_options':[sorted(x) for x in eo]})
    return evidences


def decode(evidences,competition=True,with_votes=True):
    """Exact enumeration of scoped bindings, not a quota on relations.
    Objective: unique role-binding atoms + distinct cached pass votes for pairs.
    Pair truth is not consulted. Coordination keeps multiple bindings available.
    """
    certain=set();possible=set();binding_certain=set();objective=0;branches=0
    for ev in evidences:
        hs=[h for h in ev['hypotheses'] if h['admissible']]
        if not hs:continue
        ss={h['cause'] for h in hs};ee={h['effect'] for h in hs}
        source_options=[set(x) for x in ev['source_options']] if competition else [ss]
        effect_options=[set(x) for x in ev['effect_options']] if competition else [ee]
        best=-1;opts=[]
        for s,e in itertools.product(source_options,effect_options):
            branches+=1
            selected=[h for h in hs if h['cause'] in s and h['effect'] in e]
            # Coordinated roles may share evidence; unused roles collect no credit.
            if not selected:continue
            used_s={h['cause'] for h in selected};used_e={h['effect'] for h in selected}
            p={tuple(h['pair']) for h in selected}
            value=len(used_s)+len(used_e)+(sum(h['votes'] for h in selected) if with_votes else 0)
            bs={(ev['id'],'cause',x) for x in used_s}|{(ev['id'],'effect',x) for x in used_e}
            if value>best:best=value;opts=[(p,bs)]
            elif value==best:opts.append((p,bs))
        if not opts:continue
        objective+=best
        certain|=set.intersection(*(p for p,b in opts))
        possible|=set.union(*(p for p,b in opts))
        binding_certain|=set.intersection(*(b for p,b in opts))
    return {'supported':certain,'possible':possible,'bindings':binding_certain,
            'objective':objective,'enumerated_bindings':branches}


def matched(F,A,baseline,unit_id,seed='ctb-equal-coverage-v1'):
    # Match support coverage AND baseline/new counts; therefore final positive count
    # under conservative merge is equal too. This is a CONTROL, not a relation quota.
    selected=set()
    for in_baseline in [False,True]:
        population={p for p in F['supported'] if (p in baseline)==in_baseline}
        n=sum((p in baseline)==in_baseline for p in A['supported'])
        order=sorted(population,key=lambda p:hashlib.sha256((seed+'|'+unit_id+'|'+'|'.join(p)).encode()).digest())
        selected.update(order[:n])
    assert len(selected)==len(A['supported'])
    assert len(selected-baseline)==len(A['supported']-baseline)
    return selected
