"""Technical invariants only, not corpus annotation or attribution accuracy."""
from core import decode,matched,scope_options

def run_tests():
    def ev(coord=False,tie=False):
        return {'id':'synthetic','hypotheses':[
          {'cause':'c','effect':'e1','pair':('c','e1'),'votes':2,'admissible':True},
          {'cause':'c','effect':'e2','pair':('c','e2'),'votes':2 if tie else 1,'admissible':True}],
          'source_options':[['c']], 'effect_options':[['e1','e2']] if coord else [['e1'],['e2']]}
    f=decode([ev()],False);a=decode([ev()],True)
    assert f['supported']=={('c','e1'),('c','e2')}
    assert a['supported']=={('c','e1')}
    assert decode([ev(tie=True)],True)['supported']==set()
    assert decode([ev(coord=True)],True)['supported']==f['supported']
    assert decode([],True)['supported']==set()
    assert matched(f,a,set(),'synthetic')<=f['supported']
    assert len(matched(f,a,set(),'synthetic'))==len(a['supported'])
    assert not scope_options({'x','y'},['and'])[1]
    assert not scope_options({'x','y'},['which'])[1]
    assert scope_options({'x','y'},['the','increase'])[1]
    assert decode([ev()],False)==decode([ev()],competition=False)
    return {'status':'PASS','scope':'synthetic code invariants only',
            'checks':['no_edge_F_parity','ties_abstain','multiple_effects_allowed',
                      'empty_support_not_negative','coverage_match','scope_safety_guards']}

if __name__=='__main__':print(run_tests())
