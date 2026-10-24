import copy
import io
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from pipelines.judges import offline_fusion as f


class OfflineFusion(unittest.TestCase):
    def test_features_are_symmetric_and_never_read_labels(self):
        class Forbidden:
            def __bool__(self): raise AssertionError('label read during feature extraction')
        data = SimpleNamespace(by_doc={'d': {('1','2'):dict(luna=.8,d1=.4,gold=Forbidden()),
                                             ('2','1'):dict(luna=.6,d1=.2,gold=Forbidden())}},
            suite={'d':dict(sentence_of={'1':0,'2':2},mentions=['1','2'],spans=[[0],[10]],
                            surfaces={'1':'Flood','2':'FLOOD'})})
        for kind in ['scores','structure']:
            a=f.features(data,'d','1','2',kind);b=f.features(data,'d','2','1',kind)
            np.testing.assert_allclose(a,b)
            self.assertEqual(len(a),2 if kind=='scores' else 9)

    def test_held_out_labels_cannot_change_that_folds_fit_or_threshold(self):
        docs=['1_a','2_a','3_a']
        data=SimpleNamespace(docs=docs,topic={d:d[0] for d in docs})
        rows=[dict(doc=d,topic=d[0],pair=[str(k),'x'],x=[v,float(d[0])],
                   fused=.1 if v<0 else .7,gold=v>0)
              for d in docs for k,v in enumerate([-2.,-1.,1.,2.])]
        changed=copy.deepcopy(rows)
        for row in changed:
            if row['topic']=='1':row['gold']=not row['gold']
        cfg=dict(l2_penalty=1)
        with patch.object(f,'plan',return_value={}),patch('sys.stdout',new_callable=io.StringIO):
            a=f.evaluate(cfg,data,rows);b=f.evaluate(cfg,data,changed)
        self.assertEqual(a['folds']['1'],b['folds']['1'])
        self.assertEqual(a['folds']['1']['training_documents'],['2_a','3_a'])
        self.assertEqual(a['folds']['1']['test_documents'],['1_a'])
        self.assertNotEqual(a['folds']['2']['model'],b['folds']['2']['model'])

    def test_fit_scaling_uses_only_supplied_training_rows(self):
        x=np.array([[-2.,0],[-1.,0],[1.,0],[2.,0]])
        y=np.array([0,0,1,1],bool)
        model=f.fit(x,y)
        np.testing.assert_allclose(model['mean'],[0,0])
        p=f.predict(model,x)
        self.assertTrue(np.all(np.diff(p)>0))
        self.assertLess(p[0],.5);self.assertGreater(p[-1],.5)
        self.assertTrue(np.isfinite(f.predict(model,[[1000.,1000.]])).all())

    def test_canonical_gold_mismatch_is_rejected(self):
        data=SimpleNamespace(docs=['d'],topic={'d':'1'},keys={'d':[('1','2'),('2','1')]},
            by_doc={'d':{('1','2'):dict(gold=True),('2','1'):dict(gold=False)}},
            unordered=lambda d:[('1','2')])
        with patch.object(f,'Dev44',return_value=data):
            with self.assertRaisesRegex(ValueError,'undirected'):
                f.population(dict(controls='ignored',suites_dir='ignored',feature_set='scores'))

    def test_entrypoint_blocks_network_and_restores_socket(self):
        original=socket.socket.connect
        def evaluate(*args):
            with socket.socket() as sock:
                with self.assertRaisesRegex(RuntimeError,'network forbidden'):
                    sock.connect(('127.0.0.1',1))
            return dict(api_calls=0)
        with tempfile.TemporaryDirectory() as tmp,patch.object(f,'load',return_value={}), \
                patch.object(f,'population',return_value=(None,[])),patch.object(f,'plan',return_value={}), \
                patch.object(f,'evaluate',side_effect=evaluate),patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(f.run('unused',Path(tmp)/'result.json'),0)
            self.assertTrue((Path(tmp)/'result.json').exists())
        self.assertIs(socket.socket.connect,original)

    def test_average_precision_respects_ties(self):
        self.assertEqual(f.ap(np.array([.9,.9,.1]),np.array([1,0,0],bool)),.5)
        self.assertEqual(f.ap(np.array([.9,.9,.1]),np.array([0,0,0],bool)),0)


if __name__=='__main__':unittest.main()
