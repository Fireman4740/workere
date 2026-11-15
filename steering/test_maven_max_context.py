import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from steering import maven_replay as r
from steering.test_maven_replay import bases


class NativeContext(unittest.TestCase):
    def cfg(self):
        return dict(protocol=r.MAXCTX_VERSION,arm='MAXCTX',documents=200,passes=5,
            reference_completion_sha256=r.SOURCE_HASH,engine=dict(max_model_len=262144,concurrency=4),
            gpu_kind='H100',thinking_tokens=253952,answer_tokens=4096,allocation_seconds=43200,
            lease_seconds=18000,drain_seconds=19500,thinking_request_timeout_seconds=18000,
            request_timeout_seconds=1200)

    def test_native_config_is_specific_and_keeps_short_final_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg.json';cfg=self.cfg();p.write_text(json.dumps(cfg))
            self.assertEqual(r.config(p),cfg)
            for bad in [dict(protocol=r.VERSION),dict(thinking_tokens=262144),
                        dict(request_timeout_seconds=18000),dict(drain_seconds=18000)]:
                p.write_text(json.dumps(cfg|bad))
                with self.assertRaises(ValueError):r.config(p)

    def test_all_fixed_demos_fit_without_prompt_truncation(self):
        with patch.object(r.mp,'prompt_tokens',return_value=('prompt',[1]*3425)):
            jobs=r.maximum_context_jobs(bases(),object(),self.cfg())
        self.assertEqual(len(jobs),5)
        self.assertEqual(len({j['demo_id'] for j in jobs}),5)
        self.assertEqual({j['thinking'] for j in jobs},{253952})
        self.assertTrue(all(j['prompt_tokens']+j['thinking']+j['answer_tokens']+32<=262144 for j in jobs))
        with patch.object(r.mp,'prompt_tokens',return_value=('prompt',[1]*5000)):
            with self.assertRaises(ValueError):r.maximum_context_jobs(bases(),object(),self.cfg())

    def test_native_engine_only_changes_context_and_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp);(model/'config.json').write_text(json.dumps(dict(text_config=dict(max_position_embeddings=262144))))
            cfg=self.cfg()|dict(model_path=tmp)
            ref=dict(files={'config.json':'same'},container_sha256='same',served_model='same',
                     engine=dict(max_model_len=65536,concurrency=32,kv_cache_dtype='fp8'))
            current=copy.deepcopy(ref);current['engine'].update(max_model_len=262144,concurrency=4)
            r.check_maximum_engine(cfg,ref,current)
            bad=copy.deepcopy(current);bad['files']['config.json']='changed'
            with self.assertRaises(ValueError):r.check_maximum_engine(cfg,ref,bad)
            bad=copy.deepcopy(current);bad['engine']['kv_cache_dtype']='bf16'
            with self.assertRaises(ValueError):r.check_maximum_engine(cfg,ref,bad)
            (model/'config.json').write_text(json.dumps(dict(max_position_embeddings=131072)))
            with self.assertRaises(ValueError):r.check_maximum_engine(cfg,ref,current)

    def test_absolute_allocation_deadline_reserves_long_request_drain(self):
        cfg=self.cfg()
        self.assertEqual(r.work_deadline(cfg,100,1000,44200),18100)
        self.assertEqual(r.work_deadline(cfg,100,21000,44200),3800)
        self.assertLess(r.work_deadline(cfg,100,44201,44200),100)
        with self.assertRaises(ValueError):r.work_deadline(cfg,100,1000,None)


if __name__=='__main__':unittest.main()
