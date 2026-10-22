import copy
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from pipelines.jev.cli import _configs, _execute_parallel
from pipelines.jev.confirm_luna import validate

class LunaConfirmation(unittest.TestCase):
    def test_campaign_caps_in_flight_document_batches(self):
        lock=threading.Lock();active=peak=0
        def execute(configs, **kwargs):
            nonlocal active,peak
            with lock:
                active+=1;peak=max(active,peak)
            time.sleep(.01)
            with lock:active-=1
            return {'runs':[{'status':'completed','id':configs[0]}]}
        with patch('benchmark.campaign.execute_many',side_effect=execute):
            result=_execute_parallel(list(range(40)),max_parallel=8)
        self.assertEqual(len(result['runs']),40)
        self.assertLessEqual(peak,8)
        self.assertGreater(peak,1)

    def test_frozen_reference_rejects_scientific_setting_change(self):
        path=Path('experiments/jev/luna-confirmation.yaml')
        configs=_configs(path)
        self.assertEqual(len(configs),40)
        tampered=copy.deepcopy(configs)
        tampered[0].data['pipeline']['relation_predictor']['params']['client']['reasoning_effort']='low'
        with self.assertRaisesRegex(ValueError,'scientific settings differ'):validate(path,tampered)

if __name__=='__main__':unittest.main()
