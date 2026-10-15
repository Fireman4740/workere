"""Generate synthetic YAML examples for all three protocols, campaigns and sweeps."""
import argparse
from pathlib import Path
from subprocess import run
import sys


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    root = Path(parser.parse_args().output_dir).resolve()
    run([sys.executable, '-m', 'benchmark.examples.demo', '--output-dir', str(root)], check=True)
    def spec(plugin, **params):
        return {'plugin': plugin, 'version': '1', 'params': params}
    for protocol in ('entity_extraction', 'relations_gold', 'end_to_end'):
        pipeline = {'entity_extractor': spec('reference') if protocol == 'relations_gold' else
                    spec('lexicon', terms=['rain', 'flooding'])}
        if protocol != 'entity_extraction':
            pipeline.update(candidate_generator=spec('historical' if protocol == 'relations_gold' else 'all_ordered'),
                            relation_predictor=spec('heuristic'))
        cfg = {'schema': 'workere-experiment/v1', 'name': 'synthetic-' + protocol, 'seeds': [7, 42, 2026],
               'dataset': {'name': 'synthetic', 'split': 'test', 'suite': 'suite.json'},
               'protocol': protocol, 'pipeline': pipeline, 'models': [{'id': 'heuristic', 'version': '1'}],
               'evaluation': {'bootstrap_samples': 100}, 'output_dir': 'framework-runs'}
        (root / (protocol + '.yaml')).write_text(yaml.safe_dump(cfg, sort_keys=False))
    campaign = {'schema': 'workere-campaign/v1', 'experiments': [{'config': p + '.yaml'}
                for p in ('entity_extraction', 'relations_gold', 'end_to_end')]}
    (root / 'campaign.yaml').write_text(yaml.safe_dump(campaign, sort_keys=False))
    sweep = {'schema': 'workere-sweep/v1', 'base': 'relations_gold.yaml',
             'grid': {'pipeline.relation_predictor.params': [{'confidence': .8}, {'confidence': .9}]}}
    (root / 'sweep.yaml').write_text(yaml.safe_dump(sweep, sort_keys=False))
    print('Synthetic examples only; no measured corpus performance. Run benchmark batch ' + str(root / 'campaign.yaml'))


if __name__ == '__main__':
    main()
