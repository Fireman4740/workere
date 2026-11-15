"""Classify a finished Slurm GPU allocation without importing model libraries."""
import argparse
import csv
import io
from pathlib import Path


def gpu_resources(cfg):
    kind = cfg.get('gpu_kind', 'L40S')
    if kind == 'L40S':
        return dict(GPU_KIND=kind, GPU_PARTITION='gpu_l40s_interactive',
                    GPU_CPUS=8, GPU_MEM='96G')
    if kind == 'H100':
        return dict(GPU_KIND=kind, GPU_PARTITION='gpu_h100_interactive',
                    GPU_CPUS=4, GPU_MEM='64G')
    raise ValueError('GPU kind has not passed the site runtime preflight')


def allocation_state(accounting, job_id):
    for row in csv.DictReader(io.StringIO(accounting), delimiter='|'):
        if row.get('JobID') == str(job_id):
            return row.get('State', '').split('+')[0].split(' ')[0].upper()
    return ''


def resume_reason(return_code, state):
    if state == 'TIMEOUT':
        return 'allocation_expired'
    if return_code == 75 and state in ('', 'COMPLETED', 'FAILED'):
        return 'planned_checkpoint'
    return ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounting', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--return-code', type=int, default=0)
    parser.add_argument('--field', choices=('state', 'reason'), required=True)
    args = parser.parse_args()
    state = allocation_state(Path(args.accounting).read_text(), args.job_id)
    print(state if args.field == 'state' else resume_reason(args.return_code, state))


if __name__ == '__main__':
    main()
