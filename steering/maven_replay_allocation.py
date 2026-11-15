"""Classify a finished Slurm GPU allocation without importing model libraries."""
import argparse
import csv
import io
from pathlib import Path


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
