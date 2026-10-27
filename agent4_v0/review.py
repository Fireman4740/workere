"""Import an external independent HUMAN review; never synthesize an approval."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from .common import digest, load, require, rows, jsonl, save, sha
from .kit import pilot


def certify(baseline, selection, proposals, receipt, out):
    out = Path(out)
    review = load(receipt)
    require(review.get('reviewer_type') == 'human' and review.get('independent') is True,
            'BLOCKED_HUMAN_REVIEW: independent human required')
    require(review.get('blind_to_gold') is True and review.get('blind_to_model_outputs') is True,
            'BLOCKED_HUMAN_REVIEW: both blinding attestations required')
    require(review.get('proposals_sha256') == sha(proposals), 'review binds different proposals')
    entries = rows(proposals)
    decisions = {r['proposal_sha256']: r for r in review['decisions']}
    require(len(decisions) == len(review['decisions']), 'duplicate review decision')
    require(set(decisions) == {r['proposal_sha256'] for r in entries}, 'review must cover every proposal')
    checked = []
    for r in entries:
        require(digest({k:v for k,v in r.items() if k != 'proposal_sha256'}) == r['proposal_sha256'],
                'proposal hash mismatch')
        require(review.get('reviewer') and review['reviewer'] != r['proposer'], 'reviewer is proposer')
        decision = decisions[r['proposal_sha256']]
        require(decision['decision'] in ('approved','rejected','uncertain'), 'unknown decision')
        if decision['decision'] == 'approved':
            require(decision.get('semantic_checks_complete') is True and bool(decision.get('comment')),
                    'approval needs semantic checks and written justification')
        checked.append({**r, 'review_status': decision['decision'], 'reviewer': review['reviewer'],
                        'reviewed_blind_to_gold': True, 'reviewed_blind_to_model_outputs': True,
                        'review_receipt_sha256': sha(receipt)})
    jsonl(out / 'reviewed-edits.jsonl', checked)
    require(not (out / 'tasks.jsonl').exists(), 'new review needs fresh output directory')
    pilot.prepare(SimpleNamespace(baseline=str(baseline), selection=str(selection),
        edits=str(out/'reviewed-edits.jsonl'), out_tasks=str(out/'tasks.jsonl'),
        out_audit=str(out/'edit-audit.json'), reads=1, original_reads=2))
    save(out/'review-certification.json', {'review_receipt_sha256': sha(receipt),
         'tasks_sha256': sha(out/'tasks.jsonl'), 'baseline_sha256': sha(baseline),
         'selection_sha256': sha(selection), 'reviewed_edits_sha256': sha(out/'reviewed-edits.jsonl'),
         'reviewer': review['reviewer'], 'independent_human_attestation': True,
         'note': 'This records a human attestation, not automatic certification of semantic truth.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','selection','proposals','receipt','out'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    certify(a.baseline,a.selection,a.proposals,a.receipt,a.out)


if __name__ == '__main__':
    main()
