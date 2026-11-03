"""Tracked D0/E6 retrospective cohort analysis. Zero generation calls."""
import json
import os
from pathlib import Path
from pipelines.jev.n13 import save, export
from .contracts import require, digest
from .diagnostics import diagnose
from .mbr import decode
from .evaluation import metrics, freeze_predictions

P = Path("experiments/jev/research")


def main():
    require("ORX_RUN_ID" in os.environ,"tracked execution required")
    protocol = json.loads((P/"cached-protocol.json").read_text())
    bank = json.loads((P/"cached-readings.json").read_text())
    require(digest(bank) == protocol["readings_sha256"],"mutated input bank")
    state = Path(protocol["state_root"]).expanduser()/os.environ["ORX_RUN_ID"]
    state.mkdir(parents=True,exist_ok=False)
    save(state/"protocol.json",protocol)
    readings = bank["readings"]
    docs = {d["id"]:d for d in bank["public"]}
    families = sorted({r["family"] for r in readings})
    common = set(docs)
    for family in families:
        for calendar in ("0","1"):
            for slot in range(3):
                rs = [r for r in readings if (r["family"],r["calendar_id"],r["slot_id"]) == (family,calendar,slot) and r["status"] == "valid"]
                common &= {r["doc_id"] for r in rs}
    require(common,"empty paired cohort")
    for d in common:
        require(len({r["document_sha256"] for r in readings if r["doc_id"] == d}) == 1,"incompatible candidates/revision")
    candidates = {(bank["revision"],d,*tuple(sorted(p))) for d in common for p in docs[d]["pair_list"]}
    save(state/"cohort-before-gold.json",dict(intersection=sorted(common),exclusions=sorted(set(docs)-common),
        complete_corpus=False, selection="all families, 2 calendars, 3 slots valid; no target labels"))
    hypotheses = {}
    for f in families:
        for c in ("0","1"):
            for slot in range(3):
                rs = [r for r in readings if (r["family"],r["calendar_id"],r["slot_id"]) == (f,c,slot) and r["doc_id"] in common]
                require(len(rs) == len(common) and all(r["status"] == "valid" for r in rs),"incomplete reference")
                hypotheses[f,c,slot] = {(bank["revision"],r["doc_id"],*p) for r in rs for p in r["relations"]}
    systems, details = {}, {}
    for f in families:
        for c in ("0","1"):
            hs = [hypotheses[f,c,s] for s in range(3)]
            union = set().union(*hs)
            systems[f+":"+c+":majority"] = {p for p in union if sum(p in h for h in hs) >= 2}
            for neutral in (False,True):
                result = decode(hs,neutral_cardinalities=neutral)
                key = f+":"+c+(":MBR-neutral" if neutral else ":MBR-batch")
                systems[key] = set(result["predictions"]); details[key] = result
            result = set()
            for d in common:
                result.update(decode([{p for p in h if p[1] == d} for h in hs])["predictions"])
            systems[f+":"+c+":MBR-document"] = result
    # All decisions frozen before the gold file is opened by this controller.
    frozen = {k:sorted(v) for k,v in systems.items()}
    save(state/"predictions-before-gold.json",frozen)
    save(state/"freeze.json",dict(predictions_sha256=digest(frozen),gold_opened=False))
    gold_rows = json.loads((P/"cached-gold.json").read_text())
    gold = {(bank["revision"],r["doc"],r["a"],r["b"]) for r in gold_rows if r["doc"] in common and r["gold"]}
    scores = {k:metrics(v,gold,candidates) for k,v in systems.items()}
    controls = json.loads((P/"cached-controls.json").read_text())
    verified = []
    for name,expected in controls.items():
        if name not in systems:
            continue
        for doc,counts in expected.items():
            if doc not in common:
                continue
            observed = metrics({p for p in systems[name] if p[1] == doc},{p for p in gold if p[1] == doc},
                               {p for p in candidates if p[1] == doc})
            require([observed[k] for k in ("TP","FP","FN")] == counts,"historical control reconstruction mismatch")
            verified.append([name,doc])
    ordinary = {k:v for k,v in systems.items() if k.endswith(":majority")}
    diagnostic = diagnose(ordinary,gold,candidates)
    raw_diagnostic = diagnose({":".join(map(str,k)):v for k,v in hypotheses.items()},gold,candidates)
    result = dict(status="COMPLETE_RETROSPECTIVE_DIAGNOSTIC",documents=sorted(common),
        excluded=sorted(set(docs)-common),candidate_pairs=len(candidates),systems=scores,diagnostic=diagnostic,
        raw_reading_diagnostic=raw_diagnostic,
        source_run=bank["source_run"],new_generation_calls=0,promotion=False,
        exact_historical_control_checks=len(verified),
        scope="paired intersection only; exposed development; MBR is transductive; not external validation",
        reference_ablation="self-included, within family/calendar; separate-bank and leave-self-out require development selection",
        threshold_comparison="not fitted on this exposed short diagnostic; no superiority claim over internally optimized threshold")
    save(state/"metrics.json",result)
    save(state/"mbr-details.json",details)
    print("RESEARCH_CACHED_RESULT "+json.dumps(result),flush=True)
    export(state,"CACHED_STUDY")


if __name__ == "__main__":
    main()
