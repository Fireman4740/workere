"""Local offline CLI; deliberately has no model-provider execution path."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .core import ARMS, Document, View, aggregate, compact, prepare_view, validate_pair_view
from .eval import evaluate


def read_jsonl(path: str) -> list[dict]:
    output=[]
    with open(path, encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: output.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError("%s line %d: %s"%(path,i,e)) from e
    return output


def write_jsonl(path: Path, objs: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for o in objs: f.write(compact(o)+"\n")


def private_view_from_json(raw: dict) -> View:
    return View(**raw)


def do_prepare(args: argparse.Namespace) -> None:
    docs=[Document.from_public_json(x) for x in read_jsonl(args.input)]
    if len(set(d.doc_id for d in docs)) != len(docs):
        raise ValueError("duplicate doc IDs")
    arms=args.arms
    if not arms or len(set(arms)) != len(arms) or args.passes != 7:
        raise ValueError("This protocol requires distinct arms and exactly seven slots")
    if args.max_aux_chars < 0:
        raise ValueError("max-aux-chars must be nonnegative")
    output=Path(args.out)
    output.mkdir(exist_ok=True,parents=True)
    private=[];public_jobs=[];checks=[]
    for d in docs:
        prepared={}
        selected=None
        if "A" in arms and "R" in arms:
            # Selection is common and cut by a conservative character proxy.
            from .core import choose_mentions
            selected=choose_mentions(d)
            while selected:
                av=prepare_view(d,"A",selected)
                rv=prepare_view(d,"R",selected)
                longest=max(len(compact(av.auxiliary_records)),
                            len(compact(rv.auxiliary_records)))
                if longest<=args.max_aux_chars: break
                selected=selected[:-1]
        for arm in arms:
            v=prepare_view(d,arm,selected=selected if arm in ("A","R") and selected is not None else None)
            prepared[arm]=v
            private.append(v.to_json())
            public_payload={
                "document_text":v.document_text,
                "mentions":v.mentions,
                "candidates":v.candidates,
                "auxiliary_records":v.auxiliary_records,
                "auxiliary_instruction":v.common_instruction,
                "output_contract":"JSON list of [source_id,causes|causedby,target_id]; [] valid",
            }
            for p in range(args.passes):
                public_jobs.append({
                    "doc_id":d.doc_id,"arm":arm,"pass_index":p,
                    "demo_slot":p,"provider_adapter_required":True,
                    "public_input":public_payload,
                })
        if "A" in prepared and "R" in prepared:
            checks.append(validate_pair_view(d,prepared["A"],prepared["R"]))
    write_jsonl(output/"private_views.jsonl",private)
    write_jsonl(output/"public_jobs.jsonl",public_jobs)
    write_jsonl(output/"ar_checks.jsonl",checks)
    (output/"STATUS.txt").write_text(
        "PROTOCOL PREPARED. ZERO MODEL CALLS.\n"
        "Do NOT send private_views.jsonl wholesale to a model.\n"
        "You must add historical system prompt and out-of-topic demos via your provider adapter.\n"
        "Character caps are not exact tokenizer caps; verify on target model.\n"
        "A/R records are the same except handle IDs.\n",encoding="utf-8")
    print("PREPARED",len(docs),"docs",len(private),"views",len(public_jobs),
          "model jobs",len(checks),"A/R checks; 0 calls sent")


def do_aggregate(args: argparse.Namespace) -> None:
    raw_views = read_jsonl(args.views)
    views={(v["doc_id"],v["arm"]):private_view_from_json(v) for v in raw_views}
    if len(views) != len(raw_views):
        raise ValueError("duplicate view identity")
    slots: dict[tuple[str,str],dict[int,object]]={key:{} for key in views}
    for row in read_jsonl(args.responses):
        key=(row["doc_id"],row["arm"])
        if key not in views:
            raise ValueError("response without view: "+str(key))
        p=row["pass_index"]
        if type(p) is not int or p<0 or p>=args.passes:
            raise ValueError("invalid pass")
        if p in slots[key]:
            raise ValueError("duplicate response: "+str((key,p)))
        status=row.get("status","ok")
        if status not in ("ok","error"):
            raise ValueError("status must be ok/error")
        if status == "ok" and row.get("response") is None:
            raise ValueError("ok response missing")
        slots[key][p]=row.get("response") if status=="ok" else None
    results=[]
    for key,by_pass in slots.items():
        if set(by_pass) != set(range(args.passes)):
            raise ValueError("Unexecuted response slots (not legitimate errors): "+str(key))
        results.append(aggregate([by_pass[i] for i in range(args.passes)],
                                 views[key],args.passes,args.threshold_votes))
    write_jsonl(Path(args.out),results)
    print("AGGREGATED",len(results),"views (not verified as real LLM outputs)")


def do_evaluate(args: argparse.Namespace) -> None:
    views=[private_view_from_json(x) for x in read_jsonl(args.views)]
    candidates={};topics={}
    for v in views:
        c={tuple(x) for x in v.original_candidates}
        if v.doc_id in candidates and candidates[v.doc_id]!=c:
            raise ValueError("candidate universe differs between arms")
        candidates[v.doc_id]=c
        topics[v.doc_id]=v.topic
    report=evaluate(read_jsonl(args.gold),read_jsonl(args.results),candidates,topics)
    Path(args.out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("EVALUATED",args.out,"CAUTION: only real responses justify real scores")


def do_reserve(args: argparse.Namespace) -> None:
    def ids(p):return {x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()}
    reserved=ids(args.reserve_ids);seen=ids(args.h1_ids)
    overlap=sorted(reserved & seen)
    report={"reserve_ids":len(reserved),"h1_ids":len(seen),
            "overlap":len(overlap),"overlap_ids":overlap,
            "status":"not_untouched" if overlap else "no_overlap_in_supplied_manifests",
            "caveat":"This check does not establish human gold-label inspection."}
    Path(args.out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("HOLDOUT",len(overlap),"of",len(reserved),"reserved IDs seen in H1")


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=["A","R"])
    p.add_argument("--passes",type=int,default=7)
    p.add_argument("--max-aux-chars",type=int,default=3072,
                   help="preflight only; must count tokens with target model")
    p.set_defaults(fn=do_prepare)
    p=sub.add_parser("aggregate")
    p.add_argument("--views",required=True)
    p.add_argument("--responses",required=True)
    p.add_argument("--passes",type=int,default=7)
    p.add_argument("--threshold-votes",type=int,default=4)
    p.add_argument("--out",required=True)
    p.set_defaults(fn=do_aggregate)
    p=sub.add_parser("evaluate")
    p.add_argument("--views",required=True)
    p.add_argument("--results",required=True)
    p.add_argument("--gold",required=True)
    p.add_argument("--out",required=True)
    p.set_defaults(fn=do_evaluate)
    p=sub.add_parser("audit-reserve")
    p.add_argument("--reserve-ids",required=True)
    p.add_argument("--h1-ids",required=True)
    p.add_argument("--out",required=True)
    p.set_defaults(fn=do_reserve)
    args=ap.parse_args()
    try:args.fn(args)
    except (KeyError,ValueError) as e:
        print("ERROR:",e,file=sys.stderr)
        raise SystemExit(2)

if __name__=="__main__":
    main()
