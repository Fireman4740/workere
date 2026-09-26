#!/usr/bin/env python3
"""Build the workere standard-eci datasets from source and push each to <account>/<name>-workere.

  python dataprep/publish.py --token hf_... --account <account>
  python dataprep/publish.py --token hf_... --account <account> --only esc ctb --private
  python dataprep/publish.py --token hf_... --account <account> --dry-run    # build and check, push nothing

  key     Hub repo                                             source (option; default from workere/)
  esc     <account>/EventStoryLine-0.9-standard-eci-workere    --esl-root   ../agentere/data/EventStoryLine
  ctb     <account>/CausalTimeBank-standard-eci-workere        --ctb-root   ../litt/Causal-TimeBank
  maven   <account>/MAVEN-ERE-standard-eci-workere             --maven-dir  ../litt/MAVEN-ERE/data/MAVEN_ERE

Every selected dataset is rebuilt into data/<name>-workere/ (parquet + dataset card) and that folder
is uploaded as one commit, so the Hub holds exactly what the builders produce now. The token is
checked first: it must be valid, not read-only, and <account> must be its user or one of its
organisations. The token may also come from HF_TOKEN instead of --token (keeps it out of the shell
history).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ctb_standard_eci
import esc_standard_eci
import maven_ere_standard_eci
from standard_eci import push

WORKERE = Path(__file__).resolve().parents[1]
DATASETS = {   # key: (builder module, source option, default source)
    "esc": (esc_standard_eci, "esl_root", WORKERE.parent / "agentere" / "data" / "EventStoryLine"),
    "ctb": (ctb_standard_eci, "ctb_root", WORKERE.parent / "litt" / "Causal-TimeBank"),
    "maven": (maven_ere_standard_eci, "maven_dir", WORKERE.parent / "litt" / "MAVEN-ERE" / "data" / "MAVEN_ERE"),
}


def check_token(token: str, account: str) -> str:
    """The token's user name, after checking it can write to `account`; exits otherwise."""
    from huggingface_hub import HfApi
    try:
        who = HfApi(token=token).whoami()
    except Exception as e:
        sys.exit(f"the Hugging Face token was rejected: {e}")
    names = {who["name"]} | {org["name"] for org in who.get("orgs", [])}
    if account not in names:
        sys.exit(f"the token belongs to {who['name']!r}, which cannot write to {account!r} "
                 f"(its accounts: {', '.join(sorted(names))})")
    if who.get("auth", {}).get("accessToken", {}).get("role") == "read":
        sys.exit("the token is read-only; create one with write access")
    return who["name"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=None, help="Hugging Face access token with write access (or set HF_TOKEN)")
    ap.add_argument("--account", required=True, help="user or organisation to publish under")
    ap.add_argument("--only", nargs="+", choices=list(DATASETS), default=list(DATASETS), help="datasets to publish (default: all)")
    ap.add_argument("--private", action="store_true", help="create the repos as private")
    ap.add_argument("--dry-run", action="store_true", help="check the token and build, but push nothing")
    for key, (_, option, default) in DATASETS.items():
        ap.add_argument("--" + option.replace("_", "-"), default=str(default), help=f"{key} source (default: %(default)s)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("give the token with --token or HF_TOKEN")
    user = check_token(token, args.account)
    print(f"token ok: {user}, publishing under {args.account}")

    built = []
    for key in args.only:
        module, option, _ = DATASETS[key]
        source = getattr(args, option)
        if not Path(source).exists():
            sys.exit(f"{key}: source not found at {source} (set --{option.replace('_', '-')})")
        out = WORKERE / "data" / f"{module.NAME}-workere"
        print(f"\n== {key}: building {out.relative_to(WORKERE)} from {source}")
        module.build(source, str(out))
        built.append((key, module.NAME, out))

    print()
    for key, name, out in built:
        repo = f"{args.account}/{name}-workere"
        if args.dry_run:
            print(f"{key}: would push {out.relative_to(WORKERE)} -> {repo}{' (private)' if args.private else ''}")
            continue
        url = push(str(out), repo, args.private, f"workere dataprep: {name}", token)
        print(f"{key}: pushed -> {url}")


if __name__ == "__main__":
    main()
