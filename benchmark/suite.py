"""Freeze the evaluation inputs independently of prompts, providers and tracking."""
from __future__ import annotations

import copy
from dataclasses import asdict

from benchmark.data import Doc, doc_from_row, label_set, load_eval_docs, sample
from benchmark.io import fingerprint

SCHEMA = "workere-benchmark-suite/v1"
PROTOCOL = "workere-historical-relations/v1"


def evaluation_config(cfg):
    active = cfg["active_dataset"]
    ds = cfg["datasets"][active]
    keys = ("name", "repo_id", "split", "text_field", "ann_field", "max_examples", "kfold",
            "constrain_to_pair_list")
    return copy.deepcopy({"seed": cfg["seed"], "active_dataset": active,
                          "datasets": {active: {k: ds[k] for k in keys if k in ds}},
                          "data": cfg["data"],
                          "experiment": {"resampling": cfg["experiment"]["resampling"]}})


def freeze(cfg, rows=None):
    """Rows, when supplied, must be the FULL split in original order, before sampling."""
    cfg = evaluation_config(cfg)
    ds, data = cfg["datasets"][cfg["active_dataset"]], cfg["data"]
    if rows is None:
        docs = load_eval_docs(cfg)
    else:
        docs = [doc_from_row(row, i, text_field=ds["text_field"], ann_field=ds["ann_field"],
                             valid_labels=set(label_set(cfg)),
                             binary_undirected=data["binary_undirected"],
                             shuffle_pair_list=data["shuffle_pair_list"])
                for i, row in enumerate(rows)]
        docs = sample(docs, ds["max_examples"], data["shuffle_docs"], cfg["seed"])
    folds = ds.get("kfold", {})
    n = folds["n_folds"] if folds.get("enabled") else 1
    for doc in docs:
        doc.fold = doc.idx % n if n > 1 else -1
    if not docs or len({d.id for d in docs}) != len(docs):
        raise ValueError("suite must contain documents with unique IDs")
    suite = {"schema": SCHEMA, "protocol": PROTOCOL, "config": cfg,
             "documents": [asdict(d) for d in docs]}
    suite["fingerprint"] = fingerprint(suite)
    return suite


def documents(suite):
    body = {k: v for k, v in suite.items() if k != "fingerprint"}
    if suite.get("schema") != SCHEMA or suite.get("protocol") != PROTOCOL:
        raise ValueError("unsupported suite schema or evaluation protocol")
    if suite.get("fingerprint") != fingerprint(body):
        raise ValueError("suite fingerprint mismatch")
    docs = [Doc(**d) for d in suite["documents"]]
    if not docs or len({d.id for d in docs}) != len(docs):
        raise ValueError("suite must contain documents with unique IDs")
    return docs


def model_inputs(suite, task="relations"):
    """Gold-free inputs. Mention extraction sees only raw tokens and sentence boundaries."""
    for doc in documents(suite):
        row = {"id": doc.id, "tokens": doc.tokens, "sentences": doc.sentences, "lang": doc.lang}
        if task == "relations":
            row.update(text=doc.text, mentions=doc.mentions, spans=doc.spans,
                       surfaces=doc.surfaces, pair_list=doc.pair_list)
        elif task != "event_mentions":
            raise ValueError(f"unknown task: {task}")
        yield row
