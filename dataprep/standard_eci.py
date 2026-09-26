"""What every standard-eci builder shares: the `text` / `annots` columns, writing, publishing.

The `text` / `annots` logic is benchmark/dataprep_llm_format.py's (the step that added those two
columns to every published standard-eci dataset), kept verbatim so rebuilt datasets match.
"""
from __future__ import annotations

import os
from typing import Dict, List


def markup(tokens, spans, mentions) -> List[str]:
    """<mention_id surface> around every mention span."""
    text = list(tokens)
    for k, span in enumerate(spans):
        for i in span:
            if i - 1 not in span:
                text[i] = f"<{mentions[k]} {text[i]}"
            if i + 1 not in span:
                text[i] = f"{text[i]}>"
    return text


def annots_of(relations: Dict[str, List[List[int]]], spans, text) -> str:
    """Every `causes` item, then every `causedby` item: "<a ...> causes <b ...>, "."""
    out = ""
    for rel_type in ("causes", "causedby"):
        for a, b in relations[rel_type]:
            out += " ".join(text[i] for i in spans[a]) + f" {rel_type} " + " ".join(text[i] for i in spans[b]) + ", "
    return out


def with_text(row: dict) -> dict:
    """`row` plus its `text` and `annots` columns."""
    text = markup(row["tokens"], row["spans"], row["mentions"])
    return {**row, "text": " ".join(text), "annots": annots_of(row["relations"], row["spans"], text)}


def write(out: str, splits: Dict[str, List[dict]], features, card: str) -> None:
    """<out>/data/<split>-00000-of-00001.parquet per split, and the dataset card <out>/README.md.
    Columns come in the order `features` declares them."""
    from datasets import Dataset
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    for split, rows in splits.items():
        Dataset.from_list([{k: r[k] for k in features} for r in rows], features=features).to_parquet(
            os.path.join(out, "data", f"{split}-00000-of-00001.parquet"))
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(f"written: {out}/README.md and {out}/data/{{{','.join(splits)}}}-00000-of-00001.parquet")


def push(out: str, repo: str, private: bool, message: str, token: str) -> str:
    """Upload the folder `write` produced as one commit; parquet files the folder no longer has are
    deleted from the repo. Returns the dataset's URL."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=repo, repo_type="dataset", commit_message=message,
                      delete_patterns="data/*")
    return f"https://huggingface.co/datasets/{repo}"


def card_configs(splits) -> str:
    """The `configs:` block of a card's YAML header, one data file per split."""
    return "configs:\n- config_name: default\n  data_files:\n" + "".join(
        f"  - split: {s}\n    path: data/{s}-*\n" for s in splits)
