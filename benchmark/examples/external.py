"""Offline integration skeleton for an encoder, local classifier or decision model.

python -m benchmark.examples.external --inputs inputs.jsonl --output predictions.jsonl
Replace classify() with your model. This toy baseline always predicts norel.
"""
import argparse

from benchmark.io import read_jsonl, write_jsonl
from benchmark.labels import NOREL


def classify(tokens, spans, source, target):
    return NOREL


def predict(row, classifier=classify):
    spans = dict(zip(row["mentions"], row["spans"]))
    triples = []
    for source, target in row["pair_list"]:
        label = classifier(row["tokens"], spans, source, target)
        if label != NOREL:
            triples.append([source, label, target])
    return {"id": row["id"], "triples": triples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_jsonl(args.output, (predict(row) for row in read_jsonl(args.inputs)))


if __name__ == "__main__":
    main()
