"""Deterministic TF-IDF pair retrieval from a frozen, disjoint training pool."""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from benchmark.data import parse_annotations
from benchmark.labels import is_norel


def words(text):
    stop = {"the", "and", "was", "were", "with", "that", "this", "from", "for", "has", "had", "have", "are", "its", "into", "said"}
    return Counter(w for w in re.findall(r"[a-z]{2,}", text.lower()) if w not in stop)


class RetrievedDemonstrations:
    def __init__(self, path, selection_path="experiments/jev/document-selection.json", per_class=2):
        rows = json.loads(Path(path).read_text())
        selection = json.loads(Path(selection_path).read_text())
        excluded = set(selection["diagnostic_d15"]) | {d["id"] for d in selection["confirmation"]["docs"]}
        if any(r["id"] in excluded for r in rows):
            raise ValueError("demonstration pool overlaps a protected evaluation set")
        self.items, self.per_class = [], per_class
        for row in sorted(rows, key=lambda r: r["id"]):
            labels = {(a, b): not is_norel(lab) for a, lab, b in parse_annotations(row["annots"])}
            mids = row["mentions"]
            spans = {m: span for m, span in zip(mids, row["spans"])}
            tokens, sentences = row["tokens"], row["sentences"]
            for i, a in enumerate(mids):
                for b in mids[i + 1:]:
                    selected = [s for s, (start, end) in enumerate(sentences)
                                if any(start <= spans[m][0] < end for m in (a, b))]
                    snippet = " ".join(" ".join(tokens[sentences[s][0]:sentences[s][1]]) for s in selected)
                    sa = " ".join(tokens[t] for t in spans[a]); sb = " ".join(tokens[t] for t in spans[b])
                    text = f"Context: {snippet}\nEvent A: {sa}\nEvent B: {sb}"
                    self.items.append({"id": row["id"], "topic": row["id"].split("_")[0],
                        "source": a, "target": b, "positive": labels.get((a, b), False) or labels.get((b, a), False),
                        "text": text, "words": words(text)})
        df = Counter(w for item in self.items for w in item["words"])
        self.idf = {w: 1 + math.log((1 + len(self.items)) / (1 + n)) for w, n in df.items()}
        self.index = defaultdict(list)
        for i, item in enumerate(self.items):
            item["norm"] = math.sqrt(sum((n * self.idf[w]) ** 2 for w, n in item["words"].items())) or 1
            for w, n in item["words"].items():
                self.index[w].append((i, n * self.idf[w] / item["norm"]))

    def retrieve(self, document, a, b):
        selected = {document.sentence_of(e.token_indices) for e in (a, b)} - {None}
        text = " ".join(" ".join(document.tokens[document.sentences[s][0]:document.sentences[s][1]]) for s in sorted(selected))
        query = words(text + " " + " ".join(document.tokens[t] for e in (a, b) for t in e.token_indices))
        scores = defaultdict(float)
        for w, n in query.items():
            for i, weight in self.index.get(w, []):
                scores[i] += n * self.idf[w] * weight
        topic = document.id.split("_")[0]
        eligible = [i for i, item in enumerate(self.items) if item["topic"] != topic and item["id"] != document.id]
        eligible.sort(key=lambda i: (-scores[i], self.items[i]["id"], self.items[i]["source"], self.items[i]["target"]))
        result = []
        for positive in (False, True):
            chosen = [self.items[i] for i in eligible if self.items[i]["positive"] == positive][:self.per_class]
            if len(chosen) != self.per_class:
                raise ValueError("insufficient balanced training demonstrations outside evaluation topic")
            result += chosen
        return result

    def prefix(self, document, a, b):
        demos = self.retrieve(document, a, b)
        return "Training examples from other topics (A/B labels local to each example):\n\n" + "\n\n".join(
            f"Example {i + 1} ({d['id']}, mentions {d['source']}/{d['target']}):\n{d['text']}\n"
            f"Causal relationship: {'yes' if d['positive'] else 'no'}" for i, d in enumerate(demos)) + "\n\nTarget document:\n"
