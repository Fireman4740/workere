"""Public inputs, canonical candidates and evidence. No evaluation labels here."""
from dataclasses import dataclass
import hashlib
import json
import math

VERSION = "research-v1"
STATUSES = {"not_submitted", "submitted", "valid", "invalid_schema", "truncated", "transport_failed"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pair_key(pair):
    require(isinstance(pair, (list, tuple)) and len(pair) == 2, "pair must have two IDs")
    require(all(isinstance(x, str) and x for x in pair) and pair[0] != pair[1], "invalid mention IDs")
    return tuple(sorted(pair))


@dataclass(frozen=True)
class Document:
    doc_id: str
    corpus_revision: str
    group_id: str
    tokens: tuple
    mentions: tuple
    sentences: tuple
    candidate_pairs: frozenset

    @classmethod
    def load(cls, row):
        require(not {"gold", "labels", "relations"}.intersection(row), "public input contains target labels")
        ids = [m["id"] for m in row["mentions"]]
        require(len(ids) == len(set(ids)), "duplicate mention")
        tokens = tuple(row["tokens"])
        require(tokens and all(isinstance(t, str) for t in tokens), "corpus tokens required")
        for m in row["mentions"]:
            indices = m["token_indices"]
            require(indices and indices == sorted(set(indices)), "invalid discontinuous span")
            require(all(type(i) is int and 0 <= i < len(tokens) for i in indices), "mention outside document")
        cursor = 0
        for a, b in row["sentences"]:
            require(type(a) is int and type(b) is int and a == cursor < b <= len(tokens), "invalid sentence partition")
            cursor = b
        require(cursor == len(tokens), "incomplete sentence partition")
        candidates = frozenset(pair_key(p) for p in row["candidate_pairs"])
        require(all(set(p) <= set(ids) for p in candidates), "candidate outside inventory")
        require(all(isinstance(row[k], str) and row[k] for k in ("doc_id", "corpus_revision", "group_id")), "missing provenance")
        return cls(row["doc_id"], row["corpus_revision"], row["group_id"], tokens,
                   tuple(row["mentions"]), tuple(map(tuple, row["sentences"])), candidates)

    @property
    def identity(self):
        return digest(dict(revision=self.corpus_revision, doc=self.doc_id, tokens=self.tokens,
                           mentions=self.mentions, sentences=self.sentences,
                           candidates=sorted(self.candidate_pairs)))

    def pairs(self, relations):
        result = {pair_key(p) for p in relations}
        require(result <= self.candidate_pairs, "relation outside candidate contract")
        return result

    def evidence(self, spans):
        require(isinstance(spans, list) and spans, "nonempty evidence required")
        result = []
        for s in spans:
            a, b = s["start"], s["end"]
            require(type(a) is int and type(b) is int and 0 <= a < b <= len(self.tokens), "evidence outside document")
            text = " ".join(self.tokens[a:b])
            require("text" not in s or s["text"] == text, "nonliteral evidence")
            result.append(dict(start=a, end=b, text=text))
        return result


def load_public(manifest):
    documents = [Document.load(r) for r in manifest]
    require(len({(d.corpus_revision, d.doc_id) for d in documents}) == len(documents), "duplicate document")
    return documents


def build_request(view, demonstrations, model_contract, *, logical_id, calendar_id, slot_id, replicate_id):
    """Demonstrations are explicitly allowed public messages, excluded by group."""
    require(all(d["group_id"] != view["group_id"] for d in demonstrations), "demonstration group leakage")
    body = dict(model_contract["parameters"])
    body["messages"] = [m for d in demonstrations for m in d["messages"]] + view["messages"]
    return dict(logical_id=logical_id, calendar_id=calendar_id, slot_id=slot_id, replicate_id=replicate_id,
                body=body, body_sha256=digest(body), model_contract_sha256=digest(model_contract),
                demo_ids=[d["id"] for d in demonstrations], view_sha256=digest(view),
                candidate_set_sha256=view["candidate_set_sha256"], parser=VERSION)


def validate_response(response, document):
    """New explicit sparse extraction contract; never alters historical parsers."""
    require(isinstance(response, dict) and response.get("complete") is True, "incomplete extraction")
    require(set(response) == {"complete", "relations"} and isinstance(response["relations"], list), "invalid extraction schema")
    return document.pairs(response["relations"])


def aggregate(readings, candidates, threshold=2, expected=3):
    require(len(readings) == expected and all(r["status"] == "valid" for r in readings), "invalid or missing reading")
    require(len({r["logical_id"] for r in readings}) == expected, "duplicate reading")
    sets = [{pair_key(p) for p in r["relations"]} for r in readings]
    require(all(s <= candidates for s in sets), "outside candidates")
    require(1 <= threshold <= expected, "invalid threshold")
    return [dict(pair=p, positive=sum(p in s for s in sets) >= threshold,
                 positive_votes=sum(p in s for s in sets), valid_readings=expected,
                 policy_id=f"majority-{threshold}of{expected}") for p in sorted(candidates)]
