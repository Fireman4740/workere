"""Few-shot demonstrations: the pool, the selection, the per-document plan, the CoT synthesis.

The plan is computed up front, deterministically
------------------------------------------------
For every evaluated document the demo sets are drawn once, before any inference, from an RNG
seeded by (config seed, doc id). Two agentere defects follow from doing it this way:

  * agentere drew with Python's unseeded global RNG, so no run was reproducible, and its CoT
    pre-generation pass drew a DIFFERENT random sample than inference later used: the
    pre-generated CoTs mostly went unused and inference synthesized its demos on the fly,
    outside the in-flight budget. Here pre-generation covers exactly the planned demos.
  * a document could be its own demonstration whenever the few-shot split is the evaluated
    split (ESC and CTB evaluate "train") and k-fold exclusion is off. A document is never
    offered its own gold here.

Selection (few_shot.selection): random | similarity (TF-IDF cosine on the text) | mentions
(Jaccard over mention surfaces) | bert (sentence-transformer cosine) | smallest (debugging);
optionally density-stratified (one demo per quantile bin of causal relations per mention).
With resampling.distinct_fewshots, n_runs * n_examples demos are drawn and dealt round-robin,
one set per pass, so each set still spans the ranking (or the density range).

CoT synthesis (few_shot.cot_generation): gold-guided pass, or blind attempt + gold correction
(`blind`), optionally followed by a decontamination rewrite (`rewrite`); multi-step prompts get
one guided turn per step, each decontaminated independently. `negation` also asks the
reasoning to argue the tempting norel pairs. Results are cached on disk under agentere's key
scheme (doc id + fingerprint of model, prompt and rendered demo input), so an agentere
cot_cache.json can be reused by copying it to few_shot.cot_generation.cache_path.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from eci import prompts as P
from eci.config import PromptSpec, n_folds, n_passes
from eci.data import Doc, label_set, load_split
from eci.labels import is_norel
from eci.llm import CallMeta

# ── CoT synthesis prompts (byte-identical to agentere: they are part of the cache contract,
#    and trace readers recognise synthesis calls by their prefixes) ──────────────────────────

COT_GOLD_HINT = (
    "The correct label assignments for this document are:\n\n{gold_json}\n\n"
    "Write your response for this step exactly as required — reasoning that leads naturally "
    "to these labels from the text alone. Do not mention or imply that you already know the answer."
)
COT_GOLD_HINT_CONTINUE = (
    "Continue. The correct labels are still:\n\n{gold_json}\n\n"
    "Write your response for this step, consistent with the gold labels and your prior reasoning. "
    "Do not reveal foreknowledge."
)
COT_DECONTAM_PROMPT = (
    "Rewrite your response from scratch to remove all privileged knowledge. "
    "Replace any phrasing that reveals foreknowledge "
    "(e.g. \"we see in the JSON that X is NoRel\", \"the answer shows\", "
    "\"as indicated by the labels\", \"we know that\", \"the expected output\") "
    "with genuine reasoning grounded in the text and rules. "
    "Every sentence should read as if you discovered the label through analysis, "
    "not confirmed it from a pre-known answer. Output only the rewritten response, nothing else."
)
COT_CORRECT_HINT = (
    "The correct label assignments for this document are:\n\n{gold_json}\n\n"
    "Your response above was written without access to these labels. Revise it so the "
    "reasoning leads naturally to the correct labels — fix any wrong conclusions, add "
    "missing evidence, and correct the final answer. Keep anything that already agrees "
    "with the correct labels. Do not mention or imply that you were given the answer; "
    "write as though this is a refined version of your own analysis. "
    "Output the full revised response, nothing else."
)
# Appended (never prepended: the hint prefixes above identify synthesis calls in traces).
NEGATION_CLAUSE = (
    "\n\nThe label list above is exhaustive: by the omission rule, every ordered pair of "
    "mentions that does not appear in it is \"norel\".\n"
    "In Step 2, do not reason only about the pairs that ARE causal. Reason about rejected "
    "pairs as well:\n"
    "- Work through the pairs a careful reader would be most tempted to link — mentions "
    "that share a participant, sit adjacent in the text, stand in temporal succession, or "
    "are topically associated — and say, for each, what is missing that makes it norel "
    "rather than causal: succession without causation, correlation, a shared participant "
    "only, two descriptions of the same event, or plausibility with no mechanism stated in "
    "the text.\n"
    "- Give each rejection its own reason, the way each acceptance gets one. Do not emit a "
    "bare list of norel pairs.\n"
    "- Spend a comparable amount of Step 2 on rejected pairs as on accepted ones.\n"
    "Leave Step 3 as it is: the reverse-direction (\"causedby\") mirror of the accepted pairs."
)
NEGATION_DECONTAM_CLAUSE = (
    " Preserve the Step 2 reasoning about pairs judged norel, rejections included, at the "
    "same level of detail — rewrite those sentences so they read as your own analysis "
    "rather than dropping them."
)

TOOL_EXAMPLE = "--- Example {i} ---\nText:\n{text}\n\nPairs:\n{pairs}\n\nOutput:\n{output}"


# ── Hand-written rationales ───────────────────────────────────────────────────

_ANSWER = re.compile(r"^[ \t]*(?:Answer|Response|Rationale)[ \t]*:[ \t]*$", re.M)
_HEADER = re.compile(r"^[ \t]*(?:Document ID|Doc|Text)[ \t]*:[ \t]*(\S+)[ \t]*$", re.M)
_STEP = re.compile(r"^[ \t]*-{2,}[ \t]*(?:CoT[ \t]+)?[Ss]tep[ \t]*\d+[ \t]*-{2,}[ \t]*$", re.M)


def parse_rationale(path: Path) -> Tuple[str, List[str]]:
    """(doc_id, [assistant turn per prompt step]). The id is the file stem, or the value of a
    leading `Text:` / `Doc:` / `Document ID:` header; everything after an `Answer:` line is the
    assistant turn, split on `--- Step N ---` lines."""
    raw = path.read_text(encoding="utf-8")
    m = _ANSWER.search(raw)
    head, answer = (raw[:m.start()], raw[m.end():]) if m else ("", raw)
    doc_id = path.stem
    h = _HEADER.search(head if m else raw)
    if h:
        if not m:
            raise ValueError(f"{path} starts with a '{h.group(0).strip()}' header but has no "
                             f"'Answer:' line — the document text would become part of the answer")
        doc_id = h.group(1)
    parts = [p.strip() for p in _STEP.split(answer)]
    return doc_id, [p for p in parts if p] or [answer.strip()]


def load_rationales(dir_path: Optional[str]) -> Dict[str, List[str]]:
    if not dir_path or not Path(dir_path).is_dir():
        return {}
    out = {}
    for path in sorted(Path(dir_path).iterdir()):
        if path.is_file() and path.suffix.lower() in (".txt", ".md") and not path.name.startswith("."):
            doc_id, turns = parse_rationale(path)
            out[doc_id] = turns
    return out


def density(doc: Doc) -> float:
    """Causal relations per event mention."""
    n = len(doc.surfaces)
    return sum(1 for _, l, _ in doc.gold if not is_norel(l)) / n if n else 0.0


class FewShotBank:
    """The demo pool of one run, its selection index, the per-document plan and the CoTs."""

    def __init__(self, cfg: dict, prompt: PromptSpec):
        self.cfg = cfg
        self.fs = cfg["few_shot"]
        self.prompt = prompt
        self.ds = cfg["datasets"][cfg["active_dataset"]]
        self.pool: List[Doc] = []
        self.hand: Dict[str, List[str]] = {}
        self._tfidf = None
        self._tfidf_matrix = None
        self._bert = None
        self._bert_emb = None
        self._densities: List[float] = []
        self.plan: Dict[str, List[List[Doc]]] = {}   # doc id -> one demo list per set
        self.cot: Dict[str, List[str]] = {}          # cache key -> assistant turns

    # ── Pool ──────────────────────────────────────────────────────────────────

    def load(self, decorate) -> None:
        """Load and cut the pool, fit the selection index on pristine text, then `decorate`
        (syntax / coref blocks) the pool documents only."""
        cfg = self.cfg
        docs = load_split(self.fs["repo_id"] or self.ds["repo_id"], self.fs["split"],
                          text_field=self.ds["text_field"], ann_field=self.ds["ann_field"],
                          valid_labels=set(label_set(cfg)),
                          binary_undirected=cfg["data"]["binary_undirected"],
                          shuffle_pair_list=cfg["data"]["shuffle_pair_list"],
                          revision=self.ds.get("revision") if not self.fs["repo_id"] else None)
        if self.fs["human_rationales"]["only"]:
            self.hand = load_rationales(self.fs["human_rationales"]["dir"])
            docs = [d for d in docs if d.id in self.hand]
            if not docs:
                raise ValueError(f"few_shot.human_rationales.only: no rationale file matches a "
                                 f"document of the '{self.fs['split']}' split")
        else:
            if self.fs["shuffle_pool"]:
                random.Random(self.fs["shuffle_pool_seed"]).shuffle(docs)
            if self.fs["pool_size"]:
                docs = docs[: self.fs["pool_size"]]
        self.pool = docs

        selection = self.fs["selection"]
        if selection == "similarity":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer()
            self._tfidf_matrix = self._tfidf.fit_transform([d.text for d in docs])
        elif selection == "bert":
            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer
            self._bert = SentenceTransformer(self.fs["bert_model"],
                                             device="cuda" if torch.cuda.is_available() else "cpu")
            emb = self._bert.encode([d.text for d in docs], convert_to_numpy=True, show_progress_bar=True)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self._bert_emb = emb / np.where(norms == 0, 1, norms)
        if self.fs["density_stratify"]:
            self._densities = [density(d) for d in docs]
        decorate(docs)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _scores(self, doc: Doc, idx: List[int]):
        selection = self.fs["selection"]
        if selection == "similarity":
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(self._tfidf.transform([doc.text]), self._tfidf_matrix[idx])[0]
        if selection == "mentions":
            q = frozenset(doc.surfaces.values())
            out = []
            for i in idx:
                s = frozenset(self.pool[i].surfaces.values())
                union = q | s
                out.append(len(q & s) / len(union) if union else 0.0)
            return out
        if selection == "bert":
            import numpy as np
            q = self._bert.encode([doc.text], convert_to_numpy=True)
            q = q / np.linalg.norm(q, keepdims=True).clip(min=1e-12)
            return (self._bert_emb[idx] @ q.T).ravel()
        return None

    def select(self, doc: Doc, n: int, rng: random.Random) -> List[Doc]:
        folds = n_folds(self.cfg)
        idx = [i for i, d in enumerate(self.pool) if d.id != doc.id
               and (folds < 2 or d.idx % folds != doc.idx % folds)]
        if self.fs["density_stratify"]:
            order = sorted(idx, key=lambda i: self._densities[i])
            n_bins = min(n, len(order))
            bounds = [round(b * len(order) / n_bins) for b in range(n_bins + 1)]
            scores = self._scores(doc, idx)
            score_of = dict(zip(idx, scores)) if scores is not None else None
            picks = []
            for b in range(n_bins):
                in_bin = order[bounds[b]:bounds[b + 1]] or order
                picks.append(max(in_bin, key=lambda i: score_of[i]) if score_of else rng.choice(in_bin))
            return [self.pool[i] for i in picks]
        if self.fs["selection"] == "smallest":
            non_empty = [self.pool[i] for i in idx if self.pool[i].pair_list]
            return sorted(non_empty, key=lambda d: len(d.pair_list))[:n]
        scores = self._scores(doc, idx)
        if scores is not None:
            top = sorted(range(len(idx)), key=lambda j: scores[j], reverse=True)[:n]
            return [self.pool[idx[j]] for j in top]
        return rng.sample([self.pool[i] for i in idx], min(n, len(idx)))

    def n_sets(self) -> int:
        rs = self.cfg["experiment"]["resampling"]
        return n_passes(self.cfg) if (rs["distinct_fewshots"] and n_passes(self.cfg) > 1) else 1

    def build_plan(self, docs: List[Doc]) -> None:
        n, n_sets = self.fs["n_examples"], self.n_sets()
        for doc in docs:
            rng = random.Random(f"{self.cfg['seed']}::{doc.id}")
            chosen = self.select(doc, n * n_sets, rng)
            if n_sets > 1 and len(chosen) < n * n_sets:
                raise ValueError(f"resampling.distinct_fewshots needs {n * n_sets} demos "
                                 f"({n_sets} passes x {n}) but only {len(chosen)} are eligible for {doc.id}")
            self.plan[doc.id] = [chosen[i::n_sets] for i in range(n_sets)]

    # ── Demo content ──────────────────────────────────────────────────────────

    def demo_input(self, demo: Doc) -> Tuple[str, str]:
        """(human message, gold answer) of one demonstration — exactly what the synthesis LLM
        and the cache key see."""
        cfg = self.cfg
        human = P.render(self.prompt, demo, demo.pair_list,
                         binary_undirected=cfg["data"]["binary_undirected"],
                         constrain=self.ds["constrain_to_pair_list"])
        gold = demo.gold
        if self.fs["intra_only"]:
            gold = P.same_sentence(gold, demo.sentence_of)
        return human, P.gold_output(gold, bool(demo.pair_list))

    def cot_key(self, demo: Doc, human: str, gold_output: str) -> str:
        """<doc_id>::<fingerprint>. Same fields and hashing as agentere's _cot_cache_key, so an
        agentere cache stays reachable; `negation` is added only when on, for the same reason."""
        cfg, gen = self.cfg, self.fs["cot_generation"]
        fields = {
            "model_id": cfg["model"]["default_model_id"],
            "temperature": cfg["model"]["temperature"],
            "blind": gen["blind"],
            "rewrite": gen["rewrite"],
            "intra_only": self.fs["intra_only"],
            "dataset": cfg["active_dataset"],
            "system": self.prompt.system,
            "user_template": self.prompt.user_template,
            "steps": cfg["prompt"].get("steps"),
            "binary_undirected": cfg["data"]["binary_undirected"],
            "constrain_to_pair_list": self.ds["constrain_to_pair_list"],
        }
        if cfg["model"]["reasoning_effort"] is not None:
            fields["reasoning_effort"] = cfg["model"]["reasoning_effort"]
        if gen["negation"]:
            fields["negation"] = True
        if demo.syntax_block:
            fields["syntax_block"] = sha256(demo.syntax_block.encode("utf-8")).hexdigest()[:16]
        fields["user_content"] = sha256(human.encode("utf-8")).hexdigest()[:16]
        fields["gold_output"] = sha256(gold_output.encode("utf-8")).hexdigest()[:16]
        fp = json.dumps(fields, sort_keys=True, ensure_ascii=False)
        return f"{demo.id}::{sha256(fp.encode('utf-8')).hexdigest()[:16]}"

    def n_turns(self) -> int:
        return len(self.prompt.steps) + 1 if self.prompt.steps else 1

    def turns(self, demo: Doc) -> List[str]:
        """The assistant turns of one demonstration (hand-written, synthesized or plain gold)."""
        human, gold = self.demo_input(demo)
        if demo.id in self.hand:
            hand = self.hand[demo.id]
            if len(hand) != self.n_turns():
                raise ValueError(f"the rationale for {demo.id} has {len(hand)} turn(s) but the "
                                 f"prompt needs {self.n_turns()} — separate them with '--- Step N ---'")
            return hand
        if self.fs["cot_generation"]["enabled"]:
            return self.cot[self.cot_key(demo, human, gold)]
        return [gold] + [""] * (self.n_turns() - 1)

    def demo_sets(self, doc: Doc) -> List[P.DemoSet]:
        return [[P.demo_exchanges(self.demo_input(d)[0], self.turns(d), self.prompt.steps)
                 for d in demo_set] for demo_set in self.plan[doc.id]]

    def tool_examples(self, doc: Doc) -> str:
        """The few_shot_examples tool's output: the document's first planned set, as text."""
        cfg = self.cfg
        parts = []
        for i, d in enumerate(self.plan[doc.id][0], 1):
            gold = P.same_sentence(d.gold, d.sentence_of) if self.fs["intra_only"] else d.gold
            parts.append(TOOL_EXAMPLE.format(
                i=i, text=d.prompt_text,
                pairs=P.pair_lines(d, d.pair_list, cfg["data"]["binary_undirected"],
                                   self.ds["constrain_to_pair_list"]),
                output=P.gold_output(gold, bool(d.pair_list))))
        return "\n\n".join(parts)

    # ── CoT synthesis ─────────────────────────────────────────────────────────

    def load_cache(self) -> None:
        path = self.fs["cot_generation"]["cache_path"]
        if path and Path(path).exists():
            self.cot.update(json.loads(Path(path).read_text(encoding="utf-8")))
            print(f"[few_shot] loaded {len(self.cot)} CoT entries from {path}")

    def save_cache(self) -> None:
        path = self.fs["cot_generation"]["cache_path"]
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(path) + ".tmp")
            tmp.write_text(json.dumps(self.cot, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)   # atomic: a crash mid-write cannot corrupt the cache

    def missing_cots(self) -> List[Doc]:
        if not (self.fs["cot_generation"]["enabled"] and self.cfg["few_shot"]["systematic"]):
            return []
        seen, out = set(), []
        for sets in self.plan.values():
            for demo_set in sets:
                for d in demo_set:
                    if d.id in seen or d.id in self.hand:
                        continue
                    seen.add(d.id)
                    if self.cot_key(d, *self.demo_input(d)) not in self.cot:
                        out.append(d)
        return out

    async def synthesize(self, client, demo: Doc, tools, retries: int, retry_wait: float,
                         dump_dir: Optional[Path]) -> None:
        human, gold = self.demo_input(demo)
        gen = self.fs["cot_generation"]
        neg = gen["negation"]
        gold_hint = COT_GOLD_HINT.format(gold_json=gold) + (NEGATION_CLAUSE if neg else "")
        cont_hint = COT_GOLD_HINT_CONTINUE.format(gold_json=gold)
        correct_hint = COT_CORRECT_HINT.format(gold_json=gold) + (NEGATION_CLAUSE if neg else "")
        decontam = COT_DECONTAM_PROMPT + (NEGATION_DECONTAM_CLAUSE if neg else "")
        system = self.prompt.system

        async def ask(msgs, step: str) -> str:
            out, _ = await client.call(msgs, CallMeta(role="cot", doc_id=demo.id, block=step),
                                       retries=retries, retry_wait=retry_wait, tools=tools)
            return out[-1].content

        if not self.prompt.steps:
            if gen["blind"]:
                base = [SystemMessage(content=system), HumanMessage(content=human)]
                blind_raw = await ask(base, "blind")
                convo = base + [AIMessage(content=blind_raw), HumanMessage(content=correct_hint)]
                raw = await ask(convo, "correct")
            else:
                convo = [SystemMessage(content=system), HumanMessage(content=human),
                         HumanMessage(content=gold_hint)]
                raw = await ask(convo, "guided")
            if gen["rewrite"]:
                clean = [await ask(convo + [AIMessage(content=raw), HumanMessage(content=decontam)],
                                   "decontaminate")]
            else:
                clean = [raw]
        else:
            convo = [SystemMessage(content=system), HumanMessage(content=human),
                     HumanMessage(content=gold_hint)]
            raws = [await ask(convo, "guided-0")]
            convo.append(AIMessage(content=raws[-1]))
            for i, step in enumerate(self.prompt.steps, 1):
                convo += [HumanMessage(content=step["prompt"]), HumanMessage(content=cont_hint)]
                raws.append(await ask(convo, f"guided-{i}"))
                convo.append(AIMessage(content=raws[-1]))
            clean = []
            for i, raw in enumerate(raws):
                if gen["rewrite"]:
                    clean.append(await ask([SystemMessage(content=system), HumanMessage(content=raw),
                                            HumanMessage(content=decontam)], f"decontaminate-{i}"))
                else:
                    clean.append(raw)

        self.cot[self.cot_key(demo, human, gold)] = clean
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w.-]", "_", demo.id) or "unknown"
            cot_text = ("\n\n".join(f"--- CoT step {i} ---\n{s}" for i, s in enumerate(clean, 1))
                        if len(clean) > 1 else clean[0])
            (dump_dir / f"{safe}.txt").write_text(
                f"Document ID: {demo.id}\n\n--- Document text ---\n{demo.prompt_text}\n\n"
                f"--- Synthesized CoT ---\n{cot_text}\n", encoding="utf-8")

    async def pregenerate(self, client, tools_for, retries: int, retry_wait: float,
                          dump_dir: Optional[Path]) -> None:
        """Synthesize every planned demo that the cache does not hold, concurrently (the client's
        in-flight budget bounds it), saving the cache after each success so a late failure
        loses nothing already paid for."""
        self.load_cache()
        todo = self.missing_cots()
        if not todo:
            print(f"[few_shot] all {sum(len(s) for v in self.plan.values() for s in v)} planned "
                  f"demos have their CoT cached")
            return
        print(f"[few_shot] synthesizing CoT for {len(todo)} demo documents")
        lock = asyncio.Lock()

        async def one(demo: Doc):
            await self.synthesize(client, demo, tools_for(demo), retries, retry_wait, dump_dir)
            async with lock:
                self.save_cache()

        results = await asyncio.gather(*[one(d) for d in todo], return_exceptions=True)
        failed = [(d.id, r) for d, r in zip(todo, results) if isinstance(r, BaseException)]
        if failed:
            raise RuntimeError(f"CoT synthesis failed for {len(failed)} demo(s): "
                               + "; ".join(f"{i}: {e}" for i, e in failed[:5]))
