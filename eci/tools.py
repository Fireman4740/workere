"""The agentic tools, bound to one document.

agentere's tools read the current document from contextvars and their sub-LLMs from a global
config registry; here each tool is a closure over a `ToolContext` built for the document, so a
tool can never see another document's state — agentere's CoT synthesis, for one, ran its tools
under the context of whichever TEST document the pre-generation loop had set last.

Names, argument schemas and docstrings are agentere's verbatim: they are what the model reads
when it decides to call a tool, so changing them would change behaviour.

Tool sub-calls go through a ChatClient (traced and accounted — agentere never counted the
sub-calls of counterfactual_check, bare_causes or eci) with the prompt-keyed LLM cache, at
temperature 0, on the run's model (eci_extractor: tools_config.eci_extractor.model_id).
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from eci.data import Doc
from eci.llm import CallMeta
from eci.parse import first_json_object, outer_json_array

TOOL_NAMES = ("coherence", "counterfactual_check", "encoder", "bare_causes", "eci",
              "eci_extractor", "eci_extractor_intra", "few_shot_examples", "placeholder", "reprompt")


@dataclass
class ToolContext:
    doc: Doc
    user_prompt: str
    cfg: dict
    sub_llm: Any = None                   # ChatClient for counterfactual / bare_causes / eci
    extractor_llm: Any = None             # ChatClient for eci_extractor(_intra)
    fewshot: Any = None                   # FewShotBank
    encoder_preds: Optional[Callable[[], dict]] = None
    extractor_slots: Optional[asyncio.Semaphore] = None


async def _ask(client, ctx: ToolContext, name: str, system: str, user: str) -> str:
    out, _ = await client.call([SystemMessage(content=system), HumanMessage(content=user)],
                               CallMeta(role=f"tool:{name}", doc_id=ctx.doc.id),
                               retries=0, retry_wait=0)
    return out[-1].content


# ── placeholder / reprompt / few_shot_examples ────────────────────────────────

def _placeholder(ctx: ToolContext) -> BaseTool:
    @tool
    def placeholder(message: str = "No-op") -> str:
        """
        A no-op placeholder tool for testing tool routing.
        It MUST NOT be called by the model.
        """
        return f"[PlaceholderTool] Called successfully. Message: {message}"
    return placeholder


def _reprompt(ctx: ToolContext) -> BaseTool:
    @tool
    async def reprompt(comment: str = "") -> str:
        """
        Returns the original task prompt (text + pairs to classify) exactly as it was given.
        Call this to refocus on the task after a long reasoning chain or tool sequence.
        """
        return ctx.user_prompt
    return reprompt


def _few_shot_examples(ctx: ToolContext) -> BaseTool:
    @tool
    async def few_shot_examples(comment: str = "") -> str:
        """
        Retrieves labelled examples from the training set to ground your predictions.
        Call this before analysing the document.
        """
        return ctx.fewshot.tool_examples(ctx.doc)
    return few_shot_examples


# ── coherence ─────────────────────────────────────────────────────────────────

def check_coherence(pairs: List[Dict[str, str]], rules: List[dict]) -> str:
    """Two-mention (symmetry) and three-mention (transitivity) rules over a predicted graph."""
    if not rules:
        return "No rules configured. Cannot perform check."
    rel: Dict[Tuple[str, str], str] = {}
    for o in pairs or []:
        p = (o.get("pair") or "").strip()
        lab = (o.get("label") or "").strip()
        if "," not in p or not lab:
            continue
        a, b = (x.strip() for x in p.split(",", 1))
        if a and b:
            rel[(a, b)] = lab
    mentions = sorted({m for pair in rel for m in pair})

    def pick(m1, m2, m3, d):
        return {"12": (m1, m2), "21": (m2, m1), "13": (m1, m3), "31": (m3, m1),
                "23": (m2, m3), "32": (m3, m2)}[d]

    violations, triggered = [], 0
    for r in rules:
        neg = bool(r["negated"])
        if r["type"] == "two":
            r1, r2, d = r["if"]["rel"], r["then"]["rel"], r["then"]["dir"]
            for (m1, m2), lab in list(rel.items()):
                if lab != r1:
                    continue
                triggered += 1
                a, b = pick(m1, m2, None, d)
                found = rel.get((a, b))
                ok = (found == r2) if not neg else (found is not None and found != r2)
                if not ok:
                    violations.append((f"Two-mention rule: If '{m1}' to '{m2}' is '{r1}', then "
                                       f"'{a}' to '{b}' should be '{r2}'.", found))
        elif r["type"] == "three":
            r1, r2, r3, d = r["if1"]["rel"], r["if2"]["rel"], r["then"]["rel"], r["then"]["dir"]
            succ: Dict[str, List[str]] = {}
            for (a, b), lab in rel.items():
                if lab == r2:
                    succ.setdefault(a, []).append(b)
            for (m1, m2), lab in list(rel.items()):
                if lab != r1:
                    continue
                for m3 in succ.get(m2, []):
                    if m3 in (m1, m2):
                        continue
                    triggered += 1
                    a, b = pick(m1, m2, m3, d)
                    found = rel.get((a, b))
                    ok = (found == r3) if not neg else (found is not None and found != r3)
                    if not ok:
                        violations.append((f"Three-mention rule (Transitivity): If '{m1}'->'{m2}' is "
                                           f"'{r1}' and '{m2}'->'{m3}' is '{r2}', then '{a}'->'{b}' "
                                           f"should be '{r3}'.", found))
    if triggered == 0:
        return "No relations matched the rule premises, so no coherence checks were triggered."
    if not violations:
        return f"Checked {triggered} potential rule applications. No incoherence found; the graph is logically consistent."
    lines = [f"I found **{len(violations)} incoherence violations** out of {triggered} checks.\n",
             "Here are the specific issues detected:\n"]
    for desc, found in violations:
        lines += [f"- **Violation:** {desc}", "  - **Expected:** Relation between the indicated mentions.",
                  f"  - **Found:** {found if found is not None else '<missing>'}\n"]
    return "\n".join(lines)


def _coherence(ctx: ToolContext) -> BaseTool:
    rules = ctx.cfg["rule_sets"][ctx.cfg["datasets"][ctx.cfg["active_dataset"]]["rule_set"]] or []

    @tool
    def coherence(pairs: List[Dict[str, str]]) -> str:
        """
        Analyzes the logical coherence of extracted relations.
        This tool uses the rules defined for the currently active dataset in config.yaml.

        Args:
            pairs: A list of relations in the format [{"pair": "ID1,ID2", "label": "REL"}, ...].

        Returns:
            A human-readable summary of any incoherence found.
        """
        return check_coherence(pairs, rules)
    return coherence


# ── encoder ───────────────────────────────────────────────────────────────────

def _encoder(ctx: ToolContext) -> BaseTool:
    enc_cfg = ctx.cfg["encoder"]

    @tool
    async def encoder(comment: str = "") -> str:
        """
        Retrieves the encoder-based classifier's relation predictions
        for the current document. Call this tool with no arguments to get
        a second opinion from a trained neural model before making your
        final predictions.

        Returns:
            A summary listing each mention pair with the encoder's predicted
            label and per-class confidence scores.
        """
        try:
            data = ctx.encoder_preds()
        except FileNotFoundError as e:
            return f"Error: {e}"
        samples, labels = data.get("samples", {}), data.get("label_list", [])
        threshold = data.get("threshold", 0.5)
        doc_id = ctx.doc.id
        resolved = doc_id
        if doc_id not in samples:
            cands = [k for k in samples if doc_id in k or k in doc_id]
            if len(cands) == 1:
                resolved = cands[0]
            elif cands:
                return f"Ambiguous doc_id '{doc_id}'. Candidates: {cands[:5]}"
            else:
                return f"Document '{doc_id}' not found in encoder predictions. Available: {list(samples)[:10]}..."
        rows = samples[resolved]
        total = len(rows)
        if enc_cfg["filter_norel"]["enabled"]:
            cutoff = threshold - float(enc_cfg["filter_norel"]["delta"])
            rows = [r for r in rows if r.get("pred_label", "NoRel") != "NoRel"
                    or not r.get("pred_prob") or max(r["pred_prob"]) >= cutoff]
        lines = [f"Encoder classifier predictions for document \"{doc_id}\":\n"]
        for r in rows:
            probs = r.get("pred_prob", [])
            conf = (f"[{', '.join(f'{l}: {p:.2f}' for l, p in zip(labels, probs))}]"
                    if probs and labels and len(probs) == len(labels)
                    else f"(confidence: {max(probs) if probs else 0:.2f})")
            lines.append(f'  {r.get("src", "?")} ("{r.get("src_text", "")}") → {r.get("tgt", "?")} '
                         f'("{r.get("tgt_text", "")}"): {r.get("pred_label", "NoRel")} {conf}')
        if not rows:
            lines.append("  (no pairs)")
        if total - len(rows) > 0:
            lines.append(f"\n  [{total - len(rows)} highly-confident NoRel pairs omitted (far below threshold)]")
        return "\n".join(lines) + (f"\n\n[Model: {data.get('config', {}).get('model', '?')}, "
                                   f"threshold: {threshold:.3f}, labels: {labels}, "
                                   f"pairs shown: {len(rows)}/{total}]")
    return encoder


# ── LLM-backed tools ──────────────────────────────────────────────────────────

_CF_SYSTEM = ("You are a logical reasoning assistant. Your task is to perform 'but-for' counterfactual "
              "tests on event pairs. Do not provide labels like 'Cause' or 'NoRel'. Provide ONLY the "
              "factual necessity found in the text.")


def _counterfactual_check(ctx: ToolContext) -> BaseTool:
    @tool
    async def counterfactual_check(pair: str, context_text: str, event_i_text: Optional[str] = None,
                                   event_j_text: Optional[str] = None) -> str:
        """
        Performs a 'but-for' counterfactual analysis for a pair of event triggers.
        This tool answers if one event is a necessary precondition or cause for another.

        Args:
            pair: The event IDs being tested, e.g., "e1,e2" or "T1,T2".
            context_text: The document text provided in the prompt.
            event_i_text: The specific text/trigger span for the first event.
            event_j_text: The specific text/trigger span for the second event.

        Returns:
            A JSON string containing the 'but-for' results for both directions and reasoning.
        """
        if "," not in pair:
            return "Error: Pair parameter must be 'ID1,ID2'."
        ei, ej = (p.strip() for p in pair.split(",", 1))
        user = f"""
Context:
{context_text}

Analyze the dependency between:
Event A: {ei} (Text: "{event_i_text or 'Refer to context'}")
Event B: {ej} (Text: "{event_j_text or 'Refer to context'}")

Tasks:
1. Test (B without A): If Event A had not occurred, would Event B still have happened?
2. Test (A without B): If Event B had not occurred, would Event A still have happened?

Output ONLY a JSON object:
{{
  "pair": "{pair}",
  "reasoning": "brief semantic analysis",
  "b_is_dependent_on_a": "yes/no/unclear",
  "a_is_dependent_on_b": "yes/no/unclear"
}}
"""
        try:
            return json.dumps(first_json_object(await _ask(ctx.sub_llm, ctx, "counterfactual_check", _CF_SYSTEM, user)))
        except Exception as e:
            return f"Error executing counterfactual analysis: {e}"
    return counterfactual_check


_BARE_SYSTEM = (
    "You are a causal-semantics analyst. Your task is to read a document "
    "and extract ONLY the bare causal skeleton: the minimal set of core "
    "events and the causal relations between them.\n\n"
    "Rules:\n"
    "1. Identify the FEWEST events necessary to capture all causal "
    "information in the text (typically 3-8 events).\n"
    "2. Describe each event in ≤10 words, abstracting away specific "
    "names, dates, and counts unless they are causally relevant.\n"
    "3. For every causal link, state which event causes which and give a "
    "≤15-word basis grounded in the text.\n"
    "4. Do NOT invent causal links that are not supported by the text.\n"
    "5. Do NOT include temporal-only relations (e.g., 'A happened before B' "
    "without causation).\n"
    "6. Output ONLY valid JSON — no commentary, no markdown fences."
)


def _bare_causes(ctx: ToolContext) -> BaseTool:
    @tool
    async def bare_causes(context_text: str) -> str:
        """
        Extracts the bare causal semantics from a document.

        Use this tool to reduce a complex narrative to its core causal skeleton:
        a minimal set of abstract events and the causal links between them.
        All narrative detail, attribution, hedging, and temporal filler is stripped.
        The output helps you reason about which event pairs are causally connected
        before committing to specific relation labels.

        Args:
            context_text: The full document text to distil.

        Returns:
            A JSON string containing:
            - "events": a list of {"id": str, "description": str} with short
              abstract descriptions of the core events.
            - "causal_links": a list of {"cause": event_id, "effect": event_id,
              "basis": str} capturing why the link holds.
            - "summary": a one-sentence plain-language summary of the causal
              chain in the document.
        """
        user = f"""Document:
{context_text}

Extract the bare causal skeleton. Output a JSON object with exactly three keys:

{{
  "events": [
    {{"id": "E1", "description": "..."}},
    {{"id": "E2", "description": "..."}}
  ],
  "causal_links": [
    {{"cause": "E1", "effect": "E2", "basis": "..."}}
  ],
  "summary": "One sentence describing the overall causal chain."
}}"""
        try:
            return json.dumps(first_json_object(await _ask(ctx.sub_llm, ctx, "bare_causes", _BARE_SYSTEM, user)), indent=2)
        except json.JSONDecodeError as e:
            return f"Error: LLM returned invalid JSON — {e}"
        except Exception as e:
            return f"Error executing bare_causes analysis: {e}"
    return bare_causes


_ECI_SYSTEM = (
    "You are an Event Causality Identification (ECI) specialist.\n"
    "Your task has THREE ordered steps — follow them exactly:\n\n"
    "STEP 1 — Context Selection (sentence level):\n"
    "  Read the full document and identify the 1-4 sentences most\n"
    "  relevant to the TARGET mention's causal role. Prefer sentences\n"
    "  semantically close to the target trigger and containing causal\n"
    "  cues (e.g., 'caused', 'led to', 'because', 'result', 'due to').\n\n"
    "STEP 2 — Per-candidate evidence (word level):\n"
    "  For EACH candidate mention, determine whether causal evidence\n"
    "  connects it to the target. Extract the specific bridging words\n"
    "  or phrases that support or refute a causal link.\n\n"
    "STEP 3 — Binary Causal Judgement:\n"
    "  For each candidate, answer: 'Is there a causal relation between\n"
    "  the target and this candidate?' (Yes / No).\n"
    "  Do NOT determine direction or specific relation type.\n\n"
    "Output ONLY a valid JSON object — no markdown fences, no commentary."
)


def _eci(ctx: ToolContext) -> BaseTool:
    @tool
    async def eci(target_mention: str, target_text: str, all_mentions: List[Dict[str, str]],
                  context_text: str) -> str:
        """
        Identifies all event mentions causally related to a target mention.

        Inspired by the HOTECI generative ECI framework, this tool answers:
        "Which of the other mentions in this document are causally related
        to the target mention?" — regardless of direction or specific type.

        It also returns the key context sentences and words that ground each
        detected causal link, following the HOTECI principle that selecting
        important context improves causal judgement.

        Call this tool ONCE PER MENTION (linear complexity) instead of once
        per pair (quadratic complexity).

        Args:
            target_mention: The ID of the target event mention (e.g. "T3").
            target_text: The trigger/span text of the target mention (e.g. "earthquake").
            all_mentions: List of ALL other mentions in the document, each a dict
                          with keys "id" and "text".
                          Example: [{"id": "T0", "text": "collapse"}, {"id": "T1", "text": "flood"}]
            context_text: The full document text.

        Returns:
            A JSON string containing:
            - "target": the target mention ID
            - "causal_mentions": list of objects for each mention found to be
              causally related, each with:
                - "id": the related mention ID
                - "text": its trigger text
                - "confidence": "high" / "medium" / "low"
                - "important_words": salient bridging words
                - "reasoning": ≤1-sentence explanation
            - "non_causal_mentions": list of mention IDs with no causal link
            - "important_sentences": the 1-4 most relevant sentences selected
              from the document for the target mention
        """
        if not all_mentions:
            return json.dumps({"target": target_mention, "causal_mentions": [], "non_causal_mentions": [],
                               "important_sentences": [], "note": "No other mentions provided."})
        lines = "\n".join(f'  - {m["id"]} ("{m.get("text", "?")}")' for m in all_mentions)
        user = f"""Document:
{context_text}

TARGET mention: {target_mention} ("{target_text}")

Candidate mentions:
{lines}

For EACH candidate, determine whether there is a causal relation with the
TARGET (in either direction). Output JSON:

{{
  "target": "{target_mention}",
  "important_sentences": ["sentence1", "sentence2"],
  "causal_mentions": [
    {{
      "id": "mention_id",
      "text": "trigger text",
      "confidence": "high / medium / low",
      "important_words": ["word1", "word2"],
      "reasoning": "brief explanation"
    }}
  ],
  "non_causal_mentions": ["id1", "id2"]
}}"""
        try:
            data = first_json_object(await _ask(ctx.sub_llm, ctx, "eci", _ECI_SYSTEM, user))
            data.setdefault("target", target_mention)
            return json.dumps(data)
        except Exception as e:
            return json.dumps({"target": target_mention, "error": f"ECI analysis failed: {e}"})
    return eci


_EXTRACTOR_SYSTEM = (
    "You are an Event Causality Identification (ECI) specialist.\n"
    "You are given ONE target event mention and a list of candidate event\n"
    "mentions, all tagged <ID text> in the document. Use the exact IDs given\n"
    "— never invent, renumber, or paraphrase one.\n\n"
    "For EACH candidate, decide the causal relation to the target:\n"
    "  - \"causes\":    the TARGET causes/enables/leads to the candidate.\n"
    "  - \"causedby\":  the candidate causes/enables/leads to the TARGET.\n"
    "  - \"norel\":     no causal link either way.\n\n"
    "Two examples of common mistakes to avoid:\n\n"
    "NOT CAUSAL (commonly mislabeled as causal): \"The massacre contributed to\n"
    "calls for reprisals, leading to the 1779 Sullivan Expedition which drove\n"
    "the Iroquois out of western New York.\" Here 'contributed' and 'drove' are\n"
    "NOT directly causally linked — they are two separate links in the same\n"
    "chain (massacre -> reprisals -> Expedition -> drove out), each pointing to\n"
    "the Expedition, not to each other. Do not flatten a multi-step chain into\n"
    "a direct link between two events that are each only adjacent to a third.\n\n"
    "CAUSAL (commonly missed): \"...the air strikes flown during deny flight\n"
    "led to operation deliberate force, a massive nato bombing campaign in\n"
    "bosnia that played a key role in ending the war.\" Here 'operation'\n"
    "(deliberate force) DOES cause 'ending' (the war) — phrases like 'played a\n"
    "key role in' are just as causal as 'led to' or 'caused', and a generic\n"
    "trigger word like 'operation' can still be the causal subject.\n\n"
    "Output ONLY a JSON array, one entry per candidate, in the form\n"
    "{\"id\": \"<candidate id>\", \"relation\": \"causes\"|\"causedby\"|\"norel\"}.\n"
    "No markdown fences, no commentary, no reasoning fields."
)


async def _extract(ctx: ToolContext, document_text: str, mentions: List[Dict[str, str]],
                   same_sentence_only: bool) -> str:
    """Shared body of eci_extractor(_intra): one sub-call per target mention, concurrently on the
    extractor model, aggregated into one fact per unordered pair (first report wins, so two
    sub-calls disagreeing on direction cannot both enter), emitted in both directions."""
    if len(mentions) < 2:
        return json.dumps([])
    sent = ctx.doc.sentence_of

    def candidates(target):
        return [m for m in mentions if m["id"] != target["id"] and (
            not same_sentence_only or (target["id"] in sent and m["id"] in sent
                                       and sent[target["id"]] == sent[m["id"]]))]

    async def one(target) -> List[Tuple[str, str]]:
        cands = candidates(target)
        if not cands:
            return []
        lines = "\n".join(f'  - {c["id"]} ("{c.get("text", "?")}")' for c in cands)
        user = (f"Document:\n{document_text}\n\nTARGET mention: {target['id']} (\"{target.get('text', '?')}\")"
                f"\n\nCandidate mentions:\n{lines}\n\nFor each candidate, output its relation to the target.")
        async with ctx.extractor_slots:
            try:
                judgements = outer_json_array(await _ask(ctx.extractor_llm, ctx, "eci_extractor",
                                                         _EXTRACTOR_SYSTEM, user))
            except Exception as e:
                print(f"[eci_extractor] sub-call failed for target {target.get('id')}: {type(e).__name__}: {e}")
                return []
        facts = []
        for j in judgements:
            if not isinstance(j, dict) or not j.get("id") or j.get("relation") not in ("causes", "causedby"):
                continue
            facts.append((target["id"], j["id"]) if j["relation"] == "causes" else (j["id"], target["id"]))
        return facts

    canonical: Dict[frozenset, Tuple[str, str]] = {}
    for facts in await asyncio.gather(*[one(t) for t in mentions]):
        for cause, effect in facts:
            canonical.setdefault(frozenset((cause, effect)), (cause, effect))
    out = []
    for cause, effect in canonical.values():
        out += [{"pair": f"{cause},{effect}", "label": "causes"},
                {"pair": f"{effect},{cause}", "label": "causedby"}]
    return json.dumps(out)


def _eci_extractor(ctx: ToolContext) -> BaseTool:
    @tool
    async def eci_extractor(document_text: str, mentions: List[Dict[str, str]]) -> str:
        """
        Extracts ALL causal relations in a document. Call this tool exactly ONCE
        per document, passing every event mention — internally it evaluates each
        mention against all others concurrently (on a separate, smaller model)
        and aggregates the results itself, so none of that intermediate work
        enters your context.

        Its returned JSON array IS the final answer: already deduplicated,
        already bidirectional (causes + causedby). Forward it as-is rather than
        re-deriving or rewriting it.

        Args:
            document_text: The full document text with mentions tagged inline
                            as <ID text> (e.g. <e3 collapsed>).
            mentions: List of ALL event mentions in the document, each a dict
                      with keys "id" and "text".

        Returns:
            A JSON string: a flat array of
            {"pair": "A,B", "label": "causes"|"causedby"} objects.
        """
        return await _extract(ctx, document_text, mentions, same_sentence_only=False)
    return eci_extractor


def _eci_extractor_intra(ctx: ToolContext) -> BaseTool:
    @tool
    async def eci_extractor_intra(document_text: str, mentions: List[Dict[str, str]]) -> str:
        """
        Intra-sentence causal relation extractor. Same contract as `eci_extractor`
        — call it EXACTLY ONCE per document, passing every event mention — but it
        only predicts relations between mentions that occur in the SAME sentence.
        Cross-sentence pairs are dropped internally, never sent to the model, and
        never appear in the output, so you can safely pass the full mention list.

        Its returned JSON array IS the final answer: already deduplicated, already
        bidirectional (causes + causedby). Forward it as-is rather than re-deriving
        or rewriting it.

        Args:
            document_text: The full document text with mentions tagged inline
                            as <ID text> (e.g. <e3 collapsed>).
            mentions: List of ALL event mentions in the document, each a dict
                      with keys "id" and "text".

        Returns:
            A JSON string: a flat array of
            {"pair": "A,B", "label": "causes"|"causedby"} objects.
        """
        return await _extract(ctx, document_text, mentions, same_sentence_only=True)
    return eci_extractor_intra


_BUILDERS: Dict[str, Callable[[ToolContext], BaseTool]] = {
    "coherence": _coherence, "counterfactual_check": _counterfactual_check, "encoder": _encoder,
    "bare_causes": _bare_causes, "eci": _eci, "eci_extractor": _eci_extractor,
    "eci_extractor_intra": _eci_extractor_intra, "few_shot_examples": _few_shot_examples,
    "placeholder": _placeholder, "reprompt": _reprompt,
}


def build(names: List[str], ctx: ToolContext) -> List[BaseTool]:
    unknown = [n for n in names if n not in _BUILDERS]
    if unknown:
        raise ValueError(f"unknown tool(s) {unknown}; available: {list(TOOL_NAMES)}")
    return [_BUILDERS[n](ctx) for n in names]


def encoder_loader(path: str) -> Callable[[], dict]:
    cache: Dict[str, Any] = {}

    def load() -> dict:
        if "data" not in cache:
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"Encoder predictions file not found at '{path}'. "
                                        f"Set 'encoder.path' in config.yaml.")
            with open(path, encoding="utf-8") as f:
                cache["data"] = json.load(f)
        return cache["data"]
    return load
