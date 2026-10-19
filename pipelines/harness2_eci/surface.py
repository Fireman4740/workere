"""Map a Harness² component tree (OpenCode surface, c1..c8) onto the two ECI executors.

Neither executor has a tool loop: the always-on text is composed into the model request,
skills are inlined, and `scripts/postprocess.py` is the only executable lever. The base tree
of each executor reproduces the frozen R3 request bytes exactly, so base rollouts reuse caches.
"""
import re

from pipelines.jev.state_builder import NOUL_CRITERIA, NOUL_INSTRUCTIONS

SYSTEM, GUARDRAILS, MEMORY, GUIDANCE = ("systemprompt.md", "guardrails.md", "LongTermMEMORY.md",
                                        "tool_descriptions/tool_guidance.md")
POSTPROCESS = "scripts/postprocess.py"
DOC_MARK = "Text:\n{doc_text}"
DECISION_TEXT_MAX = 6000
_SKILL = re.compile(r"skills/([a-z0-9][a-z0-9_\-]*)/SKILL\.md")
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_CRITERION = re.compile(r"^(true|false):\s*(\S.*)$")


def _unescape(text):
    return text.replace("{{", "{").replace("}}", "}")


def split_user_template(template):
    """(editable rules before the document, fixed tail after it), with format braces resolved."""
    if template.count(DOC_MARK) != 1 or "{pair_lines}" in template or "{doc_id}" in template:
        raise ValueError("user template must contain the document slot exactly once and no other slot")
    prefix, suffix = template.split(DOC_MARK)
    return _unescape(prefix), _unescape(suffix)


def skill_bodies(tree):
    """Skill bodies inlined in name order: without a tool loop a skill cannot load on demand."""
    out = []
    for rel in sorted(tree):
        match = _SKILL.fullmatch(rel)
        if match:
            out.append(f"### Procedure: {match.group(1)}\n{_FRONTMATTER.sub('', tree[rel]).strip()}")
    return out


def _sections(tree):
    parts = []
    for rel, title in ((GUARDRAILS, "Standing rules"), (MEMORY, "Durable domain knowledge")):
        if tree.get(rel, "").strip():
            parts.append(f"## {title}\n{tree[rel].strip()}")
    skills = skill_bodies(tree)
    if skills:
        parts.append("## Procedures\n" + "\n\n".join(skills))
    return parts


def luna_base_tree(prompt):
    rules, _ = split_user_template(prompt["user_template"])
    return {SYSTEM: prompt["system"], GUARDRAILS: "", MEMORY: "", GUIDANCE: rules}


def luna_request_parts(tree, prompt):
    """(system message, user prefix, user suffix): the document goes between prefix and suffix."""
    _, suffix = split_user_template(prompt["user_template"])
    extra = _sections(tree)
    system = tree.get(SYSTEM, "")
    if extra:
        system = system.rstrip("\n") + "\n\n" + "\n\n".join(extra) + "\n"
    return system, tree.get(GUIDANCE, ""), suffix


def decision_base_tree():
    criteria = "".join(f"{key}: {NOUL_CRITERIA[key]}\n" for key in ("true", "false"))
    return {SYSTEM: NOUL_INSTRUCTIONS, GUARDRAILS: "", MEMORY: "", GUIDANCE: criteria}


def parse_criteria(text):
    found = {}
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        match = _CRITERION.match(line)
        if not match or match.group(1) in found:
            raise ValueError(
                f"{GUIDANCE} must hold exactly two non-empty lines, `true: <when the pair is causal>` "
                f"and `false: <when it is not>`; offending line: {line[:120]!r}")
        found[match.group(1)] = match.group(2)
    if set(found) != {"true", "false"}:
        raise ValueError(f"{GUIDANCE} must define both `true:` and `false:` criteria lines")
    return found


def decision_question(tree):
    """The noul question sent with every pair. Base tree -> the frozen R3 question exactly."""
    instructions = tree.get(SYSTEM, "").strip()
    extra = _sections(tree)
    if extra:
        instructions = instructions + "\n\n" + "\n\n".join(extra)
    if not instructions:
        raise ValueError(f"{SYSTEM} must state the decision instruction; it is empty")
    if len(instructions) > DECISION_TEXT_MAX:
        raise ValueError(
            f"decision instruction text is {len(instructions)} characters (system prompt + guardrails + "
            f"memory + skills); the decision model accepts at most {DECISION_TEXT_MAX}. Shorten it.")
    return {"causal": {"type": "noul", "instructions": instructions,
                       "criteria": parse_criteria(tree.get(GUIDANCE, ""))}}


def validate(tree, executor):
    """Every problem in actionable prose; Harness² feeds the message back to the editor."""
    problems = []
    for rel in sorted(tree):
        if rel.startswith("plugin/") or rel.startswith("sub_agents/"):
            problems.append(f"{rel}: this executor answers in a single model request with no tool loop, "
                            "so plugins and subagents never run. Remove it and put the guidance in "
                            "always-on text or scripts/postprocess.py.")
        elif rel.startswith("scripts/") and rel != POSTPROCESS:
            problems.append(f"{rel}: only {POSTPROCESS} is executed (on the model's raw pair scores). "
                            "Merge this logic into it or remove the file.")
    if not tree.get(SYSTEM, "").strip():
        problems.append(f"{SYSTEM} is empty; the executor needs its instruction text.")
    if POSTPROCESS in tree and "def main" not in tree[POSTPROCESS] and "__main__" not in tree[POSTPROCESS]:
        problems.append(f"{POSTPROCESS} must be a runnable script reading JSON on stdin and writing JSON "
                        "on stdout (see the contract in its docstring requirements).")
    if executor == "decision":
        try:
            decision_question(tree)
        except ValueError as exc:
            problems.append(str(exc))
    elif DOC_MARK in tree.get(GUIDANCE, "") or "{doc_text}" in tree.get(GUIDANCE, ""):
        problems.append(f"{GUIDANCE} must not contain the document slot; the document is appended "
                        "after it automatically.")
    if problems:
        raise ValueError("Harness rejected:\n- " + "\n- ".join(problems))
