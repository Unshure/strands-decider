#!/usr/bin/env python3
"""Generate hard decision questions over realistic documents through OpenRouter.

WHAT THIS SCRIPT NEEDS TO DO (read this before changing anything on the other box)
---------------------------------------------------------------------------------
Goal: training data like the unreleased stage 2 of decider-2b v11 -- realistic business
documents, each with 3-4 typed questions whose answers need careful reading. The data is
for Hobson, a System One model that answers typed decisions (yes/no "noul", or "choice"
among named options) in one forward pass. decider's writer was Qwen3.6-27B with thinking;
the default writer here is the same model, served through OpenRouter.

This is the second pilot. The first (gen_documents.py, Nova Premier on Bedrock) failed on
label quality: 7 of 30 hand-checked kept labels were wrong or ill-posed, because the
verifiers were the writer's own model and shared its misreadings. Documents were short
(median 536 words), a third of choice questions had letter-only option names, and many
questions were single-clause lookups. This version changes, in order of importance:
  * verification by a DIFFERENT model with thinking (--verify-models; default
    Qwen3.5-397B-A17B, a different size and generation from the writer);
  * thinking on for writer and verifiers;
  * validation: 700-2,200 words; meaningful option names (no "A", "Option B", "Both A and
    B"); each answer cites at least two sections (one for "underdetermined"); each
    choice question names its tempting wrong option ("lure"), which must be an option
    and must differ from the answer;
  * prompt: tradeoff/precedence answers must rest on a priority or precedence rule the
    document states; the case must not restate what the document says.

Stages, each resumable (rerun the same command to continue):
  1. write   For each document index, the SAME domain, kind and skills as the first pilot
             (same seed), so the two pilots compare document for document. The writer
             returns one document and one question per skill as JSON.
  2. verify  Every question is answered once by each model in --verify-models (twice by
             one model if only one is given), each in a fresh context, options shuffled,
             WITHOUT the writer's answer. A question is kept only if every answer agrees
             with the writer's.
  3. export  Kept questions become strands-decider training rows (JSONL, one Example per line:
             kind, state, instructions, options [[name, description], ...], label, task),
             split by domain: documents in HELDOUT_DOMAINS go to the eval file only.

Rules that must survive any fix:
  * NO JEVBENCH CONTENT. The skills were written from our own reading of which kinds of
    decision fail, not from benchmark items. Never paste benchmark items into prompts.
  * Every label needs the writer AND all verifiers to agree. Don't relax this to raise
    yield; a wrong label teaches the wrong reading, which is worse than no row.
  * Open-weight, Apache-2.0 models only (licensing). Qwen's API-only models ("max",
    "plus", "flash") are under Alibaba Cloud's terms instead, and Qwen3.8-2.4T-A95B has a
    custom licence. Check the licence before changing --writer / --verify-models.
  * Difficulty is measured on the training box, not here.

Things likely to need fixing on the other box:
  * API KEY: export OPENROUTER_API_KEY=... (never put it in this file).
  * PROVIDERS. OpenRouter routes each call to one of several hosts, some serving quantised
    weights. --quantizations (default bf16,fp16,fp32,fp8) excludes int4/int8 hosts; if a
    model then has no eligible host, calls fail with a 404/400 naming the provider
    filter. Widen the list, or pin hosts with --providers. The provider that served each
    call is recorded in docs.jsonl / verify.jsonl.
  * REASONING. Thinking is requested with {"reasoning": {"effort": ...}}. If a provider
    rejects it, try --reasoning-effort none (not recommended for verifiers).
  * JSON FROM THE WRITER. Malformed output is retried up to 3 times, then logged in
    failures.jsonl. If most documents fail, read those errors first.
  * RATE LIMITS. 429s are retried with backoff. Lower --workers if they persist.
  * COST. Each call asks OpenRouter to report its cost; the total is printed at the end.

After the second pilot (1/30 wrong labels, median 1,066 words; see
research/preregistrations/PREREGISTRATION-v16.md):
  * each question is assigned its type (~30% yes/no, the pilot's share; "underdetermined"
    always choice) and, if yes/no, its answer (50/50) -- the pilot got 34 "no" to 13
    "yes", and a yes/no target on every question made the writer use yes/no for 86% of
    them (documents doc00000-doc00070 of the v16 run were written that way). Questions
    that ignore their assigned type or answer are dropped at write;
  * writing and verification share one pool, and a document is verified as soon as it
    is written (verification jobs go first); --workers calls are in flight at once;
  * --max-cost stops starting new calls past a spend; rerun with a higher cap to resume.

--skill-set targeted swaps in a second set of ten skills aimed at where v16 still trails
(withdrawn instructions, nested exceptions, definitions, judging a described action,
facts across artefacts, the rule version in force, running totals, tie-breaks, missing
facts, boundary cases), with five extra document kinds that carry amendments and
bundled artefacts. --skill-set mixed adds back five v16 skills (all conditions, exception,
precedence, dates, lookup chain) so two thirds of questions are targeted. --skill-set weak
draws the mixed skills with the five v16 did worst on in the mixed pilot four times as
likely (63% of questions). The default, v16, reproduces the specs of data/generators/gen_v16/
exactly, as mixed does data/generators/gen_mixed_pilot/'s.

After the mixed pilot (2 of 30 hand-checked labels wrong, both a case silently omitting
a precondition the answer depends on, missed by writer and verifiers alike): the writer
must state every fact the deciding rules' conditions depend on, and the verifier must
not assume an unstated fact.

Usage:
    export OPENROUTER_API_KEY=...
    python gen_documents_openrouter.py --out gen_v16 --docs 1000 --seed 1 --max-cost 90
    python gen_documents_openrouter.py --out gen_targeted_pilot --docs 50 --seed 2 \\
        --skill-set targeted --max-cost 9
    # copy the whole output directory back to the training box.

Backends (llm_client.py): --backend openrouter (default), bedrock, or bedrock-mantle, or the
environment variable HOBSON_LLM_BACKEND. Bedrock takes a Bedrock API key in
AWS_BEARER_TOKEN_BEDROCK (or, on bedrock-runtime, the AWS credential chain if botocore is
installed) and --region. Its default models are not the ones that produced the committed
data; every row records backend, model and provider:
    export AWS_BEARER_TOKEN_BEDROCK=...
    python gen_documents_openrouter.py --backend bedrock --region us-west-2 --out gen_bedrock_pilot \\
        --docs 20 --price-in <USD per M input tokens> --price-out <USD per M output tokens> --max-cost 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# ---- what to write about (identical to gen_documents.py: same specs per index) ----------

DOMAINS = [
    "employee benefits",
    "travel and expenses",
    "procurement",
    "vendor contracts",
    "customer refunds and returns",
    "IT access control",
    "data retention and privacy",
    "information security incidents",
    "facilities and building access",
    "fleet vehicles",
    "warehouse operations",
    "product warranties",
    "subscription billing",
    "loan servicing",
    "insurance claims",
    "property leasing",
    "construction subcontracting",
    "clinical scheduling",
    "pharmacy dispensing",
    "school admissions",
    "university grading",
    "grant funding",
    "event ticketing",
    "airline disruption compensation",
    "hotel reservations",
    "software release management",
    "customer support escalation",
    "payroll",
    "equipment maintenance",
    "food safety",
    "laboratory safety",
    "export compliance",
    "nonprofit volunteering",
    "municipal permits",
]
HELDOUT_DOMAINS = ["insurance claims", "municipal permits", "laboratory safety", "payroll"]

KINDS = [
    "policy document with numbered sections",
    "service agreement",
    "internal procedure",
    "terms and conditions",
    "eligibility guidelines",
    "approval matrix with narrative rules",
    "rate card with conditions",
    "incident report",
    "email thread with an attached policy excerpt",
    "meeting minutes recording decisions",
    "FAQ with exceptions",
    "handbook chapter",
    "amended policy (original text plus a later amendment)",
    "request form with reviewer notes",
    "service-level agreement",
    "scheduling rules",
    "escalation runbook",
    "pricing schedule",
    "audit findings",
    "benefits summary",
    "onboarding checklist with conditions",
    "vendor evaluation memo",
]

SKILLS = {
    "all_conditions": "An action is permitted only if several conditions all hold; either one "
    "fails in a way that is easy to overlook, or all hold although one appears "
    "to fail on a first reading.",
    "exception": "A general rule is stated, and a later clause creates an exception that changes "
    "the answer for the case asked about.",
    "precedence": "Two provisions conflict for the case; the document says (elsewhere) which one "
    "prevails, e.g. an amendment over the original, or a schedule over the body.",
    "dates": "The answer needs date arithmetic: notice periods, business days, deadlines counted "
    "from an event, effective dates of changes.",
    "numbers": "The answer needs arithmetic over figures in the document: totals across line "
    "items, percentages, caps, thresholds, pro-rating.",
    "lookup_chain": "The answer needs two or three facts from different sections chained "
    "together (e.g. role -> approver -> that approver's limit).",
    "tradeoff": "Several options each satisfy some requirements; the document's stated "
    "priorities or weights decide which is best.",
    "underdetermined": "The document does not settle the question for the case, and the correct "
    "option is that it cannot be determined; OR it looks unsettled but a "
    "clause does settle it (choose one direction; vary it across documents).",
    "surface_trap": "A prominent keyword, number or heading points to a wrong answer; only careful "
    "reading of the relevant clause gives the right one.",
    "rubric": "The document defines criteria or a rubric; a described case must be classified "
    "under it, and at least one criterion is borderline.",
}

# Targeted set (draft, after v16). Written from where v16 still trails, at the level of
# JevBench *family* descriptions and our own error analysis -- never from benchmark
# items -- plus v16's weakest skills on its held-out generated eval (underdetermined
# 0.71, tradeoff 0.75, numbers 0.79, all_conditions 0.79, rubric 0.79). The comment on
# each names the gap it aims at.
SKILLS_TARGETED = {
    # conditions and exceptions that change after the fact (policy, judge_hard, trap)
    "withdrawn_instruction": "A later part of the document (an amendment, a follow-up email, a "
    "correction, a revised schedule) withdraws, suspends or narrows an "
    "earlier instruction. The case falls inside the withdrawn scope, or "
    "just outside it, so applying the earlier instruction gives the wrong "
    "answer (or, just outside it, the right one).",
    "nested_exception": "An exception has its own exception ('...except X, unless Y'). The case "
    "turns on the inner clause, which is stated in a different section from "
    "the outer one.",
    "definition_scope": "A term is defined in a definitions section more narrowly or broadly "
    "than its everyday meaning (e.g. 'business day', 'dependant', 'incident', "
    "'written notice'), and the definition, not the everyday meaning, "
    "decides the case.",
    # evaluating a described action against the document's standard (judge_hard, policy)
    "judge_action": "The case describes what a person already did or wrote (a reply, an "
    "approval, a refund, a log entry). The question asks whether it complied "
    "with the document, or which requirement it broke. Exactly one requirement "
    "is broken, or none is, and the violation is not the most obvious "
    "candidate.",
    # facts spread over separate artefacts (multi_hop, long_policy)
    "cross_artefact": "The document bundles two or three artefacts (e.g. a policy, a later memo "
    "and a filled-in form or email). The answer needs one fact from each; one "
    "artefact partly supersedes another, and the case gives no hint which "
    "artefact matters.",
    # time and running quantities (temporal_numeric)
    "version_in_force": "Rules changed on a stated effective date. The answer depends on "
    "which version applied on the date in the case (the event date, not the "
    "filing or decision date, unless the document says otherwise).",
    "running_total": "A limit applies to an accumulated quantity (a quarterly cap, the third "
    "occurrence within twelve months, hours in a rolling week). The case lists "
    "several dated events; the answer needs the right ones summed or counted "
    "within the right window.",
    # sharper versions of v16's weakest skills (tradeoff, underdetermined, rubric)
    "tie_break": "Two options each win on a different stated priority; the document states "
    "the order of priorities or a tie-break rule, which decides. Give figures, so "
    "the comparison needs arithmetic as well as the rule.",
    "missing_fact": "The rule needs a fact the case does not give, so the answer is 'cannot "
    "be determined from the document'; OR the fact looks missing but follows "
    "from something stated elsewhere (a definition, a default, a date). Vary "
    "the direction across documents.",
    "borderline_rubric": "A rubric or classification table with numeric boundaries; the case "
    "sits exactly on a boundary, and a separate clause says how boundaries "
    "are resolved.",
}
# Choice only: the answer may be "cannot be determined", which a yes/no cannot express.
CHOICE_ONLY = {"underdetermined", "missing_fact"}

KINDS_TARGETED = [
    *KINDS,
    "policy followed by a later memo that amends it",
    "email thread in which an earlier instruction is corrected",
    "contract with a side letter",
    "procedure with a revision history table",
    "case file: a policy excerpt, a completed form and reviewer notes",
]
# Mixed: the targeted ten plus the five v16 skills they neither sharpen nor replace (the
# targeted set has its own versions of tradeoff, underdetermined and rubric; running_total
# and tie_break carry the arithmetic of numbers; trap was already 8/8 on JevBench for v16).
SKILLS_MIXED = {
    **SKILLS_TARGETED,
    **{
        s: SKILLS[s] for s in ("all_conditions", "exception", "precedence", "dates", "lookup_chain")
    },
}
SKILL_SETS = {
    "v16": (SKILLS, KINDS),
    "targeted": (SKILLS_TARGETED, KINDS_TARGETED),
    "mixed": (SKILLS_MIXED, KINDS_TARGETED),
    "weak": (SKILLS_MIXED, KINDS_TARGETED),
}
# Weighted draws for the "weak" set: the mixed skills, with the five v16 scored lowest on
# in the mixed pilot (version_in_force 0.53, missing_fact 0.60, tie_break 0.67,
# cross_artefact 0.69, running_total 0.75) four times as likely as the rest. Sets without
# weights draw uniformly, exactly as before, so their specs are unchanged.
SKILL_WEIGHTS = {
    "weak": {
        s: (
            4.0
            if s
            in ("version_in_force", "missing_fact", "tie_break", "cross_artefact", "running_total")
            else 1.0
        )
        for s in SKILLS_MIXED
    }
}
WEIGHTS: dict = {}

SYSTEM_WRITER = """You write realistic workplace documents and hard, exact questions about them.
Output a single JSON object and nothing else."""

WRITER_PROMPT = """Write one realistic {kind} for an organisation, in the domain of {domain}.

The document:
- 900 to 1,600 words of plain text, in 6 to 10 numbered sections (lists and simple text
  tables are fine). Invented organisation, people, products, amounts and dates.
- Specific and internally consistent: exact amounts, dates, roles, limits, conditions,
  exceptions, and where two rules can collide, a stated rule for which one prevails.
- Contains everything needed to answer the questions below. Put the details a question
  hinges on in DIFFERENT sections, never next to each other.

Then write {n} questions about it, one for each of these skills, in this order:
{skills}

Each question:
- Applies the document to a specific case, given in "case" (1-4 sentences: who, what, when,
  amounts). The case gives facts about the situation only; it never restates, paraphrases
  or summarises the document's rules, and never hints at the answer.
- States in the case every fact that the conditions of the deciding rules depend on
  (dates and times, amounts, roles, statuses, whether a triggering event happened). Never
  rely on a condition being met, or unmet, without saying so. If the answer is meant to
  hinge on a missing fact, the correct answer is "cannot be determined from the document".
- Has exactly one correct answer that follows from the document and the case alone. The
  answer must depend on at least two separate sections (list them in "evidence"; for
  "underdetermined", the sections that come closest). For tradeoff and precedence
  questions, one of those sections must be the priority or precedence rule that decides.
- Is hard: a reader who skims, matches keywords, or applies only the most obvious clause
  gets it wrong. Name that tempting wrong answer in "lure".
- Has the TYPE given after its skill above: "noul" (yes/no) or "choice" (3 to 6 options).
  A noul question's correct answer MUST be the one given there; build the trap in the
  other direction (for "yes", something that looks disqualifying but is not).
  "choice": every option is plausible, with a meaningful short name (e.g. "Regional
  Manager", "Refund in full", "Not eligible") -- never letters or "Option A", and never
  an option that refers to other options ("Both A and B"). Include "cannot be determined
  from the document" as an option where it is a real possibility.
  "noul": give criteria -- in general terms, what makes the answer yes and what makes it no.

Before writing the JSON, check every answer against the document clause by clause. If a
question has more than one defensible answer, rewrite it.

JSON format:
{{
  "title": "...",
  "document": "...",
  "questions": [
    {{"skill": "...", "type": "noul", "case": "...", "question": "...",
      "criteria": {{"yes": "...", "no": "..."}}, "answer": "yes", "lure": "no",
      "evidence": ["3.2", "7.1"], "rationale": "..."}},
    {{"skill": "...", "type": "choice", "case": "...", "question": "...",
      "options": [{{"name": "...", "description": "..."}}, ...], "answer": "<an option name>",
      "lure": "<another option name>", "evidence": ["2", "5.3"], "rationale": "..."}}
  ]
}}"""

SYSTEM_VERIFIER = """You answer questions about documents carefully and exactly."""

VERIFY_PROMPT = """Read the document and the case, then answer the question.

DOCUMENT: {title}
{document}

CASE: {case}

QUESTION: {question}

OPTIONS:
{options}

Use only the document and the case. Check every clause that could apply, including
exceptions and rules about which provision prevails. Do not assume any fact the case does
not state: if the answer depends on one, and an option says it cannot be determined,
choose that option. Then give your final answer on the last line exactly as:
ANSWER: <option name>"""

# ---- the client ---------------------------------------------------------------------------
# chat(), USAGE and Truncated live in llm_client.py; the three scripts that import this module
# reach them as gd.chat and gd.USAGE, so they stay importable from here.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_client as llm  # noqa: E402
from llm_client import USAGE, Truncated, chat  # noqa: E402, F401

# ---- stage 1: write ---------------------------------------------------------------------


def spec_for(i: int, seed: int) -> dict:
    rng = random.Random(seed * 1_000_003 + i)
    domain = rng.choice(DOMAINS)
    kind = rng.choice(KINDS)
    n = rng.choice([3, 4])
    if WEIGHTS:  # weighted sampling without replacement (Efraimidis-Spirakis keys)
        keys = {s: rng.random() ** (1.0 / WEIGHTS[s]) for s in sorted(SKILLS)}
        skills = sorted(keys, key=keys.get, reverse=True)[:n]
    else:
        skills = rng.sample(sorted(SKILLS), n)
    # Each question's type, and for yes/no questions the answer it must have. Left open,
    # the pilot got 34 "no" to 13 "yes"; a target on every question made the writer turn
    # 86% of them into yes/no. So: ~30% yes/no (the pilot's share), "underdetermined"
    # always choice (it needs a "cannot be determined" option), targets 50/50. Separate
    # streams, so the specs above are unchanged for a given seed.
    typ = random.Random(f"{seed}/{i}/type")
    types = {s: "choice" if s in CHOICE_ONLY or typ.random() >= 0.3 else "noul" for s in skills}
    pol = random.Random(f"{seed}/{i}/noul")
    return {
        "id": f"doc{i:05d}",
        "domain": domain,
        "kind": kind,
        "skills": skills,
        "types": types,
        "noul_answers": {s: pol.choice(["yes", "no"]) for s in skills if types[s] == "noul"},
        "split": "eval" if domain in HELDOUT_DOMAINS else "train",
    }


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in output")
    return json.loads(text[start : end + 1])


LETTERISH = re.compile(r"^(option\s*)?[a-h](\)|\.|:)?$", re.I)
# "Both A and B", "None of option C": capital letters only, so "Neither a refund nor..." passes
REFERS = re.compile(r"\b(?i:both|all of|none of|neither)\b.*\b(?:(?i:option)\s*[A-Ha-h]|[A-H])\b")


def check_question(q: dict) -> str | None:
    """None if the question is well formed, else the reason it is dropped."""
    for k in ("skill", "type", "case", "question", "answer", "evidence", "rationale"):
        if not q.get(k):
            return f"missing {k}"
    ev = q["evidence"] if isinstance(q["evidence"], list) else [q["evidence"]]
    if len({str(e).strip() for e in ev}) < (1 if q["skill"] == "underdetermined" else 2):
        return "fewer than two evidence sections"
    if q["type"] == "noul":
        if str(q["answer"]).strip().lower() not in ("yes", "no"):
            return "noul answer not yes/no"
        c = q.get("criteria") or {}
        if not (c.get("yes") and c.get("no")):
            return "noul criteria missing"
        return None
    if q["type"] != "choice":
        return "unknown type"
    try:
        names = [o["name"].strip() for o in q["options"]]
    except (KeyError, TypeError, AttributeError):
        return "malformed options"
    low = [n.lower() for n in names]
    if not 3 <= len(names) <= 8 or len(set(low)) != len(low):
        return "option count or duplicate names"
    if any(LETTERISH.match(n) or REFERS.search(n) for n in names):
        return "letter-only or cross-referencing option name"
    ans, lure = str(q["answer"]).strip().lower(), str(q.get("lure", "")).strip().lower()
    if ans not in low:
        return "answer is not an option"
    if lure not in low or lure == ans:
        return "lure missing, not an option, or equal to the answer"
    return None


def write_one(args, spec: dict) -> dict:
    types, targets = spec["types"], spec["noul_answers"]
    skills = "\n".join(
        f"{k + 1}. {s}: {SKILLS[s]} TYPE: "
        + (f'noul, and its correct answer is "{targets[s]}".' if types[s] == "noul" else "choice.")
        for k, s in enumerate(spec["skills"])
    )
    prompt = WRITER_PROMPT.format(
        kind=spec["kind"], domain=spec["domain"], n=len(spec["skills"]), skills=skills
    )
    last = None
    for _ in range(3):
        try:
            text, provider = chat(
                args,
                args.writer,
                SYSTEM_WRITER,
                prompt,
                args.writer_max_tokens,
                args.writer_temperature,
            )
            d = parse_json(text)
            words = len(str(d.get("document", "")).split())
            if not 700 <= words <= 2200:
                raise ValueError(f"document is {words} words")
            good, dropped = [], []
            for q in d.get("questions", []):
                why = check_question(q) if isinstance(q, dict) else "not an object"
                if not why and spec["types"].get(q["skill"]) != q["type"]:
                    why = "type differs from the one assigned"
                if (
                    not why
                    and q["type"] == "noul"
                    and str(q["answer"]).strip().lower() != spec["noul_answers"][q["skill"]]
                ):
                    why = "yes/no answer differs from the one assigned"
                (dropped if why else good).append(why or q)
            if not good:
                raise ValueError(f"no well-formed questions: {dropped}")
            return {
                **spec,
                "title": d.get("title", ""),
                "document": d["document"],
                "questions": good,
                "dropped": dropped,
                "backend": args.backend,
                "writer": args.writer,
                "provider": provider,
            }
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    raise RuntimeError(f"writer failed 3 times: {last}")


# ---- stage 2: verify --------------------------------------------------------------------


def options_of(q: dict) -> list[list[str]]:
    if q["type"] == "noul":
        return [["no", q["criteria"]["no"]], ["yes", q["criteria"]["yes"]]]
    return [[o["name"].strip(), str(o.get("description", "")).strip()] for o in q["options"]]


def verify_one(args, model: str, doc: dict, qi: int, attempt: int) -> dict:
    q = doc["questions"][qi]
    opts = options_of(q)
    random.Random(f"{doc['id']}/{qi}/{attempt}").shuffle(opts)
    prompt = VERIFY_PROMPT.format(
        title=doc["title"],
        document=doc["document"],
        case=q["case"],
        question=q["question"],
        options="\n".join(f"- {n}: {d}" if d else f"- {n}" for n, d in opts),
    )
    text, provider = chat(
        args, model, SYSTEM_VERIFIER, prompt, args.verify_max_tokens, args.verify_temperature
    )
    answer = None
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if m:
        said = m[-1].strip().strip("*`'\".").lower()
        names = sorted((n.lower() for n, _ in opts), key=len, reverse=True)
        answer = next((n for n in names if said == n), None) or next(
            (n for n in names if said.startswith(n)), None
        )
    return {
        "id": doc["id"],
        "q": qi,
        "attempt": attempt,
        "backend": args.backend,
        "model": model,
        "provider": provider,
        "answer": answer,
        "tail": text[-600:],
    }


# ---- stage 3: export --------------------------------------------------------------------


def to_example(doc: dict, q: dict) -> dict:
    """One strands-decider Example (training/data/format.py); label indexes the canonical options."""
    state = f"{doc['title']}\n\n{doc['document'].strip()}\n\nCase: {q['case'].strip()}"
    if q["type"] == "noul":
        opts = [["false", q["criteria"]["no"]], ["true", q["criteria"]["yes"]]]
        label, kind = int(str(q["answer"]).strip().lower() == "yes"), "noul"
    else:
        opts = options_of(q)
        label = [n.lower() for n, _ in opts].index(str(q["answer"]).strip().lower())
        kind = "choice"
    return {
        "kind": kind,
        "state": state,
        "instructions": q["question"].strip(),
        "options": opts,
        "label": label,
        "task": f"gen:{q['skill']}",
        "weight": 1.0,
        "instruction_variants": [],
    }


# ---- driver -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="gen_pilot_qwen")
    ap.add_argument("--docs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="0 = the same specs as the first pilot")
    ap.add_argument(
        "--skill-set",
        choices=sorted(SKILL_SETS),
        default="v16",
        help="v16: the set gen_v16/ was written with; targeted: the post-v16 draft",
    )
    ap.add_argument(
        "--writer", default=None, help="default: the backend's (llm_client.DEFAULT_MODELS)"
    )
    ap.add_argument(
        "--verify-models",
        default=None,
        help="comma-separated; one answer from each (a single model answers twice); "
        "default: the backend's",
    )
    ap.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "none"])
    ap.add_argument("--quantizations", default="bf16,fp16,fp32,fp8")
    ap.add_argument(
        "--providers", default="", help="comma-separated host order; disables fallbacks"
    )
    ap.add_argument("--workers", type=int, default=32, help="calls in flight")
    ap.add_argument(
        "--max-cost",
        type=float,
        default=float("inf"),
        help="USD for this run; no new call starts past it",
    )
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--writer-temperature", type=float, default=0.7)
    ap.add_argument(
        "--writer-max-tokens",
        type=int,
        default=None,
        help="default: the model's cap in llm_client.MODELS, else 32000",
    )
    ap.add_argument("--verify-temperature", type=float, default=0.6)
    ap.add_argument(
        "--verify-max-tokens", type=int, default=None, help="as --writer-max-tokens, else 16000"
    )
    ap.add_argument("--stage", choices=["all", "write", "verify", "export"], default="all")
    llm.add_args(ap)
    args = ap.parse_args()
    global SKILLS, KINDS, WEIGHTS  # spec_for and write_one read these
    SKILLS, KINDS = SKILL_SETS[args.skill_set]
    WEIGHTS = SKILL_WEIGHTS.get(args.skill_set, {})
    args.writer = args.writer or llm.default_model(args, "writer")
    args.verify_models = args.verify_models or llm.default_model(args, "verifier")
    args.writer_max_tokens = llm.max_tokens(args.writer, args.writer_max_tokens, 32000)
    args.verify_max_tokens = llm.max_tokens(args.verify_models, args.verify_max_tokens, 16000)
    if args.stage != "export":
        llm.check_credentials(args)
        print(llm.describe(args, {"writer": args.writer, "verifiers": args.verify_models}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs_path, ver_path, fail_path = (
        out / "docs.jsonl",
        out / "verify.jsonl",
        out / "failures.jsonl",
    )
    verifiers = args.verify_models.split(",")
    if len(verifiers) == 1:
        verifiers *= 2
    file_lock = threading.Lock()

    def append(path: Path, row: dict) -> None:
        with file_lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    t0 = time.time()
    docs = {d["id"]: d for d in read_jsonl(docs_path)}
    if args.stage != "export":
        # One pool for both stages: a document's questions are verified as soon as it is
        # written, instead of after the slowest document. At most --workers calls are in
        # flight; verification jobs go first so finished documents drain before new ones
        # start. No new call starts once --max-cost is reached (in-flight calls finish).
        verified = {(v["id"], v["q"], v["attempt"]) for v in read_jsonl(ver_path)}
        writes = (
            deque(
                s for s in (spec_for(i, args.seed) for i in range(args.docs)) if s["id"] not in docs
            )
            if args.stage in ("all", "write")
            else deque()
        )
        checks: deque = deque()

        def queue_checks(d: dict) -> None:
            if args.stage in ("all", "verify"):
                checks.extend(
                    (d, qi, a)
                    for qi in range(len(d["questions"]))
                    for a in range(1, len(verifiers) + 1)
                    if (d["id"], qi, a) not in verified
                )

        for d in docs.values():
            queue_checks(d)
        print(
            f"write {len(writes)} documents with {args.writer} ({len(docs)} done); "
            f"verify {len(checks)} answers queued, with {verifiers}; up to ${args.max_cost:.2f}"
        )
        n_written = n_checked = 0
        with ThreadPoolExecutor(args.workers) as ex:
            running: dict = {}
            while True:
                while len(running) < args.workers and USAGE["cost"] < args.max_cost:
                    if checks:
                        d, qi, a = checks.popleft()
                        running[ex.submit(verify_one, args, verifiers[a - 1], d, qi, a)] = (
                            "v",
                            (d, qi, a),
                        )
                    elif writes:
                        s = writes.popleft()
                        running[ex.submit(write_one, args, s)] = ("w", s)
                    else:
                        break
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    kind, job = running.pop(f)
                    try:
                        row = f.result()
                        if kind == "w":
                            append(docs_path, row)
                            docs[row["id"]] = row
                            queue_checks(row)
                            n_written += 1
                        else:
                            append(ver_path, row)
                            n_checked += 1
                    except Exception as e:
                        info = (
                            {"id": job["id"]}
                            if kind == "w"
                            else {"id": job[0]["id"], "q": job[1], "attempt": job[2]}
                        )
                        append(
                            fail_path,
                            {
                                "stage": "write" if kind == "w" else "verify",
                                **info,
                                "error": repr(e)[:2000],
                            },
                        )
                    if (n_written + n_checked) % 25 == 0:
                        print(
                            f"  written {n_written}, verified {n_checked}, "
                            f"{len(writes)} + {len(checks)} queued, ${USAGE['cost']:.2f}, "
                            f"{time.time() - t0:.0f} s",
                            flush=True,
                        )
        if USAGE["cost"] >= args.max_cost and (writes or checks):
            print(
                f"stopped at the cost cap: {len(writes)} documents and {len(checks)} answers "
                f"not started; rerun with a higher --max-cost to resume"
            )

    if args.stage in ("all", "export"):
        answers: dict = {}
        for v in read_jsonl(ver_path):
            answers.setdefault((v["id"], v["q"]), {})[v["attempt"]] = v["answer"]
        kept = {"train": [], "eval": []}
        stats = {
            "documents": len(docs),
            "questions": 0,
            "kept": 0,
            "by_skill": {},
            "dropped_at_write": {},
            "doc_words": sorted(len(d["document"].split()) for d in docs.values()),
        }
        for d in docs.values():
            for why in d.get("dropped", []):
                stats["dropped_at_write"][why] = stats["dropped_at_write"].get(why, 0) + 1
            for qi, q in enumerate(d["questions"]):
                gold = str(q["answer"]).strip().lower()
                a = answers.get((d["id"], qi), {})
                ok = len(a) == len(verifiers) and all(x == gold for x in a.values())
                s = stats["by_skill"].setdefault(q["skill"], [0, 0])
                s[0] += 1
                s[1] += ok
                stats["questions"] += 1
                stats["kept"] += ok
                if q["type"] == "noul":
                    target = d.get("noul_answers", {}).get(q["skill"])
                    key = f"{'kept' if ok else 'dropped'} {gold}" + (
                        ""
                        if target is None
                        else f" (target {'met' if gold == target else 'missed'})"
                    )
                    stats.setdefault("noul", {})[key] = stats.get("noul", {}).get(key, 0) + 1
                if ok:
                    kept[d["split"]].append(to_example(d, q))
        for split, rows in kept.items():
            with (out / f"gen_{split}.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n, k, w = stats["questions"], stats["kept"], stats["doc_words"]
        print(
            f"export: {len(docs)} documents (median {w[len(w) // 2] if w else 0} words), {n} "
            f"questions, kept {k} ({k / max(n, 1):.0%}); train {len(kept['train'])}, "
            f"eval {len(kept['eval'])}"
        )
        for skill, (a, b) in sorted(stats["by_skill"].items()):
            print(f"  {skill:<16} kept {b}/{a}")
        if stats["dropped_at_write"]:
            print("  dropped at write:", stats["dropped_at_write"])
        if stats.get("noul"):
            print("  yes/no questions:", dict(sorted(stats["noul"].items())))
        (out / "stats.json").write_text(json.dumps(stats, indent=1))

    print(
        f"this run: {USAGE['calls']} calls, {USAGE['in']:,} in / {USAGE['out']:,} out tokens, "
        f"${USAGE['cost']:.2f} {llm.cost_note(args)}, {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    sys.exit(main())
