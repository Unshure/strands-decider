#!/usr/bin/env python3
"""Generate hard decision questions over realistic documents with a Bedrock model.

WHAT THIS SCRIPT NEEDS TO DO (read this before changing anything on the other box)
---------------------------------------------------------------------------------
Goal: training data like the unreleased stage 2 of decider-2b v11 -- realistic business
documents, each with 3-4 typed questions whose answers need careful reading. The data is
for Hobson, a System One model that answers typed decisions (yes/no "noul", or "choice"
among named options) in one forward pass. See research/preregistrations/PREREGISTRATION-v15.md for
why: short public rule data (ShARC) did not transfer to JevBench; decider's long, varied
documents did.

It runs in three stages, each resumable (rerun the same command to continue):

  1. write   For each document, sample a domain, a document kind and 3-4 skills. Ask the
             model for ONE document (roughly 700-1,600 words) and one question per skill,
             each with its options and the writer's answer. Store the raw output.
  2. verify  Ask every question twice more, each in a fresh context, with the options
             shuffled and WITHOUT the writer's answer. Keep a question only if both
             answers agree with the writer's (decider kept 89-91% this way).
  3. export  Write kept questions as strands-decider training rows (JSONL, one Example per line:
             kind, state, instructions, options [[name, description], ...], label, task),
             split by domain: documents in --heldout-domains go to the eval file only.

Rules that must survive any fix:
  * NO JEVBENCH CONTENT. The skills below were written from our own reading of which
    kinds of decisions fail, not from JevBench items. Never paste benchmark items into
    these prompts, and never use them as few-shot examples.
  * Every label comes from the writer AND two agreeing verifiers. Don't relax the filter
    to raise yield. Hard questions the model cannot answer consistently are dropped,
    which caps difficulty at the model's own ability. That's the known cost.
  * Difficulty is measured back on the training box, not here: v14's accuracy on the
    kept questions should be well below 90% or the data is too easy to help.

Things likely to need fixing on the other box:
  * MODEL ID / REGION. Nova Premier is reached through a cross-region inference profile
    ("us.amazon.nova-premier-v1:0" in US regions). If the Converse call fails with a
    validation or access error, check the model ID in the Bedrock console, and that model
    access is granted in the region.
  * QUOTAS. Nova Premier's default tokens-per-minute quota is low. Throttling is retried
    (adaptive retries), but if it crawls, lower --workers or request a quota increase.
  * TIMEOUTS. A writer call produces ~3-5k tokens; read_timeout is 600 s.
  * JSON FROM THE WRITER. The writer is asked for a single JSON object. Malformed output
    is retried up to 3 times, then the document is logged as failed and skipped. If most
    fail to parse, look at failures.jsonl. Switching to Converse tool use with a forced
    tool (toolChoice) is the robust alternative, if the model supports it.
  * PRICES in --price-in / --price-out (USD per million tokens) are only for the cost
    estimate printed at the end. Set them from current Bedrock pricing.

This is the Nova pilot, kept as the record of its method; its export was removed before
publication because the rows did not record the model. For a new Bedrock run
use gen_documents_openrouter.py --backend bedrock (llm_client.py): the current prompts,
verification by a different model, and no boto3.

Usage (pilot):
    pip install boto3
    python gen_documents.py --out gen_pilot --docs 50
    # then copy gen_pilot/gen_train.jsonl, gen_pilot/gen_eval.jsonl (and docs.jsonl,
    # verify.jsonl for inspection) back to the training box.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config

# ---- what to write about --------------------------------------------------------------

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
# Held out: documents in these domains go to the eval file only (decider held out domains
# the same way). Change freely, but keep the list fixed between pilot and full runs.
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

# One question per chosen skill. Written from our own analysis of where one-pass decisions
# fail (conditions, exceptions, arithmetic, lookups, weighing, underdetermination), not
# from any benchmark's items.
SKILLS = {
    "all_conditions": "An action is permitted only if several conditions all hold; the case "
    "satisfies most of them but one fails in a way that is easy to overlook.",
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

SYSTEM_WRITER = """You write realistic workplace documents and hard, exact questions about them.
Output a single JSON object and nothing else."""

WRITER_PROMPT = """Write one realistic {kind} for an organisation, in the domain of {domain}.

The document:
- 700 to 1,600 words, plain text (numbered sections, lists and simple tables in text are fine).
- Invented organisation, people, products, amounts and dates; nothing about real companies.
- Specific and internally consistent: exact amounts, dates, roles, limits, conditions,
  exceptions. It must contain everything needed to answer the questions below, including
  the details the questions hinge on, placed naturally (not all next to each other).

Then write {n} questions about it, one for each of these skills, in this order:
{skills}

Each question:
- Applies the document to a specific case. Put the case facts in "case" (1-4 sentences:
  who, what, when, amounts). The case must not restate the answer.
- Has exactly one correct answer that follows from the document and the case alone.
  Work it out carefully before choosing it, and state that reasoning in "rationale".
- Is hard: a reader who skims, matches keywords, or ignores a clause should get it wrong.
- Is either type "noul" (a yes/no question) or type "choice" (3 to 6 named options).
  For "choice", every option is plausible and has a short name and a one-line description;
  include a "cannot be determined from the document" option where it is a real possibility.
  For "noul", give criteria: what makes the answer yes, and what makes it no, in general terms.

JSON format:
{{
  "title": "...",
  "document": "...",
  "questions": [
    {{"skill": "...", "type": "noul", "case": "...", "question": "...",
      "criteria": {{"yes": "...", "no": "..."}}, "answer": "yes", "rationale": "..."}},
    {{"skill": "...", "type": "choice", "case": "...", "question": "...",
      "options": [{{"name": "...", "description": "..."}}, ...], "answer": "<an option name>",
      "rationale": "..."}}
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

Reason step by step, briefly, using only the document and the case. Then give your final
answer on the last line exactly as: ANSWER: <option name>"""

# ---- Bedrock ----------------------------------------------------------------------------

_usage_lock = threading.Lock()
USAGE = {"in": 0, "out": 0, "calls": 0}


def make_client(region: str):
    cfg = Config(
        read_timeout=600, connect_timeout=30, retries={"max_attempts": 10, "mode": "adaptive"}
    )
    return boto3.client("bedrock-runtime", region_name=region, config=cfg)


def converse(
    client, model: str, system: str, prompt: str, max_tokens: int, temperature: float
) -> str:
    resp = client.converse(
        modelId=model,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": 0.95},
    )
    u = resp.get("usage", {})
    with _usage_lock:
        USAGE["in"] += u.get("inputTokens", 0)
        USAGE["out"] += u.get("outputTokens", 0)
        USAGE["calls"] += 1
    if resp.get("stopReason") == "max_tokens":
        raise ValueError("output truncated at max_tokens")
    return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])


# ---- stage 1: write ---------------------------------------------------------------------


def spec_for(i: int, seed: int) -> dict:
    rng = random.Random(seed * 1_000_003 + i)
    domain = rng.choice(DOMAINS)
    return {
        "id": f"doc{i:05d}",
        "domain": domain,
        "kind": rng.choice(KINDS),
        "skills": rng.sample(sorted(SKILLS), rng.choice([3, 4])),
        "split": "eval" if domain in HELDOUT_DOMAINS else "train",
    }


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in output")
    return json.loads(text[start : end + 1])


def check_doc(d: dict) -> None:
    """Reject malformed writer output; questions with a bad shape are dropped individually."""
    words = len(d.get("document", "").split())
    if not 400 <= words <= 2000:
        raise ValueError(f"document is {words} words")
    good = []
    for q in d.get("questions", []):
        try:
            if q["type"] == "noul":
                assert str(q["answer"]).lower() in ("yes", "no")
                assert q["criteria"]["yes"] and q["criteria"]["no"]
            elif q["type"] == "choice":
                names = [o["name"].strip() for o in q["options"]]
                assert 3 <= len(names) <= 8 and len(set(n.lower() for n in names)) == len(names)
                assert q["answer"].strip().lower() in [n.lower() for n in names]
            else:
                raise AssertionError
            assert q["question"].strip() and q["case"].strip()
            good.append(q)
        except (AssertionError, KeyError, TypeError, AttributeError):
            continue
    if not good:
        raise ValueError("no well-formed questions")
    d["questions"] = good


def write_one(client, args, spec: dict) -> dict:
    skills = "\n".join(f"{k + 1}. {s}: {SKILLS[s]}" for k, s in enumerate(spec["skills"]))
    prompt = WRITER_PROMPT.format(
        kind=spec["kind"], domain=spec["domain"], n=len(spec["skills"]), skills=skills
    )
    last = None
    for _ in range(3):
        try:
            d = parse_json(
                converse(
                    client,
                    args.model,
                    SYSTEM_WRITER,
                    prompt,
                    args.writer_max_tokens,
                    args.writer_temperature,
                )
            )
            check_doc(d)
            return {**spec, **d}
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    raise RuntimeError(f"writer failed 3 times: {last}")


# ---- stage 2: verify --------------------------------------------------------------------


def options_of(q: dict) -> list[list[str]]:
    if q["type"] == "noul":
        return [["no", q["criteria"]["no"]], ["yes", q["criteria"]["yes"]]]
    return [[o["name"].strip(), o.get("description", "").strip()] for o in q["options"]]


def verify_one(client, args, doc: dict, qi: int, attempt: int) -> str | None:
    q = doc["questions"][qi]
    opts = options_of(q)
    random.Random(f"{doc['id']}/{qi}/{attempt}").shuffle(opts)
    prompt = VERIFY_PROMPT.format(
        title=doc["title"],
        document=doc["document"],
        case=q["case"],
        question=q["question"],
        options="\n".join(f"- {n}: {d}" for n, d in opts),
    )
    text = converse(
        client, args.model, SYSTEM_VERIFIER, prompt, args.verify_max_tokens, args.verify_temperature
    )
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if not m:
        return None
    said = m[-1].strip().strip("*`'\".").lower()
    for name, _ in opts:
        if said == name.lower():
            return name.lower()
    for name, _ in opts:  # tolerate "yes, because..." or a trailing description
        if said.startswith(name.lower()):
            return name.lower()
    return None


# ---- stage 3: export --------------------------------------------------------------------


def to_example(doc: dict, q: dict) -> dict:
    """One strands-decider Example (training/data/format.py); label indexes the canonical options."""
    state = f"{doc['title']}\n\n{doc['document'].strip()}\n\nCase: {q['case'].strip()}"
    task = f"gen:{q['skill']}"
    if q["type"] == "noul":
        opts = [["false", q["criteria"]["no"]], ["true", q["criteria"]["yes"]]]
        label = int(str(q["answer"]).lower() == "yes")
        kind = "noul"
    else:
        opts = [[o["name"].strip(), o.get("description", "").strip()] for o in q["options"]]
        label = [n.lower() for n, _ in opts].index(q["answer"].strip().lower())
        kind = "choice"
    return {
        "kind": kind,
        "state": state,
        "instructions": q["question"].strip(),
        "options": opts,
        "label": label,
        "task": task,
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
    ap.add_argument("--out", default="gen_pilot")
    ap.add_argument("--docs", type=int, default=50)
    ap.add_argument("--model", default="us.amazon.nova-premier-v1:0")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--writer-temperature", type=float, default=0.8)
    ap.add_argument("--writer-max-tokens", type=int, default=8000)
    ap.add_argument("--verify-temperature", type=float, default=0.3)
    ap.add_argument("--verify-max-tokens", type=int, default=1500)
    ap.add_argument("--price-in", type=float, default=2.5, help="USD per 1M input tokens (check)")
    ap.add_argument(
        "--price-out", type=float, default=12.5, help="USD per 1M output tokens (check)"
    )
    ap.add_argument("--stage", choices=["all", "write", "verify", "export"], default="all")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs_path, ver_path, fail_path = (
        out / "docs.jsonl",
        out / "verify.jsonl",
        out / "failures.jsonl",
    )
    client = make_client(args.region)
    lock = threading.Lock()

    def append(path: Path, row: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    t0 = time.time()
    if args.stage in ("all", "write"):
        done = {d["id"] for d in read_jsonl(docs_path)}
        todo = [
            s for s in (spec_for(i, args.seed) for i in range(args.docs)) if s["id"] not in done
        ]
        print(f"write: {len(todo)} documents to go ({len(done)} done)")
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(write_one, client, args, s): s for s in todo}
            for k, f in enumerate(as_completed(futs), 1):
                s = futs[f]
                try:
                    append(docs_path, f.result())
                except Exception as e:
                    append(fail_path, {"stage": "write", "id": s["id"], "error": repr(e)})
                if k % 10 == 0:
                    print(f"  {k}/{len(todo)}  tokens in {USAGE['in']:,} out {USAGE['out']:,}")

    docs = {d["id"]: d for d in read_jsonl(docs_path)}
    if args.stage in ("all", "verify"):
        done = {(v["id"], v["q"], v["attempt"]) for v in read_jsonl(ver_path)}
        jobs = [
            (d, qi, a)
            for d in docs.values()
            for qi in range(len(d["questions"]))
            for a in (1, 2)
            if (d["id"], qi, a) not in done
        ]
        print(f"verify: {len(jobs)} answers to go")

        def run(job):
            d, qi, a = job
            return {
                "id": d["id"],
                "q": qi,
                "attempt": a,
                "answer": verify_one(client, args, d, qi, a),
            }

        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(run, j): j for j in jobs}
            for k, f in enumerate(as_completed(futs), 1):
                d, qi, a = futs[f]
                try:
                    append(ver_path, f.result())
                except Exception as e:
                    append(fail_path, {"stage": "verify", "id": d["id"], "q": qi, "error": repr(e)})
                if k % 50 == 0:
                    print(f"  {k}/{len(jobs)}  tokens in {USAGE['in']:,} out {USAGE['out']:,}")

    if args.stage in ("all", "export"):
        answers: dict = {}
        for v in read_jsonl(ver_path):
            answers.setdefault((v["id"], v["q"]), {})[v["attempt"]] = v["answer"]
        kept = {"train": [], "eval": []}
        stats = {"questions": 0, "kept": 0, "by_skill": {}}
        for d in docs.values():
            for qi, q in enumerate(d["questions"]):
                gold = str(q["answer"]).strip().lower()
                a = answers.get((d["id"], qi), {})
                ok = a.get(1) == gold and a.get(2) == gold
                s = stats["by_skill"].setdefault(q["skill"], [0, 0])
                s[0] += 1
                s[1] += ok
                stats["questions"] += 1
                stats["kept"] += ok
                if ok:
                    kept[d["split"]].append(to_example(d, q))
        for split, rows in kept.items():
            with (out / f"gen_{split}.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n, k = stats["questions"], stats["kept"]
        print(
            f"export: {len(docs)} documents, {n} questions, kept {k} ({k / max(n, 1):.0%}); "
            f"train {len(kept['train'])}, eval {len(kept['eval'])}"
        )
        for skill, (a, b) in sorted(stats["by_skill"].items()):
            print(f"  {skill:<16} kept {b}/{a}")
        (out / "stats.json").write_text(json.dumps(stats, indent=1))

    cost = USAGE["in"] / 1e6 * args.price_in + USAGE["out"] / 1e6 * args.price_out
    print(
        f"this run: {USAGE['calls']} calls, {USAGE['in']:,} in / {USAGE['out']:,} out tokens, "
        f"~${cost:.2f} at the given prices, {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    sys.exit(main())
