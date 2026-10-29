#!/usr/bin/env python3
"""Paraphrase the generated document questions through OpenRouter (v20).

WHAT THIS IS FOR
----------------
Every generated document question (data/generators/gen_v16/, gen_pilot_qwen/, gen_mixed_pilot/,
gen_weak/) carries exactly one phrasing, while the short tasks and the adequacy rows
carry several, sampled per epoch at collate time (Example.instruction_variants). JevBench
asks each standard-tier item two ways and v19 answers some pairs differently; the
question-sensitivity probe finds it gives the same answer to a changed question 95% of
the time. This writes up to three paraphrases per question, each checked to ask exactly
the same thing, so the model sees each question worded several ways.

Same models, client and rules as gen_documents_openrouter.py (imported from it): writer
Qwen3.6-27B, checker Qwen3.5-397B-A17B, open-weight Apache-2.0 only, NO JEVBENCH
CONTENT. The checker sees the question, its type and its options -- not the document --
and must judge each paraphrase SAME (the same option, or the same yes/no, is correct
for any case) or DIFFERENT; only SAME paraphrases are kept. For yes/no questions a
paraphrase that flips polarity ("Is X permitted?" -> "Is X prohibited?") is DIFFERENT.

Output: paraphrases.jsonl, one {"question": ..., "paraphrases": [...]} per distinct
question text; `python -m training.data.generated --paraphrases` attaches them.

Usage:
    export OPENROUTER_API_KEY=...
    python gen_paraphrases_openrouter.py --out gen_paraphrases --max-cost 10
    # or, on Amazon Bedrock (see llm_client.py; other models, so new data, not the committed data):
    export AWS_BEARER_TOKEN_BEDROCK=...
    python gen_paraphrases_openrouter.py --backend bedrock --region us-west-2 --out gen_paraphrases_bedrock --limit 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_documents_openrouter as gd
import llm_client as llm

SOURCES = ["gen_v16", "gen_pilot_qwen", "gen_mixed_pilot", "gen_weak"]
PER_CALL = 8

SYSTEM_WRITER = "You rewrite questions without changing what they ask. Output a single JSON object and nothing else."
WRITER_PROMPT = """Rewrite each question below three different ways. Each rewrite must ask exactly the
same thing, so that for any situation the same answer is correct: for a yes/no question
the same "yes" and "no"; for a choice question the same option. Vary the wording and the
sentence structure (a direct question, an instruction such as "Determine whether...",
a different order of clauses), but:
- keep every name, number, date and condition;
- do not add or drop any condition, and do not hint at the answer;
- for yes/no questions keep the polarity: never turn "permitted" into "prohibited",
  "eligible" into "ineligible", and so on;
- do not refer to the options by position or letter.

{items}

JSON format: {{"rewrites": [{{"id": <id>, "paraphrases": ["...", "...", "..."]}}, ...]}}"""

SYSTEM_CHECK = "You check whether rewritten questions ask exactly the same thing as the original."
CHECK_PROMPT = """For each original question below, decide for each of its rewrites whether it asks
exactly the same thing: for ANY situation, the same answer would be correct for the
rewrite as for the original (for yes/no questions, "yes" must mean the same thing in
both). A rewrite that adds, drops or changes a condition, a name, a number or a date,
flips the polarity, or hints at the answer is DIFFERENT.

{blocks}

Give one line per rewrite, exactly as:
Q1 R1: SAME or DIFFERENT
Q1 R2: SAME or DIFFERENT
..."""


def questions() -> list[dict]:
    """Distinct question texts across the generator exports, with type and options."""
    seen: dict[str, dict] = {}
    here = Path(__file__).resolve().parent
    for src in SOURCES:
        for split in ("train", "eval"):
            p = here / src / f"gen_{split}.jsonl"
            for r in gd.read_jsonl(p):
                q = r["instructions"].strip()
                if q not in seen:
                    seen[q] = {"question": q, "kind": r["kind"], "options": r["options"]}
    return sorted(seen.values(), key=lambda x: x["question"])


def describe(q: dict) -> str:
    if q["kind"] == "noul":
        crit = {o[0]: o[1] for o in q["options"]}
        return f'  "yes" means: {crit.get("true", "")}\n  "no" means: {crit.get("false", "")}'
    return "  options: " + "; ".join(o[0] for o in q["options"])


def write_batch(args, batch: list[tuple[int, dict]]) -> list[dict]:
    items = "\n\n".join(
        f"id {i} ({'yes/no' if q['kind'] == 'noul' else 'choice'}): {q['question']}\n{describe(q)}"
        for i, q in batch
    )
    wargs = argparse.Namespace(**{**vars(args), "reasoning_effort": args.writer_reasoning})
    for attempt in range(3):
        try:
            text, provider = gd.chat(
                wargs,
                args.writer,
                SYSTEM_WRITER,
                WRITER_PROMPT.format(items=items),
                args.writer_max_tokens,
                args.writer_temperature,
            )
            got = {
                int(r["id"]): r.get("paraphrases") or []
                for r in gd.parse_json(text).get("rewrites", [])
                if isinstance(r, dict) and "id" in r
            }
            break
        except (ValueError, json.JSONDecodeError):
            if attempt == 2:
                raise
    out = []
    for i, q in batch:
        ps = [p.strip() for p in got.get(i, []) if isinstance(p, str) and p.strip()]
        ps = [p for p in dict.fromkeys(ps) if p != q["question"]][:3]
        if ps:
            out.append(
                {
                    "id": i,
                    "question": q["question"],
                    "kind": q["kind"],
                    "options": q["options"],
                    "candidates": ps,
                    "backend": args.backend,
                    "writer": args.writer,
                    "provider": provider,
                }
            )
    return out


def check_batch(args, rows: list[dict]) -> list[dict]:
    blocks = "\n\n".join(
        f"Q{j + 1} ORIGINAL ({'yes/no' if r['kind'] == 'noul' else 'choice'}): {r['question']}\n{describe(r)}\n"
        + "\n".join(f"Q{j + 1} R{k + 1}: {p}" for k, p in enumerate(r["candidates"]))
        for j, r in enumerate(rows)
    )
    text, provider = gd.chat(
        args,
        args.checker,
        SYSTEM_CHECK,
        CHECK_PROMPT.format(blocks=blocks),
        args.check_max_tokens,
        0.3,
    )
    verdict = {
        (int(q), int(k)): v.upper()
        for q, k, v in re.findall(r"Q(\d+)\s*R(\d+)\s*:\s*\**\s*(SAME|DIFFERENT)", text, re.I)
    }
    out = []
    for j, r in enumerate(rows):
        v = [verdict.get((j + 1, k + 1)) for k in range(len(r["candidates"]))]
        if all(x is None for x in v):
            continue  # unparsed: left for a rerun rather than recorded as rejected
        out.append(
            {
                "question": r["question"],
                "paraphrases": [p for p, x in zip(r["candidates"], v, strict=False) if x == "SAME"],
                "candidates": r["candidates"],
                "verdicts": v,
                "backend": args.backend,
                "writer": r.get("writer"),
                "checker": args.checker,
                "provider": provider,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="gen_paraphrases")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (pilot)")
    ap.add_argument(
        "--writer", default=None, help="default: the backend's (llm_client.DEFAULT_MODELS)"
    )
    ap.add_argument("--checker", default=None, help="default: the backend's verifier")
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high", "none"])
    ap.add_argument("--writer-reasoning", default="none", choices=["low", "medium", "high", "none"])
    ap.add_argument("--quantizations", default="bf16,fp16,fp32,fp8")
    ap.add_argument("--providers", default="")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-cost", type=float, default=float("inf"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--writer-temperature", type=float, default=0.8)
    ap.add_argument(
        "--writer-max-tokens",
        type=int,
        default=None,
        help="default: the model's cap in llm_client.MODELS, else 16000",
    )
    ap.add_argument(
        "--check-max-tokens", type=int, default=None, help="as --writer-max-tokens, else 16000"
    )
    llm.add_args(ap)
    args = ap.parse_args()
    args.writer = args.writer or llm.default_model(args, "writer")
    args.checker = args.checker or llm.default_model(args, "checker")
    args.writer_max_tokens = llm.max_tokens(args.writer, args.writer_max_tokens, 16000)
    args.check_max_tokens = llm.max_tokens(args.checker, args.check_max_tokens, 16000)
    llm.check_credentials(args)
    print(llm.describe(args, {"writer": args.writer, "checker": args.checker}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cand_path, done_path, fail_path = (
        out / "candidates.jsonl",
        out / "paraphrases.jsonl",
        out / "failures.jsonl",
    )
    lock = threading.Lock()

    def append(path: Path, row: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    t0 = time.time()
    qs = questions()
    if args.limit:
        qs = qs[: args.limit]
    have_cand = {r["question"]: r for r in gd.read_jsonl(cand_path)}
    have_done = {r["question"] for r in gd.read_jsonl(done_path)}
    todo = [(i, q) for i, q in enumerate(qs) if q["question"] not in have_cand]
    batches = [todo[k : k + PER_CALL] for k in range(0, len(todo), PER_CALL)]
    checks = [r for r in have_cand.values() if r["question"] not in have_done]
    print(
        f"{len(qs)} distinct questions; {len(batches)} writer calls and {len(checks)} checks queued; "
        f"up to ${args.max_cost:.2f}",
        flush=True,
    )
    n = 0
    with ThreadPoolExecutor(args.workers) as ex:
        running: dict = {}
        while True:
            while len(running) < args.workers and gd.USAGE["cost"] < args.max_cost:
                if len(checks) >= PER_CALL or (checks and not batches):
                    rs = [checks.pop() for _ in range(min(PER_CALL, len(checks)))]
                    running[ex.submit(check_batch, args, rs)] = ("c", rs)
                elif batches:
                    b = batches.pop()
                    running[ex.submit(write_batch, args, b)] = ("w", b)
                else:
                    break
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                kind, job = running.pop(f)
                try:
                    res = f.result()
                    if kind == "w":
                        for r in res:
                            append(cand_path, r)
                            checks.append(r)
                    else:
                        for r in res:
                            append(done_path, r)
                except Exception as e:
                    append(
                        fail_path,
                        {
                            "stage": kind,
                            "error": repr(e)[:1500],
                            "question": (
                                job[0][1]["question"] if kind == "w" else job[0]["question"]
                            )[:200],
                        },
                    )
                n += 1
                if n % 50 == 0:
                    print(
                        f"  {n} calls done, {len(batches)} + {len(checks)} queued, ${gd.USAGE['cost']:.2f}, "
                        f"{time.time() - t0:.0f} s",
                        flush=True,
                    )

    done = gd.read_jsonl(done_path)
    kept = sum(len(r["paraphrases"]) for r in done)
    cands = sum(len(r["candidates"]) for r in done)
    with_any = sum(1 for r in done if r["paraphrases"])
    print(
        f"checked {len(done)} questions: {kept} of {cands} paraphrases judged SAME ({kept / max(cands, 1):.0%}); "
        f"{with_any} questions have at least one"
    )
    print(
        f"this run: {gd.USAGE['calls']} calls, ${gd.USAGE['cost']:.2f} {llm.cost_note(args)}, {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    sys.exit(main())
