#!/usr/bin/env python3
"""Generate instruction-flip pairs through OpenRouter (v20): one short policy and case,
asked twice with instructions that reverse the correct yes/no answer.

WHAT THIS IS FOR
----------------
strands-decider reads documents better than questions: with the state and options fixed, v19
gives the same answer to a changed question about 95% of the time, and it answers some
of JevBench's paired standard-tier items differently depending on the wording. Every
row in the corpus pairs one state with one question, so the state alone nearly always
predicts the answer. Here the state is identical across a pair and only an instruction
in the question differs -- the kind a policy check states ("treat unproven conditions
as not satisfied") -- so the answer cannot be read off the state.

The flip types were written from the `policy` family's description ("is the requested
action permitted under the stated policy?"), not from benchmark items. NO JEVBENCH
CONTENT.

Same models, client and keep rule as gen_documents_openrouter.py (imported from it):
writer Qwen3.6-27B, two judgements per variant by Qwen3.5-397B-A17B in fresh contexts,
without the writer's answer; a pair is kept only if all four judgements agree with the
writer. Each batch is one policy in one domain with three items, each assigned a flip
type and which variant is "yes". Domains in HELDOUT_DOMAINS go to the eval file.

Usage:
    export OPENROUTER_API_KEY=...
    python gen_flips_openrouter.py --out gen_flips --batches 250 --max-cost 15
    # or, on Amazon Bedrock (see llm_client.py; other models, so new data, not the committed data):
    export AWS_BEARER_TOKEN_BEDROCK=...
    python gen_flips_openrouter.py --backend bedrock --region us-west-2 --out gen_flips_bedrock --batches 5
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_documents_openrouter as gd
import llm_client as llm

FLIPS = {
    "unproven": (
        "Treat any required condition that the case does not establish as NOT satisfied.",
        "Treat any required condition that the case does not contradict as satisfied.",
        "The case leaves one condition the policy requires unstated (neither established "
        "nor contradicted); everything else is settled.",
    ),
    "no_rule": (
        "If no clause of the policy covers the request, the request is permitted.",
        "If no clause of the policy covers the request, the request is not permitted.",
        "The request falls outside every clause of the policy, though a clause nearby "
        "looks as if it might cover it.",
    ),
    "conflict": (
        "Where two clauses conflict, the stricter one applies.",
        "Where two clauses conflict, the more specific one applies.",
        "Two clauses apply to the case and give opposite answers; one is stricter and "
        "the OTHER is more specific.",
    ),
    "reference_date": (
        "Apply the version of the policy in force on the date the request was made.",
        "Apply the version of the policy in force on the date the decision is made.",
        "The policy was amended between the request date and the decision date, and "
        "the amendment changes the answer for this case.",
    ),
    "examples": (
        "Treat any list of examples in the policy as complete: only what is listed qualifies.",
        "Treat any list of examples in the policy as illustrative: similar items qualify too.",
        "The item in the case is not in a list of examples but is clearly similar to them.",
    ),
    "days": (
        "Count every period in the policy in business days (Monday to Friday).",
        "Count every period in the policy in calendar days.",
        'The policy gives a period in plain "days"; the case\'s dates fall inside the period '
        "counted one way and outside it counted the other.",
    ),
}

SYSTEM_WRITER = """You write short, realistic workplace policies and exact yes/no test cases. Output a
single JSON object and nothing else."""

WRITER_PROMPT = """Write one short, realistic policy for an organisation in the domain of {domain}: 150 to
400 words, in 4 to 8 numbered clauses. Invented organisation, people, amounts and dates.

Then write {n} test items. Each item has a case (1-4 sentences: who, what, when,
amounts) and ONE yes/no question about it (e.g. "Is the request permitted?"). The item
is asked twice, each time with one extra instruction appended to the question. The two
instructions, the correct answer under each (they are opposite) and how the case must
be built are given with each item below. Everything else about the case must be
unambiguous, so that the instruction alone decides.

{items}

Rules: the case states facts only and never hints at the answer or restates the
policy; the question itself must not mention the instruction; the two answers must
follow from the policy, the case and the instruction alone. The policy's own wording
must leave open exactly what the instruction settles: no "including but not limited to"
or "only the following" around a list of examples, no statement of how days are
counted, which clause prevails, which version applies, how unproven conditions or
uncovered requests are treated. Check both answers clause
by clause before writing the JSON.

JSON format:
{{"title": "...", "policy": "...", "items": [
  {{"flip": "<the flip type given>", "case": "...", "question": "...",
    "criteria": {{"yes": "<what makes the answer yes>", "no": "<what makes it no>"}},
    "answer_a": "yes|no", "answer_b": "yes|no", "rationale": "..."}}
]}}"""

SYSTEM_VERIFIER = "You answer questions about policies carefully and exactly."
VERIFY_PROMPT = """Read the policy and the case, then answer the question, following its instruction.

POLICY: {title}
{policy}

CASE: {case}

QUESTION: {question}

Use only the policy, the case and the instruction. Check every clause that could apply.
Then give your final answer on the last line exactly as:
ANSWER: yes
or
ANSWER: no"""


def spec_for(i: int, seed: int) -> dict:
    rng = random.Random(f"flips/{seed}/{i}")
    domain = rng.choice(gd.DOMAINS)
    flips = rng.sample(sorted(FLIPS), 3)
    return {
        "id": f"flp{i:05d}",
        "domain": domain,
        "flips": flips,
        "ans_a": [rng.choice(["yes", "no"]) for _ in flips],
        "split": "eval" if domain in gd.HELDOUT_DOMAINS else "train",
    }


def write_one(args, spec: dict) -> dict:
    items = "\n".join(
        f'{k + 1}. flip type "{f}". Instruction A: "{FLIPS[f][0]}" Instruction B: "{FLIPS[f][1]}" '
        f'Correct answer under A: "{a}" (under B: "{"no" if a == "yes" else "yes"}"). How: {FLIPS[f][2]}'
        for k, (f, a) in enumerate(zip(spec["flips"], spec["ans_a"], strict=False))
    )
    prompt = WRITER_PROMPT.format(domain=spec["domain"], n=len(spec["flips"]), items=items)
    last = None
    for _ in range(3):
        try:
            text, provider = gd.chat(
                args, args.writer, SYSTEM_WRITER, prompt, args.writer_max_tokens, 0.7
            )
            d = gd.parse_json(text)
            words = len(str(d.get("policy", "")).split())
            if not 120 <= words <= 600:
                raise ValueError(f"policy is {words} words")
            good, dropped = [], []
            for k, it in enumerate(d.get("items", [])[: len(spec["flips"])]):
                why = None
                if not isinstance(it, dict) or not all(
                    it.get(x) for x in ("case", "question", "criteria", "answer_a", "answer_b")
                ):
                    why = "missing field"
                elif (
                    str(it["answer_a"]).lower() != spec["ans_a"][k]
                    or str(it["answer_b"]).lower() == spec["ans_a"][k]
                ):
                    why = "answers differ from the ones assigned"
                elif not (
                    isinstance(it["criteria"], dict)
                    and it["criteria"].get("yes")
                    and it["criteria"].get("no")
                ):
                    why = "criteria missing"
                (dropped if why else good).append(
                    why
                    or {
                        **it,
                        "flip": spec["flips"][k],
                        "answer_a": spec["ans_a"][k],
                        "answer_b": "no" if spec["ans_a"][k] == "yes" else "yes",
                    }
                )
            if not good:
                raise ValueError(f"no well-formed items: {dropped}")
            return {
                **spec,
                "title": d.get("title", ""),
                "policy": d["policy"],
                "items": good,
                "dropped": dropped,
                "backend": args.backend,
                "writer": args.writer,
                "provider": provider,
            }
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    raise RuntimeError(f"writer failed 3 times: {last}")


def question_text(it: dict, variant: str) -> str:
    return f"{it['question'].strip()} {FLIPS[it['flip']][0 if variant == 'a' else 1]}"


def verify_one(args, model: str, b: dict, qi: int, variant: str, attempt: int) -> dict:
    it = b["items"][qi]
    text, provider = gd.chat(
        args,
        model,
        SYSTEM_VERIFIER,
        VERIFY_PROMPT.format(
            title=b["title"],
            policy=b["policy"],
            case=it["case"],
            question=question_text(it, variant),
        ),
        args.verify_max_tokens,
        0.6,
    )
    m = re.findall(r"ANSWER:\s*\**\s*(yes|no)\b", text, re.I)
    return {
        "id": b["id"],
        "q": qi,
        "variant": variant,
        "attempt": attempt,
        "backend": args.backend,
        "model": model,
        "provider": provider,
        "answer": m[-1].lower() if m else None,
        "tail": text[-500:],
    }


def to_example(b: dict, it: dict, variant: str) -> dict:
    ans = it["answer_a"] if variant == "a" else it["answer_b"]
    state = f"{b['title']}\n\n{b['policy'].strip()}\n\nCase: {it['case'].strip()}"
    return {
        "kind": "noul",
        "state": state,
        "instructions": question_text(it, variant),
        "options": [["false", it["criteria"]["no"]], ["true", it["criteria"]["yes"]]],
        "label": int(ans == "yes"),
        "task": "gen:instruction_flip",
        "weight": 1.0,
        "instruction_variants": [],
    }  # a pair's two rows are written next to each other


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="gen_flips")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--writer", default=None, help="default: the backend's (llm_client.DEFAULT_MODELS)"
    )
    ap.add_argument("--verify-models", default=None, help="comma-separated; default: the backend's")
    ap.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "none"])
    ap.add_argument("--quantizations", default="bf16,fp16,fp32,fp8")
    ap.add_argument("--providers", default="")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-cost", type=float, default=float("inf"))
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument(
        "--writer-max-tokens",
        type=int,
        default=None,
        help="default: the model's cap in llm_client.MODELS, else 24000",
    )
    ap.add_argument(
        "--verify-max-tokens", type=int, default=None, help="as --writer-max-tokens, else 12000"
    )
    ap.add_argument("--stage", choices=["all", "export"], default="all")
    llm.add_args(ap)
    args = ap.parse_args()
    args.writer = args.writer or llm.default_model(args, "writer")
    args.verify_models = args.verify_models or llm.default_model(args, "verifier")
    args.writer_max_tokens = llm.max_tokens(args.writer, args.writer_max_tokens, 24000)
    args.verify_max_tokens = llm.max_tokens(args.verify_models, args.verify_max_tokens, 12000)
    if args.stage != "export":
        llm.check_credentials(args)
        print(llm.describe(args, {"writer": args.writer, "verifiers": args.verify_models}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bat_path, ver_path, fail_path = (
        out / "batches.jsonl",
        out / "verify.jsonl",
        out / "failures.jsonl",
    )
    verifiers = args.verify_models.split(",")
    if len(verifiers) == 1:
        verifiers *= 2
    lock = threading.Lock()

    def append(path: Path, row: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    t0 = time.time()
    batches = {b["id"]: b for b in gd.read_jsonl(bat_path)}
    if args.stage == "all":
        verified = {(v["id"], v["q"], v["variant"], v["attempt"]) for v in gd.read_jsonl(ver_path)}
        writes = deque(
            s
            for s in (spec_for(i, args.seed) for i in range(args.batches))
            if s["id"] not in batches
        )
        checks: deque = deque()

        def queue_checks(b: dict) -> None:
            checks.extend(
                (b, qi, var, a)
                for qi in range(len(b["items"]))
                for var in "ab"
                for a in range(1, len(verifiers) + 1)
                if (b["id"], qi, var, a) not in verified
            )

        for b in batches.values():
            queue_checks(b)
        print(
            f"write {len(writes)} batches ({len(batches)} done); verify {len(checks)} queued; "
            f"up to ${args.max_cost:.2f}",
            flush=True,
        )
        n = 0
        with ThreadPoolExecutor(args.workers) as ex:
            running: dict = {}
            while True:
                while len(running) < args.workers and gd.USAGE["cost"] < args.max_cost:
                    if checks:
                        b, qi, var, a = checks.popleft()
                        running[ex.submit(verify_one, args, verifiers[a - 1], b, qi, var, a)] = (
                            "v",
                            (b, qi, var, a),
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
                            append(bat_path, row)
                            batches[row["id"]] = row
                            queue_checks(row)
                        else:
                            append(ver_path, row)
                    except Exception as e:
                        info = (
                            {"id": job["id"]}
                            if kind == "w"
                            else {"id": job[0]["id"], "q": job[1], "variant": job[2]}
                        )
                        append(fail_path, {"stage": kind, **info, "error": repr(e)[:1500]})
                    n += 1
                    if n % 50 == 0:
                        print(
                            f"  {n} calls, {len(writes)} + {len(checks)} queued, ${gd.USAGE['cost']:.2f}, "
                            f"{time.time() - t0:.0f} s",
                            flush=True,
                        )
        if gd.USAGE["cost"] >= args.max_cost and (writes or checks):
            print(
                f"stopped at the cost cap: {len(writes)} batches and {len(checks)} judgements not started"
            )

    answers: dict = {}
    for v in gd.read_jsonl(ver_path):
        answers.setdefault((v["id"], v["q"], v["variant"]), {})[v["attempt"]] = v["answer"]
    kept = {"train": [], "eval": []}
    stats = {"batches": len(batches), "pairs": 0, "kept": 0, "by_flip": {}, "dropped_at_write": {}}
    for b in batches.values():
        for why in b.get("dropped", []):
            stats["dropped_at_write"][why] = stats["dropped_at_write"].get(why, 0) + 1
        for qi, it in enumerate(b["items"]):
            ok = all(
                len(answers.get((b["id"], qi, var), {})) == len(verifiers)
                and all(
                    x == (it["answer_a"] if var == "a" else it["answer_b"])
                    for x in answers[(b["id"], qi, var)].values()
                )
                for var in "ab"
            )
            s = stats["by_flip"].setdefault(it["flip"], [0, 0])
            s[0] += 1
            s[1] += ok
            stats["pairs"] += 1
            stats["kept"] += ok
            if ok:
                kept[b["split"]] += [to_example(b, it, "a"), to_example(b, it, "b")]
    for split, rows in kept.items():
        with (out / f"gen_{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n, k = stats["pairs"], stats["kept"]
    print(
        f"export: {len(batches)} batches, {n} pairs, kept {k} ({k / max(n, 1):.0%}); "
        f"train {len(kept['train'])} rows, eval {len(kept['eval'])} rows"
    )
    for f, (a, b) in sorted(stats["by_flip"].items()):
        print(f"  {f:15} kept {b}/{a}")
    if stats["dropped_at_write"]:
        print("  dropped at write:", stats["dropped_at_write"])
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(
        f"this run: {gd.USAGE['calls']} calls, ${gd.USAGE['cost']:.2f} {llm.cost_note(args)}, {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    sys.exit(main())
