#!/usr/bin/env python3
"""Generate answer-adequacy judgements (request, response, adequate?) through OpenRouter.

WHAT THIS IS FOR
----------------
JevBench's `adequacy` family asks whether a saved answer adequately answers a request.
Nothing in Hobson's training corpus asks that, and v17 answers "adequate" to 92% of
HelpSteer2's validation responses (catching 7% of the inadequate ones) and to 11 of the
family's 12 public tasks. This script writes the missing kind of row: a realistic
request to an assistant, a response to it, and whether the response is adequate.

Same models, client and rules as gen_documents_openrouter.py (imported from it):
  * NO JEVBENCH CONTENT. Request categories, defects and "adequate despite appearances"
    kinds are our own, written from the family's one-line description ("answer adequacy
    judging"), never from benchmark items.
  * Writer Qwen3.6-27B; each item is judged twice by Qwen3.5-397B-A17B, in fresh
    contexts, without the writer's verdict. An item is kept only if both judgements
    agree with the writer's. Apache-2.0 open-weight models only.

Stages (resumable; rerun the same command to continue):
  1. write   Each batch has one request category and six items, each ASSIGNED its
             verdict (three adequate, three not, shuffled) and a kind: for inadequate
             items a specific defect, for adequate ones either a plain good answer or
             one that looks flawed but is not (terse, informal, corrects a false
             premise...). Assigned verdicts keep the labels balanced, and the
             look-flawed kinds keep "looks rough" from predicting "no".
  2. verify  Each item judged twice.
  3. export  Kept items become strands-decider noul rows (task "gen:adequacy"), in the format of
             training/data/adequacy.py. Batches in HELDOUT_CATEGORIES go to the eval
             file only.

Usage:
    export OPENROUTER_API_KEY=...
    python gen_adequacy_openrouter.py --out gen_adequacy_pilot --batches 20 --max-cost 3
    # or, on Amazon Bedrock (see llm_client.py; other models, so new data, not the committed data):
    export AWS_BEARER_TOKEN_BEDROCK=...
    python gen_adequacy_openrouter.py --backend bedrock --region us-west-2 --out gen_adequacy_bedrock --batches 5
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

CATEGORIES = [
    "a factual question about history, science, geography or everyday knowledge",
    "step-by-step instructions for a practical task",
    "writing with explicit constraints (length, format, tone, audience or what to leave out)",
    "summarising or extracting from a passage supplied in the request",
    "a calculation or estimate with figures given in the request",
    "a short piece of code, a shell command or a spreadsheet formula",
    "advice for a situation with stated constraints (budget, time, health, location)",
    "comparing two or three options against criteria the user states",
    "explaining a concept to a specified audience",
    "a reply to a customer or colleague, drafted from details the user gives",
    "transforming data the user supplies (reformat, sort, convert units, tabulate)",
    "planning or scheduling with constraints the user gives",
]
HELDOUT_CATEGORIES = [CATEGORIES[10], CATEGORIES[11]]

DEFECTS = {
    "factual_error": "a confident, wrong fact or figure that the answer depends on",
    "wrong_question": "a fluent answer to a nearby but different question",
    "missed_constraint": "ignores one explicit constraint in the request (length, format, "
    "audience, something to exclude) while doing the rest well",
    "incomplete": "the request has several parts; one part is skipped or only gestured at",
    "calculation_error": "a wrong step in arithmetic or logic gives a wrong result",
    "needless_refusal": "declines, or hedges without answering, although the request is "
    "benign and answerable",
    "ignores_context": "contradicts or ignores information the user supplied in the request",
    "accepts_false_premise": "goes along with a false premise in the request, so the answer "
    "is wrong",
}
GOOD_KINDS = {
    "plain": "a good, complete, correct answer",
    "terse": "short, but fully answers everything asked",
    "informal": "casual or unpolished (a typo or two, loose structure), but correct and complete",
    "corrects_premise": "the request contains a false premise; the response points it out "
    "and then gives the correct answer",
    "states_assumption": "the request is slightly ambiguous; the response states a reasonable "
    "assumption and answers under it",
    "caveated": "correct and complete, with a caveat or two that is warranted",
}

SYSTEM_WRITER = """You write realistic requests to an AI assistant, responses to them, and exact
verdicts on whether each response is adequate. Output a single JSON object and nothing else."""

WRITER_PROMPT = """Write {n} items. Every request is of this kind: {category}.

Each item has a user's request (1-8 sentences; include any passage, data or figures the
request depends on) and an assistant's response (40-350 words). Requests must differ
from one another in topic and wording. Invent names and details.

Each item is assigned a verdict and a kind; follow them exactly:
{items}

An item is ADEQUATE if the response correctly and completely does what the request asks,
respects every explicit constraint in it, and has no error that matters. Style, brevity
or small typos alone never make a response inadequate. An item is INADEQUATE if it has
the assigned defect, and the defect is unambiguous once noticed, but not flagged,
apologised for or hinted at in the response. Inadequate responses should read as fluent
and confident as adequate ones; adequate responses of the rough-looking kinds must be
genuinely adequate.

Check each response against its request, line by line, before writing the JSON. If a
verdict could reasonably be disputed, rewrite the item.

JSON format:
{{"items": [
  {{"verdict": "adequate" | "inadequate", "kind": "<the kind assigned>",
    "request": "...", "response": "...",
    "flaw": "<for inadequate: the defect, quoted and explained; for adequate: why the
             response is adequate despite anything that might look wrong>"}}
]}}"""

SYSTEM_VERIFIER = """You judge whether an assistant's response adequately answers a request."""

VERIFY_PROMPT = """REQUEST:
{request}

RESPONSE:
{response}

Is the response adequate? It is adequate if it correctly and completely does what the
request asks, respects every explicit constraint in the request, and has no error that
matters. Style, brevity or small typos alone do not make it inadequate. Check facts,
figures and every part of the request. Then give your final answer on the last line
exactly as:
ANSWER: yes   (adequate)
or
ANSWER: no    (inadequate)"""

TASK = "gen:adequacy"
QUESTION = "Does the response adequately answer the request: correct, complete and on point?"
VARIANTS = [
    QUESTION,
    "Is this response good enough to send as the answer to the request?",
    "Would a careful reviewer accept this response as an adequate answer to the request?",
    "Does the response do what the request asks, without significant errors or omissions?",
]
OPTIONS = [["false", "the response is not adequate"], ["true", "the response is adequate"]]


def spec_for(i: int, seed: int, n: int = 6) -> dict:
    rng = random.Random(f"adequacy/{seed}/{i}")
    category = rng.choice(CATEGORIES)
    verdicts = ["adequate"] * (n // 2) + ["inadequate"] * (n - n // 2)
    rng.shuffle(verdicts)
    defects = rng.sample(sorted(DEFECTS), n - n // 2)
    goods = ["plain", *rng.sample(sorted(set(GOOD_KINDS) - {"plain"}), n // 2 - 1)]
    rng.shuffle(goods)
    kinds = [defects.pop() if v == "inadequate" else goods.pop() for v in verdicts]
    return {
        "id": f"ade{i:05d}",
        "category": category,
        "verdicts": verdicts,
        "kinds": kinds,
        "split": "eval" if category in HELDOUT_CATEGORIES else "train",
    }


def write_one(args, spec: dict) -> dict:
    items = "\n".join(
        f"{k + 1}. {v.upper()}, kind {kind}: "
        f"{DEFECTS[kind] if v == 'inadequate' else GOOD_KINDS[kind]}."
        for k, (v, kind) in enumerate(zip(spec["verdicts"], spec["kinds"], strict=False))
    )
    prompt = WRITER_PROMPT.format(n=len(spec["verdicts"]), category=spec["category"], items=items)
    last = None
    for _ in range(3):
        try:
            text, provider = gd.chat(
                args,
                args.writer,
                SYSTEM_WRITER,
                prompt,
                args.writer_max_tokens,
                args.writer_temperature,
            )
            got = gd.parse_json(text).get("items", [])
            good, dropped = [], []
            for k, it in enumerate(got[: len(spec["verdicts"])]):
                why = None
                if not isinstance(it, dict) or not all(
                    it.get(x) for x in ("verdict", "request", "response", "flaw")
                ):
                    why = "missing field"
                elif str(it["verdict"]).strip().lower() != spec["verdicts"][k]:
                    why = "verdict differs from the one assigned"
                elif not 25 <= len(str(it["response"]).split()) <= 500:
                    why = "response length"
                (dropped if why else good).append(
                    why or {**it, "kind": spec["kinds"][k], "verdict": spec["verdicts"][k]}
                )
            if not good:
                raise ValueError(f"no well-formed items: {dropped}")
            return {
                **spec,
                "items": good,
                "dropped": dropped,
                "backend": args.backend,
                "writer": args.writer,
                "provider": provider,
            }
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    raise RuntimeError(f"writer failed 3 times: {last}")


def verify_one(args, model: str, batch: dict, qi: int, attempt: int) -> dict:
    it = batch["items"][qi]
    text, provider = gd.chat(
        args,
        model,
        SYSTEM_VERIFIER,
        VERIFY_PROMPT.format(request=it["request"], response=it["response"]),
        args.verify_max_tokens,
        args.verify_temperature,
    )
    m = re.findall(r"ANSWER:\s*\**\s*(yes|no)\b", text, re.I)
    answer = {"yes": "adequate", "no": "inadequate"}[m[-1].lower()] if m else None
    return {
        "id": batch["id"],
        "q": qi,
        "attempt": attempt,
        "backend": args.backend,
        "model": model,
        "provider": provider,
        "answer": answer,
        "tail": text[-600:],
    }


def to_example(it: dict) -> dict:
    state = f"Request:\n{it['request'].strip()}\n\nResponse:\n{it['response'].strip()}"
    return {
        "kind": "noul",
        "state": state,
        "instructions": QUESTION,
        "options": OPTIONS,
        "label": int(it["verdict"] == "adequate"),
        "task": TASK,
        "weight": 1.0,
        "instruction_variants": VARIANTS,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="gen_adequacy_pilot")
    ap.add_argument("--batches", type=int, default=20, help="six items per batch")
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
    ap.add_argument("--writer-temperature", type=float, default=0.8)
    ap.add_argument(
        "--writer-max-tokens",
        type=int,
        default=None,
        help="default: the model's cap in llm_client.MODELS, else 24000",
    )
    ap.add_argument("--verify-temperature", type=float, default=0.6)
    ap.add_argument(
        "--verify-max-tokens", type=int, default=None, help="as --writer-max-tokens, else 12000"
    )
    ap.add_argument("--stage", choices=["all", "write", "verify", "export"], default="all")
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
    if args.stage != "export":
        verified = {(v["id"], v["q"], v["attempt"]) for v in gd.read_jsonl(ver_path)}
        writes = (
            deque(
                s
                for s in (spec_for(i, args.seed) for i in range(args.batches))
                if s["id"] not in batches
            )
            if args.stage in ("all", "write")
            else deque()
        )
        checks: deque = deque()

        def queue_checks(b: dict) -> None:
            if args.stage in ("all", "verify"):
                checks.extend(
                    (b, qi, a)
                    for qi in range(len(b["items"]))
                    for a in range(1, len(verifiers) + 1)
                    if (b["id"], qi, a) not in verified
                )

        for b in batches.values():
            queue_checks(b)
        print(
            f"write {len(writes)} batches ({len(batches)} done); verify {len(checks)} "
            f"judgements queued; up to ${args.max_cost:.2f}"
        )
        n_w = n_v = 0
        with ThreadPoolExecutor(args.workers) as ex:
            running: dict = {}
            while True:
                while len(running) < args.workers and gd.USAGE["cost"] < args.max_cost:
                    if checks:
                        b, qi, a = checks.popleft()
                        running[ex.submit(verify_one, args, verifiers[a - 1], b, qi, a)] = (
                            "v",
                            (b, qi, a),
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
                            n_w += 1
                        else:
                            append(ver_path, row)
                            n_v += 1
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
                    if (n_w + n_v) % 25 == 0:
                        print(
                            f"  written {n_w}, verified {n_v}, {len(writes)} + {len(checks)} "
                            f"queued, ${gd.USAGE['cost']:.2f}, {time.time() - t0:.0f} s",
                            flush=True,
                        )
        if gd.USAGE["cost"] >= args.max_cost and (writes or checks):
            print(
                f"stopped at the cost cap: {len(writes)} batches and {len(checks)} judgements "
                f"not started; rerun with a higher --max-cost to resume"
            )

    if args.stage in ("all", "export"):
        answers: dict = {}
        for v in gd.read_jsonl(ver_path):
            answers.setdefault((v["id"], v["q"]), {})[v["attempt"]] = v["answer"]
        kept = {"train": [], "eval": []}
        stats = {
            "batches": len(batches),
            "items": 0,
            "kept": 0,
            "by_kind": {},
            "dropped_at_write": {},
        }
        for b in batches.values():
            for why in b.get("dropped", []):
                stats["dropped_at_write"][why] = stats["dropped_at_write"].get(why, 0) + 1
            for qi, it in enumerate(b["items"]):
                a = answers.get((b["id"], qi), {})
                ok = len(a) == len(verifiers) and all(x == it["verdict"] for x in a.values())
                s = stats["by_kind"].setdefault(f"{it['verdict']}/{it['kind']}", [0, 0])
                s[0] += 1
                s[1] += ok
                stats["items"] += 1
                stats["kept"] += ok
                if ok:
                    kept[b["split"]].append(to_example(it))
        for split, rows in kept.items():
            with (out / f"gen_{split}.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n, k = stats["items"], stats["kept"]
        yes = {s: sum(r["label"] for r in rows) for s, rows in kept.items()}
        print(
            f"export: {len(batches)} batches, {n} items, kept {k} ({k / max(n, 1):.0%}); "
            f"train {len(kept['train'])} ({yes['train']} adequate), "
            f"eval {len(kept['eval'])} ({yes['eval']} adequate)"
        )
        for kind, (a, b) in sorted(stats["by_kind"].items()):
            print(f"  {kind:<32} kept {b}/{a}")
        if stats["dropped_at_write"]:
            print("  dropped at write:", stats["dropped_at_write"])
        (out / "stats.json").write_text(json.dumps(stats, indent=1))

    print(
        f"this run: {gd.USAGE['calls']} calls, {gd.USAGE['in']:,} in / {gd.USAGE['out']:,} out "
        f"tokens, ${gd.USAGE['cost']:.2f} {llm.cost_note(args)}, {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    sys.exit(main())
