#!/usr/bin/env python3
"""Paired per-task comparison of two JevBench runs, with an exact McNemar test.

Each side is either a JevBench `results.jsonl` (one record per task, `task_id` and
`correct`) or the repo's `research/data/jevbench_results.csv` (columns run, task_id,
correct) together with a run name, e.g. `--a-run v17`.

    python evaluation/jevbench/paired.py --a research/data/jevbench_results.csv --a-run v17 \
        --b OUT/results.jsonl [--tasks research/data/jevbench_tasks.csv] [--json out.json]

The exact McNemar p-value is the two-sided binomial test on the discordant tasks:
p = min(1, 2 * P(X <= min(b, c))), X ~ Binomial(b + c, 1/2); p = 1 when b + c = 0.
Standard library only. Exit code 2 if the two task sets differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = REPO / "research" / "data" / "jevbench_tasks.csv"


def load(path: str, run: str | None) -> dict[str, bool]:
    p = Path(path)
    if p.suffix == ".csv":
        if not run:
            sys.exit(f"{path}: a .csv source needs a run name (--a-run / --b-run)")
        out = {}
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["run"] == run:
                    out[r["task_id"]] = r["correct"] in ("1", "True", "true")
        if not out:
            sys.exit(f"{path}: no rows for run {run!r}")
        return out
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["task_id"] in out:
                    sys.exit(f"{path}: duplicate task_id {r['task_id']}")
                # JevBench's summarize counts correct=None as wrong; do the same.
                out[r["task_id"]] = bool(r.get("correct"))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a: dict[str, bool], b: dict[str, bool], ids) -> dict:
    both = sum(a[t] and b[t] for t in ids)
    neither = sum(not a[t] and not b[t] for t in ids)
    a_only = sum(a[t] and not b[t] for t in ids)
    b_only = sum(b[t] and not a[t] for t in ids)
    return {
        "n": len(ids),
        "a_correct": both + a_only,
        "b_correct": both + b_only,
        "both_right": both,
        "both_wrong": neither,
        "a_only": a_only,
        "b_only": b_only,
        "discordant": a_only + b_only,
        "net_b_minus_a": b_only - a_only,
        "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="baseline: results.jsonl or jevbench_results.csv")
    ap.add_argument("--a-run", default=None)
    ap.add_argument("--b", required=True, help="candidate: results.jsonl or jevbench_results.csv")
    ap.add_argument("--b-run", default=None)
    ap.add_argument(
        "--tasks",
        default=str(DEFAULT_TASKS),
        help="task_id,family,tier CSV for the per-tier split ('' to skip)",
    )
    ap.add_argument("--list", action="store_true", help="print the discordant task ids")
    ap.add_argument("--json", default=None, help="also write the result as JSON here")
    args = ap.parse_args()

    a, b = load(args.a, args.a_run), load(args.b, args.b_run)
    la = args.a_run or Path(args.a).parent.name or args.a
    lb = args.b_run or Path(args.b).parent.name or args.b
    if set(a) != set(b):
        print(
            f"task sets differ: {len(a)} vs {len(b)}, {len(set(a) & set(b))} shared",
            file=sys.stderr,
        )
        return 2
    ids = sorted(a)

    rows = [("all", compare(a, b, ids))]
    tier = {}
    if args.tasks and Path(args.tasks).exists():
        with open(args.tasks, newline="", encoding="utf-8") as f:
            tier = {r["task_id"]: r["tier"] for r in csv.DictReader(f)}
        for t in ("easy", "standard", "hard"):
            sub = [i for i in ids if tier.get(i) == t]
            if sub:
                rows.append((t, compare(a, b, sub)))

    print(f"A = {la}   B = {lb}")
    hdr = f"{'slice':<9}{'n':>5}{'A':>6}{'B':>6}{'both+':>7}{'both-':>7}{'A only':>8}{'B only':>8}{'net B-A':>9}{'McNemar p':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows:
        print(
            f"{name:<9}{r['n']:>5}{r['a_correct']:>6}{r['b_correct']:>6}{r['both_right']:>7}"
            f"{r['both_wrong']:>7}{r['a_only']:>8}{r['b_only']:>8}{r['net_b_minus_a']:>+9}"
            f"{r['mcnemar_exact_p']:>11.4f}"
        )
    if args.list:
        for i in ids:
            if a[i] != b[i]:
                print(f"  {'A only' if a[i] else 'B only'}  {tier.get(i, '?'):<8} {i}")
    if args.json:
        out = {
            "a": {"source": args.a, "run": args.a_run},
            "b": {"source": args.b, "run": args.b_run},
            "slices": dict(rows),
            "discordant_ids": {
                "a_only": [i for i in ids if a[i] and not b[i]],
                "b_only": [i for i in ids if b[i] and not a[i]],
            },
        }
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
