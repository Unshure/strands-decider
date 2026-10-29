#!/usr/bin/env python3
"""Check one live run of data/generators/gen_flips_openrouter.py: every row parses and records the
backend, the model and the provider; then report tokens and truncations.

    python .github/scripts/check_generator_rows.py <export dir> <expected backend>

Exit 1 when a row is malformed, names another backend, lacks a model or provider, or when
the writer produced nothing (failures.jsonl explains why). The summary goes to stdout and,
in GitHub Actions, to the step summary."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main(out: str, backend: str) -> int:
    d = Path(out)
    batches, verify, failures = (
        rows(d / "batches.jsonl"),
        rows(d / "verify.jsonl"),
        rows(d / "failures.jsonl"),
    )
    problems = []
    if not batches:
        problems.append("the writer produced no batch")
    for r in batches:
        if r.get("backend") != backend or not r.get("writer") or not r.get("provider"):
            problems.append(
                f"batch {r.get('id')}: backend={r.get('backend')!r} writer={r.get('writer')!r} "
                f"provider={r.get('provider')!r}"
            )
    if not verify:
        problems.append("no verifier answer")
    for r in verify:
        if r.get("backend") != backend or not r.get("model") or not r.get("provider"):
            problems.append(
                f"verify {r.get('id')}/{r.get('q')}: backend={r.get('backend')!r} "
                f"model={r.get('model')!r} provider={r.get('provider')!r}"
            )
    truncated = [f for f in failures if "truncated" in f.get("error", "")]
    parsed = sum(1 for r in verify if r.get("answer") in ("yes", "no"))
    models = sorted(
        {r["writer"] for r in batches if r.get("writer")}
        | {r["model"] for r in verify if r.get("model")}
    )
    providers = sorted({r["provider"] for r in batches + verify if r.get("provider")})
    stats = json.loads((d / "stats.json").read_text()) if (d / "stats.json").exists() else {}
    lines = [
        f"## {backend}",
        f"- batches {len(batches)}, verifier answers {len(verify)} ({parsed} parsed as yes/no), "
        f"pairs kept {stats.get('kept', 0)} of {stats.get('pairs', 0)}",
        f"- models {', '.join(models) or 'none'}; providers {', '.join(providers) or 'none'}",
        f"- failures {len(failures)}, of which truncated at max_tokens {len(truncated)}",
    ]
    if failures:
        lines += ["- first failure: `" + failures[0].get("error", "")[:300].replace("`", "'") + "`"]
    if problems:
        lines += ["- **problems:**"] + [f"  - {p}" for p in problems]
    text = "\n".join(lines) + "\n"
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
