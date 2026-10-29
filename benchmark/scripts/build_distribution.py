"""Build a token-length distribution for prod-sim from an eval dataset.

Reads one or more decider eval JSONL files, renders each record through
`render_state` + `render_question`, tokenizes the result, buckets the lengths at
powers of two, and writes a JSON file for the prod-sim workload to sample from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from strands_decider.prompting import render_question, render_state
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

DEFAULT_BUCKETS = (256, 512, 1024, 2048, 4096, 8192)


def _build_question(record: dict[str, Any]) -> Any:
    kind = record.get("kind") or "choice"
    raw_options = record["options"]
    instructions = record.get("instructions") or ""
    if kind == "score":
        criteria = [desc for _, desc in raw_options]
        return ScoreQuestion(instructions=instructions, criteria=criteria)
    criteria = dict(raw_options)
    if kind == "choice":
        return ChoiceQuestion(instructions=instructions, criteria=criteria)
    if kind == "noul":
        return NoulQuestion(instructions=instructions, criteria=criteria)
    raise ValueError(f"unknown eval record kind: {kind}")


def _record_tokens(record: dict[str, Any], tokenizer: Any) -> int:
    state_text = render_state(record["state"])
    question_text = render_question(_build_question(record)).text
    state_ids = tokenizer(state_text, add_special_tokens=True)["input_ids"]
    question_ids = tokenizer(question_text, add_special_tokens=False)["input_ids"]
    return len(state_ids) + len(question_ids)


def _bucket(tokens: int, edges: list[int]) -> int:
    for edge in edges:
        if tokens <= edge:
            return edge
    return edges[-1]


def build(sources: list[Path], tokenizer_id: str, buckets: list[int]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    counts = {edge: 0 for edge in buckets}
    total = 0
    raw_tokens: list[int] = []
    sha = hashlib.sha256()
    for source in sources:
        data = source.read_bytes()
        sha.update(data)
        for line in data.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            tokens = _record_tokens(record, tokenizer)
            raw_tokens.append(tokens)
            counts[_bucket(tokens, buckets)] += 1
            total += 1
    if not total:
        raise ValueError("no eval records found in any source")
    weights = {edge: round(count / total, 6) for edge, count in counts.items()}
    raw_tokens.sort()

    def pct(q: float) -> int:
        idx = min(len(raw_tokens) - 1, max(0, int(q * len(raw_tokens))))
        return raw_tokens[idx]

    return {
        "name": "decider-eval-v1",
        "description": "Request token-length distribution derived from the decider eval sets.",
        "built_from": [str(source) for source in sources],
        "sources_sha256": sha.hexdigest(),
        "tokenizer": tokenizer_id,
        "records": total,
        "percentiles": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": raw_tokens[-1]},
        "buckets": [{"tokens": edge, "weight": weights[edge]} for edge in buckets],
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="build_distribution", description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="eval JSONL path; repeat to combine files",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-3B",
        help="Hugging Face tokenizer id",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        action="append",
        help=f"target token bucket; repeat; default {DEFAULT_BUCKETS}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination JSON path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    sources = [Path(path).resolve() for path in args.source]
    buckets = sorted(args.bucket or list(DEFAULT_BUCKETS))
    document = build(sources, args.tokenizer, buckets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
