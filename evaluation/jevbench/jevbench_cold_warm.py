"""Run each JevBench task twice in a row, to separate first-request cost from steady state.

On MPS the first request at a new input length pays a one-off compile (~1.5 s measured
on an M3 Pro) that later requests of the same length do not, and nearly every JevBench
task has a unique length. Asking each task twice back to back, and scoring the two
passes separately, splits the latency into the cold cost and the warm cost without
modifying the harness.

    python evaluation/jevbench/jevbench_cold_warm.py double ALL.jsonl DOUBLED.jsonl
    python -m jevbench.cli run --tasks DOUBLED.jsonl ... --results RUN/results.jsonl
    python evaluation/jevbench/jevbench_cold_warm.py split DOUBLED.jsonl RUN/results.jsonl RUN/
    python -m jevbench.cli summarize --tasks ALL.jsonl --results RUN/cold.jsonl ...
    python -m jevbench.cli summarize --tasks ALL.jsonl --results RUN/warm.jsonl ...

`summarize` refuses duplicate task ids, so the second copy is renamed `<id>~warm` (and
its paraphrase group likewise, so pairs stay within one pass); `split` maps the ids
back, so each pass is scored against the unmodified public task file. `split` also
checks the two copies really were the same request: identical input-token counts.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

SUFFIX = "~warm"


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def double(src: str, dst: str) -> None:
    out = []
    for t in _read(src):
        if SUFFIX in t["id"]:
            raise SystemExit(f"{t['id']}: already doubled")
        warm = dict(t, id=t["id"] + SUFFIX)
        if t.get("group") is not None:
            warm["group"] = t["group"] + SUFFIX
        out += [t, warm]
    _write(dst, out)
    print(f"wrote {len(out)} tasks ({len(out) // 2} x 2) to {dst}")


def _pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


def split(doubled: str, results: str, outdir: str) -> None:
    order = [t["id"] for t in _read(doubled)]
    rows = {r["task_id"]: r for r in _read(results)}
    missing = [i for i in order if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} tasks have no result, e.g. {missing[:3]}")

    cold, warm, mismatched = [], [], []
    for cid in order[0::2]:
        c, w = rows[cid], rows[cid + SUFFIX]
        if (c.get("usage") or {}).get("input_tokens") != (w.get("usage") or {}).get("input_tokens"):
            mismatched.append(cid)
        cold.append(c)
        warm.append(dict(w, task_id=cid, group=c.get("group")))
    if mismatched:
        raise SystemExit(f"cold and warm requests differ in length for {mismatched[:3]}")

    os.makedirs(outdir, exist_ok=True)
    _write(os.path.join(outdir, "cold.jsonl"), cold)
    _write(os.path.join(outdir, "warm.jsonl"), warm)

    # The very first request of the run also warms up the server itself.
    first = min(cold, key=lambda r: r["ts"])["task_id"]
    pairs = [
        (c["latency_s"] * 1000, w["latency_s"] * 1000, c["usage"]["input_tokens"])
        for c, w in zip(cold, warm, strict=True)
        if c["task_id"] != first
    ]
    same = sum(c["predicted"] == w["predicted"] for c, w in zip(cold, warm, strict=True))
    print(f"{len(cold)} tasks; same prediction on both passes: {same}/{len(cold)}")
    print(f"(excluding the run's first request, {first})")
    print(f"{'':12s} {'p50':>8s} {'p95':>8s} {'mean':>8s}   ms")
    for name, xs in (
        ("cold", [p[0] for p in pairs]),
        ("warm", [p[1] for p in pairs]),
        ("cold - warm", [p[0] - p[1] for p in pairs]),
    ):
        print(f"{name:12s} {_pct(xs, 0.5):8.0f} {_pct(xs, 0.95):8.0f} {statistics.mean(xs):8.0f}")
    print(f"\n{'tokens':>11s} {'n':>4s} {'cold p50':>9s} {'warm p50':>9s} {'extra p50':>10s}")
    for lo, hi in ((0, 300), (300, 1000), (1000, 2500), (2500, 5000)):
        band = [p for p in pairs if lo <= p[2] < hi]
        if band:
            print(
                f"{lo:>5d}-{hi:<5d} {len(band):4d} {_pct([p[0] for p in band], 0.5):9.0f}"
                f" {_pct([p[1] for p in band], 0.5):9.0f}"
                f" {_pct([p[0] - p[1] for p in band], 0.5):10.0f}"
            )


def main(argv) -> int:
    if len(argv) == 3 and argv[0] == "double":
        double(argv[1], argv[2])
    elif len(argv) == 4 and argv[0] == "split":
        split(argv[1], argv[2], argv[3])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
