#!/usr/bin/env python3
"""Gather Hobson's saved results into research/data/ as plain CSV. Nothing is re-run.

Sources, all written earlier by the runs themselves:

  JEVBENCH_ROOT   one jb_<run>/ directory per JevBench run (summary.json, results.jsonl),
                  plus the public task file all4096.jsonl. They are not committed; point
                  --jevbench-root at the directory that holds them.
  reports/        held-out short-task and question-sensitivity reports (repo, Windows)
  WSL reports     the same for the runs trained under WSL2 (v13-v16), and the per-row
                  multi-step and generated-eval outputs
  WSL data        the multi-step and generated eval files, read only for each row's
                  source name (no text is copied)

No JevBench task text is written: tasks are identified by id, family and tier only.
Numbers that survive only as text (v13's multi-step baselines in
research/preregistrations/PREREGISTRATION-v14.md) are entered by hand below and tagged with that source.

    python research/scripts/collect.py --jevbench-root DIR --wsl-root DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research" / "data"

# Generation by generation, in the order they were run. kind: "default" = adopted as the
# default recipe (the lineage); "experiment" = trained, measured, not adopted.
RUNS = [
    # id,    JevBench dir,  kind,         parent, torso,               what changed
    (
        "v5",
        "jb_3072o",
        "default",
        "v4",
        "Qwen3-1.7B-Base",
        "RuleTaker compositional rows; 3,072-token window",
    ),
    (
        "v6",
        "jb_v6c",
        "default",
        "v5",
        "Qwen3-1.7B-Base",
        "slot head seeded from the LM's digit rows + KL anchor to the frozen torso",
    ),
    ("v7", "jb_v7", "default", "v6", "Qwen3-1.7B-Base", "pointer readout"),
    ("v8", "jb_v8", "experiment", "v7", "Qwen3-1.7B", "instruction-tuned torso"),
    (
        "v9",
        "jb_v9",
        "experiment",
        "v7",
        "Qwen3-1.7B-Base",
        "+20k generated rule-execution documents",
    ),
    ("v10", "jb_v10", "experiment", "v7", "Qwen3-1.7B-Base", "+20k generated minimal pairs"),
    (
        "v11a",
        "jb_v11a",
        "experiment",
        "v7",
        "Qwen3-1.7B-Base",
        "question transforms replace 30% of the corpus",
    ),
    (
        "v11b",
        "jb_v11b",
        "experiment",
        "v7",
        "Qwen3-1.7B-Base",
        "KL toward the frozen torso on changed questions",
    ),
    ("v12", "jb_v12", "experiment", "v7", "Qwen3-1.7B-Base", "distilled from a frozen Qwen3.5-4B"),
    ("v13", "jb_v13", "default", "v7", "Qwen3.5-2B-Base", "Qwen3.5-2B-Base torso"),
    (
        "v14",
        "jb_v14",
        "default",
        "v13",
        "Qwen3.5-2B-Base",
        "+12,909 multi-step rows with a 4B teacher",
    ),
    ("v15", "jb_v15", "experiment", "v14", "Qwen3.5-2B-Base", "+4,566 ShARC / ConditionalQA rows"),
    ("v16", "jb_v16", "default", "v14", "Qwen3.5-2B-Base", "+2,148 generated document questions"),
    (
        "v17",
        "jb_v17",
        "default",
        "v16",
        "Qwen3.5-2B-Base",
        "v16 with its multi-step rows replayed toward v14's own answers",
    ),
    (
        "v18",
        "jb_v18",
        "default",
        "v17",
        "Qwen3.5-2B-Base",
        "v17 + 1,667 generated questions on the weak skills",
    ),
    (
        "v19",
        "jb_v19",
        "default",
        "v18",
        "Qwen3.5-2B-Base",
        "v18 + 6,166 answer-adequacy rows (HelpSteer2 and generated)",
    ),
]
# Side runs: measured on JevBench, not generations.
SIDE = [
    ("v5 @1024", "jb_v5", "v5 at the old 1,024-token window"),
    ("v5 @1024 +fix", "jb_fix", "v5, question reserved before truncating the state, 1,024 window"),
    ("v5 @4096", "jb_4096", "v5 at a 4,096-token window"),
    ("v6 uncalibrated", "jb_v6", "v6 before temperature fitting"),
    ("v6 arm: instruct", "jb_arm_instruct", "v5 corpus, instruct torso"),
    ("v6 arm: lm_head", "jb_arm_lmhead", "v5 corpus, head seeded from LM digit rows"),
    ("v6 arm: KL", "jb_arm_kl", "v5 corpus, KL anchor to the frozen torso"),
    ("scaling: 1.7B", "jb_head17", "controlled scaling test, Qwen3-1.7B"),
    ("scaling: 4B", "jb_head4b", "controlled scaling test, Qwen3-4B"),
    ("v14 prefix cache", "jb_v14pc", "v14 served through the shared-prefix cache"),
]
REFERENCES = [
    ("decider-2b v10", "jb_dv10", "Mapika decider-2b v10, same Qwen3.5-2B-Base torso"),
    ("decider-2b v11", "jb_dv11", "Mapika decider-2b v11: v10 + document-question stage"),
    ("Qwen3.5-4B, frozen", "jb_semif35", "frozen model read through the chat template (SemIf)"),
    ("Qwen3.5-2B, frozen", "jb_semif35_2b", "frozen model read through the chat template (SemIf)"),
    ("Qwen3-8B, frozen", "jb_semif8", "frozen model read through the chat template (SemIf)"),
    ("Qwen3-1.7B, frozen", "jb_semif17", "frozen model read through the chat template (SemIf)"),
]
# Held-out classification tasks: emotion, hate_severity, massive_intent, sarcasm (6,000
# rows). v4 and v7 onwards were scored on the same file. v5's held-out set added three
# RuleTaker splits (in-family, same generator as its training data); its four common
# tasks are recombined from the per-task breakdown below -- exact for accuracy, NLL and
# mean confidence, which are per-row means, but not for ECE, which is left blank. The
# scaling-test checkpoints were scored on the same four tasks, a different row sample.
HELDOUT_TASKS = ("emotion", "hate_severity", "massive_intent", "sarcasm")
# The question type of each held-out task, for rebuilding a by-type split from per-task
# results (v5's report mixes in RuleTaker, so its own by_kind is not comparable).
HELDOUT_KIND_TASKS = {
    "noul": ("sarcasm",),
    "choice": ("emotion", "massive_intent"),
    "score": ("hate_severity",),
}
HELDOUT = {
    "v4": ("repo", "reports/heldout_v4.json"),
    "v5": ("repo", "reports/heldout_v5.json"),
    "scaling: 1.7B": ("repo", "reports/heldout_head_linear.json"),
    "scaling: 1.7B, MLP head": ("repo", "reports/heldout_head_mlp512.json"),
    "scaling: 4B": ("repo", "reports/heldout_head_4b.json"),
    "v7": ("repo", "reports/v7_heldout.json"),
    "v11a": ("repo", "reports/v11a_heldout.json"),
    "v11b": ("repo", "reports/v11b_heldout.json"),
    "v12": ("repo", "reports/v12_heldout.json"),
    "v13": ("repo", "reports/hobson-2b-v13_heldout.json"),
    "v14": ("repo", "reports/hobson-2b-v14_heldout.json"),
    "v15": ("wsl", "reports/hobson-2b-v15_heldout.json"),
    "v16": ("wsl", "reports/hobson-2b-v16_heldout.json"),
    "v17": ("wsl", "reports/hobson-2b-v17_heldout.json"),
    "v18": ("wsl", "reports/hobson-2b-v18_heldout.json"),
    "v19": ("wsl", "reports/hobson-2b-v19_heldout.json"),
}
PROBE = {  # question-sensitivity probe reports, and the model key inside each
    "v7": ("repo", "reports/qsens_v7_v8.json", "hobson-1.7b-v7"),
    "v8": ("repo", "reports/qsens_v7_v8.json", "hobson-1.7b-v8"),
    "v11a": ("repo", "reports/qsens_v11a.json", None),
    "v11b": ("repo", "reports/qsens_v11b.json", None),
    "v12": ("repo", "reports/qsens_v12.json", None),
    "v13": ("wsl", "reports/qsens_hobson-2b-v13.json", None),
    "v14": ("wsl", "reports/qsens_hobson-2b-v14.json", None),
    "v15": ("wsl", "reports/qsens_hobson-2b-v15.json", None),
    "v16": ("wsl", "reports/qsens_hobson-2b-v16.json", None),
    "v17": ("wsl", "reports/qsens_hobson-2b-v17.json", None),
    "v18": ("wsl", "reports/qsens_hobson-2b-v18.json", None),
    "v19": ("wsl", "reports/qsens_hobson-2b-v19.json", None),
}
# Multi-step sets (built in v14). v13 was measured before v14 trained; its numbers survive
# only in research/preregistrations/PREREGISTRATION-v14.md.
# In-distribution: a sample over the training tasks (21 from v5 on; v4 had 18, the
# scaling checkpoints 14). Not measured for v8-v12, v15, v16.
INDIST = {
    "v4": ("repo", "reports/indist_v4.json"),
    "v5": ("repo", "reports/indist_v5.json"),
    "v7": ("repo", "reports/v7_indist.json"),
    "v13": ("repo", "reports/hobson-2b-v13_indist.json"),
    "v14": ("repo", "reports/hobson-2b-v14_indist.json"),
    "v16": ("wsl", "reports/hobson-2b-v16_indist.json"),
    "v17": ("wsl", "reports/hobson-2b-v17_indist.json"),
    "v18": ("wsl", "reports/hobson-2b-v18_indist.json"),
    "scaling: 1.7B": ("repo", "reports/indist_head_linear.json"),
    "scaling: 1.7B, MLP head": ("repo", "reports/indist_head_mlp512.json"),
    "scaling: 4B": ("repo", "reports/indist_head_4b.json"),
}
MULTISTEP_TEXT = {
    "v13": {"boardgame": 0.519, "contractnli": 0.564, "hotpotqa": 0.705, "musique": 0.202}
}
MULTISTEP_ROWS = {
    r: f"reports/{r}_multistep_v14_eval.json" for r in ("v14", "v15", "v16", "v17", "v18", "v19")
}
GENERATED_ROWS = {
    r: f"reports/{r}_generated_v16_eval.json" for r in ("v14", "v16", "v17", "v18", "v19")
}

# Training histories (WSL checkpoints): the logged loss is the whole objective, and the two
# KL terms are logged unweighted, each averaged over every micro-batch, so the
# cross-entropy part is exactly loss - kl_frozen_weight * kl - teacher_weight * teacher_kl.
TRAINING = {"v17": "checkpoints/hobson-2b-v17"}

# Latency against question count, v14 on the RTX 3090 (WSL): one ~2,000-token state (90
# policy clauses) with N choice questions, mean of 5 runs after a warm-up, shared-prefix
# path against plain batched encoding. The script (bench_prefix.py, not committed)
# printed these and saved nothing, so they are entered by hand from its output, 2026-09-26.
PREFIX_BENCH = [  # (questions, prefix ms, batched ms, same answer on both paths)
    (1, 278, 278, 1),
    (4, 364, 998, 4),
    (8, 369, 1964, 8),
    (16, 445, 4086, 15),
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_csv(name, rows, fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jevbench-root", required=True)
    ap.add_argument("--wsl-root", required=True)
    args = ap.parse_args()
    jb = Path(args.jevbench_root)
    roots = {"repo": REPO, "wsl": Path(args.wsl_root)}

    # --- JevBench tasks: id, family, tier (no task text) ---
    tasks = {}
    for t in read_jsonl(jb / "all4096.jsonl"):
        src = t["provenance"]["source"]
        tier = "hard" if "hard tier" in src else "easy" if "easy tier" in src else "standard"
        tasks[t["id"]] = {"task_id": t["id"], "family": t["family"], "tier": tier}
    write_csv("jevbench_tasks.csv", list(tasks.values()), ["task_id", "family", "tier"])

    # --- JevBench per run ---
    runs, results, families, latency = [], [], [], []
    catalogue = (
        [(r, d, kind, parent, torso, change) for r, d, kind, parent, torso, change in RUNS]
        + [(r, d, "side", "", "", change) for r, d, change in SIDE]
        + [(r, d, "reference", "", "", change) for r, d, change in REFERENCES]
    )
    for order, (run, d, kind, parent, torso, change) in enumerate(catalogue):
        res = read_jsonl(jb / d / "results.jsonl")
        assert len(res) == 231, (d, len(res))
        s = json.load(open(jb / d / "summary.json", encoding="utf-8"))
        correct = {r["task_id"]: bool(r["correct"]) for r in res}
        tier_n = defaultdict(lambda: [0, 0])
        fam = defaultdict(lambda: [0, 0])
        for tid, ok in correct.items():
            t = tasks[tid]
            tier_n[t["tier"]][0] += ok
            tier_n[t["tier"]][1] += 1
            fam[(t["family"], t["tier"])][0] += ok
            fam[(t["family"], t["tier"])][1] += 1
            results.append({"run": run, "task_id": tid, "correct": int(ok)})
        first = min(res, key=lambda r: r["ts"])["task_id"]  # the request that warms the server up
        latency += [
            {
                "run": run,
                "task_id": r["task_id"],
                "input_tokens": r["usage"]["input_tokens"],
                "latency_ms": round(r["latency_s"] * 1000, 1),
                "first_request": int(r["task_id"] == first),
            }
            for r in res
        ]
        for (f, tier), (k, n) in sorted(fam.items()):
            families.append(
                {
                    "run": run,
                    "family": f,
                    "tier": tier,
                    "correct": k,
                    "n": n,
                    "accuracy": round(k / n, 4),
                }
            )
        mtime = datetime.fromtimestamp(os.path.getmtime(jb / d / "results.jsonl"))
        runs.append(
            {
                "order": order,
                "run": run,
                "kind": kind,
                "parent": parent,
                "torso": torso,
                "change": change,
                "jevbench_date": mtime.strftime("%Y-%m-%d"),
                "n_correct": int(s["n_correct"]),
                "accuracy": round(s["accuracy"], 4),
                "macro_accuracy": round(s["macro_accuracy"], 4),
                "ece": round(s["ece"]["ece"], 4),
                "brier": round(s["brier_mean"], 4),
                "ordinal_mae": round(s["ordinal_mae"], 4),
                "paraphrase_consistency": round(s["paraphrase_consistency"]["agreement"], 4),
                "latency_p50_s": round(s["latency"]["p50_s"], 3),
                "latency_p95_s": round(s["latency"]["p95_s"], 3),
                **{f"{t}_correct": tier_n[t][0] for t in ("easy", "standard", "hard")},
                **{f"{t}_n": tier_n[t][1] for t in ("easy", "standard", "hard")},
                "source": f"{d}/summary.json",
            }
        )
    write_csv("runs.csv", runs, list(runs[0]))
    write_csv("jevbench_results.csv", results, ["run", "task_id", "correct"])
    write_csv(
        "jevbench_families.csv", families, ["run", "family", "tier", "correct", "n", "accuracy"]
    )
    write_csv("jevbench_latency.csv", latency, list(latency[0]))
    write_csv(
        "latency_questions.csv",
        [
            {
                "run": "v14",
                "questions": q,
                "prefix_ms": p,
                "batched_ms": b,
                "same_answer": s,
                "source": "bench_prefix.py output, 2026-09-26",
            }
            for q, p, b, s in PREFIX_BENCH
        ],
        ["run", "questions", "prefix_ms", "batched_ms", "same_answer", "source"],
    )

    # --- held-out short tasks ---
    held = []
    for run, (root, rel) in HELDOUT.items():
        d = json.load(open(roots[root] / rel, encoding="utf-8"))
        extra = sorted(set(d["by_task"]) - set(HELDOUT_TASKS))
        if extra:  # recombine the four common tasks from per-task means
            t = [d["by_task"][k] for k in HELDOUT_TASKS]
            n = sum(x["n"] for x in t)

            def mean(key, t=t, n=n):
                return sum(x[key] * x["n"] for x in t) / n

            def kind_acc(kind, d=d):
                ts = [d["by_task"][k] for k in HELDOUT_KIND_TASKS[kind]]
                return round(sum(x["accuracy"] * x["n"] for x in ts) / sum(x["n"] for x in ts), 4)

            row = {
                "run": run,
                "n": n,
                "accuracy": round(mean("accuracy"), 4),
                "ece": "",
                "nll": round(mean("nll"), 4),
                "mean_confidence": round(mean("mean_confidence"), 4),
                **{f"{k}_accuracy": kind_acc(k) for k in ("noul", "choice", "score")},
                "note": f"four common tasks recombined; report also held {', '.join(extra)}",
            }
        else:
            o = d["overall"]
            row = {
                "run": run,
                "n": o["n"],
                "accuracy": round(o["accuracy"], 4),
                "ece": round(o["ece"], 4),
                "nll": round(o["nll"], 4),
                "mean_confidence": round(o["mean_confidence"], 4),
                **{
                    f"{k}_accuracy": round(d["by_kind"][k]["accuracy"], 4)
                    for k in ("noul", "choice", "score")
                },
                "note": "",
            }
        row["source"] = rel
        held.append(row)
    write_csv("heldout_short.csv", held, list(held[0]))

    indist = []
    for run, (root, rel) in INDIST.items():
        d = json.load(open(roots[root] / rel, encoding="utf-8"))
        o = d["overall"]
        indist.append(
            {
                "run": run,
                "n": o["n"],
                "n_tasks": len(d["by_task"]),
                "accuracy": round(o["accuracy"], 4),
                "ece": round(o["ece"], 4),
                "nll": round(o["nll"], 4),
                **{
                    f"{k}_accuracy": round(d["by_kind"][k]["accuracy"], 4)
                    for k in ("noul", "choice", "score")
                },
                **{f"{k}_n": d["by_kind"][k]["n"] for k in ("noul", "choice", "score")},
                "source": rel,
            }
        )
    write_csv("indist.csv", indist, list(indist[0]))

    # --- question-sensitivity probe: held-out choice tasks ---
    probe = []
    for run, (root, rel, key) in PROBE.items():
        d = json.load(open(roots[root] / rel, encoding="utf-8"))["choice, held-out tasks"]
        key = key or next(k for k in d["real/acc"] if not k.endswith("frozen"))
        for form in ("first", "last", "not", "irrelevant"):
            probe.append(
                {
                    "run": run,
                    "form": form,
                    "same_answer_as_real": round(d[f"{form}/same_as_real"][key], 4),
                    "frozen_base": round(d[f"{form}/same_as_real"][f"{key} frozen"], 4),
                    "real_accuracy": round(d["real/acc"][key], 4),
                    "source": rel,
                }
            )
    write_csv("probe_heldout_choice.csv", probe, list(probe[0]))

    # --- multi-step sets and generated eval ---
    wsl = roots["wsl"]
    multi = [
        {"run": r, "source": s, "n": "", "accuracy": v, "measured_by": "PREREGISTRATION-v14.md"}
        for r, vals in MULTISTEP_TEXT.items()
        for s, v in vals.items()
    ]
    ms_tasks = [r["task"] for r in read_jsonl(wsl / "data" / "multistep_v14_eval.jsonl")]
    for run, rel in MULTISTEP_ROWS.items():
        ok = json.load(open(wsl / rel))
        by = defaultdict(lambda: [0, 0])
        for t, x in zip(ms_tasks, ok, strict=True):
            if x is None:
                continue
            by[t][0] += x
            by[t][1] += 1
        multi += [
            {"run": run, "source": t, "n": n, "accuracy": round(k / n, 4), "measured_by": rel}
            for t, (k, n) in sorted(by.items())
        ]
    gen_rows = read_jsonl(wsl / "data" / "generated_v16_eval.jsonl")
    for run, rel in GENERATED_ROWS.items():
        ok = json.load(open(wsl / rel))
        k, n = sum(bool(x) for x in ok), sum(x is not None for x in ok)
        multi.append(
            {
                "run": run,
                "source": "generated (held-out domains)",
                "n": n,
                "accuracy": round(k / n, 4),
                "measured_by": rel,
            }
        )
        by = defaultdict(lambda: [0, 0])
        for r, x in zip(gen_rows, ok, strict=True):
            by[r["task"]][0] += bool(x)
            by[r["task"]][1] += 1
        multi += [
            {"run": run, "source": t, "n": n2, "accuracy": round(k2 / n2, 4), "measured_by": rel}
            for t, (k2, n2) in sorted(by.items())
        ]
    write_csv(
        "multistep_and_generated.csv", multi, ["run", "source", "n", "accuracy", "measured_by"]
    )

    # --- training histories ---
    train = []
    for run, rel in TRAINING.items():
        cfg = json.load(open(wsl / rel / "train_config.json"))
        wk, wt = cfg.get("kl_frozen_weight", 0.0), cfg.get("teacher_weight", 0.0)
        for e in json.load(open(wsl / rel / "history.json")):
            row = {
                "run": run,
                "step": e["step"],
                "kind": "val" if "val_loss" in e else "train",
                "loss": "",
                "ce": "",
                "kl_frozen": "",
                "kl_teacher": "",
                "kl_frozen_weight": wk,
                "teacher_weight": wt,
                "val_loss": "",
                "val_acc": "",
                "source": f"{rel}/history.json",
            }
            if row["kind"] == "val":
                row.update(val_loss=round(e["val_loss"], 5), val_acc=round(e["val_acc"], 5))
            else:
                kl, tkl = e.get("kl", 0.0), e.get("teacher_kl", 0.0)
                row.update(
                    loss=round(e["loss"], 5),
                    kl_frozen=round(kl, 5),
                    kl_teacher=round(tkl, 5),
                    ce=round(e["loss"] - wk * kl - wt * tkl, 5),
                )
            train.append(row)
    write_csv("training_history.csv", train, list(train[0]))


if __name__ == "__main__":
    main()
