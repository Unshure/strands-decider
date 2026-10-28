#!/usr/bin/env python3
"""Draw the report's figures from research/data/*.csv into research/figures/*.svg.

Reads only the CSVs written by collect.py, so it can be re-run anywhere.

    python research/scripts/make_figures.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from svgchart import dot_chart, heatmap, line_chart, scatter_chart, small_multiples, trajectory

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
RESOLUTION = "differences below ~0.043 (10 tasks) are within sampling noise"


def rows(name):
    with open(DATA / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    FIG.mkdir(exist_ok=True)
    runs = rows("runs.csv")
    gens = [r for r in runs if r["kind"] in ("default", "experiment")]
    ref = {r["run"]: r for r in runs if r["kind"] == "reference"}
    labels = [r["run"] for r in gens]

    def pts(metric, fmt="{:.3f}", scale=1.0, subset=None, value=None):
        out = []
        for r in gens:
            if subset and r["run"] not in subset:
                continue
            v = value(r) if value else float(r[metric]) * scale
            if v is None:
                continue
            out.append(
                {
                    "x_label": r["run"],
                    "value": v,
                    "kind": r["kind"],
                    "tip": f"{r['run']} ({r['kind']}): {r['change']} - {fmt.format(v)}",
                }
            )
        return out

    def refs(metric, names=("decider-2b v11", "Qwen3.5-4B, frozen")):
        return [(n, float(ref[n][metric])) for n in names]

    # 1. JevBench accuracy
    acc = pts("accuracy")
    last_default = [r["run"] for r in gens if r["kind"] == "default"][-1]
    for p, r in zip(acc, gens, strict=True):
        p["tip"] += f" ({r['n_correct']}/231)"
        if r["run"] == last_default:
            p["label"] = f"{float(r['accuracy']):.3f} ({r['n_correct']}/231)"
    (FIG / "jevbench_accuracy.svg").write_text(
        dot_chart(
            "JevBench accuracy by generation",
            f"Public set, 231 tasks. {RESOLUTION}.",
            labels,
            acc,
            refs("accuracy"),
            ylabel="accuracy",
            cap=1.0,
        ),
        encoding="utf-8",
    )

    # 2. Accuracy by tier
    tiers = []
    for t, n in (("easy", 48), ("standard", 72), ("hard", 111)):
        tiers.append(
            {
                "title": f"{t} tier ({n} tasks)",
                "cap": 1.0,
                "points": pts(None, value=lambda r, t=t: int(r[f"{t}_correct"]) / int(r[f"{t}_n"])),
                "refs": [
                    (n_, int(ref[n_][f"{t}_correct"]) / int(ref[n_][f"{t}_n"]))
                    for n_ in ("decider-2b v11", "Qwen3.5-4B, frozen")
                ],
            }
        )
    (FIG / "jevbench_tiers.svg").write_text(
        small_multiples(
            "JevBench accuracy by tier",
            "The easy tier saturated at v5; the gains since are in the hard tier.",
            tiers,
            labels,
        ),
        encoding="utf-8",
    )

    # 3-4. Calibration and Brier
    (FIG / "jevbench_ece.svg").write_text(
        dot_chart(
            "JevBench calibration error (ECE) by generation",
            "Expected calibration error after per-primitive temperature fitting on held-out classification.",
            labels,
            pts("ece"),
            refs("ece"),
            ylabel="ECE",
            lower_is_better=True,
        ),
        encoding="utf-8",
    )
    (FIG / "jevbench_brier.svg").write_text(
        dot_chart(
            "JevBench Brier score by generation",
            "Mean Brier score over the answer distribution.",
            labels,
            pts("brier"),
            refs("brier"),
            ylabel="Brier",
            lower_is_better=True,
        ),
        encoding="utf-8",
    )

    # 5. Secondary JevBench metrics
    (FIG / "jevbench_secondary.svg").write_text(
        small_multiples(
            "Other JevBench metrics by generation",
            "Each panel on its own scale.",
            [
                {
                    "title": "macro accuracy (mean over families)",
                    "points": pts("macro_accuracy"),
                    "refs": refs("macro_accuracy"),
                    "cap": 1.0,
                },
                {
                    "title": "ordinal MAE (score questions)",
                    "lower_is_better": True,
                    "points": pts("ordinal_mae"),
                    "refs": refs("ordinal_mae"),
                },
                {
                    "title": "paraphrase consistency (same answer to reworded tasks)",
                    "points": pts("paraphrase_consistency"),
                    "refs": refs("paraphrase_consistency"),
                    "cap": 1.0,
                },
            ],
            labels,
        ),
        encoding="utf-8",
    )

    # 6. Held-out short tasks
    held = {r["run"]: r for r in rows("heldout_short.csv")}
    hl = [x for x in labels if x in held]
    hp = [
        {
            "x_label": r,
            "value": float(held[r]["accuracy"]),
            "kind": next(g["kind"] for g in gens if g["run"] == r),
            "tip": f"{r}: {float(held[r]['accuracy']):.3f} on {held[r]['n']} rows",
        }
        for r in hl
    ]
    (FIG / "heldout_short.svg").write_text(
        dot_chart(
            "Held-out classification tasks by generation",
            "emotion, hate_severity, massive_intent, sarcasm: tasks no model trained on. v5 on these four tasks only; v6 and v8-v10 not measured.",
            hl,
            hp,
            ylabel="accuracy",
            cap=1.0,
        ),
        encoding="utf-8",
    )

    # 7. Question-sensitivity probe
    probe = defaultdict(list)
    frozen = []
    for r in rows("probe_heldout_choice.csv"):
        probe[r["run"]].append(float(r["same_answer_as_real"]))
        frozen.append(float(r["frozen_base"]))
    ql = [x for x in labels if x in probe]
    qp = [
        {
            "x_label": r,
            "value": sum(probe[r]) / len(probe[r]),
            "kind": next(g["kind"] for g in gens if g["run"] == r),
            "tip": f"{r}: same answer to a changed question {sum(probe[r]) / len(probe[r]):.3f} of the time",
        }
        for r in ql
    ]
    (FIG / "question_sensitivity.svg").write_text(
        dot_chart(
            "Same answer to a changed question",
            "Held-out choice tasks, state and options fixed, question changed (first / last / NOT / irrelevant).",
            ql,
            qp,
            [("frozen torsos (mean)", sum(frozen) / len(frozen))],
            ylabel="share unchanged",
            lower_is_better=True,
            y_range=(0.0, 1.0),
        ),
        encoding="utf-8",
    )

    # 8. Multi-step sets and generated questions (v13 onwards)
    ms = defaultdict(dict)
    for r in rows("multistep_and_generated.csv"):
        ms[r["source"]][r["run"]] = float(r["accuracy"])
    ml = ["v13", "v14", "v15", "v16", "v17", "v18", "v19"]
    panels = []
    for src, name in (
        ("hotpotqa", "HotpotQA (held out: never trained on)"),
        ("musique", "MuSiQue (dev pairs)"),
        ("contractnli", "ContractNLI (dev)"),
        ("boardgame", "BoardgameQA (valid)"),
        ("generated (held-out domains)", "generated questions, held-out domains"),
    ):
        panels.append(
            {
                "title": name,
                "cap": 1.0,
                "points": [
                    {
                        "x_label": r,
                        "value": v,
                        "kind": next(g["kind"] for g in gens if g["run"] == r),
                        "tip": f"{r}: {v:.3f}",
                    }
                    for r, v in ms[src].items()
                ],
            }
        )
    (FIG / "multistep.svg").write_text(
        small_multiples(
            "Multi-step and generated-document evaluations, v13-v19",
            "v13 figures from PREREGISTRATION-v14.md; generated questions measured for v14 and v16-v19 only.",
            panels,
            ml,
            cols=2,
            panel_height=150,
            width=860,
        ),
        encoding="utf-8",
    )

    # 9. Hard-tier families heatmap
    fam = defaultdict(dict)
    for r in rows("jevbench_families.csv"):
        if r["tier"] == "hard":
            fam[r["family"]][r["run"]] = (
                float(r["accuracy"]),
                f"{r['family']}, {r['run']}: {r['correct']}/{r['n']}",
            )
    fams = sorted(fam, key=lambda f: -sum(v for v, _ in fam[f].values()))
    cols = [*labels, "decider-2b v11"]
    cells = {(f, c): fam[f][c] for f in fams for c in cols if c in fam[f]}
    heat = heatmap(
        "JevBench hard tier by family and generation",
        "Accuracy within each hard-tier family (5-19 tasks each); decider-2b v11 at right for reference.",
        fams,
        [c if c != "decider-2b v11" else "dv11" for c in cols],
        {(f, "dv11" if c == "decider-2b v11" else c): v for (f, c), v in cells.items()},
        gap_before="dv11",
    )
    (FIG / "hard_families.svg").write_text(heat, encoding="utf-8")

    # 10. Trajectory: accuracy against Brier and against ECE
    on_chart = ("decider-2b v10", "decider-2b v11", "Qwen3.5-4B, frozen", "Qwen3.5-2B, frozen")
    off_chart = [ref[n] for n in ("Qwen3-8B, frozen", "Qwen3-1.7B, frozen")]
    lineage = [r["run"] for r in gens if r["kind"] == "default"]

    # label placement (dx, dy in px; negative dx anchors the label's end) where points crowd
    place = {
        "brier": {
            "v5": (-10, -30, True),
            "v6": (-2, -40, True),
            "v7": (-8, 16),
            "v13": (6, -8),
            "v14": (-8, 16),
            "v16": (-2, 20),
            "v17": (8, -6),
            "v18": (8, 18),
            "decider-2b v10": (8, -8),
            "decider-2b v11": (8, -8),
            "Qwen3.5-4B, frozen": (-8, -9),
            "Qwen3.5-2B, frozen": (8, -8),
        },
        "ece": {
            "v5": (2, -12),
            "v6": (2, -12),
            "v7": (8, 14),
            "v13": (-8, -9),
            "v14": (-8, -9),
            "v16": (0, 16),
            "v17": (8, -6),
            "v18": (8, 6),
            "v19": (0, 16),
            "decider-2b v10": (8, -8),
            "decider-2b v11": (8, -8),
            "Qwen3.5-4B, frozen": (-8, -9),
            "Qwen3.5-2B, frozen": (-8, -8),
        },
    }

    def tpoints(metric):
        out = []
        for r in gens + [ref[n] for n in on_chart]:
            kind = r["kind"]
            x, y = float(r["accuracy"]), float(r[metric])
            dx, dy, *lead = place[metric].get(r["run"], (8, -8))
            out.append(
                {
                    "id": r["run"],
                    "x": x,
                    "y": y,
                    "kind": kind,
                    "dx": dx,
                    "dy": dy,
                    "leader": bool(lead and lead[0]),
                    # the newest run is labelled too, whatever its kind
                    "label": r["run"]
                    if kind in ("default", "reference") or r["run"] == gens[-1]["run"]
                    else None,
                    "tip": f"{r['run']}: accuracy {x:.3f} ({r['n_correct']}/231), {metric} {y:.3f}"
                    + (f" - {r['change']}" if r["change"] else ""),
                }
            )
        return out

    note = (
        "Off the chart, frozen and badly calibrated: "
        + "; ".join(
            f"{r['run']} accuracy {float(r['accuracy']):.3f}, Brier {float(r['brier']):.3f}, "
            f"ECE {float(r['ece']):.3f}"
            for r in off_chart
        )
        + "."
    )
    (FIG / "trajectory.svg").write_text(
        trajectory(
            "Accuracy against calibration, generation by generation",
            "JevBench public set. The line follows the default recipe "
            + " -> ".join(lineage)
            + ".",
            [
                {
                    "title": "Brier score (lower is better)",
                    "points": tpoints("brier"),
                    "x_range": (0.60, 0.82),
                    "y_range": (0.24, 0.50),
                },
                {
                    "title": "ECE (lower is better)",
                    "points": tpoints("ece"),
                    "x_range": (0.60, 0.82),
                    "y_range": (0.04, 0.16),
                },
            ],
            lineage,
            note=note,
        ),
        encoding="utf-8",
    )

    # 10b-c. the same two panels, one per file
    for metric, name, y_range in (
        ("brier", "Brier score", (0.24, 0.50)),
        ("ece", "ECE", (0.04, 0.16)),
    ):
        solo_note = (
            "Off the chart, frozen and badly calibrated: "
            + "; ".join(
                f"{r['run'].removesuffix(', frozen')} {float(r['accuracy']):.3f} accuracy, {float(r[metric]):.3f} {name}"
                for r in off_chart
            )
            + "."
        )
        (FIG / f"trajectory_{metric}.svg").write_text(
            trajectory(
                f"Accuracy against {name}, generation by generation",
                "JevBench public set. The line follows the default recipe "
                + " -> ".join(lineage)
                + ".",
                [
                    {
                        "title": f"{name} (lower is better)",
                        "points": tpoints(metric),
                        "x_range": (0.60, 0.82),
                        "y_range": y_range,
                    }
                ],
                lineage,
                note=solo_note,
                width=680,
            ),
            encoding="utf-8",
        )

    # 11-12. Held-out classification: accuracy against calibration; fit against generalisation
    held_rows = {r["run"]: r for r in rows("heldout_short.csv")}
    ind_rows = {r["run"]: r for r in rows("indist.csv")}
    kind_of = {r["run"]: r["kind"] for r in gens} | {"v4": "default"}
    change_of = {r["run"]: r["change"] for r in gens} | {
        "v4": "instruction variants, noul 4 -> 8 types"
    }
    scaling = {"scaling: 1.7B": "Qwen3-1.7B, scaling test", "scaling: 4B": "Qwen3-4B, scaling test"}
    # v5 is left out here: its temperatures were fitted on a held-out set two-fifths RuleTaker,
    # so its calibration on these four tasks is not comparable (its accuracy is, see below)
    h_lineage = ["v4"] + [r for r in lineage if r in held_rows and r != "v5"]
    place_h = {  # v14, v13, v4, v7 sit in a row at NLL 0.882: alternate labels above and below
        "nll": {
            "v4": (0, 18),
            "v7": (4, -10),
            "v13": (0, -10),
            "v14": (-14, 16),
            "v16": (-8, -6),
            "v17": (-10, 34, True),
            "v18": (4, 18),
            "Qwen3-1.7B, scaling test": (8, -8),
            "Qwen3-4B, scaling test": (-8, -8),
        },
        "ece": {
            "v4": (8, 4),
            "v7": (8, 4),
            "v13": (-8, -6),
            "v14": (0, 18),
            "v16": (-8, -6),
            "v17": (8, 4),
            "v18": (8, -8),
            "v19": (8, 8),
            "Qwen3-1.7B, scaling test": (10, 55, True),
            "Qwen3-4B, scaling test": (-8, -8),
        },
    }

    def hpoints(metric):
        out = []
        for run, r in held_rows.items():
            if run in ("scaling: 1.7B, MLP head", "v5") or r[metric] == "":
                continue
            name = scaling.get(run, run)
            kind = "reference" if run in scaling else kind_of[run]
            x, y = float(r["accuracy"]), float(r[metric])
            dx, dy, *lead = place_h[metric].get(name, (8, -8))
            out.append(
                {
                    "id": run,
                    "x": x,
                    "y": y,
                    "kind": kind,
                    "dx": dx,
                    "dy": dy,
                    "leader": bool(lead and lead[0]),
                    "label": name if kind in ("default", "reference") else None,
                    "tip": f"{name}: held-out accuracy {x:.3f}, {metric.upper()} {y:.3f} (n={r['n']})"
                    + (f" - {change_of[run]}" if run in change_of else ""),
                }
            )
        return out

    (FIG / "heldout_trajectory.svg").write_text(
        trajectory(
            "Held-out classification: accuracy against calibration",
            "emotion, hate_severity, massive_intent, sarcasm: tasks no model trained on. "
            "Line: default recipe " + " -> ".join(h_lineage) + " (v6 and v8-v10 were not scored).",
            [
                {
                    "title": "NLL (log loss, lower is better)",
                    "points": hpoints("nll"),
                    "xlabel": "held-out accuracy",
                    "x_range": (0.61, 0.69),
                    "y_range": (0.78, 0.94),
                },
                {
                    "title": "ECE (lower is better)",
                    "points": hpoints("ece"),
                    "xlabel": "held-out accuracy",
                    "x_range": (0.61, 0.69),
                    "y_range": (0.04, 0.08),
                },
            ],
            h_lineage,
            ref_label="scaling test: same corpus and steps, torso only",
            note="v5 left out: its temperatures were fitted on a held-out set that was two-fifths RuleTaker. "
            "Scaling runs: same four tasks, a different row sample.",
        ),
        encoding="utf-8",
    )

    place_g = {
        "v4": (-8, -8),
        "v5": (8, 4),
        "v7": (8, -8),
        "v13": (-8, -6),
        "v14": (8, 4),
        "v16": (-8, 12),
        "v18": (-12, 4),
        "Qwen3-1.7B, scaling test": (0, 18),
        "Qwen3-4B, scaling test": (8, -8),
    }
    # v17 and v18 sit 0.0007 apart on both axes, so their dots overlap: one label for both
    shared_label = {"v17": None, "v18": "v17, v18"}
    gpts = []
    for run, r in ind_rows.items():
        if run == "scaling: 1.7B, MLP head" or run not in held_rows:
            continue
        name = scaling.get(run, run)
        kind = "reference" if run in scaling else kind_of[run]
        x, y = float(r["accuracy"]), float(held_rows[run]["accuracy"])
        dx, dy = place_g.get(name, (8, -8))
        gpts.append(
            {
                "id": run,
                "x": x,
                "y": y,
                "kind": kind,
                "label": shared_label.get(run, name),
                "dx": dx,
                "dy": dy,
                "tip": f"{name}: in-distribution {x:.3f} ({r['n_tasks']} training tasks), held-out {y:.3f}",
            }
        )
    measured = [r for r in ind_rows if r in held_rows and r not in scaling and r in kind_of]
    g_lineage = [r for r in ["v4", *lineage] if r in measured]
    (FIG / "fit_vs_generalisation.svg").write_text(
        trajectory(
            "Fitting the training tasks against generalising to unseen ones",
            "In-distribution: a fixed 6,000-row sample of the short-task training file, mostly rows trained on. "
            "Held-out: four tasks no model trained on.",
            [
                {
                    "title": "held-out accuracy (higher is better)",
                    "points": gpts,
                    "xlabel": "in-distribution accuracy",
                    "better": ("better: right and up", "top"),
                    "x_range": (0.84, 0.89),
                    "y_range": (0.61, 0.69),
                }
            ],
            g_lineage,
            ref_label="scaling test: torso only",
            note=f"Measured for {', '.join(measured)}. Training tasks: v4 18, v5 onwards 21, scaling runs 14. "
            "v5 held-out: its four common tasks, recombined.",
        ),
        encoding="utf-8",
    )

    # 12b. The same, split by question type, one panel each on its own scale. Held-out, each
    # type is one or two tasks, so these are close to per-task results.
    kinds = [("noul", "sarcasm"), ("choice", "emotion, massive_intent"), ("score", "hate_severity")]
    ref_short = {"scaling: 1.7B": "1.7B", "scaling: 4B": "4B"}
    first, last = g_lineage[0], g_lineage[-1]
    place_k = {
        "noul": {last: (8, 14), "1.7B": (-8, 4)},  # label offsets where points crowd
        "choice": {last: (0, -10), "1.7B": (-8, 4), "4B": (-8, -4)},
    }

    def kind_points(kind):
        pts = []
        for run in g_lineage + list(ref_short):
            x, y = (
                float(ind_rows[run][f"{kind}_accuracy"]),
                float(held_rows[run][f"{kind}_accuracy"]),
            )
            label = ref_short.get(run) or (run if run in (first, last) else None)
            dx, dy = place_k.get(kind, {}).get(label, (8, -8))
            pts.append(
                {
                    "id": run,
                    "x": x,
                    "y": y,
                    "label": label,
                    "dx": dx,
                    "dy": dy,
                    "kind": "reference" if run in ref_short else "default",
                    "tip": f"{scaling.get(run, run)}, {kind}: in-distribution {x:.3f}, held-out {y:.3f}",
                }
            )
        return pts

    def span(vals, pad):
        return min(vals) - pad, max(vals) + pad

    panels = []
    for kind, tasks in kinds:
        pts = kind_points(kind)
        panels.append(
            {
                "title": f"{kind} (held-out: {tasks})",
                "points": pts,
                "xlabel": "in-distribution accuracy",
                "better": ("better: right and up", "top"),
                "x_range": span([p["x"] for p in pts], 0.01),
                "y_range": span([p["y"] for p in pts], 0.02),
            }
        )
    (FIG / "fit_vs_generalisation_by_kind.svg").write_text(
        trajectory(
            "Fitting against generalising, by question type",
            "Each panel has its own scale. In-distribution: the fixed training-file sample, mostly rows trained on.",
            panels,
            g_lineage,
            ref_label="scaling test: torso only",
            width=1040,
            note="Held-out rows per type: noul 1,104 (one standard error ~0.015), choice 3,467 (~0.008), score "
            f"1,429 (~0.013). Line: {' -> '.join(g_lineage)}; labelled at its ends.",
        ),
        encoding="utf-8",
    )

    # 13-15. v17's training run: the objective by term, then validation loss and accuracy
    hist = [r for r in rows("training_history.csv") if r["run"] == "v17"]
    tr = [r for r in hist if r["kind"] == "train"]
    wk = float(tr[0]["kl_frozen_weight"])
    last = int(tr[-1]["step"])
    # the final evaluation repeats the one a step before it; keep the later of any such pair
    va = [
        r
        for i, r in enumerate(hist)
        if r["kind"] == "val"
        and not any(
            o["kind"] == "val" and 0 < int(o["step"]) - int(r["step"]) <= 10 for o in hist[i + 1 :]
        )
    ]

    def line(key, scale=1.0):
        return [(int(r["step"]), float(r[key]) * scale) for r in tr]

    x_axis = "optimiser step (32 examples each; one epoch)"
    (FIG / "training_v17_loss.svg").write_text(
        line_chart(
            "v17 training loss, by term",
            "Each point averages 20 optimiser steps. The objective is the sum of the three terms below it.",
            [
                {
                    "label": "total objective",
                    "end_label": "total",
                    "cls": "0",
                    "points": line("loss"),
                },
                {
                    "label": "cross-entropy on the gold option",
                    "end_label": "cross-entropy",
                    "cls": "1",
                    "points": line("ce"),
                },
                {
                    "label": f"{wk:g} × KL to the frozen torso",
                    "end_label": "KL, frozen torso",
                    "cls": "2",
                    "points": line("kl_frozen", wk),
                },
                {
                    "label": "KL to v14 (multi-step rows)",
                    "end_label": "KL, v14",
                    "cls": "3",
                    "points": line("kl_teacher", float(tr[0]["teacher_weight"])),
                },
            ],
            x_axis,
            "loss (nats)",
            (0, last),
            (0, 1.6),
            y_ticks=8,
            notes=(
                "KL to v14 is averaged over every batch, including those with no multi-step rows, so it reads "
                "lower than on the rows it applies to.",
            ),
            width=820,
        ),
        encoding="utf-8",
    )
    val_note = (
        "The same 400 rows at every checkpoint: the first 50 batches of the 3% of training files held "
        "back. In-distribution, so this tracks fitting, not generalisation."
    )
    (FIG / "training_v17_val_loss.svg").write_text(
        line_chart(
            "v17 validation loss",
            "Cross-entropy on held-back training rows, every 500 steps and at the end of the epoch.",
            [
                {
                    "label": "validation loss",
                    "end_label": "",
                    "cls": "1",
                    "points": [(int(r["step"]), float(r["val_loss"])) for r in va],
                }
            ],
            x_axis,
            "cross-entropy (nats)",
            (0, last),
            (0.35, 0.55),
            notes=(val_note,),
            markers=True,
            width=820,
        ),
        encoding="utf-8",
    )
    (FIG / "training_v17_val_acc.svg").write_text(
        line_chart(
            "v17 validation accuracy",
            "Top-option accuracy on held-back training rows, every 500 steps and at the end of the epoch. "
            "At 400 rows, one standard error is about 0.017.",
            [
                {
                    "label": "validation accuracy",
                    "end_label": "",
                    "cls": "1",
                    "points": [(int(r["step"]), float(r["val_acc"])) for r in va],
                }
            ],
            x_axis,
            "accuracy",
            (0, last),
            (0.80, 0.90),
            notes=(val_note,),
            cap=1.0,
            y_ticks=5,
            markers=True,
            width=820,
        ),
        encoding="utf-8",
    )
    # 16-17. Latency: against question count (v14 benchmark), against prompt length (v18 JevBench)
    lq = rows("latency_questions.csv")
    ms = "{:,.0f}".format
    (FIG / "latency_vs_questions.svg").write_text(
        line_chart(
            "Latency against number of questions",
            "One ~2,000-token state with N choice questions over it. v14 on an RTX 3090, mean of 5 runs; "
            "v17 and v18 have the same architecture.",
            [
                {
                    "label": "shared prefix (the default)",
                    "end_label": "shared prefix",
                    "cls": "1",
                    "points": [(int(r["questions"]), float(r["prefix_ms"])) for r in lq],
                },
                {
                    "label": "batched, no prefix cache",
                    "end_label": "batched",
                    "cls": "2",
                    "points": [(int(r["questions"]), float(r["batched_ms"])) for r in lq],
                },
            ],
            "questions per request",
            "latency (ms)",
            (0, 17),
            (0, 4100),
            y_ticks=5,
            x_ticks=[1, 4, 8, 16],
            y_fmt=ms,
            markers=True,
            x_name="questions",
            width=820,
            notes=(
                "Shared prefix: the state is encoded once and every question reads its cache. Batched: state "
                "and question encoded once per question.",
                "Answers agree on both paths except one of 16 (bf16 rounding). Timings from bench_prefix.py "
                "output, 2026-09-26; not re-measured on v18.",
            ),
        ),
        encoding="utf-8",
    )

    lat = [r for r in rows("jevbench_latency.csv") if r["run"] == "v18"]
    warm = [r for r in lat if r["first_request"] == "1"]
    lat = [r for r in lat if r["first_request"] == "0"]
    ordered = sorted(float(r["latency_ms"]) for r in lat)

    def pct(q):  # nearest rank
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    p50, p95, p99 = pct(0.50), pct(0.95), pct(0.99)
    (FIG / "latency_vs_tokens_v18.svg").write_text(
        scatter_chart(
            "v18 latency against prompt length",
            f"JevBench public set: one question per request through the local server, HTTP round trip "
            f"included. {len(lat)} requests on an RTX 3090.",
            [
                {
                    "x": int(r["input_tokens"]),
                    "y": float(r["latency_ms"]),
                    "tip": f"{r['task_id']}: {int(r['input_tokens']):,} tokens, {float(r['latency_ms']):.0f} ms",
                }
                for r in lat
            ],
            "input tokens",
            "latency (ms)",
            (0, 3100),
            (0, 450),
            y_ticks=5,
            y_fmt=ms,
            refs=[(f"p50 {p50:.0f} ms", p50), (f"p95 {p95:.0f} ms", p95)],
            width=820,
            notes=(
                f"Left out: the first request ({float(warm[0]['latency_ms']) / 1000:.1f} s, server warm-up). "
                f"p99 is {p99:.0f} ms, set by the top three requests, so it is not drawn.",
                "Percentiles are nearest-rank over these requests; JevBench's own summary includes the "
                "warm-up request.",
            ),
        ),
        encoding="utf-8",
    )
    print("\n".join(sorted(p.name for p in FIG.glob("*.svg"))))


if __name__ == "__main__":
    main()
