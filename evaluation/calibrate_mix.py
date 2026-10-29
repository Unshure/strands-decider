"""Refit a checkpoint's noul/choice temperatures on the calibration halves of every
held-out set, and score old against new on their test halves
(research/preregistrations/PREREGISTRATION-v19-calmix.md). Score keeps its temperature. Writes the refitted
temperatures into a copy of the checkpoint, never the original.

    python -m evaluation.calibrate_mix [CHECKPOINT] [COPY]
      (defaults: checkpoints/hobson-2b-v19 checkpoints/hobson-2b-v19-calmix)

The six sets are built by recipe.sh (build, multistep, generated, adequacy); halves are
evaluate.partition_examples's deterministic split, per file; at most 500 calibration
rows and 3,000 test rows per set. The rule fixed before the v19 fit: adopt if the
equal-weight mean ECE over the six test halves falls and mean NLL rises by at most 0.005.
"""

import json
import shutil
import sys

from strands_decider.modeling import StrandsDeciderModel
from training.data.format import read_jsonl
from training.evaluate import (
    collect_logits,
    fit_temperature_by_kind,
    partition_examples,
    predictions_from_logits,
    sample_examples,
    summarise,
)

SETS = [
    "holdout_v5_norule",
    "multistep_v14_eval",
    "generated_v16_eval",
    "generated_v18_eval",
    "adequacy_hs2_eval",
    "adequacy_gen_eval",
]
CAP, TEST_CAP = 500, 3000


def brier(preds):
    return sum(sum((q - (i == p.label)) ** 2 for i, q in enumerate(p.probs)) for p in preds) / len(
        preds
    )


def main(argv):
    src = argv[0] if argv else "checkpoints/hobson-2b-v19"
    dst = argv[1] if len(argv) > 1 else src + "-calmix"
    model = StrandsDeciderModel.load(src)
    old = dict(model.config.temperature_by_kind)
    eps = model.config.ordinal_smoothing
    print("temperatures:", old, "window", model.config.max_length, flush=True)

    def logits_for(exs):
        return collect_logits(model, exs, batch_size=16, max_length=model.config.max_length)

    calib, test = [], {}
    for name in SETS:
        rows = list(read_jsonl(f"data/{name}.jsonl"))
        calib += sample_examples(partition_examples(rows, "calib"), CAP)
        test[name] = sample_examples(partition_examples(rows, "test"), TEST_CAP)
    print(
        "calibration mix:",
        len(calib),
        {k: sum(e.kind == k for e in calib) for k in ("noul", "choice", "score")},
    )

    L, Y, S, E = logits_for(calib)
    fit = fit_temperature_by_kind(L, Y, S, E, objective="ece", ordinal_smoothing=eps)
    new = {"noul": fit["noul"], "choice": fit["choice"], "score": old["score"]}
    print("refitted:", fit, "->", new, flush=True)

    report = {}
    for name, exs in test.items():
        L, Y, S, E = logits_for(exs)
        row = {}
        for tag, t in (("old", old), ("mix", new)):
            preds = predictions_from_logits(L, Y, S, E, t, eps)
            o = summarise(preds, by_task=False)["overall"]
            row[tag] = {
                "n": o["n"],
                "acc": o["accuracy"],
                "ece": o["ece"],
                "nll": o["nll"],
                "brier": brier(preds),
                "conf": o["mean_confidence"],
            }
        report[name] = row
        print(
            f"{name:20} n={row['old']['n']:5} acc {row['old']['acc']:.3f} | ECE {row['old']['ece']:.4f} -> "
            f"{row['mix']['ece']:.4f} | NLL {row['old']['nll']:.4f} -> {row['mix']['nll']:.4f} | "
            f"conf {row['old']['conf']:.3f} -> {row['mix']['conf']:.3f}",
            flush=True,
        )

    mean = lambda tag, k: sum(r[tag][k] for r in report.values()) / len(report)  # noqa: E731
    d_ece, d_nll = mean("mix", "ece") - mean("old", "ece"), mean("mix", "nll") - mean("old", "nll")
    adopt = d_ece < 0 and d_nll <= 0.005
    print(
        f"mean ECE {mean('old', 'ece'):.4f} -> {mean('mix', 'ece'):.4f} ({d_ece:+.4f}); "
        f"NLL {mean('old', 'nll'):.4f} -> {mean('mix', 'nll'):.4f} ({d_nll:+.4f}) => "
        f"{'ADOPT' if adopt else 'KEEP'} by the rule"
    )

    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    cfg = json.load(open(f"{dst}/hobson_config.json"))
    cfg["temperature_by_kind"] = new
    json.dump(cfg, open(f"{dst}/hobson_config.json", "w"), indent=2)
    json.dump(
        {"old": old, "new": new, "fit": fit, "report": report, "adopt": adopt},
        open(f"reports/{dst.rstrip('/').split('/')[-1]}.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
