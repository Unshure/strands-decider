"""Paired evaluations (v20), where rows come in adjacent pairs.

    python -m evaluation.pair_eval CHECKPOINT FILE [FILE ...]

  data/para_pairs_*_eval.jsonl  (original wording, a checked paraphrase), same label:
                                `same_answer` is paraphrase consistency (v20 prediction 1)
  data/flips_v20_eval.jsonl     (instruction A, instruction B), opposite labels:
                                `both_right` is pair accuracy (v20 prediction 2)

Both are built by `training/recipe.sh generated`. Prints one JSON line per file: pairs,
accuracy of each half, the share of pairs given the same answer, the share with both
halves right. Uses the checkpoint's own temperatures and window.
"""

import json
import sys

from strands_decider.modeling import StrandsDeciderModel
from training.data.format import read_jsonl
from training.evaluate import collect_logits, predictions_from_logits


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    ckpt, files = argv[0], argv[1:]
    model = StrandsDeciderModel.load(ckpt)
    t = model.config.temperature_by_kind or model.config.temperature
    for f in files:
        exs = list(read_jsonl(f))
        if len(exs) % 2:
            raise ValueError(f"{f}: odd number of rows, not pairs")
        L, Y, S, E = collect_logits(model, exs, batch_size=16, max_length=model.config.max_length)
        P = predictions_from_logits(L, Y, S, E, t, model.config.ordinal_smoothing)
        a, b = P[0::2], P[1::2]
        n = len(a)
        res = {
            "file": f,
            "pairs": n,
            "acc_first": sum(p.correct for p in a) / n,
            "acc_second": sum(p.correct for p in b) / n,
            "same_answer": sum(x.pred == y.pred for x, y in zip(a, b, strict=False)) / n,
            "both_right": sum(x.correct and y.correct for x, y in zip(a, b, strict=False)) / n,
        }
        print(
            json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}),
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
