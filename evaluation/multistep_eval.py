"""Accuracy on the multi-step evaluation sets, per source (v14's pre-registered checks).

HotpotQA is the held-out transfer test: no HotpotQA row is ever trained on. MuSiQue is
split into its answerable and unanswerable halves, since answering "cannot be
determined" to everything scores 0.5. ContractNLI keeps its natural label distribution;
guessing each claim's most common label scores 0.679 on it.

Rows whose prompt exceeds the model's window are skipped, never truncated.

    python -m evaluation.multistep_eval CHECKPOINT [--data data/multistep_v14_eval.jsonl]
"""

import argparse
import json
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
from strands_decider.modeling import StrandsDeciderModel  # noqa: E402
from strands_decider.prompting import build_prompt  # noqa: E402
from training.data.format import load_examples  # noqa: E402
from training.data.multistep import CANNOT  # noqa: E402
from training.evaluate import collect_logits  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/multistep_v14_eval.jsonl")
    ap.add_argument("--out", help="write per-row correctness (JSON list, null = skipped)")
    args = ap.parse_args()

    rows = load_examples([args.data])
    model = StrandsDeciderModel.load(args.checkpoint)
    fits = [
        len(model.tokenizer(build_prompt(e.state, e.to_question())[0])["input_ids"])
        <= model.config.max_length
        for e in rows
    ]
    keep = [e for e, f in zip(rows, fits, strict=True) if f]
    logits, *_ = collect_logits(model, keep, max_length=model.config.max_length, batch_size=4)
    preds = logits.argmax(-1).tolist()  # collect_logits masks the columns past each row's options

    by = defaultdict(list)
    for ex, p in zip(keep, preds, strict=True):
        ok = p == ex.label
        by[ex.task + (" (held out)" if ex.task == "hotpotqa" else "")].append(ok)
        if ex.task == "musique":
            half = "unanswerable" if ex.options[ex.label][0] == CANNOT else "answerable"
            by[f"musique, {half}"].append(ok)
        if ex.task in ("sharc", "conditionalqa"):  # v15: per answer, "depends" above all
            by[f"{ex.task}, {ex.options[ex.label][0]}"].append(ok)
    print(f"{args.checkpoint}  ({len(rows) - len(keep)} rows over the window skipped)")
    for name in sorted(by):
        v = by[name]
        # Two or more spaces before the value, whatever the name's length: that is what
        # hf_export.internal_evals, and any script that copies its regex, split the line on.
        print(f"  {name:<26}  {sum(v) / len(v):.3f}  (n={len(v):,})")
    if args.out:  # per-row correctness, in the order of --data, for paired comparisons
        it = iter(p == e.label for e, p in zip(keep, preds, strict=True))
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump([next(it) if f else None for f in fits], fh)


if __name__ == "__main__":
    main()
