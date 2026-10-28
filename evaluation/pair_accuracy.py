"""Item and pair accuracy on a minimal-pair eval file.

Pair accuracy counts a pair only when BOTH halves are answered correctly. The halves
differ in one decisive fact and have different answers, so a model that learned the
generator's surface -- and therefore answers both halves the same way -- can get at most
one of the two right, and scores near zero here however good its item accuracy looks.
That makes it the sharpest available test for the failure v9 exhibited.

Pairs are consecutive rows: `generate_pairs` writes each pair's two halves together.

    python -m evaluation.pair_accuracy CHECKPOINT EVAL.jsonl
"""

import json
import sys
import warnings

warnings.filterwarnings("ignore")
from strands_decider.modeling import StrandsDeciderModel  # noqa: E402
from training.data.format import load_examples  # noqa: E402
from training.evaluate import collect_logits, predictions_from_logits  # noqa: E402

ckpt, path = sys.argv[1], sys.argv[2]
examples = load_examples([path])
tasks = [json.loads(line)["task"] for line in open(path, encoding="utf-8")]
model = StrandsDeciderModel.load(ckpt)
logits, labels, slots, exs = collect_logits(model, examples, max_length=model.config.max_length)
preds = predictions_from_logits(logits, labels, slots, exs)
ok = [p.correct for p in preds]

by = {}
for t in sorted(set(tasks)):
    idx = [i for i, x in enumerate(tasks) if x == t]
    pairs = [(idx[k], idx[k + 1]) for k in range(0, len(idx) - 1, 2)]
    item = sum(ok[i] for i in idx) / len(idx)
    both = sum(ok[a] and ok[b] for a, b in pairs) / len(pairs)

    # Compare the chosen option's NAME: each half of a cross-reference pair has its own
    # option list, so equal indices can mean different answers and vice versa.
    def chosen(i):
        return exs[i].options[preds[i].pred][0]

    same = sum(chosen(a) == chosen(b) for a, b in pairs) / len(pairs)
    by[t] = (item, both, same, len(pairs))
    print(
        f"  {t:<14} item {item:.4f}   pair (both right) {both:.4f}   "
        f"same answer to both halves {same:.4f}   pairs={len(pairs)}"
    )
allp = [(i, i + 1) for i in range(0, len(ok) - 1, 2)]
print(
    f"  {'overall':<14} item {sum(ok) / len(ok):.4f}   pair (both right) "
    f"{sum(ok[a] and ok[b] for a, b in allp) / len(allp):.4f}"
)
