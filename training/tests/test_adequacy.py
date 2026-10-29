"""HelpSteer2 -> answer-adequacy rows (training/data/adequacy.py). No GPU, no download."""

from training.data.adequacy import OPTIONS, QUESTION, TASK, balanced, to_example, verdict


def row(prompt, h, c, response="r"):
    return {"prompt": prompt, "response": response, "helpfulness": h, "correctness": c}


def test_verdict_thresholds():
    assert verdict(row("p", 3, 3)) is True
    assert verdict(row("p", 4, 3)) is True
    assert verdict(row("p", 1, 4)) is False  # unhelpful
    assert verdict(row("p", 4, 0)) is False  # incorrect
    assert verdict(row("p", 2, 3)) is None  # middle band dropped
    assert verdict(row("p", 3, 2)) is None


def test_balanced_prefers_pair_mates():
    rows = [
        row("a", 0, 0),
        row("a", 4, 4, "good a"),
        row("b", 1, 1),
        row("b", 3, 3, "good b"),
        row("c", 4, 4, "good c"),
        row("d", 4, 4, "good d"),
    ]
    out = balanced(rows, seed=0)
    bad = [r for r in out if not verdict(r)]
    good = [r for r in out if verdict(r)]
    assert len(bad) == len(good) == 2
    # the adequate responses come from the prompts that also have an inadequate one
    assert {r["prompt"] for r in good} == {"a", "b"}


def test_to_example():
    e = to_example(row("What is 2+2?", 4, 4, "4"))
    assert e.kind == "noul" and e.task == TASK and e.label == 1
    assert e.options == [list(o) for o in OPTIONS] and e.options[1][0] == "true"
    assert e.state == "Request:\nWhat is 2+2?\n\nResponse:\n4"
    assert e.instructions == QUESTION and QUESTION in e.instruction_variants
    assert to_example(row("q", 0, 0)).label == 0
