"""Catch-all option rows (training/data/catchall.py). No GPU."""

import random

from training.data.catchall import DESCRIPTION, NAMES, build, transform
from training.data.format import Example


def ex(label=1, task="ag_news", names=("world", "sports", "business", "tech")):
    return Example(
        kind="choice",
        state="s",
        instructions="Which topic?",
        options=[[n, f"about {n}"] for n in names],
        label=label,
        task=task,
    )


def test_absent_removes_gold_and_makes_catchall_right():
    r = transform(ex(label=1), "absent", random.Random(0))
    names = [n for n, _ in r.options]
    assert "sports" not in names and len(names) == 4
    assert names[r.label] in NAMES and r.options[r.label][1] == DESCRIPTION
    assert r.task == "ag_news+catchall:absent"


def test_present_keeps_gold():
    r = transform(ex(label=2), "present", random.Random(0))
    assert r.options[r.label][0] == "business" and r.options[-1][0] in NAMES


def test_skips_out_of_scope_and_too_few_options():
    assert transform(ex(names=("oos", "balance", "transfer")), "present", random.Random(0)) is None
    assert transform(ex(label=0, names=("a", "b")), "absent", random.Random(0)) is None


def test_build_is_balanced():
    rows = build([ex(label=i % 4) for i in range(40)], ["ag_news"], per_task=20, seed=0)
    assert len(rows) == 20
    assert sum(r.task.endswith(":absent") for r in rows) == 10


def test_present_stays_within_max_options():
    names = tuple(f"c{i}" for i in range(24))
    for label in (0, 5, 23):
        r = transform(ex(label=label, names=names), "present", random.Random(label), max_options=24)
        assert (
            len(r.options) == 24
            and r.options[r.label][0] == f"c{label}"
            and r.options[-1][0] in NAMES
        )
