"""Question-varied rows: every label re-derived from the rendered text, not trusted.

The derivation here reads only the transformed instruction, the transformed options and
the SOURCE row -- never the transform's own bookkeeping -- so a bug that produced a
wrong label consistently would still be caught. v9 and v10 each shipped a generator
with a wrong-label bug that only an independent check like this found.
"""

from __future__ import annotations

import os
import random
import re

import pytest

from training.data.format import Example, read_jsonl
from training.data.question_transforms import (
    COMPLEMENT,
    GENERIC_NOUL,
    IS_ANSWER,
    POLARITY,
    PROBE_FORMS,
    THRESHOLD,
    _clean,
    build,
    transform,
)

NEGATIVE = re.compile(
    r"\b(wrong|mistaken|incorrect|an error|NOT|does not apply|lower than|"
    r"fall short|negative|is no|Is it wrong)\b"
)


def _source_rows():
    rows = [
        Example(
            kind="choice",
            state="Arsenal won 3-1 at home.",
            instructions="Which section?",
            options=[
                ["World", "international"],
                ["Sports", "matches"],
                ["Business", "markets"],
                ["Sci/Tech", "science"],
            ],
            label=1,
            task="ag_news",
            instruction_variants=["What is it about?"],
        ),
        Example(
            kind="score",
            state="Great food, slow service.",
            instructions="How positive is this review?",
            options=[
                [str(i), d]
                for i, d in enumerate(
                    ["very negative", "negative", "mixed", "positive", "very positive"]
                )
            ],
            label=2,
            task="yelp_stars",
        ),
        Example(
            kind="noul",
            state="WIN a FREE cruise, reply now",
            instructions="Is this spam?",
            options=[["false", "genuine"], ["true", "promotional"]],
            label=1,
            task="spam",
        ),
        Example(
            kind="noul",
            state="Paris is in France.",
            instructions='Does the state imply "Paris is in Europe"?',
            options=[["false", "no"], ["true", "yes"]],
            label=0,
            task="mnli_entail",
        ),
    ]
    path = "data/train_v5.jsonl"
    if os.path.exists(path):  # real rows too, when the corpus has been built
        real = []
        for ex in read_jsonl(path):
            real.append(ex)
            if len(real) >= 20000:
                break
        rows += random.Random(1).sample(real, 3000)
    return rows


def _is_negative(text: str) -> bool:
    return bool(NEGATIVE.search(text))


def skeleton(src: Example, out: Example):
    """The instruction with the source question, option and scale replaced by
    placeholders -- so words inside them ("wrong exchange rate", "very negative") cannot
    be mistaken for the template's polarity. Returns (skeleton, option index or None).

    Templates are used only to locate the option name, never for polarity or label.
    """
    s = out.instructions.replace(_clean(src.instructions), "Q")
    if src.kind == "score":
        scale = "; ".join(f"{i} = {_clean(d)}" for i, (_, d) in enumerate(src.options))
        s = s.replace(scale, "S")
    if src.kind == "choice" and out.kind == "noul":
        # Longest names first, so "cancel transfer" is not matched inside a longer one.
        order = sorted(range(src.n_options), key=lambda i: -len(src.options[i][0]))
        for i in order:
            name = _clean(src.options[i][0])
            for t, _ in IS_ANSWER:
                if t.format(q="Q", x=name) == s:
                    # Not s.replace(name, "X"): a short name like "en" also occurs
                    # inside words ("mistaken") and would erase the polarity.
                    return t.format(q="Q", x="X"), i
        raise AssertionError(f"no option name found in {out.instructions!r}")
    return s, None


def derive(src: Example, out: Example) -> int:
    """The label the rendered text implies, given the source row's gold answer."""
    s, x = skeleton(src, out)
    neg = _is_negative(s)
    if src.kind == "choice" and out.kind == "noul":
        return int((x == src.label) != neg)
    if src.kind == "choice":
        gold = src.options[src.label]
        return next(i for i, o in enumerate(out.options) if (o == gold) != neg)
    if src.kind == "score":
        (k,) = map(int, re.findall(r"\d+", s))  # the only number left is the threshold
        return int(src.label < k) if neg else int(src.label >= k)
    return 1 - src.label if neg else src.label


def test_every_label_matches_an_independent_derivation():
    rng = random.Random(0)
    rows = _source_rows()
    checked = 0
    for src in rows:
        for _ in range(3):
            out, _info = transform(src, rng)
            assert out.label == derive(src, out), (src.task, out.instructions, out.options)
            checked += 1
    assert checked >= 12


def test_the_same_state_gets_opposite_answers():
    """The point of the corpus: within one source row, the question decides the label."""
    rng = random.Random(0)
    for src in _source_rows()[:4]:
        labels = {transform(src, rng)[0].label for _ in range(40)}
        assert len(labels) == 2, src.task


def test_labels_are_balanced():
    rng = random.Random(0)
    rows = _source_rows()
    for kind in ("choice", "score", "noul"):
        pool = [r for r in rows if r.kind == kind]
        outs = [transform(pool[i % len(pool)], rng)[0] for i in range(2000)]
        noul = [o.label for o in outs if o.kind == "noul"]
        assert 0.4 < sum(noul) / len(noul) < 0.6, (kind, sum(noul) / len(noul))


def test_rows_are_well_formed():
    rng = random.Random(0)
    for src in _source_rows():
        out, _ = transform(src, rng)
        # A source variant would phrase the ORIGINAL question, whose answer differs.
        assert out.instruction_variants == []
        assert out.state == src.state
        if out.kind == "noul":
            assert out.options == GENERIC_NOUL
        else:
            assert len(out.options) == 2 and src.options[src.label] in out.options
        assert out.task.startswith(src.task + "+")


def test_probe_forms_never_generated():
    templates = [t for t, _ in IS_ANSWER + COMPLEMENT + THRESHOLD + POLARITY]
    for form in PROBE_FORMS:
        assert not any(form.lower() in t.lower() for t in templates), form
    rng = random.Random(0)
    for src in _source_rows():
        text = transform(src, rng)[0].instructions
        assert not any(form in text for form in PROBE_FORMS if form not in src.instructions)


def test_every_template_polarity_is_classified_correctly():
    """The derivation's negation detector must agree with each template's own polarity."""
    for t, positive in IS_ANSWER + COMPLEMENT + THRESHOLD + POLARITY:
        filled = t.format(q="Q", x="X", k=2, scale="S")
        assert _is_negative(filled) != positive, t


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
def test_build_replaces_the_chosen_fraction_deterministically(fraction):
    rows = _source_rows()[:4] * 25
    kept, made = build(rows, fraction, seed=3)
    assert len(kept) + len(made) == len(rows)
    assert len(made) == round(fraction * len(rows))
    _, made2 = build(rows, fraction, seed=3)
    assert [m[0] for m in made] == [m[0] for m in made2]
    assert [m[1].instructions for m in made] == [m[1].instructions for m in made2]
