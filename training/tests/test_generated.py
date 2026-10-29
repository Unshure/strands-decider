"""Generated document questions (v16): the export loads, and yes/no balancing removes the
skill-only shortcut without touching choice rows."""

from __future__ import annotations

import json
from collections import Counter

from training.data import generated as gen
from training.data.format import Example

YES_NO = [["false", "no"], ["true", "yes"]]


def _noul(skill, label, i):
    return Example(
        kind="noul",
        state=f"doc {skill} {i}",
        instructions="q",
        options=YES_NO,
        label=label,
        task=f"gen:{skill}",
    )


def test_balance_noul_equalises_each_skill_and_keeps_choice():
    exs = (
        [_noul("all_conditions", 0, i) for i in range(30)]
        + [_noul("all_conditions", 1, i) for i in range(10)]
        + [_noul("dates", 1, i) for i in range(7)]
    )  # one label only: dropped whole
    choice = [
        Example(
            kind="choice",
            state=f"c{i}",
            instructions="q",
            options=[["a", ""], ["b", ""], ["c", ""]],
            label=i % 3,
            task="gen:lookup_chain",
        )
        for i in range(12)
    ]
    out = gen.balance_noul(exs + choice)
    per = Counter((e.task, e.label) for e in out if e.kind == "noul")
    assert per[("gen:all_conditions", 0)] == per[("gen:all_conditions", 1)] == 10
    assert not any(t == "gen:dates" for t, _ in per)
    assert sorted(e.state for e in out if e.kind == "choice") == sorted(e.state for e in choice)


def _choice(label, lengths, i):
    return Example(
        kind="choice",
        state=f"s{i}",
        instructions="q",
        options=[[f"o{k}", "x" * n] for k, n in enumerate(lengths)],
        label=label,
        task="gen:numbers",
    )


def test_balance_length_brings_the_longest_answer_cue_to_chance():
    cued = [_choice(0, (40, 5, 5, 5), i) for i in range(60)]  # answer is the longest
    clean = [_choice(1, (40, 5, 5, 5), 100 + i) for i in range(40)]
    out = gen.balance_length(cued + clean)
    choice = [e for e in out if e.kind == "choice"]
    rate = sum(gen._longest(e) == e.label for e in choice) / len(choice)
    assert rate <= 0.25 + 1e-9 and len(choice) > 40  # 4 options: chance is 0.25
    assert all(e in out for e in clean)


def test_shuffle_options_moves_the_label_with_its_option():
    rows = [_choice(i % 4, (1, 2, 3, 4), i) for i in range(40)]
    out = gen.shuffle_options(rows, seed=3)
    for a, b in zip(rows, out, strict=True):
        assert b.options[b.label] == a.options[a.label]
    assert sum(b.label == 0 for b in out) < 40  # the order did change
    assert gen.shuffle_options(rows, seed=3)[5].options == out[5].options  # fixed by the seed


def test_load_reads_the_generator_export(tmp_path):
    row = {
        "kind": "choice",
        "state": "Title\n\nDoc.\n\nCase: c",
        "instructions": "Who approves?",
        "options": [
            ["Manager", "m"],
            ["Director", "d"],
            ["cannot be determined from the document", ""],
        ],
        "label": 1,
        "task": "gen:exception",
        "weight": 1.0,
        "instruction_variants": [],
    }
    p = tmp_path / "gen_train.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (ex,) = gen.load(str(p))
    assert ex.options[ex.label][0] == "Director" and ex.task == "gen:exception"


def test_attach_paraphrases_keeps_rows_and_order():
    from training.data.format import Example
    from training.data.generated import attach_paraphrases, paraphrase_pairs

    rows = [
        Example(
            kind="noul",
            state=f"s{i}",
            instructions=f"Is {i} allowed?",
            options=[["false", "no"], ["true", "yes"]],
            label=i % 2,
            task="gen:x",
        )
        for i in range(3)
    ]
    table = {"Is 1 allowed?": ["May 1 be allowed?", "Is 1 permitted?"]}
    out = attach_paraphrases(rows, table)
    assert [e.state for e in out] == ["s0", "s1", "s2"] and [e.label for e in out] == [0, 1, 0]
    assert out[1].instruction_variants == ["Is 1 allowed?", "May 1 be allowed?", "Is 1 permitted?"]
    assert out[0].instruction_variants == [] and out[1].instructions == "Is 1 allowed?"
    pairs = paraphrase_pairs(rows, table)
    assert [(p.state, p.instructions) for p in pairs] == [
        ("s1", "Is 1 allowed?"),
        ("s1", "May 1 be allowed?"),
    ]
