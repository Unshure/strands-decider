"""Teacher distributions where the teacher agrees with gold (training/data/distill.py)."""

import json

import pytest

from training.data.distill import agreeing, main


def rows():
    return [
        {"kind": "noul", "label": 1, "options": [["false", ""], ["true", ""]]},
        {"kind": "noul", "label": 0, "options": [["false", ""], ["true", ""]]},
        {"kind": "choice", "label": 2, "options": [["a", ""], ["b", ""], ["c", ""]]},
        {"kind": "score", "label": 1, "options": [["0", ""], ["1", ""], ["2", ""]]},
    ]


def test_keeps_only_agreeing_noul_and_choice():
    teacher = {0: [0.2, 0.8], 1: [0.3, 0.7], 2: [0.1, 0.1, 0.8], 3: [0.1, 0.8, 0.1]}
    kept = agreeing(rows(), teacher)
    assert [r["i"] for r in kept] == [0, 2]  # row 1 disagrees; row 3 is a score row


def test_width_mismatch_is_an_error():
    with pytest.raises(ValueError):
        agreeing(rows()[:1], {0: [0.2, 0.3, 0.5]})


def test_main_appends_and_rejects_overlap(tmp_path):
    corpus, teacher, extra, out = (
        tmp_path / n for n in ("c.jsonl", "t.jsonl", "r.jsonl", "o.jsonl")
    )
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows()))
    teacher.write_text(json.dumps({"i": 0, "probs": [0.2, 0.8]}) + "\n")
    extra.write_text(json.dumps({"i": 10, "probs": [0.5, 0.5]}) + "\n")
    main(
        [
            "--corpus",
            str(corpus),
            "--teacher",
            str(teacher),
            "--append",
            str(extra),
            "--out",
            str(out),
        ]
    )
    assert [json.loads(line)["i"] for line in out.read_text().splitlines()] == [0, 10]
    extra.write_text(json.dumps({"i": 0, "probs": [0.5, 0.5]}) + "\n")
    with pytest.raises(ValueError):
        main(
            [
                "--corpus",
                str(corpus),
                "--teacher",
                str(teacher),
                "--append",
                str(extra),
                "--out",
                str(out),
            ]
        )
