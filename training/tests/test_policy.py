"""Policy sources (v15): labels map as documented, windows keep what the answer turns on,
and no cell shortcut survives balancing."""

from __future__ import annotations

from collections import Counter

from training.data import policy as pol
from training.data.format import Example


def _sharc(answer, scenario="", history=0):
    return {
        "snippet": "# Rule\n\nYou qualify if you are over 18 and resident.",
        "question": "Do I qualify?",
        "scenario": scenario,
        "answer": answer,
        "history": [
            {"follow_up_question": f"Q{i}?", "follow_up_answer": "Yes"} for i in range(history)
        ],
    }


def test_sharc_labels_and_irrelevant_dropped():
    rows = [
        _sharc("Yes", "I am 30."),
        _sharc("No", history=1),
        _sharc("Are you resident?"),
        _sharc("Irrelevant"),
    ]
    out = list(pol.sharc(rows))
    assert [e.label for e in out] == [pol.YES, pol.NO, pol.DEPENDS]
    assert all(e.options == pol.POLICY_OPTIONS[:3] for e in out)
    assert "User's situation: I am 30." in out[0].state
    assert "Q: Q0?\nA: Yes" in out[1].state
    assert '"Do I qualify?"' in out[0].instructions


def test_balance_removes_the_cell_shortcut():
    exs = []
    for cell, counts in {"a": (100, 40, 60), "b": (0, 0, 500), "c": (30, 300, 10)}.items():
        for label, n in enumerate(counts):
            exs += [
                Example(
                    kind="choice",
                    state=f"{cell}{i}",
                    instructions=cell,
                    options=pol.POLICY_OPTIONS[:3],
                    label=label,
                )
                for i in range(n)
            ]
    out = pol.balance(exs, key=lambda e: e.instructions, min_count=25)
    per = Counter((e.instructions, e.label) for e in out)
    assert per[("a", 0)] == per[("a", 1)] == per[("a", 2)] == 40
    assert not any(k == "b" for k, _ in per)  # one label only: dropped whole
    assert per[("c", 0)] == per[("c", 1)] == 30 and per[("c", 2)] == 0  # 10 is below min_count


def test_cap_per():
    exs = [
        Example(
            kind="noul",
            state=str(i),
            instructions=str(i % 3),
            options=[["f", ""], ["t", ""]],
            label=0,
        )
        for i in range(30)
    ]
    assert Counter(e.instructions for e in pol.cap_per(exs, lambda e: e.instructions, 4)) == {
        "0": 4,
        "1": 4,
        "2": 4,
    }


def _cq(answers, not_answerable=False, evidences=("<p>Rule 150.</p>",)):
    return {
        "url": "u",
        "scenario": "I am 17.",
        "question": "Can I apply?",
        "not_answerable": not_answerable,
        "answers": answers,
        "evidences": list(evidences),
    }


def test_conditionalqa_labels():
    L = pol.conditionalqa_label
    assert L(_cq([["yes", []]])) == pol.YES
    assert L(_cq([["no", []]])) == pol.NO
    assert L(_cq([["yes", ["<p>You must be 18.</p>"]]])) == pol.DEPENDS
    assert L(_cq([["yes", ["<p>c1</p>"]], ["no", ["<p>c2</p>"]]])) == pol.DEPENDS
    assert L(_cq([], not_answerable=True)) == pol.NOT_SAID
    assert L(_cq([["the county court", []]])) is None


def test_conditionalqa_window_keeps_evidence_and_conditions():
    contents = [
        f"<p>Filler sentence number {i} about something else entirely.</p>" for i in range(400)
    ]
    contents[250] = "<p>Rule 150.</p>"
    contents[262] = "<p>You must be 18.</p>"
    docs = [{"url": "u", "title": "Guide", "contents": contents}]
    q = _cq([["yes", ["<p>You must be 18.</p>"]]])
    (ex,) = pol.conditionalqa([q], docs, budget=3000)
    assert "Rule 150." in ex.state and "You must be 18." in ex.state
    assert len(ex.state) <= 3000 + 100
    assert ex.options[ex.label][0] == "depends"
    # evidence missing from the page: skipped, not guessed
    assert (
        list(pol.conditionalqa([_cq([["yes", []]], evidences=["<p>Not there.</p>"])], docs)) == []
    )
