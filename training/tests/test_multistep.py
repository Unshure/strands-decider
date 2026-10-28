"""Multi-step sources: labels must survive windowing, and no shortcut may survive balancing."""

from __future__ import annotations

import json
import os
from collections import Counter

import pytest

from training.data import multistep as ms
from training.data.format import Example

RAW = "data/raw"


def _nda(tmp_path, text, spans, annotations):
    doc = {"id": 1, "text": text, "spans": spans, "annotation_sets": [{"annotations": annotations}]}
    data = {
        "documents": [doc],
        "labels": {k: {"short_description": k, "hypothesis": f"Claim {k}."} for k in annotations},
    }
    path = tmp_path / "nda.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_contractnli_window_keeps_every_evidence_span(tmp_path):
    sentences = [f"Sentence {i} of the agreement says something. " for i in range(400)]
    text = "".join(sentences)
    spans, pos = [], 0
    for s in sentences:
        spans.append([pos, pos + len(s)])
        pos += len(s)
    evidence = [300, 305]  # deep in a document far longer than the budget
    path = _nda(
        tmp_path,
        text,
        spans,
        {
            "nda-1": {"choice": "Entailment", "spans": evidence},
            "nda-2": {"choice": "NotMentioned", "spans": []},
        },
    )
    rows = list(ms.contractnli(path, budget=2000))
    assert len(text) > 2000
    by = {r.label: r for r in rows}
    for i in evidence:
        assert sentences[i].strip() in by[0].state
    assert all(len(r.state) <= 2000 for r in rows)
    assert by[2].options[by[2].label][0] == "not mentioned"


def test_contractnli_drops_evidence_wider_than_the_budget(tmp_path):
    text = "a" * 5000
    spans = [[0, 10], [4990, 5000]]
    path = _nda(tmp_path, text, spans, {"nda-1": {"choice": "Contradiction", "spans": [0, 1]}})
    assert list(ms.contractnli(path, budget=1000)) == []


def test_balance_by_claim_removes_the_claim_only_shortcut():
    exs = []
    for claim, counts in {"A": (300, 40, 3), "B": (30, 200, 60), "C": (500, 0, 0)}.items():
        for label, n in enumerate(counts):
            exs += [
                Example(
                    kind="choice",
                    state=f"{claim}{i}",
                    instructions=claim,
                    options=ms.NLI_OPTIONS,
                    label=label,
                )
                for i in range(n)
            ]
    out = ms.balance_by_claim(exs, min_count=25)
    per = Counter((e.instructions, e.label) for e in out)
    assert per[("A", 0)] == per[("A", 1)] == 40 and per[("A", 2)] == 0  # 3 is below min_count
    assert per[("B", 0)] == per[("B", 1)] == per[("B", 2)] == 30
    assert not any(e.instructions == "C" for e in out)  # one answer only: nothing to learn
    # knowing the claim now predicts the label no better than chance among its answers
    for claim in ("A", "B"):
        labels = Counter(e.label for e in out if e.instructions == claim)
        assert len(set(labels.values())) == 1


def test_musique_pairs_keep_both_versions_and_label_cannot():
    def row(qid, answerable):
        paras = [
            {
                "idx": i,
                "title": f"T{qid}{i}",
                "paragraph_text": "text " * 20,
                "is_supporting": answerable and i < 2,
            }
            for i in range(6)
        ]
        return {
            "id": qid,
            "question": f"Q{qid}?",
            "answer": "Paris",
            "answerable": answerable,
            "paragraphs": paras,
            "question_decomposition": [{"answer": "France"}, {"answer": "Paris"}],
        }

    rows = [row(q, a) for q in ("x", "y", "z") for a in (True, False)]
    picked = ms.musique_pairs(rows, 2)
    assert Counter(r["id"] for r in picked) == {k: 2 for k in {r["id"] for r in picked}}
    for ex, r in zip(ms.musique(picked), picked, strict=True):
        names = [o[0] for o in ex.options]
        assert "Paris" in names and "France" in names and ms.CANNOT in names
        assert names[ex.label] == ("Paris" if r["answerable"] else ms.CANNOT)


def test_hotpotqa_uses_only_exact_answers():
    def row(ans, titles):
        return {
            "type": "comparison",
            "question": "Which is older?",
            "answer": ans,
            "supporting_facts": {"title": titles, "sent_id": [0, 0]},
            "context": {"title": [*titles, "D"], "sentences": [["s."], ["t."], ["u."]]},
        }

    yes, pick, vague = (list(ms.hotpotqa([row(a, ["A", "B"])])) for a in ("yes", "B", "1901"))
    assert yes[0].kind == "noul" and yes[0].label == 1
    assert pick[0].options[pick[0].label][0] == "B"
    assert vague == []  # an answer that is neither entity would need invented options


def test_boardgame_splits_off_the_question():
    r = {
        "example": "Facts here. Rule1: x. " + ms.BOARDGAME_Q + "does the cat win?",
        "label": "unknown",
    }
    (ex,) = ms.boardgame([r])
    assert ex.instructions.startswith(ms.BOARDGAME_Q) and ms.BOARDGAME_Q not in ex.state
    assert ex.options[ex.label][0] == "unknown"


@pytest.mark.skipif(not os.path.exists(f"{RAW}/contract-nli/train.json"), reason="raw data absent")
def test_real_contractnli_balance_defeats_the_prior():
    exs = ms.balance_by_claim(list(ms.contractnli(f"{RAW}/contract-nli/train.json")))
    per = Counter((e.instructions, e.label) for e in exs)
    majority = sum(max(n for (q2, _), n in per.items() if q2 == q) for q in {q for q, _ in per})
    assert majority / len(exs) < 0.55  # the per-claim prior was 0.68 before balancing
