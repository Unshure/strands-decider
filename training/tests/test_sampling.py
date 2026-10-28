"""Length-grouped batching must be faster without being a different training run."""

from __future__ import annotations

import random

import pytest

from training.data.sampling import LengthGroupedBatchSampler, padding_fraction


def _mixed_lengths(n: int = 4000, long_share: float = 0.09, seed: int = 0):
    """A short-heavy corpus with a minority of long documents, like v10's."""
    rng = random.Random(seed)
    return [
        rng.randint(1800, 2600) if rng.random() < long_share else rng.randint(150, 400)
        for _ in range(n)
    ]


def test_every_index_appears_once_per_epoch():
    lengths = _mixed_lengths()
    s = LengthGroupedBatchSampler(lengths, 8, drop_last=False)
    seen = [i for b in s.batches(0) for i in b]
    assert sorted(seen) == list(range(len(lengths)))


def test_drop_last_gives_full_batches_and_matches_len():
    lengths = _mixed_lengths(n=4003)  # not a multiple of 8
    s = LengthGroupedBatchSampler(lengths, 8, drop_last=True)
    batches = s.batches(0)
    assert all(len(b) == 8 for b in batches)
    assert len(batches) == len(s) == 4003 // 8
    assert len({i for b in batches for i in b}) == 8 * len(batches)  # no repeats


def test_deterministic_per_epoch_and_different_across_epochs():
    lengths = _mixed_lengths()
    a = LengthGroupedBatchSampler(lengths, 8, seed=3)
    b = LengthGroupedBatchSampler(lengths, 8, seed=3)
    assert a.batches(0) == b.batches(0)
    assert a.batches(0) != a.batches(1)
    # __iter__ advances the epoch, as shuffle=True reshuffles every pass
    first, second = list(a), list(a)
    assert first != second


def test_grouping_cuts_padding_substantially():
    lengths = _mixed_lengths()
    grouped = LengthGroupedBatchSampler(lengths, 8).batches(0)
    order = list(range(len(lengths)))
    random.Random(0).shuffle(order)
    ungrouped = [order[i : i + 8] for i in range(0, len(order) - len(order) % 8, 8)]
    g, u = padding_fraction(lengths, grouped), padding_fraction(lengths, ungrouped)
    assert g < u / 3, (g, u)


def test_longest_batch_comes_first():
    lengths = _mixed_lengths()
    batches = LengthGroupedBatchSampler(lengths, 8).batches(0)
    top = max(max(lengths[i] for i in b) for b in batches)
    assert max(lengths[i] for i in batches[0]) == top


def test_long_documents_stay_spread_across_the_epoch():
    """Grouping must not become a global sort.

    Sorting the whole corpus would put every long document into a few consecutive steps
    -- a different training distribution, not a faster version of the same one. With
    megabatches, batches holding long documents should be scattered from start to end.
    """
    lengths = _mixed_lengths()
    batches = LengthGroupedBatchSampler(lengths, 8).batches(0)
    long_at = [k for k, b in enumerate(batches) if max(lengths[i] for i in b) > 1500]
    assert len(long_at) > 20
    span = (long_at[-1] - long_at[0]) / len(batches)
    assert span > 0.8, span
    # and no single stretch holds most of them
    quarter = len(batches) // 4
    per_quarter = [
        sum(1 for k in long_at if q * quarter <= k < (q + 1) * quarter) for q in range(4)
    ]
    assert max(per_quarter) < 0.5 * len(long_at), per_quarter


def test_rejects_nonsense_sizes():
    with pytest.raises(ValueError):
        LengthGroupedBatchSampler([1, 2, 3], 0)
