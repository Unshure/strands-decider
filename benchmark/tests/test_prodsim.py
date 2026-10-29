from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from benchmark.runner.prodsim import (
    Distribution,
    LengthSampler,
    load_distribution,
    poisson_schedule,
)


def test_poisson_schedule_is_increasing_and_in_bounds():
    rng = random.Random(1)
    arrivals = poisson_schedule(10.0, 5.0, rng)
    assert arrivals
    assert all(0 <= value < 5.0 for value in arrivals)
    assert arrivals == sorted(arrivals)


def test_poisson_schedule_matches_rate_at_scale():
    rng = random.Random(42)
    arrivals = poisson_schedule(50.0, 20.0, rng)
    # 50 rps * 20 s = 1000 arrivals (±5 sigma ≈ 158).
    assert 800 < len(arrivals) < 1200


def test_poisson_schedule_is_deterministic():
    first = poisson_schedule(8.0, 4.0, random.Random(7))
    second = poisson_schedule(8.0, 4.0, random.Random(7))
    assert first == second


def test_poisson_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        poisson_schedule(0.0, 1.0, random.Random(0))
    with pytest.raises(ValueError):
        poisson_schedule(1.0, 0.0, random.Random(0))


def _distribution(buckets):
    return Distribution(
        name="test",
        tokenizer="",
        buckets=tuple(buckets),
        raw={},
    )


def test_length_sampler_only_emits_positive_weight_targets():
    sampler = LengthSampler(
        _distribution([(256, 0.0), (1024, 0.5), (2048, 0.5)]),
        seed=0,
    )
    seen = {sampler.next() for _ in range(100)}
    assert seen <= {1024, 2048}


def test_length_sampler_weights_track_distribution():
    buckets = [(256, 0.1), (1024, 0.3), (2048, 0.6)]
    sampler = LengthSampler(_distribution(buckets), seed=1234)
    counts: dict[int, int] = {}
    for _ in range(5000):
        counts[sampler.next()] = counts.get(sampler.next(), 0) + 1
    # Allow a wide tolerance; just confirm order of magnitude per bucket.
    assert counts.get(2048, 0) > counts.get(1024, 0) > counts.get(256, 0)


def test_load_distribution_round_trip(tmp_path: Path):
    document = {
        "name": "demo",
        "tokenizer": "fake/tokenizer",
        "buckets": [
            {"tokens": 256, "weight": 0.25},
            {"tokens": 1024, "weight": 0.75},
        ],
    }
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(document))
    distribution = load_distribution(path)
    assert distribution.name == "demo"
    assert distribution.targets == (256, 1024)


def test_load_distribution_rejects_zero_weight(tmp_path: Path):
    document = {
        "name": "empty",
        "buckets": [{"tokens": 256, "weight": 0.0}],
    }
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_distribution(path)
