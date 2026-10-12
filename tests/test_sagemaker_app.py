"""The SageMaker routes on the real app, against a stub engine: no GPU and no weights."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_server import _StubEngine, _StubModel

from strands_decider import sagemaker_app, server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server.StrandsDeciderModel, "load", lambda *a, **k: _StubModel())
    monkeypatch.setattr(server, "SystemOneEngine", lambda model, cfg: _StubEngine())
    return TestClient(sagemaker_app.create_sagemaker_app({"DECIDER_DEVICE": "cpu"}))


def test_ping(client):
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json()["checkpoint"] == sagemaker_app.DEFAULT_CHECKPOINT


def test_invocations_matches_systemone(client):
    body = {
        "state": "Help! My payouts have been failing for 3 days.",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    }
    r = client.post("/invocations", json=body)
    assert r.status_code == 200
    expected = client.post("/v1/systemone", json=body).json()
    got = r.json()
    assert got["answers"] == expected["answers"]
    assert "latency_ms" in got


def test_invocations_rejects_bad_request(client):
    r = client.post("/invocations", json={"state": "x", "questions": {"q": {"type": "nope"}}})
    assert r.status_code == 422


def test_settings_from_env():
    assert sagemaker_app.settings_from_env({}) == {
        "checkpoint": sagemaker_app.DEFAULT_CHECKPOINT,
        "device": "cuda",
        "model_name": None,
        "max_batch": 32,
        "max_batch_tokens": None,
        "strict_window": False,
    }
    s = sagemaker_app.settings_from_env(
        {
            "DECIDER_CHECKPOINT": "/opt/model",
            "DECIDER_MODEL_NAME": "strands-decider-2B-qwen3.5-v1-2610",
            "DECIDER_MAX_BATCH": "8",
            "DECIDER_MAX_BATCH_TOKENS": "65536",
            "DECIDER_STRICT_WINDOW": "true",
        }
    )
    assert s["checkpoint"] == "/opt/model"
    assert s["model_name"] == "strands-decider-2B-qwen3.5-v1-2610"
    assert (s["max_batch"], s["max_batch_tokens"], s["strict_window"]) == (8, 65536, True)
