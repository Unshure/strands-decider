"""The generators' chat client (data/generators/llm_client.py), offline: a fake urlopen captures the
request and returns canned JSON. No network, no keys beyond the fake ones set here."""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "generators"))
import llm_client as llm


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ok(text="ANSWER: yes", finish="stop", usage=None, provider="DeepInfra"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
        "provider": provider,
    }


@pytest.fixture
def transport(monkeypatch):
    """Replaces urlopen; `sent` collects (url, headers, body); `replies` are popped in order."""
    sent, replies = [], []

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, dict(req.header_items()), json.loads(req.data)))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return FakeResponse(json.dumps(r).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "USAGE", {"in": 0, "out": 0, "calls": 0, "cost": 0.0})
    return argparse.Namespace(sent=sent, replies=replies)


def args(**over):
    base = {
        "quantizations": "bf16,fp16,fp32,fp8",
        "providers": "",
        "reasoning_effort": "high",
        "timeout": 9,
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_openrouter_request_is_unchanged(transport, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    transport.replies.append(ok())
    text, provider = llm.chat(args(), "qwen/qwen3.6-27b", "sys", "hi", 100, 0.7)
    assert (text, provider) == ("ANSWER: yes", "DeepInfra")
    url, headers, body = transport.sent[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer or-key"
    assert headers["X-title"] == "hobson-gen-pilot"
    assert body["model"] == "qwen/qwen3.6-27b"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert body["max_tokens"] == 100 and body["temperature"] == 0.7 and body["top_p"] == 0.95
    assert body["usage"] == {"include": True}
    assert body["provider"] == {
        "quantizations": ["bf16", "fp16", "fp32", "fp8"],
        "allow_fallbacks": True,
    }
    assert body["reasoning"] == {"effort": "high", "exclude": True}
    assert llm.USAGE == {"in": 10, "out": 5, "calls": 1, "cost": 0.001}


def test_openrouter_pinned_providers_and_no_reasoning(transport, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    transport.replies.append(ok())
    llm.chat(args(providers="Venice,Chutes", reasoning_effort="none"), "m", "s", "p", 10, 0.1)
    body = transport.sent[0][2]
    assert body["provider"] == {
        "quantizations": ["bf16", "fp16", "fp32", "fp8"],
        "allow_fallbacks": False,
        "order": ["Venice", "Chutes"],
    }
    assert "reasoning" not in body


def test_retries_429_then_succeeds(transport, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    err = urllib.error.HTTPError("u", 429, "busy", {}, io.BytesIO(b"rate limited"))
    transport.replies += [err, ok(text="second")]
    assert llm.chat(args(), "m", "s", "p", 10, 0.1)[0] == "second"
    assert len(transport.sent) == 2 and llm.USAGE["calls"] == 1


def test_400_is_not_retried(transport, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    transport.replies.append(
        urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"no such model"))
    )
    with pytest.raises(RuntimeError, match="HTTP 400"):
        llm.chat(args(), "m", "s", "p", 10, 0.1)
    assert len(transport.sent) == 1


def test_truncation_raises_after_counting_usage(transport, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    transport.replies.append(ok(finish="length"))
    with pytest.raises(llm.Truncated):
        llm.chat(args(), "m", "s", "p", 10, 0.1)
    assert llm.USAGE["calls"] == 1


# ---- bedrock and bedrock-mantle ---------------------------------------------------------


def bedrock_reply():
    return {
        "choices": [{"message": {"content": "ANSWER: no"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }


def test_bedrock_runtime_request(transport, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    transport.replies.append(bedrock_reply())
    a = args(backend="bedrock", region="us-west-2", price_in=1.0, price_out=4.0)
    text, provider = llm.chat(a, "qwen.qwen3-32b-v1:0", "sys", "hi", 8000, 0.7)
    assert (text, provider) == ("ANSWER: no", "bedrock:us-west-2")
    url, headers, body = transport.sent[0]
    assert url == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer bedrock-key"
    assert "X-title" not in headers
    assert body == {
        "model": "qwen.qwen3-32b-v1:0",
        "max_tokens": 8000,
        "temperature": 0.7,
        "top_p": 0.95,
        "reasoning_effort": "high",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
    }
    for k in ("usage", "provider", "reasoning"):
        assert k not in body
    # cost from the given prices: 1000 in at $1/M plus 500 out at $4/M
    assert llm.USAGE == {"in": 1000, "out": 500, "calls": 1, "cost": pytest.approx(0.003)}


def test_bedrock_mantle_request_and_no_reasoning(transport, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    transport.replies.append(bedrock_reply())
    a = args(
        backend="bedrock-mantle",
        region="eu-west-1",
        reasoning_effort="none",
        price_in=0.0,
        price_out=0.0,
    )
    _, provider = llm.chat(a, "qwen.qwen3-32b", "s", "p", 8000, 0.5)
    url, headers, body = transport.sent[0]
    assert url == "https://bedrock-mantle.eu-west-1.api.aws/v1/chat/completions"
    assert headers["Authorization"] == "Bearer bedrock-key"
    assert "reasoning_effort" not in body and provider == "bedrock-mantle:eu-west-1"
    assert llm.USAGE["cost"] == 0.0 and llm.USAGE["in"] == 1000


def test_reasoning_effort_goes_only_to_models_that_think(transport, monkeypatch):
    """Qwen3 235B A22B 2507 and Qwen3 Next 80B A3B are Instruct releases: with reasoning_effort
    in the request they never answer (measured 2026-09-29), so the client drops the field for
    them; Qwen3 32B keeps it."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    transport.replies += [bedrock_reply()] * 4
    a = args(backend="bedrock", region="us-west-2", price_in=0.0, price_out=0.0)
    llm.chat(a, "qwen.qwen3-235b-a22b-2507-v1:0", "s", "p", 8000, 0.6)
    llm.chat(a, "qwen.qwen3-next-80b-a3b", "s", "p", 8000, 0.6)
    llm.chat(a, "qwen.qwen3-32b-v1:0", "s", "p", 8000, 0.6)
    llm.chat(a, "some.unknown-model", "s", "p", 8000, 0.6)  # not in the table: the flag is trusted
    efforts = [b.get("reasoning_effort") for _, _, b in transport.sent]
    assert efforts == [None, None, "high", "high"]
    # the same on Mantle, with its model ids
    transport.replies += [bedrock_reply()] * 2
    m = args(backend="bedrock-mantle", region="us-west-2", price_in=0.0, price_out=0.0)
    transport.replies.append(bedrock_reply())
    llm.chat(m, "qwen.qwen3-235b-a22b-2507", "s", "p", 8000, 0.6)
    llm.chat(m, "qwen.qwen3-next-80b-a3b-instruct", "s", "p", 8000, 0.6)
    llm.chat(m, "qwen.qwen3-32b", "s", "p", 8000, 0.6)
    assert [b.get("reasoning_effort") for _, _, b in transport.sent[4:]] == [None, None, "high"]


def test_inline_reasoning_block_is_stripped_on_bedrock(transport, monkeypatch):
    """bedrock-runtime returns Qwen3 32B's thinking inline, as <reasoning>...</reasoning> before
    the answer; callers get the answer alone, as OpenRouter's exclude gives it."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    thought = (
        '<reasoning>\nLet me think {"policy": "not this"}...\n</reasoning>\n\n{"policy": "this"}'
    )
    transport.replies.append(
        bedrock_reply() | {"choices": [{"message": {"content": thought}, "finish_reason": "stop"}]}
    )
    a = args(backend="bedrock", region="us-west-2", price_in=0.0, price_out=0.0)
    assert llm.chat(a, "qwen.qwen3-32b-v1:0", "s", "p", 8000, 0.7)[0] == '{"policy": "this"}'
    # whitespace ahead of the block is still a leading block
    transport.replies.append(
        bedrock_reply()
        | {
            "choices": [
                {
                    "message": {"content": "\n  <reasoning>x</reasoning>\nANSWER: no"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert llm.chat(a, "qwen.qwen3-32b-v1:0", "s", "p", 8000, 0.7)[0] == "ANSWER: no"
    # the strip ends at the first closing tag; a later one is content
    transport.replies.append(
        bedrock_reply()
        | {
            "choices": [
                {
                    "message": {"content": "<reasoning>x</reasoning>ANSWER: yes </reasoning> tail"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert (
        llm.chat(a, "qwen.qwen3-32b-v1:0", "s", "p", 8000, 0.7)[0]
        == "ANSWER: yes </reasoning> tail"
    )
    # only a leading block is thinking; the same tag later in an answer is content
    transport.replies.append(
        bedrock_reply()
        | {
            "choices": [
                {
                    "message": {"content": "ANSWER: yes <reasoning>x</reasoning>"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert (
        llm.chat(a, "qwen.qwen3-32b-v1:0", "s", "p", 8000, 0.7)[0]
        == "ANSWER: yes <reasoning>x</reasoning>"
    )
    # a thinking block cut off by max_tokens is still a truncation
    transport.replies.append(
        bedrock_reply()
        | {"choices": [{"message": {"content": "<reasoning>\nendless"}, "finish_reason": "length"}]}
    )
    with pytest.raises(llm.Truncated):
        llm.chat(a, "qwen.qwen3-32b-v1:0", "s", "p", 8000, 0.7)
    # OpenRouter content is passed through untouched
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    transport.replies.append(ok(text="<reasoning>kept</reasoning> body"))
    assert llm.chat(args(), "m", "s", "p", 10, 0.1)[0] == "<reasoning>kept</reasoning> body"


def test_check_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setitem(sys.modules, "botocore", None)  # not installed
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        llm.check_credentials(args(backend="openrouter"))
    with pytest.raises(SystemExit, match=r"AWS_BEARER_TOKEN_BEDROCK.*botocore"):
        llm.check_credentials(args(backend="bedrock"))
    with pytest.raises(SystemExit, match="AWS_BEARER_TOKEN_BEDROCK"):
        llm.check_credentials(args(backend="bedrock-mantle"))
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    llm.check_credentials(args(backend="bedrock"))
    llm.check_credentials(args(backend="bedrock-mantle"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    llm.check_credentials(args(backend="openrouter"))


def test_defaults_per_backend_and_max_tokens_by_model():
    assert llm.default_model(args(backend="openrouter"), "writer") == "qwen/qwen3.6-27b"
    assert llm.default_model(args(backend="openrouter"), "checker") == "qwen/qwen3.5-397b-a17b"
    assert llm.default_model(args(backend="bedrock"), "writer") == "qwen.qwen3-235b-a22b-2507-v1:0"
    assert (
        llm.default_model(args(backend="bedrock"), "verifier") == "qwen.qwen3-235b-a22b-2507-v1:0"
    )
    assert (
        llm.default_model(args(backend="bedrock-mantle"), "writer") == "qwen.qwen3-235b-a22b-2507"
    )
    assert (
        llm.default_model(args(backend="bedrock-mantle"), "verifier") == "qwen.qwen3-235b-a22b-2507"
    )
    # the flag wins; then the table; then the generator's own default
    assert llm.max_tokens("qwen.qwen3-32b-v1:0", 12345, 32000) == 12345
    assert llm.max_tokens("qwen.qwen3-32b-v1:0", None, 32000) == 8000
    assert llm.max_tokens("qwen/qwen3.6-27b", None, 32000) == 32000
    assert (
        llm.max_tokens("qwen/qwen3.5-397b-a17b,qwen.qwen3-235b-a22b-2507-v1:0", None, 16000) == 8000
    )


def test_describe_flags_models_other_than_the_recorded_ones():
    same = llm.describe(
        args(backend="openrouter"),
        {"writer": "qwen/qwen3.6-27b", "verifiers": "qwen/qwen3.5-397b-a17b"},
    )
    assert same.startswith("backend openrouter: writer qwen/qwen3.6-27b") and "NOTE" not in same
    other = llm.describe(
        args(backend="bedrock", region="us-east-1"),
        {"writer": "qwen.qwen3-32b-v1:0", "verifiers": "qwen.qwen3-235b-a22b-2507-v1:0"},
    )
    assert other.startswith("backend bedrock (us-east-1): writer qwen.qwen3-32b-v1:0")
    assert "not the models that produced the committed data" in other
    assert "NOTE" in llm.describe(
        args(backend="openrouter"), {"writer": "qwen/qwen3.6-27b", "verifiers": "other/model"}
    )


def test_add_args_reads_the_environment(monkeypatch):
    monkeypatch.setenv("HOBSON_LLM_BACKEND", "bedrock-mantle")
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    ap = argparse.ArgumentParser()
    llm.add_args(ap)
    a = ap.parse_args([])
    assert (a.backend, a.region, a.price_in, a.price_out) == (
        "bedrock-mantle",
        "ap-southeast-2",
        0.0,
        0.0,
    )
    a = ap.parse_args(["--backend", "openrouter", "--region", "us-east-2", "--price-in", "0.2"])
    assert (a.backend, a.region, a.price_in) == ("openrouter", "us-east-2", 0.2)


def test_region_falls_back_to_us_west_2(monkeypatch):
    # as of 2026-09 us-east-1 offers neither default (it lists only Qwen3 32B).
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    ap = argparse.ArgumentParser()
    llm.add_args(ap)
    assert ap.parse_args([]).region == "us-west-2"
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    ap = argparse.ArgumentParser()
    llm.add_args(ap)
    assert ap.parse_args([]).region == "eu-central-1"
    assert llm._with_defaults(argparse.Namespace()).region == "us-west-2"


# ---- a generator end to end, offline ------------------------------------------------------


def test_flips_generator_records_backend_model_and_provider(transport, monkeypatch, tmp_path):
    """gen_flips_openrouter.py on the fake transport: writer and verifier rows carry the
    backend, the model and the provider, and the export stage still runs."""
    import gen_flips_openrouter as gf

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    spec = gf.spec_for(0, 0)
    doc = {
        "title": "T",
        "policy": " ".join(["clause"] * 200),
        "items": [
            {
                "flip": f,
                "case": "c",
                "question": "Is it permitted?",
                "criteria": {"yes": "y", "no": "n"},
                "answer_a": a,
                "answer_b": "no" if a == "yes" else "yes",
                "rationale": "r",
            }
            for f, a in zip(spec["flips"], spec["ans_a"], strict=True)
        ],
    }

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        transport.sent.append((req.full_url, dict(req.header_items()), body))
        writer = body["messages"][0]["content"] == gf.SYSTEM_WRITER
        return FakeResponse(
            json.dumps(
                bedrock_reply()
                | {
                    "choices": [
                        {
                            "message": {"content": json.dumps(doc) if writer else "ANSWER: yes"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gen_flips",
            "--backend",
            "bedrock",
            "--region",
            "us-east-2",
            "--out",
            str(tmp_path),
            "--batches",
            "1",
            "--workers",
            "2",
        ],
    )
    gf.main()
    batches = [json.loads(line) for line in (tmp_path / "batches.jsonl").read_text().splitlines()]
    verify = [json.loads(line) for line in (tmp_path / "verify.jsonl").read_text().splitlines()]
    assert len(batches) == 1 and len(verify) == 3 * 2 * 2  # 3 items, 2 variants, 2 judgements
    assert {(b["backend"], b["writer"], b["provider"]) for b in batches} == {
        ("bedrock", "qwen.qwen3-235b-a22b-2507-v1:0", "bedrock:us-east-2")
    }
    assert {(v["backend"], v["model"], v["provider"]) for v in verify} == {
        ("bedrock", "qwen.qwen3-235b-a22b-2507-v1:0", "bedrock:us-east-2")
    }
    # the model table capped both roles at 8000 without a flag and dropped reasoning_effort
    # (the generator's default, high): the default models do not think
    assert {
        (b["model"], b["max_tokens"], b.get("reasoning_effort")) for _, _, b in transport.sent
    } == {("qwen.qwen3-235b-a22b-2507-v1:0", 8000, None)}
    assert all(
        u.startswith("https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/")
        for u, _, _ in transport.sent
    )
    assert (tmp_path / "gen_train.jsonl").exists() and (tmp_path / "stats.json").exists()
