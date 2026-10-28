#!/usr/bin/env python3
"""The chat-completion client shared by the data generators.

Every generator in this directory (gen_documents_openrouter.py and the three scripts that
import it) sends its writer and verifier prompts through `chat()` and counts tokens and
cost in `USAGE`. Keeping the client in one place means one retry policy, one place to add a
backend, and one thing to test offline.

Backends (`--backend`, or the environment variable HOBSON_LLM_BACKEND; default openrouter):
  openrouter      https://openrouter.ai/api/v1/chat/completions. One key, many models.
                  Auth: OPENROUTER_API_KEY. Cost: reported by OpenRouter per call.
  bedrock         Amazon Bedrock, the OpenAI-compatible Chat Completions API on the
                  bedrock-runtime endpoint (the one AWS recommends):
                  https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions
                  Auth: a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK, or, if botocore is
                  installed, SigV4 with the default AWS credential chain (IAM roles,
                  profiles). Cost: tokens times --price-in / --price-out.
  bedrock-mantle  The same API on the bedrock-mantle endpoint:
                  https://bedrock-mantle.{region}.api.aws/v1/chat/completions
                  Auth: a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK. Model ids drop the
                  "-v1:0" suffix. Cost: as bedrock.
The region comes from --region, else AWS_REGION, else AWS_DEFAULT_REGION, else us-west-2:
as of 2026-09 us-east-1 offers neither default (it lists only Qwen3 32B).

All three speak the same JSON, so the request differs only in the URL, the auth header,
and the OpenRouter-only fields (provider routing, quantization filter, cost reporting),
which are sent to OpenRouter alone. Reasoning is requested as {"reasoning": {"effort"}} on
OpenRouter and as "reasoning_effort" on Bedrock, where MODELS says which models take it: the
field is dropped for the listed models that do not think, and a model not in the table gets it
as given.

Nothing here reads a key at import time: keys are read when a call is made, so the export
stage of a generator runs without one.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

BACKENDS = ("openrouter", "bedrock", "bedrock-mantle")

# The models each backend uses when a generator is not given --writer / --verify-models /
# --checker. The committed exports in data/generators/gen_*/ were all written and verified with the
# OpenRouter pair; Bedrock does not offer Qwen3.6-27B or Qwen3.5-397B-A17B, so its defaults
# are the nearest open-weight Qwen3 models it does offer. They are different models: a
# Bedrock run gives new data with its own labels, not the committed data.
#
# Bedrock's writer and verifier are both Qwen3 235B A22B 2507, chosen on gen_flips_openrouter.py
# runs in us-west-2 on 2026-09-29 (4 batches per run, one policy and up to 3 pairs per batch).
# As writer it gave a policy of the required length in 12 of 12 batches (3 seeds); Qwen3 32B
# fell short of 120 words in 3 of 8 batches (thinking or not) and Qwen3 Next 80B A3B in 1 of
# 4, and every verifier kept 0-25% of those writers' pairs. On the 235B writer's 31 pairs the
# verifiers kept 15 (235B, which does not think; 84 of 124 judgements agreed with the writer)
# and 13 (32B with reasoning_effort high; 80 of 124). The OpenRouter pair kept 52% of the
# committed gen_flips pairs. The two roles share one model here, unlike on OpenRouter; the
# verifier still judges in a fresh context, without the writer's answer.
DEFAULT_MODELS = {
    "openrouter": {"writer": "qwen/qwen3.6-27b", "verifier": "qwen/qwen3.5-397b-a17b"},
    "bedrock": {
        "writer": "qwen.qwen3-235b-a22b-2507-v1:0",
        "verifier": "qwen.qwen3-235b-a22b-2507-v1:0",
    },
    "bedrock-mantle": {
        "writer": "qwen.qwen3-235b-a22b-2507",
        "verifier": "qwen.qwen3-235b-a22b-2507",
    },
}

# Per-model defaults for the Bedrock models. A generator's --writer-max-tokens /
# --verify-max-tokens / --check-max-tokens flag always wins; a model not listed here keeps the
# generator's own default (16k-32k, sized for the OpenRouter models, whose thinking counts
# against it).
#   max_tokens  The Qwen3 models on Bedrock and Mantle cap output at 8K tokens (their model
#               cards), and the API rejects a larger request.
#   reasoning   Whether the model takes "reasoning_effort". Qwen3 32B does (a hybrid thinking
#               model; on bedrock-runtime the thinking comes back inline in the content, in a
#               <reasoning> block that chat() strips). The Qwen3 235B A22B 2507 and Qwen3 Next
#               80B A3B weights are the Instruct releases: they do not think, and a request that
#               carries reasoning_effort never returns (measured 2026-09-29: a 4-token answer
#               takes 0.4 s without the field and times out at 900 s with it, on both endpoints).
#               The field is dropped for them whatever --reasoning-effort says. A model that
#               is not listed gets --reasoning-effort as given.
# Edit this table when a model's cap or its handling of reasoning_effort changes.
MODELS = {
    "qwen.qwen3-32b-v1:0": {"max_tokens": 8000, "reasoning": True},
    "qwen.qwen3-32b": {"max_tokens": 8000, "reasoning": True},
    "qwen.qwen3-235b-a22b-2507-v1:0": {"max_tokens": 8000, "reasoning": False},
    "qwen.qwen3-235b-a22b-2507": {"max_tokens": 8000, "reasoning": False},
    "qwen.qwen3-next-80b-a3b": {"max_tokens": 8000, "reasoning": False},
    "qwen.qwen3-next-80b-a3b-instruct": {"max_tokens": 8000, "reasoning": False},
}

_INLINE_REASONING = re.compile(r"\A\s*<reasoning>.*?</reasoning>", re.S)
_lock = threading.Lock()
USAGE = {"in": 0, "out": 0, "calls": 0, "cost": 0.0}


class Truncated(ValueError):
    pass


# ---- arguments shared by the generators -------------------------------------------------


def add_args(ap) -> None:
    ap.add_argument(
        "--backend",
        choices=BACKENDS,
        default=os.environ.get("HOBSON_LLM_BACKEND", "openrouter"),
        help="where the calls go (default: $HOBSON_LLM_BACKEND, else openrouter)",
    )
    ap.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2",
        help="AWS region for the bedrock backends",
    )
    ap.add_argument(
        "--price-in",
        type=float,
        default=0.0,
        help="USD per million input tokens, for the cost estimate on bedrock (OpenRouter reports its own)",
    )
    ap.add_argument(
        "--price-out", type=float, default=0.0, help="USD per million output tokens, as --price-in"
    )


def default_model(args, role: str) -> str:
    """The backend's default writer or verifier ("checker" is a verifier)."""
    return DEFAULT_MODELS[args.backend]["writer" if role == "writer" else "verifier"]


def max_tokens(models: str, explicit, fallback: int) -> int:
    """The explicit flag if given; else the smallest table value over the comma-separated
    models; else the generator's own default."""
    if explicit is not None:
        return explicit
    caps = [MODELS[m]["max_tokens"] for m in models.split(",") if m in MODELS]
    return min(caps) if caps else fallback


def describe(args, models: dict) -> str:
    """One startup line: the backend, the models, and whether they are the ones that
    produced the committed data."""
    recorded = DEFAULT_MODELS["openrouter"]
    same = args.backend == "openrouter" and all(
        set(m.split(",")) <= {recorded["writer"], recorded["verifier"]} for m in models.values()
    )
    where = args.backend + ("" if args.backend == "openrouter" else f" ({args.region})")
    line = f"backend {where}: " + ", ".join(f"{k} {v}" for k, v in models.items())
    if not same:
        line += (
            f". NOTE: these are not the models that produced the committed data "
            f"(OpenRouter, writer {recorded['writer']}, verifier {recorded['verifier']}); "
            f"this run gives new data with its own labels"
        )
    return line


def cost_note(args) -> str:
    """How the printed cost was obtained."""
    return (
        "reported by OpenRouter"
        if args.backend == "openrouter"
        else "at the given --price-in/--price-out"
    )


def check_credentials(args) -> None:
    """Exit with one line if the chosen backend cannot authenticate."""
    if args.backend == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("set OPENROUTER_API_KEY")
        return
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return
    if args.backend == "bedrock-mantle":
        raise SystemExit(
            "set AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key; bedrock-mantle takes no other auth here)"
        )
    try:
        import botocore.session  # noqa: F401
    except ImportError:
        raise SystemExit(
            "set AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key), or install botocore "
            "to sign with the default AWS credential chain"
        ) from None
    if _credentials() is None:
        raise SystemExit(
            "no AWS credentials found: set AWS_BEARER_TOKEN_BEDROCK, or configure "
            "the default credential chain (AWS_PROFILE, a role, ...)"
        )


# ---- the request ------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_URL = _OPENROUTER_URL  # kept for readers of the old name


def url_for(args) -> str:
    if args.backend == "openrouter":
        return _OPENROUTER_URL
    if args.backend == "bedrock":
        return f"https://bedrock-runtime.{args.region}.amazonaws.com/openai/v1/chat/completions"
    return f"https://bedrock-mantle.{args.region}.api.aws/v1/chat/completions"


def _body(args, model, system, prompt, max_tokens, temperature) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    if args.backend == "openrouter":
        body["usage"] = {"include": True}
        body["provider"] = {"quantizations": args.quantizations.split(","), "allow_fallbacks": True}
        if args.reasoning_effort != "none":
            body["reasoning"] = {"effort": args.reasoning_effort, "exclude": True}
        if args.providers:
            body["provider"].update(order=args.providers.split(","), allow_fallbacks=False)
    elif args.reasoning_effort != "none" and MODELS.get(model, {}).get("reasoning", True):
        body["reasoning_effort"] = args.reasoning_effort
    return body


def _credentials():
    import botocore.session

    return botocore.session.get_session().get_credentials()


def _headers(args, url: str, data: bytes) -> dict:
    headers = {"Content-Type": "application/json"}
    if args.backend == "openrouter":
        headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"
        headers["X-Title"] = "hobson-gen-pilot"
        return headers
    key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if key or args.backend == "bedrock-mantle":
        headers["Authorization"] = f"Bearer {key}"
        return headers
    # SigV4 with the default credential chain; signed per attempt, the signature carries a date
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    req = AWSRequest(method="POST", url=url, data=data, headers=headers)
    SigV4Auth(_credentials().get_frozen_credentials(), "bedrock", args.region).add_auth(req)
    return dict(req.headers)


def chat(
    args, model: str, system: str, prompt: str, max_tokens: int, temperature: float
) -> tuple[str, str]:
    """One chat completion; returns (answer text, provider). Retries transient failures.

    The provider is the host OpenRouter routed to, or "<backend>:<region>" on Bedrock."""
    args = _with_defaults(args)
    url = url_for(args)
    data = json.dumps(_body(args, model, system, prompt, max_tokens, temperature)).encode()
    delay = 5.0
    for attempt in range(8):
        req = urllib.request.Request(
            url, data=data, method="POST", headers=_headers(args, url, data)
        )
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                resp = json.loads(r.read())
            if "error" in resp:
                raise RuntimeError(f"api error: {resp['error']}")
            choice = resp["choices"][0]
            text = choice["message"].get("content") or ""
            if args.backend != "openrouter":  # bedrock-runtime puts Qwen3 32B's thinking here
                text = _INLINE_REASONING.sub("", text)
            text = text.strip()
            u = resp.get("usage", {})
            with _lock:
                USAGE["in"] += u.get("prompt_tokens", 0)
                USAGE["out"] += u.get("completion_tokens", 0)
                USAGE["cost"] += (
                    float(u.get("cost") or 0)
                    if args.backend == "openrouter"
                    else (
                        u.get("prompt_tokens", 0) * args.price_in
                        + u.get("completion_tokens", 0) * args.price_out
                    )
                    / 1e6
                )
                USAGE["calls"] += 1
            if choice.get("finish_reason") == "length":
                raise Truncated(f"output truncated at max_tokens={max_tokens}")
            if not text:
                raise RuntimeError("empty content")
            provider = (
                resp.get("provider", "?")
                if args.backend == "openrouter"
                else f"{args.backend}:{args.region}"
            )
            return text, provider
        except Truncated:
            raise
        except urllib.error.HTTPError as e:
            body = e.read()
            # 402 "in_flight_budget_exhausted": credit is reserved for calls in flight, and
            # this one did not fit; it clears as they finish (a real out-of-credit 402 does not)
            busy = e.code == 402 and b"in_flight_budget" in body
            if (e.code not in (408, 429, 500, 502, 503, 504) and not busy) or attempt == 7:
                raise RuntimeError(f"HTTP {e.code}: {body[:500]!r}") from e
            if busy:
                delay = max(delay, 60.0)
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            RuntimeError,
            KeyError,
            json.JSONDecodeError,
        ):  # HTTPException: IncompleteRead
            if attempt == 7:
                raise
        time.sleep(delay + random.random() * delay)
        delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def _with_defaults(args):
    """A namespace with the bedrock fields present, for callers built before add_args."""
    import argparse

    d = {
        "backend": "openrouter",
        "region": "us-west-2",
        "price_in": 0.0,
        "price_out": 0.0,
        **vars(args),
    }
    return argparse.Namespace(**d)
