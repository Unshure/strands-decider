"""The System One server inside a SageMaker inference container (deploy/sagemaker/).

SageMaker runs the image as `docker run IMAGE serve` and sends `GET /ping` and
`POST /invocations` to port 8080. This module mounts those two paths on the app from
`server.create_app`, next to `/health` and `/v1/systemone`, so `/invocations` takes the
same request body and returns the same response. Settings come from the environment,
which the SageMaker model's `Environment` sets:

| variable | default |
| --- | --- |
| `DECIDER_CHECKPOINT` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` |
| `DECIDER_DEVICE` | `cuda` |
| `DECIDER_MODEL_NAME` | the checkpoint basename |
| `DECIDER_MAX_BATCH` | `32` |
| `DECIDER_MAX_BATCH_TOKENS` | unset |
| `DECIDER_STRICT_WINDOW` | `false` |
| `SAGEMAKER_BIND_TO_PORT` | `8080` |
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute

from .schema import SystemOneResponse
from .server import create_app

DEFAULT_CHECKPOINT = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"


def settings_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    """The `create_app` arguments that the DECIDER_* variables in `env` set."""
    max_batch_tokens = env.get("DECIDER_MAX_BATCH_TOKENS")
    return {
        "checkpoint": env.get("DECIDER_CHECKPOINT") or DEFAULT_CHECKPOINT,
        "device": env.get("DECIDER_DEVICE") or "cuda",
        "model_name": env.get("DECIDER_MODEL_NAME") or None,
        "max_batch": int(env.get("DECIDER_MAX_BATCH") or 32),
        "max_batch_tokens": int(max_batch_tokens) if max_batch_tokens else None,
        "strict_window": (env.get("DECIDER_STRICT_WINDOW") or "").lower() in ("1", "true", "yes"),
    }


def add_sagemaker_routes(app: FastAPI) -> FastAPI:
    """Mount `/ping` on the `/health` handler and `/invocations` on the `/v1/systemone` one."""
    handlers = {r.path: r.endpoint for r in app.routes if isinstance(r, APIRoute)}
    app.add_api_route("/ping", handlers["/health"], methods=["GET"])
    app.add_api_route(
        "/invocations",
        handlers["/v1/systemone"],
        methods=["POST"],
        response_model=SystemOneResponse,
    )
    return app


def create_sagemaker_app(env: Mapping[str, str] | None = None) -> FastAPI:
    settings = settings_from_env(os.environ if env is None else env)
    checkpoint = settings.pop("checkpoint")
    return add_sagemaker_routes(create_app(checkpoint, **settings))


def main() -> None:
    """The container entrypoint. SageMaker passes `serve` as the argument; it is ignored."""
    import uvicorn

    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT") or 8080)
    # Single worker, as in server.serve: the model owns the GPU.
    uvicorn.run(create_sagemaker_app(), host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
