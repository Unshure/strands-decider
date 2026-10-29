"""Open-loop prod-traffic simulation.

Generates Poisson request arrivals at a sequence of target RPS levels, sampling
input lengths from a prebuilt distribution. Records per-request latency and
failures, and reports the peak RPS whose p99 stays within an SLO.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from benchmark.runner.config import Environment
from benchmark.runner.core import (
    HTTPClient,
    ManagedServer,
    SageMakerClient,
    percentile,
    round_or_none,
)
from benchmark.runner.workload import Request, Workload

DEFAULT_RPS_STEPS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_STEP_SECONDS: float = 30.0
DEFAULT_SLO_P99_MS: float = 2000.0
WARMUP_SECONDS: float = 10.0
MAX_IN_FLIGHT: int = 1024


@dataclass(frozen=True)
class Distribution:
    name: str
    tokenizer: str
    buckets: tuple[tuple[int, float], ...]  # (tokens, weight)
    raw: dict[str, Any]

    @property
    def targets(self) -> tuple[int, ...]:
        return tuple(tokens for tokens, weight in self.buckets if weight > 0)


def load_distribution(path: Path) -> Distribution:
    document = json.loads(path.read_text())
    buckets = []
    for entry in document["buckets"]:
        buckets.append((int(entry["tokens"]), float(entry["weight"])))
    if not any(weight > 0 for _, weight in buckets):
        raise ValueError(f"{path}: distribution has no bucket with positive weight")
    return Distribution(
        name=str(document.get("name") or path.stem),
        tokenizer=str(document.get("tokenizer") or ""),
        buckets=tuple(buckets),
        raw=document,
    )


class LengthSampler:
    """Deterministic weighted sampler over the distribution's buckets."""

    def __init__(self, distribution: Distribution, seed: int):
        self._rng = random.Random(seed)
        self._targets: list[int] = []
        self._weights: list[float] = []
        for tokens, weight in distribution.buckets:
            if weight <= 0:
                continue
            self._targets.append(tokens)
            self._weights.append(weight)

    def next(self) -> int:
        return self._rng.choices(self._targets, weights=self._weights, k=1)[0]


@dataclass
class ProdSample:
    scheduled_s: float
    started_s: float
    ended_s: float
    tokens: int
    status: int
    success: bool
    latency_ms: float
    server_ms: float | None
    error: str | None


def _summarize_step(samples: list[ProdSample], step_seconds: float) -> dict[str, Any]:
    successes = [sample for sample in samples if sample.success]
    latencies = [sample.latency_ms for sample in successes]
    server = [sample.server_ms for sample in successes if sample.server_ms is not None]
    schedule_delays = [
        (sample.started_s - sample.scheduled_s) * 1000
        for sample in samples
        if sample.started_s >= sample.scheduled_s
    ]
    errors: dict[str, int] = {}
    for sample in samples:
        if sample.success:
            continue
        key = sample.error or f"status {sample.status}"
        errors[key] = errors.get(key, 0) + 1

    by_tokens: dict[str, Any] = {}
    for tokens in sorted({sample.tokens for sample in samples}):
        selected = [sample for sample in samples if sample.tokens == tokens]
        selected_ok = [sample for sample in selected if sample.success]
        shape_latencies = [sample.latency_ms for sample in selected_ok]
        by_tokens[f"{tokens}t"] = {
            "requested_input_tokens": tokens,
            "requests": len(selected),
            "successes": len(selected_ok),
            "failure_rate": round((len(selected) - len(selected_ok)) / len(selected), 6)
            if selected
            else 1.0,
            "p50_ms": round_or_none(percentile(shape_latencies, 0.50), 2),
            "p99_ms": round_or_none(percentile(shape_latencies, 0.99), 2),
        }

    return {
        "duration_seconds": round(step_seconds, 3),
        "requests": len(samples),
        "successes": len(successes),
        "failures": len(samples) - len(successes),
        "failure_rate": round((len(samples) - len(successes)) / len(samples), 6)
        if samples
        else 1.0,
        "achieved_rps": round(len(successes) / step_seconds, 4) if step_seconds else 0,
        "offered_rps": round(len(samples) / step_seconds, 4) if step_seconds else 0,
        "latency": {
            "p50_ms": round_or_none(percentile(latencies, 0.50), 2),
            "p95_ms": round_or_none(percentile(latencies, 0.95), 2),
            "p99_ms": round_or_none(percentile(latencies, 0.99), 2),
            "max_ms": round_or_none(max(latencies) if latencies else None, 2),
        },
        "server_latency": {
            "p50_ms": round_or_none(percentile(server, 0.50), 2),
            "p99_ms": round_or_none(percentile(server, 0.99), 2),
        },
        "schedule_delay": {
            "p50_ms": round_or_none(percentile(schedule_delays, 0.50), 2),
            "p99_ms": round_or_none(percentile(schedule_delays, 0.99), 2),
        },
        "errors": dict(sorted(errors.items(), key=lambda item: (-item[1], item[0]))[:20]),
        "shapes": by_tokens,
    }


def poisson_schedule(target_rps: float, duration_s: float, rng: random.Random) -> list[float]:
    """Return arrival times in [0, duration_s) for a Poisson process at target_rps."""
    if target_rps <= 0:
        raise ValueError("target_rps must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    arrivals: list[float] = []
    cursor = 0.0
    while True:
        cursor += rng.expovariate(target_rps)
        if cursor >= duration_s:
            break
        arrivals.append(cursor)
    return arrivals


class ProdSimRunner:
    def __init__(
        self,
        environment: Environment,
        model: str,
        *,
        distribution: Distribution,
        rps_steps: Sequence[int],
        step_seconds: float,
        slo_p99_ms: float,
        seed: int = 0,
        progress: Callable[[str], None] = print,
        advance: Callable[[int], None] | None = None,
    ):
        if not rps_steps:
            raise ValueError("rps_steps must be non-empty")
        if min(rps_steps) <= 0:
            raise ValueError("rps_steps must be positive")
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if slo_p99_ms <= 0:
            raise ValueError("slo_p99_ms must be positive")
        self.environment = environment
        self.model = model
        self.distribution = distribution
        self.rps_steps = list(rps_steps)
        self.step_seconds = float(step_seconds)
        self.slo_p99_ms = float(slo_p99_ms)
        self.seed = seed
        self.progress = progress
        self.advance = advance or (lambda _: None)
        self.server = ManagedServer(environment, model)
        self.client: HTTPClient | SageMakerClient | None = None
        self.workload: Workload | None = None

    def _invoke(self, request: Request, scheduled: float, start_offset: float) -> ProdSample:
        assert self.client is not None
        started = time.perf_counter()
        invocation = self.client.invoke(request.body)
        ended = time.perf_counter()
        success = (
            invocation.status == 200
            and invocation.error is None
            and invocation.answer_count == request.shape.questions
        )
        error = invocation.error
        if invocation.success and invocation.answer_count != request.shape.questions:
            error = "incomplete answer set"
        return ProdSample(
            scheduled_s=scheduled,
            started_s=started - start_offset,
            ended_s=ended - start_offset,
            tokens=request.shape.input_tokens,
            status=invocation.status,
            success=success,
            latency_ms=invocation.latency_ms,
            server_ms=invocation.server_ms,
            error=error,
        )

    def _run_step(
        self,
        target_rps: int,
        duration_s: float,
        sampler: LengthSampler,
        rng: random.Random,
    ) -> dict[str, Any]:
        assert self.workload is not None
        arrivals = poisson_schedule(target_rps, duration_s, rng)
        start_offset = time.perf_counter()
        samples: list[ProdSample] = []
        samples_lock = threading.Lock()
        in_flight = threading.Semaphore(MAX_IN_FLIGHT)
        max_workers = min(MAX_IN_FLIGHT, max(32, 4 * target_rps))
        overloaded = False

        def fire(request: Request, scheduled: float) -> None:
            try:
                sample = self._invoke(request, scheduled, start_offset)
                with samples_lock:
                    samples.append(sample)
                self.advance(1)
            finally:
                in_flight.release()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            sequence_base = 1_000_000 + target_rps * 1000
            for index, scheduled in enumerate(arrivals):
                sleep_for = scheduled - (time.perf_counter() - start_offset)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                if not in_flight.acquire(timeout=0.05):
                    overloaded = True
                    break
                tokens = sampler.next()
                repeated = (index % 2) == 0
                request = self.workload.scheduled_request(
                    tokens,
                    sequence=sequence_base + index,
                    repeated=repeated,
                )
                pool.submit(fire, request, scheduled)
            # Executor __exit__ waits for all in-flight work.

        summary = _summarize_step(samples, duration_s)
        summary["target_rps"] = target_rps
        summary["overloaded"] = overloaded
        return summary

    def run(self, output: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        base_url = self.server.start()
        if not base_url:
            raise ValueError("HTTP environment requires endpoint")
        if self.environment.sagemaker:
            endpoint_name = self.environment.sagemaker.get("endpoint_name")
            endpoints = self.environment.sagemaker.get("endpoints") or {}
            endpoint_name = endpoint_name or endpoints.get(self.model)
            if not endpoint_name:
                raise ValueError(f"no SageMaker endpoint configured for model {self.model}")
            self.client = SageMakerClient(
                endpoint_name,
                self.environment.sagemaker["region"],
                self.model,
                timeout_s=self.environment.timeout_s,
                profile=self.environment.sagemaker.get("profile"),
                max_pool_connections=MAX_IN_FLIGHT,
            )
        else:
            self.client = HTTPClient(base_url, timeout_s=self.environment.timeout_s)
        try:
            health = self.client.health()
            if self.environment.sagemaker:
                expected_instance_type = self.environment.sagemaker["instance_type"]
                actual_instance_types = {
                    variant.get("instance_type")
                    for variant in health.get("variants", [])
                    if variant.get("instance_type")
                }
                if actual_instance_types != {expected_instance_type}:
                    raise ValueError(
                        f"SageMaker endpoint for {self.model} uses "
                        f"{sorted(actual_instance_types) or ['unknown']}, expected "
                        f"{expected_instance_type}"
                    )
            tokenizer_id = health.get("checkpoint") or self.model
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            targets = self.distribution.targets
            self.workload = Workload(
                tokenizer,
                self.seed,
                throughput_tokens=targets[0],
                extra_targets=targets,
            )
            self.progress(
                "calibrating input lengths against server-reported tokens "
                f"({len(set(targets))} distribution targets)"
            )

            def measure(request: Request) -> int | None:
                assert self.client is not None
                invocation = self.client.invoke(request.body)
                if invocation.status in {413, 422}:
                    return None
                if not invocation.success:
                    raise RuntimeError(
                        "input-length calibration failed: "
                        f"status={invocation.status}, error={invocation.error}"
                    )
                if invocation.input_tokens is None:
                    raise RuntimeError(
                        "server did not report usage.input_tokens during calibration"
                    )
                return int(invocation.input_tokens)

            self.workload.calibrate(measure)

            sampler = LengthSampler(self.distribution, seed=self.seed)
            schedule_rng = random.Random(self.seed ^ 0xC0DE)
            self.progress(f"warm-up: {int(WARMUP_SECONDS)}s at {self.rps_steps[0]} rps")
            self._run_step(self.rps_steps[0], WARMUP_SECONDS, sampler, schedule_rng)

            steps: list[dict[str, Any]] = []
            peak: dict[str, Any] | None = None
            stop_reason = "completed"
            for target_rps in self.rps_steps:
                self.progress(
                    f"step: {int(self.step_seconds)}s at {target_rps} rps "
                    f"(slo p99 {int(self.slo_p99_ms)} ms)"
                )
                step = self._run_step(target_rps, self.step_seconds, sampler, schedule_rng)
                steps.append(step)
                p99 = (step["latency"] or {}).get("p99_ms")
                failure_rate = step["failure_rate"] or 0.0
                within_slo = (
                    p99 is not None
                    and p99 <= self.slo_p99_ms
                    and failure_rate < 0.01
                    and not step["overloaded"]
                )
                self.progress(
                    f"  achieved {step['achieved_rps']:.2f} rps, "
                    f"p99 {p99} ms, failures {100 * failure_rate:.2f}%"
                    + (" [overloaded]" if step["overloaded"] else "")
                )
                if within_slo:
                    peak = step
                else:
                    stop_reason = (
                        "overloaded"
                        if step["overloaded"]
                        else "slo-breach"
                        if p99 is not None and p99 > self.slo_p99_ms
                        else "failure-rate"
                    )
                    break

            headline = {
                "peak_rps_within_slo": peak["target_rps"] if peak else None,
                "peak_achieved_rps": peak["achieved_rps"] if peak else None,
                "peak_p99_ms": (peak["latency"] or {}).get("p99_ms") if peak else None,
                "slo_p99_ms": self.slo_p99_ms,
                "stop_reason": stop_reason,
            }
            summary = {
                "headline": headline,
                "lifecycle": {
                    "time_to_ready_seconds": round_or_none(self.server.ready_seconds, 3),
                },
                "distribution": {
                    "name": self.distribution.name,
                    "tokenizer": self.distribution.tokenizer,
                    "buckets": [
                        {"tokens": tokens, "weight": weight}
                        for tokens, weight in self.distribution.buckets
                    ],
                },
                "ramp": {
                    "rps_steps": list(self.rps_steps),
                    "step_seconds": self.step_seconds,
                    "warmup_seconds": WARMUP_SECONDS,
                },
                "steps": steps,
                "health": health,
            }
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
                (output / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n"
                )
            return summary, self.workload.manifest()
        finally:
            self.server.stop()


def write_prodsim_plots(summary: dict[str, Any], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = summary.get("steps") or []
    if not steps:
        return
    rps = [step["target_rps"] for step in steps]
    p50 = [(step["latency"] or {}).get("p50_ms") for step in steps]
    p99 = [(step["latency"] or {}).get("p99_ms") for step in steps]
    failures = [100 * (step.get("failure_rate") or 0) for step in steps]
    slo = summary["headline"]["slo_p99_ms"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if any(value is not None for value in p50):
        ax.plot(rps, p50, marker="o", label="p50")
    if any(value is not None for value in p99):
        ax.plot(rps, p99, marker="o", label="p99")
    ax.axhline(slo, linestyle="--", color="red", label=f"SLO p99 {slo:.0f} ms")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rps, rps)
    ax.set_xlabel("Target RPS (Poisson arrivals)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Prod-sim: latency vs offered load")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "prod-sim-latency-vs-rps.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rps, failures, marker="o", color="firebrick")
    ax.axhline(1.0, linestyle="--", color="gray", label="1% budget")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rps, rps)
    ax.set_xlabel("Target RPS (Poisson arrivals)")
    ax.set_ylabel("Failure rate (%)")
    ax.set_title("Prod-sim: failures vs offered load")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "prod-sim-failures-vs-rps.png", dpi=150)
    plt.close(fig)


def write_prodsim_csv(summary: dict[str, Any], path: Path) -> None:
    import csv

    fields = [
        "target_rps",
        "offered_rps",
        "achieved_rps",
        "requests",
        "failures",
        "failure_rate",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "server_p50_ms",
        "server_p99_ms",
        "schedule_delay_p99_ms",
        "overloaded",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        for step in summary.get("steps") or []:
            latency = step["latency"] or {}
            server = step["server_latency"] or {}
            delay = step["schedule_delay"] or {}
            writer.writerow(
                {
                    "target_rps": step["target_rps"],
                    "offered_rps": step["offered_rps"],
                    "achieved_rps": step["achieved_rps"],
                    "requests": step["requests"],
                    "failures": step["failures"],
                    "failure_rate": step["failure_rate"],
                    "p50_ms": latency.get("p50_ms"),
                    "p95_ms": latency.get("p95_ms"),
                    "p99_ms": latency.get("p99_ms"),
                    "server_p50_ms": server.get("p50_ms"),
                    "server_p99_ms": server.get("p99_ms"),
                    "schedule_delay_p99_ms": delay.get("p99_ms"),
                    "overloaded": step["overloaded"],
                }
            )
