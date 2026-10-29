from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from benchmark.runner.config import (
    accepted_baseline_dir,
    environment_runs_dir,
    load_environment,
    model_slug,
)
from benchmark.runner.core import BenchmarkRunner, git_metadata, machine_metadata
from benchmark.runner.prodsim import (
    DEFAULT_RPS_STEPS,
    DEFAULT_SLO_P99_MS,
    DEFAULT_STEP_SECONDS,
    ProdSimRunner,
    load_distribution,
    write_prodsim_csv,
    write_prodsim_plots,
)
from benchmark.runner.reporting import (
    comparison_markdown,
    compatibility_errors,
    load_run,
    promote,
    write_artifacts,
    write_plots,
    write_summary_csv,
)
from benchmark.runner.workload import DEFAULT_THROUGHPUT_INPUT_TOKENS, INPUT_LENGTHS

DEFAULT_DISTRIBUTION = (
    Path(__file__).resolve().parents[1] / "distributions" / "decider-eval-v1.json"
)

STANDARD_MEASUREMENT = {
    "requests_per_point": 16,
    "repetitions": 2,
    "concurrency_levels": [1, 4, 16, 32],
}
SMOKE_MEASUREMENT = {
    "requests_per_point": 8,
    "repetitions": 1,
    "concurrency_levels": [1],
}


def _run(args: argparse.Namespace) -> int:
    settings = SMOKE_MEASUREMENT if args.smoke else STANDARD_MEASUREMENT
    environment = load_environment(args.environment)
    if args.smoke and not args.model:
        raise ValueError("smoke runs require a model")
    models = [args.model] if args.model else list(environment.models)
    if not models:
        raise ValueError(f"{environment.path}: models are required for a full benchmark")
    if args.throughput_input_length <= 0:
        raise ValueError("--throughput-input-length must be positive")
    measurement = {
        **settings,
        "throughput_requests_per_concurrency": {
            str(concurrency): max(settings["requests_per_point"], 2 * concurrency)
            for concurrency in settings["concurrency_levels"]
        },
        "preset": "smoke" if args.smoke else "standard",
        "reference_concurrency": 1,
        "failure_rate_limit": 0.01,
        "traffic": ["closed-loop"],
        "throughput_input_tokens": args.throughput_input_length,
        "latency_input_tokens": list(INPUT_LENGTHS),
        "warmup": (
            "each latency input length once; one request per concurrent client "
            "before each throughput repetition"
        ),
    }
    requests_per_model = (
        1
        + settings["repetitions"] * (settings["requests_per_point"] + len(INPUT_LENGTHS))
        + settings["repetitions"]
        * sum(
            max(settings["requests_per_point"], 2 * concurrency) + concurrency
            for concurrency in settings["concurrency_levels"]
        )
    )
    summaries: dict[str, dict[str, Any]] = {}
    outputs: list[Path] = []
    explicit_output = Path(args.output).expanduser().resolve() if args.output else None
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    ) as progress:
        task = progress.add_task("benchmark models", total=requests_per_model * len(models))
        for model in models:
            progress.update(task, description=model.rsplit("/", 1)[-1])
            accepted = accepted_baseline_dir(environment, model)
            baseline_manifest = None
            if not args.smoke and (accepted / "manifest.json").is_file():
                baseline_manifest, _ = load_run(accepted)
                mismatches = []
                if baseline_manifest["environment"].get("fingerprint") != environment.fingerprint:
                    mismatches.append("environment configuration")
                if baseline_manifest["model"].get("id") != model:
                    mismatches.append("model")
                if baseline_manifest.get("measurement") != measurement:
                    mismatches.append("measurement settings")
                if (baseline_manifest.get("workload") or {}).get("seed") != 0:
                    mismatches.append("workload seed")
                if mismatches:
                    raise ValueError(
                        f"accepted baseline {accepted} is incompatible: {', '.join(mismatches)}"
                    )

            if explicit_output:
                out = explicit_output / model_slug(model) if len(models) > 1 else explicit_output
            elif args.smoke:
                out = accepted / "smoke"
            elif baseline_manifest is None:
                out = accepted
            else:
                out = accepted / "candidate"

            started = datetime.now(timezone.utc)
            runner = BenchmarkRunner(
                environment,
                model,
                requests_per_point=settings["requests_per_point"],
                repetitions=settings["repetitions"],
                concurrency_levels=settings["concurrency_levels"],
                seed=0,
                throughput_input_tokens=args.throughput_input_length,
                progress=progress.console.print,
                advance=lambda completed: progress.advance(task, completed),
            )
            summary, workload = runner.run()
            ended = datetime.now(timezone.utc)
            run_id = str(uuid.uuid4())
            manifest = {
                "run_id": run_id,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "duration_seconds": round((ended - started).total_seconds(), 3),
                "environment": {
                    "name": environment.name,
                    "path": str(environment.path),
                    "fingerprint": environment.fingerprint,
                    "configuration": environment.raw,
                },
                "model": {
                    "id": model,
                    "tokenizer": summary.get("health", {}).get("checkpoint") or model,
                    "health": summary.get("health"),
                },
                "measurement": measurement,
                "workload": workload,
                "client": machine_metadata(),
                "code": git_metadata(Path(__file__).resolve().parents[2]),
                "baseline_run_id": baseline_manifest.get("run_id") if baseline_manifest else None,
            }
            summary["run_id"] = run_id
            write_artifacts(out, manifest, summary)
            summaries[model] = summary
            outputs.append(out)

    if len(summaries) > 1:
        graph_dir = explicit_output or environment_runs_dir(environment)
        graph_dir.mkdir(parents=True, exist_ok=True)
        write_plots(summaries, graph_dir, prefix="models-")
        print(graph_dir)
    for out in outputs:
        print(out)
    return 0


def _prod_sim(args: argparse.Namespace) -> int:
    environment = load_environment(args.environment)
    models = [args.model] if args.model else list(environment.models)
    if not models:
        raise ValueError(f"{environment.path}: models are required for prod-sim")
    distribution_path = Path(args.distribution).expanduser().resolve()
    if not distribution_path.is_file():
        raise ValueError(f"distribution not found: {distribution_path}")
    distribution = load_distribution(distribution_path)
    rps_steps = args.rps_steps or list(DEFAULT_RPS_STEPS)
    if any(step <= 0 for step in rps_steps):
        raise ValueError("--rps-steps values must be positive")
    slo_p99_ms = (
        float(args.slo_p99_ms)
        if args.slo_p99_ms is not None
        else float((environment.raw.get("prod_sim") or {}).get("slo_p99_ms", DEFAULT_SLO_P99_MS))
    )
    outputs: list[Path] = []
    explicit_output = Path(args.output).expanduser().resolve() if args.output else None
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    ) as progress:
        estimated_requests = sum(rps_steps) * int(args.step_seconds)
        task = progress.add_task(
            "prod-sim",
            total=estimated_requests * len(models),
        )
        for model in models:
            progress.update(task, description=f"prod-sim {model.rsplit('/', 1)[-1]}")
            if explicit_output:
                out = explicit_output / model_slug(model) if len(models) > 1 else explicit_output
            else:
                out = accepted_baseline_dir(environment, model) / "prod-sim"
            started = datetime.now(timezone.utc)
            runner = ProdSimRunner(
                environment,
                model,
                distribution=distribution,
                rps_steps=rps_steps,
                step_seconds=args.step_seconds,
                slo_p99_ms=slo_p99_ms,
                seed=0,
                progress=progress.console.print,
                advance=lambda completed: progress.advance(task, completed),
            )
            summary, workload = runner.run()
            ended = datetime.now(timezone.utc)
            run_id = str(uuid.uuid4())
            manifest = {
                "run_id": run_id,
                "mode": "prod-sim",
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "duration_seconds": round((ended - started).total_seconds(), 3),
                "environment": {
                    "name": environment.name,
                    "path": str(environment.path),
                    "fingerprint": environment.fingerprint,
                    "configuration": environment.raw,
                },
                "model": {
                    "id": model,
                    "tokenizer": summary.get("health", {}).get("checkpoint") or model,
                    "health": summary.get("health"),
                },
                "measurement": {
                    "preset": "prod-sim",
                    "rps_steps": rps_steps,
                    "step_seconds": args.step_seconds,
                    "slo_p99_ms": slo_p99_ms,
                    "distribution": distribution.name,
                    "distribution_path": str(distribution_path),
                },
                "workload": workload,
                "client": machine_metadata(),
                "code": git_metadata(Path(__file__).resolve().parents[2]),
            }
            summary["run_id"] = run_id
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            write_prodsim_csv(summary, out / "summary.csv")
            write_prodsim_plots(summary, out)
            (out / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            outputs.append(out)
    for out in outputs:
        print(out)
    return 0


def _summarize(args: argparse.Namespace) -> int:
    path = Path(args.run).expanduser().resolve()
    manifest, summary = load_run(path)
    write_summary_csv(summary, path / "summary.csv")
    write_plots({manifest["model"]["id"]: summary}, path)
    (path / "report.md").unlink(missing_ok=True)
    print(path)
    return 0


def _compare(args: argparse.Namespace) -> int:
    environment = load_environment(args.environment)
    baseline = accepted_baseline_dir(environment, args.model)
    candidate = Path(args.candidate).expanduser().resolve()
    base_manifest, base_summary = load_run(baseline)
    candidate_manifest, candidate_summary = load_run(candidate)
    errors = compatibility_errors(base_manifest, candidate_manifest)
    if errors:
        raise ValueError("incompatible benchmark runs:\n- " + "\n- ".join(errors))
    report = comparison_markdown(base_summary, candidate_summary)
    if args.output:
        Path(args.output).expanduser().write_text(report)
    print(report)
    if args.update_baseline:
        promote(candidate, baseline)
        print(f"updated baseline: {baseline}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.runner",
        description="Benchmark Strands Decider inference performance.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a benchmark")
    run.add_argument("environment", help="environment name, directory, or YAML")
    run.add_argument(
        "model",
        nargs="?",
        help="optional checkpoint or model identifier; omit for all configured models",
    )
    run.add_argument(
        "--output",
        help="output directory; defaults beneath the environment's runs directory",
    )
    run.add_argument("--smoke", action="store_true", help="run the short development preset")
    run.add_argument(
        "--throughput-input-length",
        type=int,
        default=DEFAULT_THROUGHPUT_INPUT_TOKENS,
        help=(
            "token length of each throughput-sweep request; "
            f"default {DEFAULT_THROUGHPUT_INPUT_TOKENS}. "
            "Changing it creates a new baseline (compare refuses mixed lengths)."
        ),
    )
    run.set_defaults(func=_run)

    prod_sim = commands.add_parser(
        "prod-sim",
        help="open-loop prod-traffic simulation ramp",
    )
    prod_sim.add_argument("environment", help="environment name, directory, or YAML")
    prod_sim.add_argument(
        "model",
        nargs="?",
        help="optional checkpoint or model identifier; omit for all configured models",
    )
    prod_sim.add_argument(
        "--distribution",
        default=str(DEFAULT_DISTRIBUTION),
        help="input-length distribution JSON (default: decider-eval-v1)",
    )
    prod_sim.add_argument(
        "--rps-steps",
        type=lambda value: [int(part) for part in value.split(",") if part.strip()],
        default=None,
        help=f"comma-separated RPS steps; default {','.join(str(step) for step in DEFAULT_RPS_STEPS)}",
    )
    prod_sim.add_argument(
        "--step-seconds",
        type=float,
        default=DEFAULT_STEP_SECONDS,
        help=f"seconds per step; default {int(DEFAULT_STEP_SECONDS)}",
    )
    prod_sim.add_argument(
        "--slo-p99-ms",
        type=float,
        default=None,
        help=(
            f"p99 latency SLO in ms; default from environment.yaml prod_sim.slo_p99_ms "
            f"or {int(DEFAULT_SLO_P99_MS)} ms"
        ),
    )
    prod_sim.add_argument(
        "--output",
        help="output directory; defaults to <baseline>/prod-sim",
    )
    prod_sim.set_defaults(func=_prod_sim)

    summarize = commands.add_parser("summarize", help="regenerate the CSV and graphs")
    summarize.add_argument("run", help="run directory")
    summarize.set_defaults(func=_summarize)

    compare = commands.add_parser("compare", help="compare candidate with accepted baseline")
    compare.add_argument("environment", help="environment name, directory, or YAML")
    compare.add_argument("model", help="checkpoint or model identifier")
    compare.add_argument("candidate", help="candidate run directory")
    compare.add_argument("--output", help="optional comparison Markdown path")
    compare.add_argument("--update-baseline", action="store_true")
    compare.set_defaults(func=_compare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args) or 0)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
