"""`python -m training.data recipes | build | stats | peek`.

The corpus-building commands that used to live under `strands-decider data …`.
"""

from __future__ import annotations

from collections import Counter
from itertools import islice

import typer
from rich.console import Console
from rich.table import Table

from strands_decider.prompting import build_prompt

from . import recipes as _recipes
from .format import Example, read_jsonl, write_jsonl

app = typer.Typer(help="Build and inspect training corpora.", no_args_is_help=True)
console = Console()


@app.command("recipes")
def recipes_cmd() -> None:
    """List the datasets that can be converted into training examples."""
    table = Table(title="Available recipes")
    for col in ("name", "dataset", "config", "split", "max examples"):
        table.add_column(col)
    for name in sorted(_recipes.SPECS):
        s = _recipes.SPECS[name]
        table.add_row(name, s.path, s.config or "-", s.split, str(s.max_examples))
    console.print(table)


@app.command("build")
def build_cmd(
    out: str = typer.Option("data/train.jsonl", help="Output JSONL (.gz supported)."),
    recipes: list[str] | None = typer.Option(
        None,
        "--recipe",
        "-r",
        help="Recipe name; repeat. Default: all. If a recipe fails, the build writes nothing.",
    ),
    max_options: int = typer.Option(16, help="Cap options per example (large label sets)."),
    max_chars: int = typer.Option(2000, help="Truncate each state to this many characters."),
    max_examples: int | None = typer.Option(None, help="Per-recipe example cap."),
    seed: int = typer.Option(0),
    holdout: list[str] | None = typer.Option(
        None, "--holdout", help="Task to route to a separate held-out file; repeat."
    ),
) -> None:
    """Download datasets and convert them to (state, question, answer) examples."""
    names = list(recipes) if recipes else _recipes.available_recipes()
    hold = set(holdout or [])

    kept: list[Example] = []
    held: list[Example] = []
    for name in names:
        console.print(f"[cyan]building[/] {name} ...")
        try:
            examples = _recipes.build_recipe(
                name,
                max_options=max_options,
                max_chars=max_chars,
                seed=seed,
                max_examples=max_examples,
            )
        except Exception as exc:
            hint = (
                ' Install the "train" extra: pip install -e ".[train]".'
                if isinstance(exc, ModuleNotFoundError) and exc.name == "datasets"
                else ""
            )
            raise SystemExit(f"recipe {name} failed, no corpus written: {exc!r}.{hint}") from exc
        (held if name in hold else kept).extend(examples)
        console.print(f"  [green]{len(examples):,}[/] examples")

    if not kept:
        raise typer.Exit(code=1)

    n = write_jsonl(out, kept)
    console.print(f"[green]wrote[/] {n:,} examples -> {out}")
    if held:
        hpath = out.replace(".jsonl", ".holdout.jsonl")
        console.print(f"[green]wrote[/] {write_jsonl(hpath, held):,} held-out -> {hpath}")


@app.command("stats")
def stats_cmd(path: str = typer.Argument(..., help="JSONL corpus to summarise.")) -> None:
    """Show composition of a corpus: tasks, kinds, option-count distribution."""
    tasks: Counter = Counter()
    kinds: Counter = Counter()
    nopts: Counter = Counter()
    total = 0
    for ex in read_jsonl(path):
        tasks[ex.task] += 1
        kinds[ex.kind] += 1
        nopts[ex.n_options] += 1
        total += 1

    console.print(f"[bold]{total:,}[/] examples in {path}")
    t = Table(title="By task")
    t.add_column("task")
    t.add_column("n", justify="right")
    t.add_column("share", justify="right")
    for name, c in tasks.most_common():
        t.add_row(name, f"{c:,}", f"{100 * c / total:.1f}%")
    console.print(t)

    k = Table(title="By kind / option count")
    k.add_column("kind")
    k.add_column("n", justify="right")
    for name, c in kinds.most_common():
        k.add_row(name, f"{c:,}")
    console.print(k)
    console.print("option counts: " + ", ".join(f"{n}:{c:,}" for n, c in sorted(nopts.items())))


@app.command("peek")
def peek_cmd(
    path: str = typer.Argument(...),
    n: int = typer.Option(3, help="How many rendered prompts to print."),
) -> None:
    """Render a few examples exactly as the model will see them."""
    for ex in islice(read_jsonl(path), n):
        prompt, rq = build_prompt(ex.state, ex.to_question())
        console.rule(f"{ex.task} / {ex.kind} / gold slot {ex.label}")
        console.print(prompt)
        console.print(f"[dim]slots: {rq.slot_labels}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
