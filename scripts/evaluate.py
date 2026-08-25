"""Score one prompt version against the golden set.

    python -m scripts.evaluate --prompt classify-v1 --label v1
    python -m scripts.evaluate --prompt classify-v2 --label v2 --backend groq

Results are cached per (prompt, model, case), so re-running an unchanged prompt
costs nothing. Only new or edited prompts trigger real inference.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src import models
from src.golden import load, majority_baseline
from src.runner import Run, run_prompt
from src.scorers import DEFAULT_SCORERS, PRIMARY

console = Console()
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _progress(done: int, total: int, cached: bool) -> None:
    if done == 1 or done % 5 == 0 or done == total:
        tag = "cached" if cached else "calling model"
        console.print(f"  [dim]{done}/{total} {tag}[/dim]" + " " * 12, end="\r")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True,
                        help="prompt file stem, e.g. classify-v1")
    parser.add_argument("--label", help="name for this run (default: prompt stem)")
    parser.add_argument("--backend", default="ollama",
                        choices=sorted(models.BACKENDS))
    parser.add_argument("--model", help="override the backend's default model")
    parser.add_argument("--json-mode", action="store_true",
                        help="constrain decoding to valid JSON. Off by default "
                             "because several experiments measure whether the "
                             "PROMPT produces valid JSON, and this would hide it")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    path = PROMPTS_DIR / f"{args.prompt}.txt"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.txt"))
        console.print(f"[red]No prompt {args.prompt!r}.[/red] Available: {available}")
        return 1

    template = path.read_text()
    cases = load()
    label = args.label or args.prompt

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.backend == "ollama":
        kwargs["json_mode"] = args.json_mode
    model = models.build(args.backend, **kwargs)

    console.print(f"\n[bold blue]{label}[/bold blue]  "
                  f"[dim]{args.prompt} · {model.name} · {len(cases)} cases[/dim]\n")

    try:
        run = run_prompt(template, cases, model, label=label,
                         prompt_name=args.prompt, progress=_progress,
                         use_cache=not args.no_cache)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        console.print(f"\n[red]Run failed:[/red] {exc}")
        if args.backend == "ollama":
            console.print("[dim]Is `ollama serve` running?[/dim]")
        return 1

    console.print(" " * 40, end="\r")
    run.save()

    live = sum(1 for r in run.results if not r.cached)
    console.print(f"[dim]{live} model calls, {len(run.results) - live} from cache, "
                  f"{run.seconds:.1f}s[/dim]\n")

    table = Table(box=None, title="Scores, structural first")
    table.add_column("property")
    table.add_column("pass rate", justify="right")
    for scorer in DEFAULT_SCORERS:
        name = scorer.__name__
        rate = run.rate(name)
        colour = "green" if rate == 1.0 else ("yellow" if rate >= 0.9 else "red")
        marker = " [bold](primary)[/bold]" if name == PRIMARY else ""
        table.add_row(f"{name}{marker}", f"[{colour}]{rate:.1%}[/{colour}]")
    console.print(table)

    baseline = majority_baseline(cases)
    primary = run.rate(PRIMARY)
    console.print(f"\n  Guessing scores {baseline:.1%}. This prompt scores "
                  f"[bold]{primary:.1%}[/bold] "
                  f"([bold]{primary - baseline:+.1%}[/bold]).")

    table = Table(box=None, title="\nPer category")
    table.add_column("category")
    table.add_column("n", justify="right")
    table.add_column("correct", justify="right")
    for name, (n, rate) in run.by_class(PRIMARY).items():
        colour = "green" if rate >= 0.9 else ("yellow" if rate >= 0.7 else "red")
        table.add_row(name, str(n), f"[{colour}]{rate:.1%}[/{colour}]")
    console.print(table)

    broken = [r for r in run.results if not r.scores.get("parses_as_json", False)]
    if broken:
        console.print(f"\n[red]{len(broken)} replies did not parse as JSON:[/red]")
        for r in broken[:3]:
            console.print(f"  [dim]{r.case_id}[/dim] {r.detail.get('parses_as_json','')}")

    invented = [r for r in run.results
                if r.scores.get("parses_as_json") and not r.scores.get("category_is_valid")]
    if invented:
        console.print(f"\n[yellow]{len(invented)} replies invented a category:[/yellow]")
        for r in invented[:3]:
            console.print(f"  [dim]{r.case_id}[/dim] {r.detail.get('category_is_valid','')}")

    console.print(f"\n[dim]saved as run {label!r} — compare with:"
                  f"\n  python -m scripts.compare --baseline <other> --candidate {label}[/dim]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
