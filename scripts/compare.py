"""Diff two runs and fail if the prompt change made things worse.

    python -m scripts.compare --baseline v1 --candidate v2

Exit code 0 when the candidate is acceptable, 1 when it is a regression - so
this is runnable from a build. That exit code is the whole point: a report you
have to read and interpret is a report somebody eventually stops reading.
"""

import argparse
import sys

from rich.console import Console
from rich.table import Table

from src.compare import CLASS_TOLERANCE, PRIMARY_TOLERANCE, churn, compare
from src.runner import Run
from src.scorers import PRIMARY

console = Console()


def _arrow(change: float) -> str:
    if change > 0.0005:
        return f"[green]+{change:.1%}[/green]"
    if change < -0.0005:
        return f"[red]{change:.1%}[/red]"
    return "[dim]—[/dim]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="run label to compare against")
    parser.add_argument("--candidate", required=True, help="run label being judged")
    parser.add_argument("--show-broken", type=int, default=5,
                        help="how many newly-failing cases to print")
    args = parser.parse_args()

    try:
        baseline = Run.load(args.baseline)
        candidate = Run.load(args.candidate)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    result = compare(baseline, candidate)

    console.print(f"\n[bold blue]{args.baseline} → {args.candidate}[/bold blue]")
    console.print(f"[dim]{baseline.prompt_name} ({baseline.prompt_hash}) → "
                  f"{candidate.prompt_name} ({candidate.prompt_hash})"
                  f" · {candidate.model} · {len(candidate.results)} cases[/dim]\n")

    table = Table(box=None)
    table.add_column("property")
    table.add_column(args.baseline, justify="right")
    table.add_column(args.candidate, justify="right")
    table.add_column("change", justify="right")
    for delta in result.metrics:
        marker = " [bold](primary)[/bold]" if delta.name == PRIMARY else ""
        table.add_row(f"{delta.name}{marker}", f"{delta.before:.1%}",
                      f"{delta.after:.1%}", _arrow(delta.change))
    console.print(table)

    table = Table(box=None, title="\nPer category")
    table.add_column("category")
    table.add_column(args.baseline, justify="right")
    table.add_column(args.candidate, justify="right")
    table.add_column("change", justify="right")
    for delta in result.class_deltas:
        table.add_row(delta.name, f"{delta.before:.1%}", f"{delta.after:.1%}",
                      _arrow(delta.change))
    console.print(table)

    # Net movement and churn are different questions, and only reporting the
    # first is how a prompt that traded one set of failures for another gets
    # described as "no change".
    moved = churn(result)
    console.print(f"\n  [green]{len(result.improved)} fixed[/green] · "
                  f"[red]{len(result.broken)} broken[/red] · "
                  f"{moved:.0%} of cases changed verdict")
    if moved > 0.2 and abs(result.primary.change) < 0.02:
        console.print("  [yellow]Net movement is small but churn is high — this is "
                      "not the same prompt behaving slightly differently,\n  it is a "
                      "different prompt that happens to average the same.[/yellow]")

    if result.broken:
        console.print(f"\n[red]Newly failing:[/red]")
        by_id = {r.case_id: r for r in candidate.results}
        for case_id in result.broken[:args.show_broken]:
            r = by_id[case_id]
            reason = r.detail.get(PRIMARY) or r.detail.get("parses_as_json") or ""
            console.print(f"  [dim]{case_id}[/dim] {reason}")
        if len(result.broken) > args.show_broken:
            console.print(f"  [dim]…and {len(result.broken) - args.show_broken} more[/dim]")

    console.print()
    if result.is_regression:
        console.print("[bold red]  REGRESSION  [/bold red]")
        for reason in result.failures:
            console.print(f"  [red]·[/red] {reason}")
        console.print(f"\n[dim]tolerances: primary {PRIMARY_TOLERANCE:.0%}, "
                      f"per-class {CLASS_TOLERANCE:.0%}, structural 0%[/dim]\n")
        return 1

    console.print("[bold green]  PASS  [/bold green]  no regression detected")
    console.print(f"[dim]tolerances: primary {PRIMARY_TOLERANCE:.0%}, "
                  f"per-class {CLASS_TOLERANCE:.0%}, structural 0%[/dim]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
