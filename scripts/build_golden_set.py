"""Draw the evaluation set once, from hand-labelled data, and commit it.

    python -m scripts.build_golden_set --n 60

Run this once. Re-running with different arguments produces a different set, and
runs scored against different sets are not comparable - which the comparison
step refuses to do rather than silently allowing.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.golden import GOLDEN_PATH, build_from_tickets, distribution, majority_baseline, save

console = Console()

# The labels were written by a person, for a different project, before this
# harness existed. That is the property that makes them usable as ground truth:
# had they come from a model, this would measure agreement with that model
# rather than correctness.
DEFAULT_SOURCE = (Path(__file__).resolve().parent.parent.parent
                  / "ticket-classifier" / "data" / "tickets.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60,
                        help="approximate number of cases (default: 60)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing golden set")
    args = parser.parse_args()

    if GOLDEN_PATH.exists() and not args.force:
        console.print(f"[yellow]{GOLDEN_PATH} already exists.[/yellow]")
        console.print("Overwriting it invalidates every run already recorded, "
                      "because they were scored against the old set.")
        console.print("Pass --force if that is what you want.")
        return 1

    if not args.source.exists():
        console.print(f"[red]No source data at {args.source}[/red]")
        return 1

    cases = build_from_tickets(args.source, n=args.n, seed=args.seed)
    save(cases)

    console.print(f"\n[bold green]Wrote {len(cases)} cases[/bold green] to "
                  f"{GOLDEN_PATH.relative_to(Path.cwd())}\n")

    table = Table(box=None)
    table.add_column("category")
    table.add_column("cases", justify="right")
    for category, count in distribution(cases).items():
        table.add_row(category, str(count))
    console.print(table)

    baseline = majority_baseline(cases)
    console.print(f"\nGuessing the most common category scores "
                  f"[bold]{baseline:.1%}[/bold].")
    console.print("[dim]Every score this harness reports should be read against "
                  "that number.[/dim]\n")

    console.print("[bold]Two examples:[/bold]")
    for case in cases[:2]:
        console.print(f"  [dim]{case.id}[/dim] {case.text[:64]!r} "
                      f"-> [bold]{case.expected}[/bold]")
    console.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
