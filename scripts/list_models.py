"""Print the models this machine and this account can actually reach.

    python -m scripts.list_models
    python -m scripts.list_models --backend groq

WHY THIS EXISTS
A hosted model name is the shortest-lived constant in the codebase. This
project's default was `llama-3.3-70b-versatile`, which the provider had already
retired - the run failed with a 404 that reads like an auth problem and is not.
Guessing the current name from documentation or memory is how an afternoon
disappears; asking the provider takes a second.
"""

import argparse
import os
import sys

from rich.console import Console
from rich.table import Table

from src.models import _ensure_ca_bundle, load_env

console = Console()


def groq_models() -> list[str]:
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        console.print("[yellow]GROQ_API_KEY not set — skipping hosted models[/yellow]")
        return []
    return sorted(m.id for m in Groq(api_key=key).models.list().data)


def ollama_models() -> list[str]:
    import ollama

    try:
        return sorted(m["model"] for m in ollama.list()["models"])
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Ollama unreachable ({exc}) — is `ollama serve` running?[/yellow]")
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ollama", "groq", "all"), default="all")
    args = parser.parse_args()

    load_env()
    _ensure_ca_bundle()

    table = Table(box=None)
    table.add_column("backend")
    table.add_column("model")

    if args.backend in ("ollama", "all"):
        for name in ollama_models():
            table.add_row("[cyan]ollama[/cyan]  [dim]local[/dim]", name)
    if args.backend in ("groq", "all"):
        for name in groq_models():
            table.add_row("[magenta]groq[/magenta]  [dim]hosted[/dim]", name)

    console.print()
    console.print(table)
    console.print("\n[dim]use with: --backend <backend> --model <model>[/dim]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
