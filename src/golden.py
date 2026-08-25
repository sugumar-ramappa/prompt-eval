"""The evaluation set: fixed inputs with known-correct answers.

WHY THIS IS THE HARD PART
Most of a prompt evaluation harness is plumbing. The golden set is the bit that
decides whether any of the numbers mean anything, and it is the bit people skip.

Two properties matter more than size:

**Ground truth you did not generate with a model.** If the expected answers came
from an LLM, the harness measures agreement with that LLM rather than
correctness, and every prompt that drifts towards the generator's style scores
better for no real reason. These cases are hand-labelled support tickets from a
sibling project - written and categorised by a person before any of this
existed.

**A fixed sample.** Re-sampling between runs makes two measurements
incomparable: a score moves and you cannot tell whether the prompt changed or
the questions did. The sample is drawn once, with a fixed seed, and committed.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_PATH = DATA_DIR / "golden_set.json"

# The four categories the ticket data uses. Held here rather than derived from
# whatever happens to be in the sample, so that a category missing from the
# sample is still a label the model is allowed to choose - and choosing it is
# then a measurable mistake rather than an unmeasurable one.
CATEGORIES = ("billing", "account", "technical", "shipping")


@dataclass(frozen=True)
class Case:
    """One evaluation case: an input and the answer a person gave."""

    id: str
    text: str
    expected: str
    allowed: tuple[str, ...] = CATEGORIES

    def __post_init__(self):
        if self.expected not in self.allowed:
            raise ValueError(
                f"case {self.id}: expected {self.expected!r} is not in allowed "
                f"{self.allowed} - the case can never pass"
            )


def build_from_tickets(source: Path, n: int = 60, seed: int = 42) -> list[Case]:
    """Draw a stratified sample from the labelled ticket data.

    Stratified, so every category appears in proportion. A random draw of 60
    from 389 could under-represent a class badly enough that its per-class
    score is three tickets wide, and a per-class score built on three tickets
    moves twenty points when one of them changes.
    """
    rows = list(csv.DictReader(source.open()))
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    rng = random.Random(seed)
    cases: list[Case] = []
    for category in sorted(by_category):
        pool = by_category[category]
        take = max(1, round(n * len(pool) / len(rows)))
        for row in rng.sample(pool, min(take, len(pool))):
            cases.append(Case(id=f"t{row['ticket_id']}",
                              text=row["text"].strip(),
                              expected=row["category"].strip().lower()))
    cases.sort(key=lambda c: int(c.id[1:]))
    return cases


def save(cases: list[Case], path: Path = GOLDEN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{k: v for k, v in asdict(c).items() if k != "allowed"}
               for c in cases]
    path.write_text(json.dumps(payload, indent=1))
    return path


def load(path: Path = GOLDEN_PATH) -> list[Case]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python -m scripts.build_golden_set` first"
        )
    return [Case(**row) for row in json.loads(path.read_text())]


def distribution(cases: list[Case]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.expected] = counts.get(case.expected, 0) + 1
    return dict(sorted(counts.items()))


def majority_baseline(cases: list[Case]) -> float:
    """What you score by ignoring the ticket and always guessing the commonest.

    Present for the same reason it is present in every other project here: a
    score without it is not a result. If a prompt scores 30% on four balanced
    categories, it is barely beating a coin with four sides.
    """
    counts = distribution(cases)
    return max(counts.values()) / len(cases) if cases else 0.0
