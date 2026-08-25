"""How to grade an answer that is never byte-identical twice.

THE PROBLEM THIS SOLVES
`assertEquals` does not work on generated text. Ask the same model the same
question twice and you get two different sentences that mean the same thing, so
any test asserting on the exact string fails for the wrong reason.

The answer is to stop asserting on the text and start asserting on **properties
of the text**, in layers:

    1  structural   is it even parseable JSON?
    2  schema       does it have the fields we asked for?
    3  vocabulary   is the label one we actually offered, or an invention?
    4  correctness  is it the right label?

That order matters. A prompt change that drops accuracy by two points is worth
knowing about; a prompt change that stops the output parsing at all is a
different severity of problem, and an aggregate accuracy score hides it -
unparseable answers just look like wrong ones.

Keeping the layers separate is what lets the report say "accuracy barely moved
but the parse rate fell from 100% to 88%", which is the sentence a regression
harness exists to produce.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from src.golden import Case
from src.models import Reply


@dataclass(frozen=True)
class Score:
    """One graded property of one answer."""

    name: str
    passed: bool
    # Kept separate from `passed` so a scorer can be partially satisfied later
    # (a similarity score, a judge's 0-5 rating) without changing the interface.
    value: float
    detail: str = ""


Scorer = Callable[[Reply, Case], Score]


# Models are fond of wrapping JSON in a markdown fence even when told not to.
# Stripping it before parsing is a judgement call worth stating: it means the
# harness measures "did it produce JSON" rather than "did it follow the
# no-fence instruction exactly". The stricter reading would fail answers that
# are structurally fine, and the fence is trivially handled by any caller.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def extract_json(text: str) -> dict | None:
    """Parse a reply as a JSON object, tolerating a markdown fence."""
    candidate = text.strip()
    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parses_as_json(reply: Reply, case: Case) -> Score:
    """Layer 1. Nothing downstream means anything if this fails."""
    obj = extract_json(reply.text)
    if obj is None:
        preview = reply.text.strip().replace("\n", " ")[:60]
        return Score("parses_as_json", False, 0.0, f"unparseable: {preview!r}")
    return Score("parses_as_json", True, 1.0)


def has_required_fields(reply: Reply, case: Case) -> Score:
    """Layer 2. Valid JSON that is missing `category` is still useless."""
    obj = extract_json(reply.text)
    if obj is None:
        return Score("has_required_fields", False, 0.0, "did not parse")
    missing = [f for f in ("category", "confidence") if f not in obj]
    if missing:
        return Score("has_required_fields", False, 0.0,
                     f"missing {', '.join(missing)}")
    return Score("has_required_fields", True, 1.0)


def category_is_valid(reply: Reply, case: Case) -> Score:
    """Layer 3. Did it invent a label we never offered?

    Worth measuring on its own, because it is the failure that most resembles
    success. "refund" instead of "billing" is a plausible, well-formed,
    confidently-returned answer that no downstream system can route - and in an
    accuracy score it is indistinguishable from an ordinary mistake.
    """
    obj = extract_json(reply.text)
    if obj is None:
        return Score("category_is_valid", False, 0.0, "did not parse")
    value = str(obj.get("category", "")).strip().lower()
    if value in case.allowed:
        return Score("category_is_valid", True, 1.0)
    return Score("category_is_valid", False, 0.0, f"invented label {value!r}")


def category_correct(reply: Reply, case: Case) -> Score:
    """Layer 4. The one everybody quotes, and the last one to look at."""
    obj = extract_json(reply.text)
    if obj is None:
        return Score("category_correct", False, 0.0, "did not parse")
    value = str(obj.get("category", "")).strip().lower()
    ok = value == case.expected
    return Score("category_correct", ok, 1.0 if ok else 0.0,
                 "" if ok else f"said {value!r}, expected {case.expected!r}")


def confidence_in_range(reply: Reply, case: Case) -> Score:
    """A confidence of 95 rather than 0.95 breaks any threshold rule built on it."""
    obj = extract_json(reply.text)
    if obj is None:
        return Score("confidence_in_range", False, 0.0, "did not parse")
    raw = obj.get("confidence")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Score("confidence_in_range", False, 0.0, f"not a number: {raw!r}")
    if 0.0 <= value <= 1.0:
        return Score("confidence_in_range", True, 1.0)
    return Score("confidence_in_range", False, 0.0, f"out of range: {value}")


# Order is the escalation order above, and the report prints them in this
# sequence so a structural failure is read before an accuracy figure.
DEFAULT_SCORERS: list[Scorer] = [
    parses_as_json,
    has_required_fields,
    category_is_valid,
    confidence_in_range,
    category_correct,
]

# The scorer whose drop constitutes a regression worth failing a build over.
# Named here rather than assumed by position, because "the last one" is the kind
# of implicit contract that breaks silently when somebody appends a scorer.
PRIMARY = "category_correct"
