"""Decide whether a prompt change made things worse, and say so loudly.

WHAT MAKES THIS A REGRESSION HARNESS RATHER THAN A REPORT
A report tells you two numbers and leaves you to judge. A regression harness
makes the judgement, exits non-zero when the answer is "worse", and is therefore
something a build can run.

Three rules, in the order they fire:

**A structural failure outranks an accuracy change.** If the parse rate falls,
that is a regression regardless of what accuracy did - unparseable answers are
not answers, and downstream code cannot consume them. An aggregate accuracy
score hides this completely, because an unparseable reply scores exactly the
same as a wrong one.

**A per-class collapse outranks the mean.** A change can lift the average while
destroying one category. Reporting only the mean is how that ships.

**Case-level movement is reported both ways.** Net accuracy can be unchanged
while thirty cases flipped in each direction, which is not a stable prompt - it
is two different prompts that happen to average the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.runner import Run
from src.scorers import PRIMARY

# How far the primary metric may fall before it counts as a regression.
#
# Not zero, deliberately. On sixty cases one flipped answer is 1.7 points, and a
# gate that fires on a single case is a gate people learn to ignore. Not large
# either: three points is two cases, which is the smallest movement that is
# plausibly real rather than noise.
PRIMARY_TOLERANCE = 0.03

# Structural properties get no tolerance at all. A prompt that stops producing
# parseable output on even one extra case has broken a contract, not shifted a
# score.
STRUCTURAL = ("parses_as_json", "has_required_fields", "category_is_valid")

# A class may drop further than the aggregate before it is called out, because
# per-class rates on ~15 cases are inherently noisier - one case is 6.7 points.
CLASS_TOLERANCE = 0.15


@dataclass
class Delta:
    """One metric, before and after."""

    name: str
    before: float
    after: float

    @property
    def change(self) -> float:
        return self.after - self.before


@dataclass
class Comparison:
    baseline: Run
    candidate: Run
    metrics: list[Delta]
    class_deltas: list[Delta]
    improved: list[str]      # case ids that started failing and now pass
    broken: list[str]        # case ids that passed and now fail
    failures: list[str]      # human-readable reasons this is a regression

    @property
    def is_regression(self) -> bool:
        return bool(self.failures)

    @property
    def primary(self) -> Delta:
        return next(m for m in self.metrics if m.name == PRIMARY)


def compare(baseline: Run, candidate: Run) -> Comparison:
    """Diff two runs and decide whether the candidate is a regression."""
    failures: list[str] = []

    # Comparing runs scored on different evaluation sets would be meaningless,
    # and the mistake is easy to make once several runs exist on disk. Caught
    # here rather than producing a confidently wrong verdict.
    if baseline.golden_hash and candidate.golden_hash \
            and baseline.golden_hash != candidate.golden_hash:
        failures.append(
            "evaluation sets differ - these runs are not comparable "
            f"({baseline.golden_hash} vs {candidate.golden_hash})"
        )

    if baseline.model != candidate.model:
        failures.append(
            f"different models ({baseline.model} vs {candidate.model}) - this "
            "measures the provider, not the prompt"
        )

    before, after = baseline.rates(), candidate.rates()
    metrics = [Delta(name, before.get(name, 0.0), after.get(name, 0.0))
               for name in after]

    for delta in metrics:
        if delta.name in STRUCTURAL and delta.change < 0:
            failures.append(
                f"{delta.name} fell {delta.before:.1%} -> {delta.after:.1%} "
                "(structural failures have no tolerance)"
            )
        elif delta.name == PRIMARY and delta.change < -PRIMARY_TOLERANCE:
            failures.append(
                f"{delta.name} fell {delta.before:.1%} -> {delta.after:.1%}, "
                f"beyond the {PRIMARY_TOLERANCE:.0%} tolerance"
            )

    base_classes = baseline.by_class(PRIMARY)
    cand_classes = candidate.by_class(PRIMARY)
    class_deltas = [
        Delta(name, base_classes.get(name, (0, 0.0))[1], rate)
        for name, (_, rate) in cand_classes.items()
    ]
    for delta in class_deltas:
        if delta.change < -CLASS_TOLERANCE:
            failures.append(
                f"class {delta.name!r} fell {delta.before:.1%} -> "
                f"{delta.after:.1%} even though the average may not have"
            )

    base_pass = {r.case_id: r.passed_primary for r in baseline.results}
    improved = sorted(r.case_id for r in candidate.results
                      if r.passed_primary and not base_pass.get(r.case_id, False))
    broken = sorted(r.case_id for r in candidate.results
                    if not r.passed_primary and base_pass.get(r.case_id, False))

    return Comparison(baseline, candidate, metrics, class_deltas,
                      improved, broken, failures)


def churn(comparison: Comparison) -> float:
    """Share of cases whose verdict flipped in either direction.

    A prompt change with a net movement of zero and 40% churn has not left
    things alone - it has traded one set of failures for another. Worth seeing,
    because "no change" and "no effect" are different claims.
    """
    total = len(comparison.candidate.results)
    if not total:
        return 0.0
    return (len(comparison.improved) + len(comparison.broken)) / total
