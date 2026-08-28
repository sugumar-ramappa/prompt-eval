"""Measure whether the LLM judge can be trusted.

    python -m scripts.calibrate_judge --backend ollama --model llama3.1:8b

WHY CALIBRATION COMES BEFORE USE
An LLM judge is easy to write and worthless unmeasured. Pointing one at a task
with no ground truth and reporting its verdicts replaces "I don't know if the
output is good" with "I don't know if the judge is right", which is the same
problem wearing a lab coat.

This golden set has 60 human-labelled tickets, so the judge can be run where the
answer is already known and scored on whether it agrees.

WHY THE ANSWERS ARE SYNTHESISED RATHER THAN GENERATED
The obvious approach is to run a classifier, collect its answers, and judge
those. It does not work, for a reason worth stating: a decent classifier is right
about 87% of the time, so a 60-case run yields roughly 8 wrong answers - far too
few to measure how often the judge WAVES ONE THROUGH, which is the failure that
matters.

So the calibration set is built deliberately: half the answers are the human
label (the judge should accept) and half are a different category (the judge
should reject). That gives balanced positives and negatives, makes false
acceptances and false rejections separately measurable, and costs no classifier
calls at all.

The judge is never shown the human label. It sees the ticket and a category, the
same as a human grader would. Show it the answer key and it scores near-perfectly
while proving nothing.
"""

from __future__ import annotations

import argparse
import random
import sys

from src import golden
from src.judge import VERDICT_SCHEMA, ask_judge
from src.models import build


def build_calibration_answers(cases, seed: int = 42) -> dict[str, str]:
    """Half correct, half deliberately wrong, deterministically.

    Seeded so a re-run judges the identical set. A calibration figure that moves
    because the wrong answers changed is not a measurement of the judge.
    """
    rng = random.Random(seed)
    answers: dict[str, str] = {}

    for i, case in enumerate(cases):
        if i % 2 == 0:
            answers[case.id] = case.expected
        else:
            wrong = [c for c in case.allowed if c != case.expected]
            answers[case.id] = rng.choice(wrong)

    return answers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ollama",
                        help="ollama (free, local) or groq (metered)")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--limit", type=int, default=0,
                        help="judge only the first N cases (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = golden.load()
    if args.limit:
        cases = cases[:args.limit]

    answers = build_calibration_answers(cases, args.seed)
    planted_correct = sum(1 for c in cases if answers[c.id] == c.expected)

    print(f"\n  calibrating judge: {args.backend}/{args.model}")
    print(f"  {len(cases)} cases - {planted_correct} answers correct, "
          f"{len(cases) - planted_correct} deliberately wrong\n")

    # A schema, not json_mode. Ollama treats `format: "json"` as a no-op and
    # honours a schema, which is the difference between structured output and
    # prose that happens to contain braces.
    judge = build(args.backend, model=args.model, temperature=0.0,
                  json_mode=True, schema=VERDICT_SCHEMA)

    judged = agreements = false_accept = false_reject = unavailable = 0
    disagreements = []

    for n, case in enumerate(cases, start=1):
        answer = answers[case.id]
        verdict = ask_judge(judge, case, answer)

        if verdict is None:
            unavailable += 1
            print(f"  {n:3}/{len(cases)}  {case.id:6} JUDGE UNAVAILABLE")
            continue

        judged += 1
        human_says_correct = (answer == case.expected)

        if verdict.correct == human_says_correct:
            agreements += 1
            mark = "ok"
        elif verdict.correct:
            false_accept += 1
            mark = "FALSE ACCEPT"
        else:
            false_reject += 1
            mark = "false reject"

        if verdict.correct != human_says_correct:
            disagreements.append((case, answer, verdict))

        print(f"  {n:3}/{len(cases)}  {case.id:6} answered {answer:10} "
              f"truth {case.expected:10} {mark}")

    # ------------------------------------------------------------- report --
    print("\n  " + "-" * 62)
    if not judged:
        print("  the judge returned no verdicts at all - nothing to calibrate.")
        return 1

    agreement = agreements / judged
    print(f"  judged           {judged} of {len(cases)}"
          + ("" if not unavailable else f"   ({unavailable} unavailable)"))
    print(f"  agreement        {agreement:.1%}  ({agreements}/{judged})")
    print(f"  false accept     {false_accept}   judge passed a wrong answer")
    print(f"  false reject     {false_reject}   judge failed a right answer")

    if false_accept > false_reject:
        skew = "GENEROUS - it waves through answers the humans rejected"
    elif false_reject > false_accept:
        skew = "harsh - it rejects answers the humans accepted"
    else:
        skew = "balanced"
    print(f"  skew             {skew}")

    print(f"""
  WHAT THIS NUMBER LICENSES
  Agreement of {agreement:.0%} means a verdict from this judge on a task with no
  ground truth is worth about that much and no more. Quote it alongside any
  judged score, the way a poll is quoted with its margin.

  The skew matters more than the rate. A generous judge passes bad output, and
  that is the failure that reaches production; a harsh one only annoys you.
""")

    if disagreements:
        print("  where it disagreed - read these, they are the calibration:\n")
        for case, answer, verdict in disagreements[:8]:
            direction = "accepted" if verdict.correct else "rejected"
            print(f"    {case.id}  {direction} {answer!r} (truth {case.expected!r})")
            print(f"            ticket: {case.text[:68]}")
            print(f"            judge:  {verdict.reason[:68]}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
