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
import json
from datetime import datetime, timezone
from pathlib import Path
import random
import sys

from src import golden
from src.judge import VERDICT_SCHEMA, ask_judge, load_judge_prompt
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
    parser.add_argument("--judge-prompt", default="judge-v2",
                        help="which prompts/judge-*.txt to grade with. "
                             "judge-v1 is a RECONSTRUCTION - see prompts/README.md")
    args = parser.parse_args()

    cases = golden.load()
    if args.limit:
        cases = cases[:args.limit]

    answers = build_calibration_answers(cases, args.seed)
    planted_correct = sum(1 for c in cases if answers[c.id] == c.expected)

    template = load_judge_prompt(args.judge_prompt)
    print(f"\n  calibrating judge: {args.backend}/{args.model}"
          f"  prompt {args.judge_prompt}")
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
        verdict = ask_judge(judge, case, answer, template)

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

    _record(args, cases, judged, unavailable, agreements, agreement,
            false_accept, false_reject, skew, disagreements)

    if disagreements:
        print("  where it disagreed - read these, they are the calibration:\n")
        for case, answer, verdict in disagreements[:8]:
            direction = "accepted" if verdict.correct else "rejected"
            print(f"    {case.id}  {direction} {answer!r} (truth {case.expected!r})")
            print(f"            ticket: {case.text[:68]}")
            print(f"            judge:  {verdict.reason[:68]}\n")

    return 0


def _record(args, cases, judged, unavailable, agreements, agreement,
            false_accept, false_reject, skew, disagreements) -> None:
    """Write the calibration where it can be read back.

    Until 30 Aug this script only printed. The 76.7% it produced was quoted in
    five documents and existed in none of them as data - the exact failure this
    workspace hit with a sibling project's latency figures, which survived in a
    README after the result files had been overwritten with cached runs.

    **This file is not a reproducibility guarantee and must not be read as one.**
    The judge is a language model. It runs at temperature 0, which is as stable
    as sampling gets and is not the same as deterministic: a different model
    build or Ollama version can move the number. So the model, the case count and
    the date are recorded beside the figures, and re-running is expected to give
    something close rather than something identical. A result you can attribute
    beats a result you can only assert, even when it will not reproduce to the
    decimal.
    """
    out = (Path(__file__).resolve().parents[1] / "data" / "runs"
           / f"judge-calibration-{args.judge_prompt}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model": args.model,
        "judgePrompt": args.judge_prompt,
        "seed": args.seed,
        "cases": len(cases),
        "judged": judged,
        "unavailable": unavailable,
        "agreements": agreements,
        "agreement": round(agreement, 4),
        "falseAccept": false_accept,
        "falseReject": false_reject,
        "skew": skew,
        "reproducible": False,
        "note": (
            "A judge is a model, so this does not reproduce to the decimal even "
            "at temperature 0. Agreement alone is the WRONG metric for choosing "
            "a judge: v2 agreed more often than v1 (76.7% against 66.7%) and was "
            "the worse judge, because it waved through 11 wrong answers against "
            "v1's 1. A generous judge passes bad output; a harsh one only annoys "
            "you. Read falseAccept and falseReject, not agreement."
        ),
        "disagreements": [
            {"case": c.id, "judgeSaid": "accepted" if v.correct else "rejected",
             "answer": a, "truth": c.expected, "reason": v.reason}
            for c, a, v in disagreements
        ],
    }, indent=2) + "\n")
    print(f"  recorded to {out.parent.name}/{out.name}")


if __name__ == "__main__":
    sys.exit(main())
