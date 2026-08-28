"""Calibrate several judges on the same cases, and compare them to each other.

    python -m scripts.compare_judges --judge ollama:llama3.1:8b --judge ollama:qwen2.5:14b

WHY MORE THAN ONE JUDGE
A single judge's agreement rate cannot distinguish two very different situations:

    the judge is too weak for this task
    the task is genuinely ambiguous and no judge will score well

Running a second, larger judge in the same runtime separates them. If agreement
jumps, the first judge was too small. If it does not move, the difficulty is in
the task - which is the more useful finding, because it says no judge should be
gating anything here.

THE THIRD THING THIS PRODUCES, WHICH IS THE MOST VALUABLE
Cases where EVERY judge disagrees with the human label.

A judge disagreeing with a label is usually the judge being wrong. Every judge
disagreeing with the same label, independently, is evidence the LABEL is wrong -
and a wrong label is worse than a wrong judge, because every score computed
against it inherits the error silently and forever.

So this ends up auditing the golden set, which was never the intention.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from src import golden
from src.judge import VERDICT_SCHEMA, ask_judge
from src.models import build
from scripts.calibrate_judge import build_calibration_answers


def parse_judge(spec: str):
    """`backend:model` -> (backend, model). Colons are legal in model names."""
    backend, _, model = spec.partition(":")
    if not model:
        raise argparse.ArgumentTypeError(
            f"expected backend:model, got {spec!r} (e.g. ollama:qwen2.5:14b)")
    return backend, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="append", type=parse_judge, required=True,
                        help="backend:model, repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = golden.load()
    if args.limit:
        cases = cases[:args.limit]
    answers = build_calibration_answers(cases, args.seed)

    # verdicts[case_id][judge_label] = bool | None
    verdicts: dict[str, dict[str, bool | None]] = defaultdict(dict)
    labels: list[str] = []

    for backend, model in args.judge:
        label = f"{backend}/{model}"
        labels.append(label)
        print(f"\n  === {label} ===")

        judge = build(backend, model=model, temperature=0.0,
                      json_mode=True, schema=VERDICT_SCHEMA)

        for n, case in enumerate(cases, start=1):
            verdict = ask_judge(judge, case, answers[case.id])
            verdicts[case.id][label] = None if verdict is None else verdict.correct
            mark = "?" if verdict is None else ("y" if verdict.correct else "n")
            print(f"  {n:3}/{len(cases)} {case.id:6} {mark}", end="\r", flush=True)
        print(f"  {len(cases)}/{len(cases)} done{'':20}")

    # -------------------------------------------------- per-judge agreement --
    print("\n  " + "=" * 68)
    print(f"  {'judge':28} {'agreement':>10} {'f-accept':>9} {'f-reject':>9}")
    print("  " + "-" * 68)

    for label in labels:
        judged = agree = f_acc = f_rej = 0
        for case in cases:
            v = verdicts[case.id].get(label)
            if v is None:
                continue
            judged += 1
            truth = (answers[case.id] == case.expected)
            if v == truth:
                agree += 1
            elif v:
                f_acc += 1
            else:
                f_rej += 1
        rate = f"{agree / judged:.1%}" if judged else "n/a"
        print(f"  {label:28} {rate:>10} {f_acc:>9} {f_rej:>9}")

    if len(labels) < 2:
        return 0

    # ------------------------------------------------- judges vs each other --
    both_judged = [c for c in cases
                   if all(verdicts[c.id].get(l) is not None for l in labels)]
    disputed = [c for c in both_judged
                if len({verdicts[c.id][l] for l in labels}) > 1]

    print(f"\n  judges agreed with each other on "
          f"{len(both_judged) - len(disputed)}/{len(both_judged)} cases")

    # The payoff: every judge, independently, disagreeing with the human.
    unanimous_against = [
        c for c in both_judged
        if all(verdicts[c.id][l] != (answers[c.id] == c.expected) for l in labels)
    ]

    if unanimous_against:
        print(f"""
  {len(unanimous_against)} case(s) where EVERY judge disagreed with the human label.

  One judge disagreeing is usually the judge being wrong. Every judge
  disagreeing independently is evidence the LABEL is worth re-reading - and a
  wrong label is worse than a wrong judge, because every score computed against
  it inherits the error silently.
""")
        for case in unanimous_against[:10]:
            given = answers[case.id]
            print(f"    {case.id}  labelled {case.expected!r}, answer judged was {given!r}")
            print(f"            {case.text[:70]}")

    if disputed:
        print(f"\n  {len(disputed)} case(s) where the judges disagreed with EACH OTHER"
              " - genuinely ambiguous:\n")
        for case in disputed[:8]:
            marks = "  ".join(
                f"{l.split('/')[-1]}={'y' if verdicts[case.id][l] else 'n'}"
                for l in labels)
            print(f"    {case.id}  {marks}")
            print(f"            {case.text[:70]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
