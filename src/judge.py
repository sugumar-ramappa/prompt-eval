"""An LLM-as-judge scorer, and the calibration that makes it worth anything.

WHY THIS EXISTS
Every other scorer in this harness is deterministic: does it parse, does it have
the fields, does the label match. That works because classification has a correct
answer sitting in the golden set.

Summarisation, extraction and rewriting do not. There is no `assertEquals` for
"is this a good summary?", so a harness built only on deterministic scorers can
only ever evaluate tasks with a lookup key - which is most of the interesting
work excluded.

An LLM judge grades output against a rubric instead of against a key.

THE PART THAT IS ACTUALLY HARD
Writing the judge is twenty lines. A judge you have not validated is just
another unmeasured component, and swapping "I don't know if the output is good"
for "I don't know if the judge is right" is not progress.

So the judge is calibrated against cases where the truth IS known. This golden
set has 60 human-labelled tickets, so the judge can be run where the answer is
already established and scored on agreement:

    agreement            how often the judge and the human agree
    false acceptances    judge says correct, human label says wrong   <- the dangerous ones
    false rejections     judge says wrong, human label says correct

Only after that can it be pointed at a task with no ground truth and have its
verdict mean something.

WHAT THE JUDGE IS NOT TOLD
The expected label. A judge shown the answer key is not judging, it is comparing
strings - and it would score ~100% agreement while proving nothing. It sees the
ticket and the model's answer, exactly what a human grader would see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.golden import Case
from src.models import Model, Reply
from src.scorers import Score, extract_json

# The judge answers in a fixed shape. Ollama honours a JSON *schema* and ignores
# `format: "json"` entirely - a distinction that cost a day on this project
# before it was understood, so the schema is passed explicitly rather than hoped
# for.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
}

# v2. The first version scored 66.7% agreement with 19 false rejections against 1
# false acceptance, and the calibration output said why: the judge kept rejecting
# a defensible answer because it preferred a category that DOES NOT EXIST.
#
#   "Two factor authentication is not sending codes"    account -> "technical"
#   "Locked out after too many failed attempts"         account -> "security"
#   "Team member left, revoke their access"             account -> "security"
#   "Driver could not find the building"                shipping -> "navigation"
#   "What happens to my data if I stay inactive"        account -> "data retention"
#
# security, navigation and data retention are not options. The prompt listed the
# allowed categories but never said the list was EXHAUSTIVE, so the judge
# compared the answer against the best label it could imagine rather than the
# best label available. Naming a constraint is not the same as closing it.
JUDGE_PROMPT = """You are grading a support-ticket classifier.

TICKET
{ticket}

THE ONLY CATEGORIES THAT EXIST
{allowed}

This list is exhaustive. No other category is available - not "security", not
"technical support", not anything else you might think fits better. If the idea
you have in mind is not on that list, it is not an option, and the classifier
could not have chosen it.

THE CLASSIFIER ANSWERED
{answer}

Decide whether that answer is the best available fit, choosing only from the list
above.

Mark it INCORRECT only if a DIFFERENT category FROM THAT LIST is clearly better.
Do not mark it incorrect because some category outside the list would have suited
the ticket more closely - that is not a mistake the classifier was able to make.

Where two listed categories are both defensible, accept the classifier's choice.
You are checking for a wrong answer, not for the answer you would have given.

Judge only the category. Ignore formatting, ignore any confidence value, and
ignore wording - a correct category expressed awkwardly is correct.

Reply with JSON only: {{"correct": true or false, "reason": "one short sentence"}}
"""


@dataclass(frozen=True)
class Verdict:
    """One judged answer."""

    correct: bool
    reason: str


def ask_judge(judge: Model, case: Case, answer_text: str) -> Verdict | None:
    """Put one answer to the judge. None when the judge itself failed.

    None rather than False, deliberately. A judge that could not be reached has
    not decided the answer is wrong, and collapsing those two makes an outage
    look like a quality problem - the same distinction the reviewer agents in the
    sibling project keep for exactly this reason.
    """
    prompt = JUDGE_PROMPT.format(
        ticket=case.text,
        allowed=", ".join(case.allowed),
        answer=answer_text.strip() or "(the classifier returned nothing)",
    )
    try:
        reply = judge.complete(prompt)
    except Exception:
        return None

    parsed = extract_json(reply.text)
    if not isinstance(parsed, dict) or "correct" not in parsed:
        return None
    return Verdict(bool(parsed["correct"]), str(parsed.get("reason", "")))


def make_llm_judge(judge: Model) -> "callable":
    """Build a Scorer that grades with a model rather than a key.

    A factory because the Scorer signature is (Reply, Case) -> Score with no room
    for a model. Keeping that signature means the judge drops into the existing
    runner with no change to it.

    DELIBERATELY NOT IN DEFAULT_SCORERS. Adding a sixth scorer to the default
    list would change every aggregate score this harness has already recorded, so
    a re-run would not be comparable with what is on disk. It is opt-in, and the
    deterministic numbers stay exactly as they were.
    """

    def llm_judge(reply: Reply, case: Case) -> Score:
        # The judge grades the CATEGORY the classifier chose, so extract it
        # rather than handing over raw JSON - a judge asked to read JSON starts
        # grading the JSON.
        parsed = extract_json(reply.text)
        answer = (parsed or {}).get("category") if isinstance(parsed, dict) else None
        shown = str(answer) if answer else reply.text

        verdict = ask_judge(judge, case, shown)
        if verdict is None:
            # Unavailable, not failed. value stays 0.0 but the detail says why,
            # so a run degraded by an unreachable judge is legible rather than
            # looking like a collapse in quality.
            return Score("llm_judge", False, 0.0, "judge unavailable")

        return Score("llm_judge", verdict.correct,
                     1.0 if verdict.correct else 0.0, verdict.reason[:120])

    return llm_judge


# ------------------------------------------------------------- calibration --


@dataclass(frozen=True)
class Calibration:
    """How well the judge agrees with the humans who labelled the set."""

    total: int
    judged: int              # cases the judge actually returned a verdict for
    agreements: int
    false_acceptances: int   # judge said correct, the label says otherwise
    false_rejections: int    # judge said wrong, the label says correct

    @property
    def agreement(self) -> float:
        return self.agreements / self.judged if self.judged else 0.0

    @property
    def skew(self) -> str:
        """Which way the judge errs. The direction matters more than the rate.

        A generous judge passes bad answers, which is the failure that reaches
        production. A harsh judge rejects good ones, which is annoying and safe.
        Reporting only an agreement percentage hides which of those you have.
        """
        if self.false_acceptances > self.false_rejections:
            return "generous - accepts answers the humans rejected"
        if self.false_rejections > self.false_acceptances:
            return "harsh - rejects answers the humans accepted"
        return "balanced"

    def summary(self) -> str:
        lines = [
            f"  cases            {self.total}",
            f"  judged           {self.judged}"
            + ("" if self.judged == self.total
               else f"   ({self.total - self.judged} unavailable)"),
            f"  agreement        {self.agreement:.1%}  ({self.agreements}/{self.judged})",
            f"  false accept     {self.false_acceptances}   judge passed what the label failed",
            f"  false reject     {self.false_rejections}   judge failed what the label passed",
            f"  skew             {self.skew}",
        ]
        return "\n".join(lines)


def calibrate(judge: Model, cases: list[Case],
              answers: dict[str, str]) -> Calibration:
    """Measure the judge against the human labels.

    `answers` maps case id to the category a classifier produced. The judge sees
    the ticket and that category and nothing else; the human label is used only
    afterwards, to score the judge.

    That ordering is the whole method. Show the judge the expected answer and it
    scores near-perfectly while demonstrating nothing.
    """
    judged = agreements = false_accept = false_reject = 0

    for case in cases:
        answer = answers.get(case.id)
        if answer is None:
            continue

        verdict = ask_judge(judge, case, answer)
        if verdict is None:
            continue

        judged += 1
        human_says_correct = (answer == case.expected)

        if verdict.correct == human_says_correct:
            agreements += 1
        elif verdict.correct:
            false_accept += 1
        else:
            false_reject += 1

    return Calibration(len(cases), judged, agreements, false_accept, false_reject)
