# Prompts

Two different jobs, so two families. Both are prompts and both got revised, which
is why both have version numbers — and why the numbering is easy to confuse.

## `classify-*` — the prompt under test

Tells the model how to sort a ticket. This is what the harness *evaluates*: four
wordings, same model, same 60 cases, so any difference is caused by the prompt.

| | measured on `llama3.1:8b` |
|---|---|
| `classify-v1.txt` | 85.0% |
| `classify-v2.txt` | 86.7% |
| `classify-v3.txt` | 86.7% |
| `classify-v4-reasoning.txt` | 0% plain, **80%** with a schema |

**v4 is the interesting one.** It scored 0% because nothing it produced was valid
JSON — it reasoned in prose and the parser got nothing. The prompt was never
wrong; the output format was unenforceable. With constrained decoding the same
prompt scores 80%. That is why `parses_as_json` is scored separately from
`category_correct`: one combined number would have read as "terrible prompt".

**And the result worth remembering:** prompt tuning bought 1.7 points across four
versions. Running `classify-v2` on `gpt-oss-120b` instead of `llama3.1:8b` bought
**ten**.

## `judge-*` — the grader

Tells the model how to *mark* the classifier's answer. A separate experiment:
classification has a right answer and needs no judge, so this exists to find out
whether an LLM judge could be trusted on tasks that have no answer key.

Measured 30 Aug on `llama3.1:8b`, both against the same 60 cases:

| | agreement | false accepts | false rejects | skew |
|---|---|---|---|---|
| `judge-v1.txt` | 66.7% | **0** | 20 | harsh |
| `judge-v2.txt` | **76.7%** | **11** | 3 | generous |

**v1 never passed a single wrong answer.** Sixty cases, thirty of them
deliberately wrong, and it caught every one. It was annoying and it was safe.

The earlier hand-recorded figures were 1 and 19; the reconstruction gives 0 and
20 - twenty errors either way, one case falling on the other side. Close enough
to call the reconstruction faithful, and the difference is stated rather than
smoothed over.

**The higher-agreement judge is the worse judge.** v2 agrees with humans more
often and waves through eleven wrong answers against v1's *none*. A harsh judge
annoys you; a generous one lies to you. Agreement rate alone is the wrong metric
for choosing a judge — read the two error directions separately.

**Three cases defeat both versions** — `t108`, `t198`, `t253`. Everything else v1
rejected, the exhaustiveness clause fixed. So one line moved seventeen harmless
errors into eleven dangerous ones, and raised the headline number by ten points
while doing it.

### What changed between them, and why

v1 kept rejecting **correct** answers because it preferred categories that do not
exist:

```
"Two factor authentication is not sending codes"   account → wanted "technical"
"Locked out after too many failed attempts"        account → wanted "security"
"Team member left, revoke their access"            account → wanted "security"
"Driver could not find the building"              shipping → wanted "navigation"
"What happens to my data if I stay inactive"       account → wanted "data retention"
```

`security`, `navigation` and `data retention` are not options. v1 listed the four
allowed categories and never said the list was **closed**, so the judge graded
against the best label it could imagine rather than the best one available.

> **Naming a constraint is not the same as closing it.**

v2 adds exactly that closing — *"This list is exhaustive"* — plus the instruction
not to penalise the classifier for a category it could not have chosen. Rejections
fell from 19 to 3, and acceptances of wrong answers rose from 1 to 11.

## `judge-v1.txt` is RECONSTRUCTED, and that matters

**It is not the original text.** The original was rewritten before the project's
first commit, so git never recorded it — the only commit touching `JUDGE_PROMPT`
already contains v2.

This file is v2 with the exhaustiveness correction removed, rebuilt from the
failure described in the comment at `src/judge.py`. It reproduces the *defect*;
it is not evidence of what was literally run.

**It reproduced.** Run on 30 Aug it scored 66.7% — the documented figure exactly,
with the same harsh skew. So the reconstruction is faithful enough to use, and
the honest framing is *"I rebuilt the prompt from the failure it caused, and it
reproduced the score."* Say the rebuilt part.

The `classify-*` family has no such problem: all four are the files that were
actually run, and re-running them is a diff away.
