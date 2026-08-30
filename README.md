# prompt-eval

A regression harness for prompts. Change a prompt, re-run, and find out whether
you broke something — with an exit code, so a build can run it.

The gap it fills: retrieval quality and classifier accuracy can be measured
precisely. **"Did this prompt edit break anything?" usually cannot**, so prompts
get changed on the strength of trying three examples by hand.

---

## Read this first: there are TWO things numbered v1, v2

The single most confusing thing about this project. Two different prompts do two
different jobs, and both were revised, so both carry version numbers.

```
STEP 1   classify prompt + a ticket        →  llama3.1:8b  →  "billing"
STEP 2   judge prompt + ticket + "billing" →  llama3.1:8b  →  "correct? yes/no"
```

**Same model, two jobs, two separate instruction sheets.**

| | `classify-v1 … v4` | `judge-v1 … v2` |
|---|---|---|
| job | sort a ticket into a category | mark whether that sorting was right |
| where | [`prompts/classify-*.txt`](prompts/) | [`prompts/judge-*.txt`](prompts/) |
| how it is scored | string match against an answer key | 30 correct + 30 deliberately wrong cases |
| how many | 4 | 2 |
| run it with | `scripts/evaluate --prompt classify-v2` | `scripts/calibrate_judge --judge-prompt judge-v2` |
| results land in | `data/runs/v2.json` | `data/runs/judge-calibration-judge-v2.json` |

**Why a judge exists at all when classification has an answer key:** it does not
need one. The judge is a separate experiment, asking whether an LLM grader could
be trusted on tasks that have *no* answer key — summarisation, extraction,
rewriting. The answer was 77% and generous, so it is deliberately excluded from
`DEFAULT_SCORERS`.

Full explanation of both families, with what changed between each version and
why: [`prompts/README.md`](prompts/README.md).

---

## The result that justifies the project

A prompt was edited to add step-by-step reasoning. This is the most common
well-intentioned prompt change there is.

| | v1 | v4 *(added "explain your reasoning")* |
|---|---:|---:|
| parses as JSON | **100.0%** | **0.0%** |
| has required fields | 100.0% | 0.0% |
| valid category | 100.0% | 0.0% |
| **correct category** | **85.0%** | **0.0%** |

```
t6  unparseable: "To classify the customer support ticket, let's consider each..."
```

All sixty replies became prose. The model reasoned well and returned nothing any
downstream system could consume.

**Checking three examples by hand would have shown three thoughtful,
better-looking answers.** The harness exits 1 and blocks the build.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama serve &                       # local model, no API key
python -m scripts.build_golden_set   # draw the evaluation set, once

python -m scripts.evaluate --prompt classify-v1 --label v1
python -m scripts.evaluate --prompt classify-v2 --label v2
python -m scripts.compare  --baseline v1 --candidate v2
```

`compare` exits **0** when the candidate is acceptable and **1** when it is a
regression, so it drops straight into CI.

---

## How you test output that is never identical twice

`assertEquals` does not work on generated text. So the harness stops asserting
on the text and asserts on **properties of the text**, in escalating layers:

| Layer | Question | Why it is separate |
|---|---|---|
| 1 · structural | Is it parseable JSON? | An unparseable reply scores the same as a wrong one in an accuracy figure |
| 2 · schema | Does it have the fields we asked for? | Valid JSON missing `category` is still useless |
| 3 · vocabulary | Is the label one we offered? | **The failure that most resembles success** |
| 4 · correctness | Is it the right label? | The number everyone quotes, and the last one to look at |

Layer 3 deserves its own line. A model answering `"refund"` instead of
`"billing"` gives a well-formed, confident, plausible answer that **no
downstream system can route** — and in an accuracy score it is indistinguishable
from an ordinary mistake.

Keeping the layers apart is what lets the report say *"accuracy barely moved but
the parse rate fell from 100% to 88%"*, which is the sentence the harness exists
to produce.

---

## What makes it a regression harness, not a report

A report gives you two numbers and leaves you to judge. This makes the
judgement and fails the build. Three rules, in the order they fire:

**A structural failure outranks an accuracy change.** Parse rate falling is a
regression regardless of accuracy. Zero tolerance.

**A per-class collapse outranks the mean.** A change can lift the average while
destroying one category.

**Churn is reported alongside net movement.** Net accuracy can be unchanged
while thirty cases flip in each direction — that is not a stable prompt, it is a
different prompt that happens to average the same.

```
tolerances:  structural 0%   ·   primary 3%   ·   per-class 15%
```

The 3% is deliberate, not laziness. On sixty cases one flipped answer is 1.7
points, and **a gate that fires on a single case is a gate people learn to
ignore.**

The harness also refuses to compare runs scored on **different evaluation sets**
or produced by **different models** — otherwise it reports on the provider while
claiming to report on the prompt.

---

## Measured findings

All figures: local `llama3.1:8b` via Ollama, temperature 0, 60 hand-labelled
cases, guessing scores 25.0%.

### An improvement that quietly broke one category

| | v1 | v2 | change |
|---|---:|---:|---|
| **overall** | 85.0% | **86.7%** | **+1.7%** |
| account | 73.3% | 80.0% | +6.7% |
| billing | 66.7% | 73.3% | +6.7% |
| **shipping** | **100.0%** | **93.3%** | **−6.7%** |
| technical | 100.0% | 100.0% | — |

v2 adds a persona and a description of each category. The aggregate improves and
it reads like a clean win. **One category regressed**, and only a per-class
breakdown shows it.

It passes the gate — 6.7 points on 15 cases is a single ticket, inside the 15%
per-class tolerance — but it is *reported*, which is the difference between
a tolerance and a blind spot.

### The edit that destroyed everything

v4 adds step-by-step reasoning: parse rate **100% → 0%**, 51 of 60 cases newly
failing, 85% churn. Gate fires, exit 1, eight distinct reasons listed.

### And what actually fixes it — which is not a better prompt

The same v4 prompt, re-run with the response **schema passed to the decoder**
rather than described in the prompt:

| v4 — "explain your reasoning step by step" | prompt only | schema-constrained |
|---|---:|---:|
| parses as JSON | **0.0%** | **100.0%** |
| has required fields | 0.0% | 100.0% |
| valid category | 0.0% | 100.0% |
| correct category | 0.0% | **80.0%** |

Total structural repair. Which reframes the whole finding:

> **The regression was never really about the prompt.** It was about relying on
> a prompt to enforce a contract. Asking for a shape is a request; handing the
> decoder a schema is a guarantee — the tokens that would break the structure
> are never available to sample.

The `enum` on `category` carries the most weight: it makes an invented label —
the failure that most resembles success — *structurally impossible* rather than
merely discouraged.

### But chain-of-thought still lost, and that is the second finding

With the format problem removed, v4 is **still the worst prompt**:

| prompt | correct |
|---|---:|
| v2 personas + descriptions | **86.7%** |
| v3 concise | 86.7% |
| v1 baseline | 85.0% |
| **v4 reasoning, schema-constrained** | **80.0%** |

`account` and `billing` both fell to 60%, from 73.3% and 66.7%.

So "add step-by-step reasoning" hurt this task **twice** — it destroyed the
output format, and once that was fixed it was still five points behind doing
nothing. Constrained to emit only the schema, the model has nowhere to put the
reasoning it was asked for; the instruction becomes context that crowds out the
ticket without buying anything.

Chain-of-thought is widely assumed to help. On short classification it did not.
That is exactly the kind of assumption a harness exists to test.

### Local vs hosted, same prompt, same 60 cases

The harness is model-agnostic, so the same evaluation runs against a model on
this laptop and a model fifteen times its size in someone else's data centre.

| | local `llama3.1:8b` | hosted `gpt-oss-120b` |
|---|---:|---:|
| **accuracy** | **86.7%** | **96.7%** |
| account | 80.0% | 93.3% |
| billing | 73.3% | 93.3% |
| shipping | 93.3% | 100.0% |
| technical | 100.0% | 100.0% |
| **latency per call** | **689 ms** | **1,786 ms** |
| total run | 42 s | 107 s |
| API key | none | required |
| rate limit | none | free-tier cap |
| data leaves the machine | no | yes |

**Ten points of accuracy is what keeping the model local costs.** Now it is a
number rather than an argument.

**And the local model is 2.6× faster**, which was not the expected result. Groq
runs on custom silicon and is genuinely fast per token — but for a 60-token
classification the network round trip dominates, so the model on the laptop wins
on latency while losing on accuracy.

That inversion is the useful part. "Local is slower but private" is the assumed
trade-off, and on short structured requests it is simply wrong.

### Local vs hosted: the complete matrix

Same 60 cases, same prompts, temperature 0. Local is `llama3.1:8b` on an M3 Pro;
hosted is `openai/gpt-oss-120b` on Groq — fifteen times the parameters.

| prompt | local 8B | hosted 120B |
|---|---:|---:|
| v1 baseline | 85.0% | **88.3%** |
| v2 personas + descriptions | 86.7% | **96.7%** |
| v3 concise | 86.7% | *not run — identical to v2 locally* |
| **v4 reasoning** | **0.0%** | **0.0%** |
| v4 + constrained decoding | 80.0% | **impossible — see below** |
| v2 + constrained decoding | — | **98.3%** |

**The hosted model is better, consistently.** +3.3 on v1, +10.0 on v2, and 98.3%
once decoding is constrained — the best number this project has produced.

**And scale buys nothing at all against a bad prompt.** v4 scores **0.0% on
both**. A 120B model on custom inference silicon fails byte-for-byte the same way
an 8B model on a laptop does:

```
local  8B    "To classify the customer support ticket, let's consider each..."
hosted 120B  "**Step-by-step reasoning** 1. Identify the main issue –"
```

That is the strongest version of the project's central claim. The failure is not
a small-model weakness to be bought out of with a bigger model. **It is what
happens when a prompt is asked to enforce a contract**, and it is
provider-independent and scale-independent.

#### What this comparison does and does not isolate

Held constant: the 60 cases, the prompt files byte-for-byte, temperature 0, and
the scoring code.

**Not** held constant - and this is a real limitation, not a footnote:

| | local | hosted |
|---|---|---|
| model family | Llama 3.1 (Meta) | gpt-oss (OpenAI weights) |
| parameters | 8B | 120B |
| precision | 4-bit quantized | full |

**Two variables move at once.** When hosted wins by 10 points, that could be
scale, model family, or quantization, and this experiment cannot separate them.

The original plan was `llama-3.3-70b-versatile` - same family, different size,
which would have isolated scale cleanly. The provider retired that model
mid-project, and `gpt-oss-120b` was the closest available contrast.

So the defensible claims are narrower than the table suggests:

- **Supported:** "the hosted setup scored 10 points higher on identical inputs,
  was 2.6x slower per call, and offered a weaker output guarantee"

  *(2.6x is the mean and the wall-clock ratio, which agree - 107.4 s against
  41.6 s over 60 cases. The median ratio is 3.1x. Both are in `data/runs/`;
  the mean is quoted because total wall clock is what a build actually waits
  for, and it is the figure the two measures agree on.)*
- **Not supported:** "larger models are better at this task" - size was never
  isolated
- **Supported, and strengthened by the confound:** "a bad prompt scores 0% on
  both". Two different model families, two sizes, two precisions, and an
  identical failure. Varying more things and still seeing the same result makes
  that finding stronger, not weaker.

### The local model gives a *stronger* guarantee than the hosted one

This was the surprise, and it runs against the assumption that hosted is simply
the more capable option.

| | local Ollama | hosted Groq |
|---|---|---|
| `json_schema` with an `enum` | **supported** — enforced by the sampler | **rejected**, `BadRequestError` |
| `json_object` | n/a | supported, but **only if the prompt contains the word "json"** |
| Shape guaranteed | yes | no — valid JSON of any shape |
| Invented category possible | **no** | **yes** |

Two consequences, both measured:

**The rescue does not exist on the hosted model.** Locally, the broken v4 prompt
was repaired by passing the schema to the decoder: 0% → 100% parse. On Groq that
is not available. `json_schema` is refused outright, and the weaker `json_object`
mode **requires the prompt to mention JSON** — which the v4 prompt does not, by
design. So the request fails:

```
400 - 'messages' must contain the word 'json' in some form,
      to use 'response_format' of type 'json_object'
```

The one prompt that most needs constraining is the one that cannot be
constrained, because the mechanism depends on the prompt already caring about
JSON. Locally the decoder does not ask the prompt's permission.

**And where it does work, it is the weaker guarantee.** v2 constrained on Groq
reaches 98.3% with a 100% parse rate — but the `enum` is not enforced, so an
invented category remains structurally possible. Locally it is impossible.

> This is the concrete answer to "why run a model locally when the hosted one is
> better?" It is not only privacy and cost. On this task the local runtime offers
> a **contract the hosted API will not**: the output shape is guaranteed by the
> sampler rather than requested in the prompt, whatever the prompt happens to
> say.

### A hosted model name is the shortest-lived constant in the codebase

This defaulted to `llama-3.3-70b-versatile` and returned **404 mid-run** — the
provider had retired it. Not an auth failure, not a typo; the model was simply
gone.

Every baseline recorded against a hosted model is reproducible only until the
provider decides otherwise. The local GGUF will still be here in two years.
`python -m scripts.list_models` now prints what the account can actually reach,
because asking the provider takes a second and guessing costs an afternoon.

### Two cache bugs, both mine, both silent

Worth recording because the second only appeared because the first was fixed.

**The flag was not in the key.** `json_mode` changes the answer, and the cache
key was `(prompt, model_name)` where the name omitted it. A constrained run was
served an unconstrained run's replies and reported that constrained decoding
made *no difference at all* — a plausible, completely wrong result, with no
error anywhere.

**Then the behaviour was not in the key.** With `[json]` added to the name, the
implementation of json_mode then changed — from the string `"json"` to a schema
— and every entry cached by the broken version stayed valid-looking under the
same key. The second run reported 0% for the same reason as the first and a
completely different cause.

The key now hashes the schema itself, so changing the contract invalidates
exactly the entries it should. Two tests guard it.

> This is the failure this README already described from a sibling Java project
> — a cache key built from something that did not capture everything affecting
> the answer. Made here, twice, in the file that warns about it. Which is
> roughly the point: **the only reason it was caught is that a result looked too
> tidy and got checked.**

### A library flag that silently did nothing

`format: "json"` was Ollama's way of requesting JSON. In 0.32 it is a **silent
no-op** — accepted, ignored, no warning. Measured: byte-identical 1,904
characters of prose with and without it.

```
format: "json"     →  1904 chars of prose        (ignored)
format: {schema}   →  {"category":"shipping",…}  (47 chars, correct)
```

A flag that appears to work, changes nothing, and reports success is worse than
one that fails.

### The edit that did nothing, which is also worth knowing

v3 adds *"Be concise."* — a plausible candidate for breaking structured output.
It did not: **100% parse rate, 86.7% accuracy, identical to v2.** A negative
result, reported because "we tried it and it made no difference" is information.

### Three approaches to the same task

The sibling `ticket-classifier` project solves this problem two other ways:

| Approach | Labels used | Accuracy |
|---|---:|---|
| Zero-shot classifier (bart-large-mnli) | 0 | 75.6% |
| Prompted LLM (llama3.1:8b, v2) | 0 | **86.7%** |
| Trained TF-IDF + logistic regression | 311 | 86.9% |

**A prompted 8B model matches a trained classifier while using no labels at
all.** Read with two caveats: this is 60 cases against the classifier's 389, and
the classifier answers in 2ms on a CPU while the model takes ~700ms on a GPU.
Not the same trade.

### An LLM judge, and the calibration that says not to trust it

Every scorer above is deterministic — parses, has the fields, matches the label.
That works because classification has a correct answer. **Summarisation,
extraction and rewriting do not**, so a harness built only on deterministic
scorers can evaluate only tasks with a lookup key.

`src/judge.py` adds a scorer that grades against a rubric instead. Writing it was
twenty lines. **Measuring whether it can be trusted was the work**, and it is
the only reason the scorer is worth having: a judge you have not validated
replaces *"I don't know if the output is good"* with *"I don't know if the judge
is right"*.

The golden set has 60 human-labelled tickets, so the judge can be run where the
truth is already known. It is never shown the label — only the ticket and a
category, the same as a human grader.

**Half the answers fed to it are deliberately wrong.** A real classifier is right
about 87% of the time, so judging its output yields ~8 negatives from 60 cases —
far too few to measure how often the judge *waves a wrong answer through*, which
is the failure that matters. So the calibration set is built 30 correct / 30
incorrect, seeded, giving both error directions equal weight.

Two prompt versions, and they fail in opposite directions:

```
                agreement   false accept   false reject   skew
judge-v1           66.7%          0             20        harsh
judge-v2           76.7%         11              3        GENEROUS
```

**v1 never passed a single wrong answer.** Sixty cases, thirty deliberately
wrong, and it caught every one - annoying, and completely safe. The judge with
the *worse* headline number had zero dangerous errors.

Re-measured 30 Aug with both prompts as files, recorded in
`data/runs/judge-calibration-judge-v1.json` and `-judge-v2.json`. The earlier
hand-recorded split was 1/19 against this run's 0/20 - twenty errors either way,
one case landing differently. `prompts/judge-v1.txt` is a **reconstruction**, so
that is as close as it can get; see `prompts/README.md`.

**v1 was rejecting defensible answers in favour of categories that do not
exist:**

```
"Two factor authentication is not sending codes"   account -> judge wanted "technical"
"Locked out after too many failed attempts"        account -> judge wanted "security"
"Driver could not find the building"               shipping -> judge wanted "navigation"
```

`security` and `navigation` are not options. The prompt listed the allowed
categories but never said the list was **exhaustive**, so the judge compared the
answer against the best label it could imagine rather than the best one
available. Naming a constraint is not closing it.

**v2 closed that and over-corrected.** It now accepts almost anything relevant —
including one answer whose own stated reason contradicts the verdict:

```
ticket:  "The discount code was accepted but not applied to the total"
truth:   billing
verdict: ACCEPTED 'account'
reason:  "The issue is related to the discount code, which is a billing issue."
```

**The conclusion is that `llama3.1:8b` is not a reliable judge for this task** —
and the harness established that rather than assuming it. That is the result. A
judged score from this model is worth roughly 70%, which is not enough to gate a
build on.

**The methodological point is worth more than the number.** v2 looks ten points
better and is arguably worse: a generous judge passes bad output, and that is the
failure that reaches production, while a harsh one only annoys you. **Agreement
rate alone is the wrong metric for choosing a judge** — the same lesson as recall
without false positives, in a different disguise.

**Also free:** some disagreements are not judge errors at all. *"Two factor
authentication is not sending codes"* is genuinely arguable between `account` and
`technical`. Calibrating a judge surfaces ambiguity in your own labels.

**Not in `DEFAULT_SCORERS`, deliberately.** Adding a sixth scorer would change
every aggregate this harness has already recorded, so a re-run would not be
comparable with what is on disk. The judge is opt-in; the deterministic numbers
are untouched.

### A second judge, and what two judges find that one cannot

`scripts/compare_judges.py` runs the same calibration across several judges. Both
local, both free:

```
judge                  agreement   false accept   false reject
llama3.1:8b                76.7%        11              3
qwen2.5:14b                85.0%         7              2
```

**Size mattered.** Doubling the parameters bought 8.3 points, which answers the
question the single-judge run left open: 8B was too small, rather than the task
being unjudgeable. Both still skew generous, which is the direction that lets bad
output through.

**But the more valuable output is the disagreements.**

Six cases where **both judges independently** disagreed with the human label —
and all six sit on the same boundary:

```
t41   "Subscription auto-renewed but I wanted to cancel"       labelled billing
t71   "Subscription renewed automatically without warning"     labelled billing
t73   "Discount code accepted but not applied to the total"    labelled billing
t206  "Happy to move to the yearly plan"                       labelled billing
t239  "Promotional rate ended without notice"                  labelled billing
t247  "Is there a setup fee for new accounts"                  labelled billing
```

Every one is a subscription or pricing question, and both models accepted
`account` as defensible.

**Re-run 30 Aug and reproduced exactly** — 76.7%/11/3 and 85.0%/7/2, the same six
unanimous cases. Now recorded in `data/runs/judge-panel.json` rather than living
in this README, with the eleven cases where the two judges split from each other.

**The six are 10% of the answer key**, and that is the part with consequences.
Every score in this project is computed against those labels, so a wrong label is
not a judge problem — it is silently wrong arithmetic in every number the harness
has ever produced. It also very likely runs through `ticket-classifier`'s 389
hand-assigned labels, which share the same billing/account boundary.


That is not six wrong labels. It is evidence that **the `account` / `billing`
boundary is under-specified in the category scheme** - "my subscription
auto-renewed" is a billing event about an account setting, and the taxonomy never
says which wins. Two independent models converging on the same six cases is
evidence about the DATA, which no deterministic scorer can produce:
`category_correct` marks those six wrong and moves on.

Eleven further cases where the judges disagreed with **each other** are the
genuinely hard ones - a damaged package that is both shipping and goods
condition, a concurrent-edit bug that presents as an account-collaboration
problem.

**The technique, stated plainly:** a single judge reports ambiguity as an error.
A panel reports it as ambiguity. Where judges agree, trust the verdict; where
they split, you have found a case your labelling guide does not cover.

**Still not enough to gate a build.** 85% agreement means roughly one verdict in
seven is wrong, and the errors lean permissive. This measures a judge well enough
to know it should not be a gate - which was the point of measuring it.

**Where a judge SHOULD go next, and it is not here.** Classification already has
ground truth, so a judge is redundant: `category_correct` answers directly. A
judge earns its keep only where no key exists. The sibling `hybrid-rag-service`
names exactly such a gap in its own README - *nothing measures what the system
does when retrieval fails*, and 7 of its 267 queries have no correct chunk in the
top 5. Whether the answer is then "I don't know" or a confident fabrication is
unmeasured, has no deterministic scorer, and is a **containment check** ("is every
claim supported by this passage?") rather than a taxonomy judgement - which is a
far easier question for a small model than the one asked here.

---

## Why the cache is not an optimisation

The discipline this supports is *change one thing, measure again* — easy to
write in a plan, impossible to follow if every measurement costs twenty minutes.
Results are cached per **(prompt, model, case)**, so re-running anything
unchanged is free and only new or edited prompts trigger inference.

The cache key is the part worth getting right, and the failure mode is
instructive. A sibling project built one by iterating a `HashMap`, whose order
varies between JVM runs, so identical inputs produced different keys: **the cache
never hit once, every answer was still correct, and the only symptom was the
bill.** Hence the key is assembled from explicitly ordered parts, and there is a
test for it.

---

## Design decisions worth defending

**Ground truth was written by a person, not a model.** The cases are
hand-labelled support tickets from a sibling project, labelled before this
existed. Had the expected answers come from an LLM, the harness would measure
agreement with that LLM rather than correctness, and any prompt drifting toward
the generator's style would score better for no real reason.

**The evaluation set is drawn once and committed.** Re-sampling between runs
makes two measurements incomparable — a score moves and you cannot tell whether
the prompt changed or the questions did.

**Schema-constrained decoding is off by default.** `--json-mode` passes the
response schema to the decoder, which forces valid output. Turning it on by
default would make the parse-rate metric report 100% however badly a prompt was
written — hiding exactly the failure the harness exists to catch.

In production you would want it **on**: it is the difference between a request
and a guarantee, and it repaired a prompt with a 0% parse rate completely. The
point is that it conceals a broken prompt rather than fixing one, so the harness
has to be able to see the prompt naked.

**Temperature 0.** Otherwise a "regression" could just be resampling, and the
cache would store one draw from a distribution rather than an answer.

**The model is behind an interface.** A prompt comparison is only meaningful if
everything except the prompt is held still — the provider included. It also
means the 24 tests need no key, no network and no GPU: they run in 0.06s against
a deterministic stub.

---

## Layout

```
src/
  models.py     Ollama · Groq · a deterministic stub for tests
  golden.py     the evaluation set and its ground truth
  scorers.py    the four layers of grading
  runner.py     run one prompt over every case, with caching
  compare.py    the regression rules and the verdict
prompts/        one file per version — v1 baseline, v2 personas, v3 concise, v4 reasoning
scripts/        build_golden_set · evaluate · compare
tests/          24 tests, no model required
```

## Tests

```bash
pytest -q          # 24 passed in 0.06s
```

Everything runs against `StubModel`. A harness that can only be tested by
running it against a live model is a harness nobody tests.

## Backends

```bash
python -m scripts.list_models          # what this machine and account can reach

--backend ollama                       # local, no key, unlimited   (default)
--backend groq                         # hosted, needs GROQ_API_KEY in .env
--backend groq --model qwen/qwen3.6-27b
```

`.env` is gitignored. The local backend needs no key at all.

Model availability changes without notice — `list_models` asks the provider
rather than trusting a constant in the source.
