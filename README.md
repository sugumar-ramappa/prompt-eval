# prompt-eval

A regression harness for prompts. Change a prompt, re-run, and find out whether
you broke something — with an exit code, so a build can run it.

The gap it fills: retrieval quality and classifier accuracy can be measured
precisely. **"Did this prompt edit break anything?" usually cannot**, so prompts
get changed on the strength of trying three examples by hand.

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

**JSON-constrained decoding is off by default.** Ollama can force valid JSON at
the decoding level. Turning it on would make the parse-rate metric report 100%
no matter how badly a prompt was written — hiding exactly the failure the
harness exists to catch. It is available as `--json-mode`, because in production
you would want it on; the point is that it conceals a failure rather than fixing
it.

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
--backend ollama            # local, no key, unlimited   (default)
--backend groq              # hosted, needs GROQ_API_KEY in .env
```

`.env` is gitignored. The local backend needs no key at all.
