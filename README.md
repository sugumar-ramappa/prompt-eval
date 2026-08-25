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
