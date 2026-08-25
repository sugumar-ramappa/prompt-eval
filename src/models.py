"""One interface over several model providers, plus a stub for tests.

WHY AN ABSTRACTION AND NOT JUST CALLING A CLIENT
This project exists to compare prompts, and a prompt comparison is only
meaningful if everything except the prompt is held still - the provider
included. Wiring one vendor's SDK through the harness would make "does this
prompt change help?" and "does this provider handle it better?" the same
question, and neither would be answerable.

It also means the tests need no API key, no network and no GPU. StubModel is
deterministic and instant, so the scoring, caching and comparison logic - the
parts that actually contain the bugs - are testable in milliseconds.
"""

from __future__ import annotations

import hashlib
import json as _json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Reply:
    """One model response, and what it cost to get it."""

    text: str
    model: str
    latency_ms: float
    # None when the provider does not report usage. Recorded rather than
    # estimated locally, because a token count guessed client-side is a guess
    # with a decimal point on it.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Model(Protocol):
    """What the harness needs from a provider. Deliberately small."""

    name: str

    def complete(self, prompt: str) -> Reply: ...


def _ensure_ca_bundle() -> None:
    """Trust locally installed roots as well as the public ones.

    Networks that inspect TLS re-sign HTTPS traffic with their own certificate
    authority. Browsers accept it because it sits in the system keychain; Python
    does not, because it verifies against certifi, which by design contains only
    public authorities. Requests then fail with

        CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain

    which reads as a connectivity fault and is not one - it is a trust-store
    gap. Ollama is unaffected, being a Go binary that reads the system keychain
    directly. The hosted client is not.

    setdefault, never assignment: an explicitly configured bundle is already
    correct and must not be overridden.
    """
    bundle = Path.home() / ".certs" / "combined-ca.pem"
    if bundle.exists():
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            os.environ.setdefault(var, str(bundle))


class StubModel:
    """Deterministic fake. Same prompt in, same text out, no network.

    Lets the test suite exercise caching, scoring and the regression gate with
    no provider running. The default reply is derived from a hash of the prompt,
    giving the two properties tests need: stable across runs, and different when
    the prompt differs.
    """

    def __init__(self, name: str = "stub", replies: dict[str, str] | None = None):
        self.name = name
        # Canned answers take priority; anything unlisted falls back to the
        # hash, so a test can pin the cases it cares about and ignore the rest.
        self.replies = replies or {}
        self.calls = 0

    def complete(self, prompt: str) -> Reply:
        self.calls += 1
        text = self.replies.get(prompt)
        if text is None:
            digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
            text = f'{{"category": "billing", "confidence": 0.5, "_h": "{digest}"}}'
        return Reply(text=text, model=self.name, latency_ms=0.0)


class OllamaModel:
    """A model running locally through Ollama.

    No key and no per-call cost, which is what makes it usable here: a
    regression harness re-runs the whole evaluation set every time a prompt
    changes, and a metered provider makes that expensive enough to discourage
    the very habit the harness exists to support.

    Requires `ollama serve` to already be running. Deliberately not started from
    here - a library that launches background services on import is a library
    that surprises people.
    """

    def __init__(self, model: str = "llama3.1:8b", host: str | None = None,
                 temperature: float = 0.0, json_mode: bool = False,
                 schema: dict | None = None):
        # EVERY SETTING THAT CHANGES THE ANSWER BELONGS IN THE NAME.
        #
        # The name is the cache key and the identity used to refuse
        # incomparable runs, so anything affecting the output has to appear in
        # it. This originally read f"ollama/{model}" and omitted json_mode -
        # so a run with constrained decoding turned ON was served the cached
        # replies from a run with it OFF, and reported that constrained
        # decoding made no difference whatsoever. Perfectly plausible, entirely
        # wrong, and no error anywhere.
        #
        # That is the same class of bug this project's README describes from a
        # sibling codebase, made here, in the file that warns about it.
        # Temperature is included for the same reason: at 0.7 the same prompt
        # returns different text, and caching one draw under a key that does
        # not mention temperature makes it look reproducible.
        # The key must capture the BEHAVIOUR, not merely the flag.
        #
        # This first read "[json]". That was not enough: when the implementation
        # of json_mode changed from the string "json" to a schema, every entry
        # cached by the broken version stayed valid-looking under the same key,
        # and the fixed code was served the broken code's answers. A second run
        # reported 0% for the same reason the first did, and for a completely
        # different cause.
        #
        # Hashing the schema means changing the contract - a new field, a
        # different enum, a different mechanism - invalidates exactly the
        # entries it should.
        suffix = ""
        if json_mode:
            from src.golden import RESPONSE_SCHEMA

            active = schema or RESPONSE_SCHEMA
            digest = hashlib.sha256(
                _json.dumps(active, sort_keys=True).encode()
            ).hexdigest()[:6]
            suffix = f"[schema:{digest}]"
        if temperature:
            suffix += f"[t{temperature}]"
        self.name = f"ollama/{model}{suffix}"
        self.model = model
        self.host = host
        # Temperature 0 so re-running an unchanged prompt returns the same text.
        # Without it a "regression" could just be resampling, and the cache
        # would store one draw from a distribution rather than an answer.
        self.temperature = temperature
        # Constrained decoding: the schema is enforced by the sampler, so tokens
        # that would break the structure are never available to choose.
        #
        # MUST BE A SCHEMA, NOT THE STRING "json".
        # Ollama used to accept format="json". In 0.32 that string is a silent
        # no-op - it does not error, it simply does nothing, and the model
        # returns whatever it likes. Measured here: identical 1,904 characters
        # of prose with and without it. A flag that appears to work, changes
        # nothing, and reports success is worse than one that fails.
        #
        # OFF BY DEFAULT, DELIBERATELY. Several experiments measure whether a
        # PROMPT reliably produces parseable JSON. Turning this on would report
        # a 100% parse rate however badly the prompt was written, hiding the
        # exact failure the harness exists to catch. In production you would
        # want it on - the point is that it conceals a broken prompt rather
        # than fixing one.
        self.json_mode = json_mode
        self.schema = schema
        self._client = None

    def _connect(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host) if self.host else ollama
        return self._client

    def complete(self, prompt: str) -> Reply:
        client = self._connect()
        kwargs: dict = {"options": {"temperature": self.temperature}}
        if self.json_mode:
            # A schema, never the string "json" - see __init__.
            from src.golden import RESPONSE_SCHEMA
            kwargs["format"] = self.schema or RESPONSE_SCHEMA

        t0 = time.perf_counter()
        result = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return Reply(
            text=result["message"]["content"],
            model=self.name,
            latency_ms=elapsed,
            prompt_tokens=result.get("prompt_eval_count"),
            completion_tokens=result.get("eval_count"),
        )


class GroqModel:
    """A hosted model, for the comparison against running one locally.

    Groq runs on custom silicon rather than GPUs, so it is unusually fast, which
    is part of why it is worth including. The comparison is not only "is 70B
    better than 8B" - it is what you give up by keeping data on your own
    machine, and what it costs you to send it away.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.0):
        self.name = f"groq/{model}"
        self.model = model
        self.temperature = temperature
        self._client = None

    def _connect(self):
        if self._client is None:
            _ensure_ca_bundle()
            from groq import Groq

            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Put it in prompt-eval/.env "
                    "(gitignored) or export it for this shell. The local "
                    "backend needs no key: --backend ollama"
                )
            self._client = Groq(api_key=key)
        return self._client

    def complete(self, prompt: str) -> Reply:
        client = self._connect()
        t0 = time.perf_counter()
        result = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        usage = getattr(result, "usage", None)
        return Reply(
            text=result.choices[0].message.content,
            model=self.name,
            latency_ms=elapsed,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


def load_env() -> None:
    """Read .env if present. Absent is fine - the local backend needs no key."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        load_dotenv(env)


BACKENDS = {"ollama": OllamaModel, "groq": GroqModel, "stub": StubModel}


def build(backend: str, **kwargs) -> Model:
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; have {sorted(BACKENDS)}")
    load_env()
    return BACKENDS[backend](**kwargs)
