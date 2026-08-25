"""Run one prompt version over the whole evaluation set and record the result.

WHY THE CACHE IS NOT AN OPTIMISATION
The discipline this harness exists to support is "change one thing, measure
again". That is easy to write in a plan and impossible to follow if every
measurement costs twenty minutes of local inference or a slice of a rate limit.
So results are cached per (prompt, model, case) and a re-run of anything
unchanged is free.

The cache key is the part worth getting right, and the failure mode is
instructive: a sibling project built one by iterating a HashMap, whose order
varies between JVM runs, so identical inputs produced different keys. The cache
never hit once, every answer was still correct, and the only symptom was the
bill. Hence `_fingerprint` below sorts, and hence there is a test for it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.golden import Case
from src.models import Model, Reply
from src.scorers import DEFAULT_SCORERS, PRIMARY, Score, Scorer

RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


@dataclass
class CaseResult:
    """One case, its answer, and every property we graded it on."""

    case_id: str
    expected: str
    text: str
    scores: dict[str, bool]
    detail: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False

    @property
    def passed_primary(self) -> bool:
        return self.scores.get(PRIMARY, False)


@dataclass
class Run:
    """Everything about one evaluation of one prompt version."""

    label: str
    prompt_name: str
    prompt_hash: str
    model: str
    results: list[CaseResult]
    # Recorded so a run can be interpreted months later without guessing which
    # version of the evaluation set produced it.
    golden_hash: str = ""
    seconds: float = 0.0

    def rate(self, scorer_name: str) -> float:
        if not self.results:
            return 0.0
        return sum(r.scores.get(scorer_name, False) for r in self.results) / len(self.results)

    def rates(self) -> dict[str, float]:
        names = [s.__name__ for s in DEFAULT_SCORERS]
        return {n: self.rate(n) for n in names}

    def by_class(self, scorer_name: str = PRIMARY) -> dict[str, tuple[int, float]]:
        """Per-expected-label pass rate.

        An aggregate can improve while one class collapses. Reporting only the
        mean is how that goes unnoticed.
        """
        buckets: dict[str, list[bool]] = {}
        for r in self.results:
            buckets.setdefault(r.expected, []).append(r.scores.get(scorer_name, False))
        return {k: (len(v), sum(v) / len(v)) for k, v in sorted(buckets.items())}

    def save(self, path: Path | None = None) -> Path:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = path or RUNS_DIR / f"{self.label}.json"
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @staticmethod
    def load(label: str) -> "Run":
        path = RUNS_DIR / f"{label}.json"
        if not path.exists():
            available = sorted(p.stem for p in RUNS_DIR.glob("*.json")) if RUNS_DIR.exists() else []
            raise FileNotFoundError(
                f"no run labelled {label!r}. Available: {available or 'none'}"
            )
        raw = json.loads(path.read_text())
        raw["results"] = [CaseResult(**r) for r in raw["results"]]
        return Run(**raw)


def _fingerprint(*parts: str) -> str:
    """Stable hash over ordered parts.

    Everything that changes the answer goes in, and nothing that does not.
    Ordered explicitly by the caller rather than assembled from a dict, because
    a key built by iterating an unordered structure is a key that changes
    between processes for no visible reason.
    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode()).hexdigest()


def prompt_fingerprint(template: str) -> str:
    return _fingerprint(template)[:12]


def _cache_path(prompt_hash: str, model_name: str) -> Path:
    slug = model_name.replace("/", "__").replace(":", "-")
    return CACHE_DIR / f"{slug}.{prompt_hash}.json"


def run_prompt(template: str, cases: list[Case], model: Model, *,
               label: str, prompt_name: str,
               scorers: list[Scorer] | None = None,
               progress=None, use_cache: bool = True) -> Run:
    """Evaluate one prompt version against every case.

    `template` must contain `{ticket}`. That is checked rather than assumed:
    a template missing its placeholder sends the identical prompt for every
    case, which produces a plausible-looking uniform score and no error.
    """
    if "{ticket}" not in template:
        raise ValueError(
            "prompt template has no {ticket} placeholder - every case would "
            "receive an identical prompt and the run would be meaningless"
        )

    scorers = scorers or DEFAULT_SCORERS
    p_hash = prompt_fingerprint(template)
    cache_file = _cache_path(p_hash, model.name)
    cache: dict[str, str] = {}
    if use_cache and cache_file.exists():
        cache = json.loads(cache_file.read_text())

    results: list[CaseResult] = []
    dirty = False
    started = time.perf_counter()

    for i, case in enumerate(cases, 1):
        prompt = template.replace("{ticket}", case.text)
        cached_text = cache.get(case.id) if use_cache else None

        if cached_text is not None:
            reply = Reply(text=cached_text, model=model.name, latency_ms=0.0)
            was_cached = True
        else:
            reply = model.complete(prompt)
            cache[case.id] = reply.text
            dirty = True
            was_cached = False

        graded: list[Score] = [s(reply, case) for s in scorers]
        results.append(CaseResult(
            case_id=case.id,
            expected=case.expected,
            text=reply.text,
            scores={g.name: g.passed for g in graded},
            detail={g.name: g.detail for g in graded if g.detail},
            latency_ms=reply.latency_ms,
            cached=was_cached,
        ))
        if progress:
            progress(i, len(cases), was_cached)

    if dirty and use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1))

    return Run(
        label=label,
        prompt_name=prompt_name,
        prompt_hash=p_hash,
        model=model.name,
        results=results,
        golden_hash=_fingerprint(*(c.id + c.expected for c in cases))[:12],
        seconds=time.perf_counter() - started,
    )
