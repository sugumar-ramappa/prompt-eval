"""Tests for the harness itself.

None of these call a model. The whole point of StubModel is that the logic which
actually contains bugs - scoring, cache keys, the regression rules - is testable
without a GPU, an API key or a network.

A harness that can only be tested by running it against a live model is a
harness nobody tests.
"""

import json

import pytest

from src.compare import CLASS_TOLERANCE, PRIMARY_TOLERANCE, churn, compare
from src.golden import Case, majority_baseline
from src.models import Reply, StubModel
from src.runner import CaseResult, Run, prompt_fingerprint, run_prompt
from src.scorers import (category_correct, category_is_valid,
                         confidence_in_range, extract_json, has_required_fields,
                         parses_as_json)

CASE = Case(id="t1", text="my card was charged twice", expected="billing")


def reply(text: str) -> Reply:
    return Reply(text=text, model="stub", latency_ms=0.0)


# ----------------------------------------------------------------- scorers

def test_parses_plain_json():
    assert parses_as_json(reply('{"category":"billing","confidence":0.9}'), CASE).passed


def test_parses_json_wrapped_in_a_markdown_fence():
    """Models add fences even when told not to.

    Tolerating it is a deliberate choice: it means the harness measures "did it
    produce JSON" rather than "did it obey the no-fence instruction". The
    stricter reading would fail answers that are structurally fine.
    """
    fenced = '```json\n{"category":"billing","confidence":0.9}\n```'
    assert parses_as_json(reply(fenced), CASE).passed
    assert extract_json(fenced)["category"] == "billing"


def test_prose_does_not_parse():
    score = parses_as_json(reply("Let's think step by step. This is billing."), CASE)
    assert not score.passed
    assert "unparseable" in score.detail


def test_a_json_array_is_not_an_object():
    """`["billing"]` is valid JSON and still unusable - we asked for a shape."""
    assert not parses_as_json(reply('["billing"]'), CASE).passed


def test_missing_fields_are_caught_separately_from_bad_json():
    score = has_required_fields(reply('{"category":"billing"}'), CASE)
    assert not score.passed
    assert "confidence" in score.detail


def test_invented_category_is_its_own_failure():
    """The failure that most resembles success.

    'refund' is well-formed, confident and plausible, and no downstream system
    can route it. In an accuracy score it is indistinguishable from an ordinary
    mistake, which is why it is scored separately.
    """
    r = reply('{"category":"refund","confidence":0.95}')
    assert parses_as_json(r, CASE).passed
    assert has_required_fields(r, CASE).passed
    score = category_is_valid(r, CASE)
    assert not score.passed
    assert "invented" in score.detail


def test_category_matching_ignores_case_and_padding():
    assert category_correct(reply('{"category":" Billing ","confidence":0.5}'), CASE).passed


def test_confidence_must_be_zero_to_one():
    assert confidence_in_range(reply('{"category":"billing","confidence":0.5}'), CASE).passed
    # 95 instead of 0.95 silently breaks any threshold rule built on it.
    assert not confidence_in_range(reply('{"category":"billing","confidence":95}'), CASE).passed
    assert not confidence_in_range(reply('{"category":"billing","confidence":"high"}'), CASE).passed


# ----------------------------------------------------------------- golden set

def test_a_case_whose_answer_is_not_allowed_is_rejected_at_construction():
    """Better to fail loudly than to ship a case that can never pass."""
    with pytest.raises(ValueError, match="never pass"):
        Case(id="x", text="...", expected="refunds")


def test_majority_baseline():
    cases = [Case(id=str(i), text="t", expected="billing") for i in range(3)]
    cases.append(Case(id="9", text="t", expected="account"))
    assert majority_baseline(cases) == 0.75


# ----------------------------------------------------------------- runner

def test_template_without_a_placeholder_is_rejected():
    """Every case would get an identical prompt, and the run would look fine.

    This produces a uniform score and no error - the worst combination.
    """
    with pytest.raises(ValueError, match="placeholder"):
        run_prompt("Classify the ticket.", [CASE], StubModel(),
                   label="x", prompt_name="x", use_cache=False)


def test_fingerprint_changes_when_the_prompt_changes():
    assert prompt_fingerprint("Classify: {ticket}") != prompt_fingerprint("Sort: {ticket}")


def test_fingerprint_is_stable_across_calls():
    """Guards the bug a sibling project shipped: a cache key built by iterating
    an unordered structure, which changed between processes so the cache never
    hit once - every answer correct, and only the bill showed it."""
    template = "Classify: {ticket}"
    assert prompt_fingerprint(template) == prompt_fingerprint(template)


def test_cache_avoids_a_second_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr("src.runner.CACHE_DIR", tmp_path)
    model = StubModel()
    template = "Classify: {ticket}"

    first = run_prompt(template, [CASE], model, label="a", prompt_name="p")
    assert model.calls == 1
    assert not first.results[0].cached

    second = run_prompt(template, [CASE], model, label="b", prompt_name="p")
    assert model.calls == 1, "cache did not prevent a second call"
    assert second.results[0].cached


def test_editing_the_prompt_invalidates_the_cache(tmp_path, monkeypatch):
    """The regression this guards: caching on the case alone.

    Edit a prompt, re-run, and the harness would serve the old prompt's answers
    and report that the change had no effect.
    """
    monkeypatch.setattr("src.runner.CACHE_DIR", tmp_path)
    model = StubModel()
    run_prompt("Classify: {ticket}", [CASE], model, label="a", prompt_name="p")
    run_prompt("Sort into a category: {ticket}", [CASE], model, label="b", prompt_name="p")
    assert model.calls == 2


def test_a_setting_that_changes_the_answer_changes_the_cache_key():
    """The bug this project shipped, in the file that warns about it.

    `json_mode` constrains decoding, so it changes the reply. It was missing
    from the model name - which is the cache key - so a run with it ON was
    served the cached replies from a run with it OFF and reported that
    constrained decoding made no difference at all.

    Plausible, completely wrong, and no error anywhere. The only symptom was a
    result that was too tidy.
    """
    from src.models import OllamaModel

    plain = OllamaModel(model="llama3.1:8b")
    constrained = OllamaModel(model="llama3.1:8b", json_mode=True)
    hot = OllamaModel(model="llama3.1:8b", temperature=0.7)

    assert plain.name != constrained.name, "json_mode must be in the cache key"
    assert plain.name != hot.name, "temperature must be in the cache key"
    assert constrained.name != hot.name


def test_two_identically_configured_models_share_a_cache_key():
    """The other half: identical configuration must NOT miss the cache."""
    from src.models import OllamaModel

    assert OllamaModel(model="llama3.1:8b").name == OllamaModel(model="llama3.1:8b").name


def test_run_round_trips_through_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("src.runner.RUNS_DIR", tmp_path)
    monkeypatch.setattr("src.runner.CACHE_DIR", tmp_path / "c")
    run = run_prompt("Classify: {ticket}", [CASE], StubModel(),
                     label="v1", prompt_name="p")
    run.save()
    loaded = Run.load("v1")
    assert loaded.rate("category_correct") == run.rate("category_correct")
    assert loaded.prompt_hash == run.prompt_hash


# ----------------------------------------------------------------- compare

def _run(label, scores_by_case, model="stub", golden="g1"):
    """Build a Run directly, so the comparison rules can be tested in isolation."""
    results = [
        CaseResult(case_id=cid, expected=exp, text="{}", scores=dict(sc))
        for cid, exp, sc in scores_by_case
    ]
    return Run(label=label, prompt_name="p", prompt_hash="h", model=model,
               results=results, golden_hash=golden)


ALL_PASS = {"parses_as_json": True, "category_correct": True}
ALL_FAIL = {"parses_as_json": True, "category_correct": False}
UNPARSEABLE = {"parses_as_json": False, "category_correct": False}


def test_identical_runs_are_not_a_regression():
    base = _run("a", [("1", "billing", ALL_PASS)] * 1)
    assert not compare(base, _run("b", [("1", "billing", ALL_PASS)])).is_regression


def test_a_structural_drop_fails_even_when_accuracy_is_unchanged():
    """The rule that matters most.

    An unparseable reply scores the same as a wrong one in an accuracy figure,
    so a prompt that breaks the output format can look like a small accuracy
    dip - or like nothing at all.
    """
    cases = [("1", "billing", ALL_PASS), ("2", "billing", ALL_FAIL)]
    broke = [("1", "billing", {"parses_as_json": False, "category_correct": True}),
             ("2", "billing", ALL_FAIL)]
    result = compare(_run("a", cases), _run("b", broke))
    assert result.is_regression
    assert any("parses_as_json" in f for f in result.failures)


def test_small_accuracy_movement_is_tolerated():
    """A gate that fires on one flipped case is a gate people learn to ignore."""
    n = 60
    base = [(str(i), "billing", ALL_PASS) for i in range(n)]
    one_worse = [(str(i), "billing", ALL_PASS if i else ALL_FAIL) for i in range(n)]
    result = compare(_run("a", base), _run("b", one_worse))
    assert abs(result.primary.change) < PRIMARY_TOLERANCE
    assert not result.is_regression


def test_a_large_accuracy_drop_fails():
    base = [(str(i), "billing", ALL_PASS) for i in range(10)]
    worse = [(str(i), "billing", ALL_FAIL if i < 5 else ALL_PASS) for i in range(10)]
    assert compare(_run("a", base), _run("b", worse)).is_regression


def test_a_single_class_collapsing_fails_even_if_the_average_rises():
    """The finding an aggregate hides.

    'shipping' is destroyed while every other class improves, and the mean goes
    UP. Reporting only the mean is how that ships.
    """
    base = ([("s%d" % i, "shipping", ALL_PASS) for i in range(10)]
            + [("b%d" % i, "billing", ALL_FAIL) for i in range(10)])
    cand = ([("s%d" % i, "shipping", ALL_FAIL) for i in range(10)]
            + [("b%d" % i, "billing", ALL_PASS) for i in range(10)])
    result = compare(_run("a", base), _run("b", cand))
    assert abs(result.primary.change) < 0.001, "aggregate deliberately unchanged"
    assert result.is_regression
    assert any("shipping" in f for f in result.failures)


def test_runs_scored_on_different_golden_sets_are_refused():
    a = _run("a", [("1", "billing", ALL_PASS)], golden="g1")
    b = _run("b", [("1", "billing", ALL_PASS)], golden="g2")
    result = compare(a, b)
    assert result.is_regression
    assert any("not comparable" in f for f in result.failures)


def test_runs_from_different_models_are_refused():
    """Otherwise the harness reports on the provider while claiming to report
    on the prompt."""
    a = _run("a", [("1", "billing", ALL_PASS)], model="ollama/llama3.1:8b")
    b = _run("b", [("1", "billing", ALL_PASS)], model="groq/llama-3.3-70b")
    assert any("different models" in f for f in compare(a, b).failures)


def test_churn_sees_what_the_net_change_hides():
    """Twenty cases flip each way, the average does not move, and the prompt is
    not 'unchanged' - it is a different prompt that happens to average the same."""
    base = [(str(i), "billing", ALL_PASS if i < 10 else ALL_FAIL) for i in range(20)]
    cand = [(str(i), "billing", ALL_FAIL if i < 10 else ALL_PASS) for i in range(20)]
    result = compare(_run("a", base), _run("b", cand))
    assert abs(result.primary.change) < 0.001
    assert churn(result) == 1.0
