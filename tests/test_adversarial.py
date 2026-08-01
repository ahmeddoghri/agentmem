"""Tests for the salience-gate leak, and the fix for it."""

from __future__ import annotations

import pytest

from agentmem.adversarial import ADVERSARIAL_CORPUS, HOLDOUT_CORPUS
from agentmem.eval_v2 import (
    DEFAULT_THRESHOLD,
    build_report,
    run_real_gate,
    score_gate,
)
from agentmem.llm import ExtractiveLLM
from agentmem.llm_v2 import ExtractiveLLMV2, _is_pronoun_form
from agentmem.memory import MemoryStore

# --- the finding: "I" is not a proper noun ----------------------------------

def test_original_scorer_flags_capitalized_i_as_a_proper_noun():
    """The exact bug: 'I'm' triggers the named-entity bonus."""
    llm = ExtractiveLLM()
    score = llm.score_salience("Hmm, I'm not sure about that.")
    assert score >= DEFAULT_THRESHOLD  # crosses the write threshold


def test_fixed_scorer_does_not_flag_it():
    llm = ExtractiveLLMV2()
    score = llm.score_salience("Hmm, I'm not sure about that.")
    assert score < DEFAULT_THRESHOLD


@pytest.mark.parametrize("form", ["I", "I'm", "I've", "I'll", "I'd"])
def test_pronoun_forms_are_recognized(form):
    assert _is_pronoun_form(form)


def test_a_capitalized_place_name_still_counts():
    """The fix must not blind the scorer to real proper nouns."""
    llm = ExtractiveLLMV2()
    with_name = llm.score_salience("The meeting moved to Toronto next week.")
    without_name = llm.score_salience("The meeting moved to that place next week.")
    assert with_name > without_name


def test_a_fact_that_starts_with_i_still_scores_high():
    """'I' is only wrong when treated as a named entity; genuine I-led facts
    must still clear the gate on their other merits."""
    llm = ExtractiveLLMV2()
    score = llm.score_salience("I live at 42 Birchwood Lane in Toronto.")
    assert score >= DEFAULT_THRESHOLD


# --- gate precision on labeled sentences ------------------------------------

def test_original_gate_precision_is_poor_on_realistic_filler():
    score = score_gate(ExtractiveLLM(), ADVERSARIAL_CORPUS)
    assert score.precision < 0.6
    assert score.false_positive > 0


def test_fixed_gate_has_perfect_precision_on_the_adversarial_corpus():
    score = score_gate(ExtractiveLLMV2(), ADVERSARIAL_CORPUS)
    assert score.precision == 1.0
    assert score.false_positive == 0


def test_fixed_gate_does_not_sacrifice_recall_for_precision():
    """A gate that rejects everything would also show zero false positives."""
    score = score_gate(ExtractiveLLMV2(), ADVERSARIAL_CORPUS)
    assert score.recall == 1.0
    assert score.false_negative == 0


def test_fixed_gate_recall_matches_original_on_real_facts():
    v1 = score_gate(ExtractiveLLM(), ADVERSARIAL_CORPUS)
    v2 = score_gate(ExtractiveLLMV2(), ADVERSARIAL_CORPUS)
    assert v2.recall >= v1.recall


# --- held out, evaluated once ------------------------------------------------

def test_holdout_was_written_after_the_fix_was_frozen():
    # Sanity: the holdout corpus exists and is disjoint from the tuning corpus.
    adversarial_texts = {u.text for u in ADVERSARIAL_CORPUS}
    holdout_texts = {u.text for u in HOLDOUT_CORPUS}
    assert not (adversarial_texts & holdout_texts)


def test_holdout_gate_precision():
    v1 = score_gate(ExtractiveLLM(), HOLDOUT_CORPUS)
    v2 = score_gate(ExtractiveLLMV2(), HOLDOUT_CORPUS)
    assert v1.precision < 0.6
    assert v2.precision == 1.0
    assert v2.recall == 1.0


# --- the real end-to-end leak ------------------------------------------------

def test_original_scorer_leaks_filler_into_memory_end_to_end():
    """No hand-picked salience override: the scorer decides every write."""
    result = run_real_gate(ExtractiveLLM(), capacity=32, n_facts=50)
    assert result.leak_rate > 0.2


def test_fixed_scorer_does_not_leak_filler():
    result = run_real_gate(ExtractiveLLMV2(), capacity=32, n_facts=50)
    assert result.leak_rate == 0.0


def test_fixing_the_leak_does_not_hurt_fact_recall():
    v1 = run_real_gate(ExtractiveLLM(), capacity=32, n_facts=50)
    v2 = run_real_gate(ExtractiveLLMV2(), capacity=32, n_facts=50)
    assert v2.recall_at_k >= v1.recall_at_k - 0.05


# --- the consolidations bug --------------------------------------------------

def test_total_merges_is_tracked_on_the_store():
    mem = MemoryStore(capacity=100, write_threshold=0.0,
                      dedup_threshold=0.99, merge_threshold=0.5)
    mem.write("Ava lives in Toronto and works as a software engineer.", salience=0.9)
    mem.write("Ava works in Toronto as an engineer building web apps.", salience=0.9)
    assert mem.total_merges == 0
    mem.consolidate()
    assert mem.total_merges == 1


def test_eval_reports_real_consolidation_count_not_zero():
    from agentmem.eval import run

    # Same parameters as the module's own default benchmark, which is known
    # to produce merges (confirmed by direct instrumentation: 8-10 per run).
    result = run(capacity=32, n_facts=200, distractors=4)
    # The original hardcoded this field to 0 regardless of what happened.
    assert result.consolidations > 0


# --- the report ---------------------------------------------------------------

def test_report_is_reproducible():
    assert build_report(capacity=16, n_facts=60) == build_report(capacity=16, n_facts=60)


def test_report_shows_v2_with_zero_leak_and_full_precision():
    report = build_report(capacity=16, n_facts=60)
    assert report["end_to_end"]["v2"]["leak_rate"] == 0.0
    assert report["gate"]["adversarial"]["v2"]["precision"] == 1.0
    assert report["gate"]["holdout"]["v2"]["precision"] == 1.0
