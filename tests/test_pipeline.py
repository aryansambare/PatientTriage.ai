"""The safety invariants of the assembled system.

These are the tests that matter most in the whole suite. A wrong probability is a
quality problem; a triage screen that crashes, silently de-escalates a red flag, or
presents a guess as a confident number is a patient-safety problem.
"""

from __future__ import annotations

import numpy as np
import pytest

from patienttriage.features.schema import TriageSnapshot
from patienttriage.models.conformal import ConformalAbstainer
from patienttriage.rules.engine import RuleEngine
from patienttriage.service.audit import AuditLog
from patienttriage.service.pipeline import TriageAssistant


def snap(**kwargs) -> TriageSnapshot:
    base = {"patient_id": "p1", "arrived_at": "2026-08-31T09:00:00", "age_years": 45.0}
    return TriageSnapshot(**(base | kwargs))


class ExplodingModel:
    """A model that fails the way real ones do: at inference, in production."""

    def predict_proba(self, features):
        raise RuntimeError("model server unreachable")


class FixedModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return np.tile(self.probabilities, (len(features), 1))


# --- invariant 1: it never crashes -----------------------------------------------------


def test_assess_survives_a_dead_model():
    assistant = TriageAssistant(acuity_model=ExplodingModel())
    assessment = assistant.assess(
        snap(heart_rate=88, systolic_bp=120, respiratory_rate=16, spo2=98)
    )

    assert assessment.degraded
    assert "model server unreachable" in assessment.degraded_reason
    assert "RULES ONLY" in assessment.headline() or assessment.recommended_acuity is None


def test_a_dead_model_does_not_suppress_a_red_flag():
    """Degrading must not lose the deterministic layer - that is the whole point of
    having one that does not depend on the model."""
    assistant = TriageAssistant(acuity_model=ExplodingModel())
    assessment = assistant.assess(
        snap(chief_complaint="CPR in progress", arrival_mode="ambulance")
    )
    assert assessment.degraded
    assert assessment.rule_floor == 1
    assert assessment.recommended_acuity == 1


def test_no_model_at_all_still_produces_an_assessment():
    assessment = TriageAssistant().assess(snap(spo2=82))
    assert assessment.degraded
    assert assessment.rule_floor == 1


def test_assess_never_raises_on_an_empty_patient():
    """Nothing recorded but an age. Real, and the code path least likely to be tried."""
    assessment = TriageAssistant().assess(snap(age_years=0.1))
    assert assessment.latency_ms >= 0
    assert len(assessment.missing_vitals) == 5


# --- invariant 2: escalation only ------------------------------------------------------


def test_the_model_cannot_de_escalate_a_red_flag():
    """The model is certain this is a level 5. A red flag says level 1. Level 1 wins."""
    confident_nonurgent = FixedModel([0.0, 0.0, 0.0, 0.0, 1.0])
    assistant = TriageAssistant(acuity_model=confident_nonurgent)
    assessment = assistant.assess(snap(spo2=80, chief_complaint="feels fine"))

    assert assessment.rule_floor == 1
    assert assessment.recommended_acuity == 1


def test_the_model_may_escalate_beyond_the_rules():
    """No rule fires, but the model sees a pattern. It is allowed to pull forward."""
    confident_urgent = FixedModel([0.0, 1.0, 0.0, 0.0, 0.0])
    assistant = TriageAssistant(acuity_model=confident_urgent)
    assessment = assistant.assess(snap(heart_rate=84, systolic_bp=126, spo2=98))

    assert assessment.rule_floor == 5
    assert assessment.recommended_acuity == 2


# --- invariant 3: uncertainty is visible ----------------------------------------------


def test_abstention_shows_nothing_rather_than_a_hedged_number():
    """There must be nothing on screen for a busy nurse to accidentally accept."""
    abstainer = ConformalAbstainer(alpha=0.1)
    abstainer.quantile = 0.6  # threshold 0.4, so {2, 3} both survive
    assistant = TriageAssistant(
        acuity_model=FixedModel([0.0, 0.45, 0.45, 0.10, 0.0]), abstainer=abstainer
    )
    assessment = assistant.assess(
        snap(heart_rate=96, systolic_bp=112, respiratory_rate=18, spo2=97)
    )

    assert assessment.abstained
    assert assessment.recommended_acuity is None
    assert "UNCERTAIN" in assessment.headline()


def test_abstention_does_not_withdraw_a_red_flag():
    """Declining to refine a judgement is not the same as cancelling a hard rule."""
    abstainer = ConformalAbstainer(alpha=0.1)
    abstainer.quantile = 0.6
    assistant = TriageAssistant(
        acuity_model=FixedModel([0.0, 0.45, 0.45, 0.10, 0.0]), abstainer=abstainer
    )
    assessment = assistant.assess(snap(spo2=80))

    assert assessment.abstained
    assert assessment.recommended_acuity == 1


# --- effective_acuity: a number for anything that needs one, e.g. routing -------------


def test_effective_acuity_is_the_recommendation_when_there_is_one():
    assessment = TriageAssistant().assess(snap(spo2=80))  # rule floor 1
    assert assessment.effective_acuity == 1


def test_effective_acuity_falls_back_to_the_rule_floor_when_degraded():
    """No model loaded and nothing fired: still a real floor to route on, not nothing."""
    assessment = TriageAssistant().assess(
        snap(heart_rate=80, systolic_bp=118, respiratory_rate=16, spo2=98)
    )
    assert assessment.degraded
    assert assessment.recommended_acuity is None
    assert assessment.effective_acuity == 5


def test_effective_acuity_is_none_only_for_a_true_abstention_to_nothing():
    abstainer = ConformalAbstainer(alpha=0.1)
    abstainer.quantile = 0.6
    assistant = TriageAssistant(
        acuity_model=FixedModel([0.0, 0.45, 0.45, 0.10, 0.0]), abstainer=abstainer
    )
    assessment = assistant.assess(
        snap(heart_rate=96, systolic_bp=112, respiratory_rate=18, spo2=97)
    )
    assert assessment.recommended_acuity is None
    assert not assessment.degraded
    assert assessment.effective_acuity is None


def test_latency_over_budget_is_reported_not_hidden():
    assistant = TriageAssistant(
        acuity_model=FixedModel([0.1, 0.1, 0.6, 0.1, 0.1]), latency_budget_ms=0.0
    )
    assessment = assistant.assess(snap(heart_rate=80))
    assert assessment.degraded
    assert "latency budget" in assessment.degraded_reason


# --- the audit log ---------------------------------------------------------------------


def test_audit_chain_detects_a_silently_edited_record(tmp_path):
    """The question after an adverse event is what the system said *before* it."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("assessment", "p1", {"recommended_acuity": 4})
    log.append("assessment", "p2", {"recommended_acuity": 2})
    log.append("override", "p1", {"from": 4, "to": 2})

    assert log.verify() == (True, None)

    tampered = log.path.read_text(encoding="utf-8").replace(
        '"recommended_acuity": 4', '"recommended_acuity": 2'
    )
    log.path.write_text(tampered, encoding="utf-8")

    intact, problem = log.verify()
    assert not intact
    assert "modified" in problem


def test_overrides_are_recorded_as_ordinary_events(tmp_path):
    """A nurse disagreeing is the system working, and it is the next training label."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("assessment", "p1", {"recommended_acuity": 3})
    log.append("override", "p1", {"from": 3, "to": 2, "by": "nurse"})

    overrides = log.overrides()
    assert len(overrides) == 1
    assert overrides[0].payload["to"] == 2


def test_an_assessment_round_trips_into_the_log(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    assessment = TriageAssistant().assess(snap(spo2=80, chief_complaint="SOB"))
    record = log.append("assessment", assessment.patient_id, assessment.to_payload())

    assert record.payload["rule_floor"] == 1
    assert record.payload["rule_hits"]
    assert log.verify()[0]


def test_empty_watch_list_without_a_deterioration_model():
    assert TriageAssistant().watch_list([snap()]).empty


@pytest.mark.parametrize("spo2", [0.0, 50.0, 100.0])
def test_extreme_but_recordable_vitals_do_not_break_the_pipeline(spo2):
    assessment = TriageAssistant().assess(snap(spo2=spo2, heart_rate=0))
    assert assessment.rule_floor in (1, 2, 5)


# --- NEWS2: reported, never decided on --------------------------------------------------


def test_news2_reported_for_an_eligible_adult():
    assessment = TriageAssistant().assess(snap(heart_rate=75, systolic_bp=120))
    assert assessment.news2_score is not None
    assert "NEWS2" in assessment.news2_label()


def test_news2_not_reported_for_a_child():
    """Scale 1 is derived from adults; applying it to a child silently anyway is the
    same failure mode the age-stratification work in `clinical/agebands.py` exists to
    close."""
    assessment = TriageAssistant().assess(snap(age_years=10.0, heart_rate=110))
    assert assessment.news2_score is None
    assert "not applicable" in assessment.news2_label()


def test_news2_never_changes_the_recommended_acuity():
    """NEWS2 is a second opinion, not a vote. A high NEWS2 band must not, by itself,
    move the number a rules-only site puts on screen beyond what the rule engine
    already demanded from the same vitals."""
    s = snap(
        heart_rate=135, respiratory_rate=28, spo2=90, systolic_bp=105, temperature_c=39.5
    )
    engine_floor = RuleEngine().evaluate(s).acuity_floor
    assessment = TriageAssistant().assess(s)

    assert assessment.news2_band == "high"
    assert assessment.recommended_acuity == engine_floor
