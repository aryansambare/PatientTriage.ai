"""Overrides, site profiles, governance, and the demo cohort.

Each test here corresponds to a stated requirement of the prototype. Where the design
says the system "must" do something, there is a test that fails if it stops.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from patienttriage.compliance.governance import (
    GDPR_POLICY,
    HIPAA_POLICY,
    AccessDenied,
    Jurisdiction,
    Role,
    may,
    policy_for,
    pseudonymise,
    require,
    scoring_projection,
    validate_override_record,
)
from patienttriage.config.profile import PRESETS, DepartmentState, IntegrationTier, ModelTier
from patienttriage.data.cohort import build_cohort, cohort_summary, surge_cohort
from patienttriage.features.schema import TriageSnapshot
from patienttriage.service.override import (
    OverrideReason,
    capture,
    override_rate,
    reasons_breakdown,
)
from patienttriage.service.pipeline import TriageAssistant


def snap(**kwargs) -> TriageSnapshot:
    base = {"patient_id": "p1", "arrived_at": "2026-08-31T09:00:00", "age_years": 45.0}
    return TriageSnapshot(**(base | kwargs))


# --- the demo cohort's stated minimums ---------------------------------------------------


def test_cohort_meets_every_stated_minimum():
    coverage = cohort_summary(build_cohort())
    assert coverage["total"] >= 15
    assert coverage["ambiguous"] >= 1
    assert coverage["paediatric"] >= 1
    assert coverage["geriatric"] >= 1
    assert coverage["zero_history"] >= 1


def test_paediatric_cases_span_more_than_one_physiology():
    """'Paediatric' is not one population. A neonate and a fourteen-year-old share
    almost no thresholds."""
    from patienttriage.clinical import agebands

    bands = {
        agebands.band_name(c.snapshot.age_years)
        for c in build_cohort()
        if agebands.is_paediatric(c.snapshot.age_years)
    }
    assert len(bands) >= 2


def test_zero_history_patients_carry_none_not_false():
    """An absent record must not read as 'confirmed not on anticoagulants'."""
    for case in build_cohort():
        if case.has_prior_record:
            continue
        assert case.snapshot.on_anticoagulant is None
        assert case.snapshot.on_beta_blocker is None


def test_prior_record_availability_is_about_half():
    cases = build_cohort()
    with_record = sum(c.has_prior_record for c in cases)
    assert 0.35 <= with_record / len(cases) <= 0.65


def test_surge_cohort_triples_volume_without_inventing_patients():
    """A surge is the normal case mix arriving faster, not a different case mix."""
    base, surged = build_cohort(), surge_cohort(3)
    assert len(surged) == 3 * len(base)
    assert len({c.snapshot.patient_id for c in surged}) == len(surged)


# --- a confidence indicator on every output --------------------------------------------


def test_every_assessment_carries_a_confidence_indicator():
    """The brief: the prototype must not return a score without one. No exceptions,
    including the degraded path where there is no model at all."""
    assistant = TriageAssistant()
    for case in build_cohort():
        assessment = assistant.assess(case.snapshot)
        assert assessment.confidence_label()
        assert assessment.data_completeness()


def test_a_fired_red_flag_is_never_displayed_as_uncertain():
    """Regression: model abstention once masked a level-1 rule, and a cardiac arrest
    rendered as an unanswered question."""
    assistant = TriageAssistant()
    assessment = assistant.assess(
        snap(chief_complaint="VSA, CPR in progress", arrival_mode="ambulance")
    )
    assert assessment.source == "rule"
    assert assessment.recommended_acuity == 1
    assert "RED FLAG" in assessment.headline()
    assert "UNCERTAIN" not in assessment.headline()
    assert "certain" in assessment.confidence_label()


# --- overrides -------------------------------------------------------------------------


def test_capture_records_agreement_as_well_as_disagreement():
    """An override rate needs a denominator."""
    assistant = TriageAssistant()
    assessment = assistant.assess(snap(spo2=80))
    agreed = capture(assessment, assigned_acuity=1, clinician_id="RN-1")
    assert agreed.direction == "agreed"


def test_override_direction_is_classified():
    assistant = TriageAssistant()
    assessment = assistant.assess(snap(spo2=80))  # rule floor 1
    escalated = capture(assessment, 1, "RN-1")
    de_escalated = capture(assessment, 4, "RN-1")
    assert escalated.direction == "agreed"
    assert de_escalated.direction == "de_escalated"


def test_overriding_a_red_flag_downward_is_flagged_for_review():
    """Legitimate, and the one override worth surfacing routinely."""
    assistant = TriageAssistant()
    assessment = assistant.assess(snap(chief_complaint="cardiac arrest"))
    decision = capture(assessment, 4, "RN-1", OverrideReason.COMPLAINT_MISREAD,
                       "patient describing a relative's arrest, not their own")
    assert decision.overrode_a_red_flag


def test_override_payload_serialises_the_reason_as_a_value():
    """An audit trail recording a Python enum repr is recording an implementation
    detail rather than a clinical fact."""
    assistant = TriageAssistant()
    decision = capture(
        assistant.assess(snap()), 3, "RN-1", OverrideReason.CLINICAL_GESTALT
    )
    assert decision.to_payload()["reason"] == "clinical_gestalt"


def test_override_statistics_separate_adoption_from_safety():
    assistant = TriageAssistant()
    assessment = assistant.assess(snap(spo2=80))
    overrides = [
        capture(assessment, 1, "RN-1"),
        capture(assessment, 3, "RN-1", OverrideReason.DATA_WRONG),
        capture(assessment, 4, "RN-2", OverrideReason.DATA_WRONG),
    ]
    stats = override_rate(overrides)
    assert stats["n"] == 3
    assert stats["override_rate"] == pytest.approx(2 / 3)
    assert reasons_breakdown(overrides)["data_wrong"] == 2


# --- site profiles ---------------------------------------------------------------------


def test_escalation_budget_scales_with_staffing_not_volume():
    """The constraint is who does the re-assessing, not who arrives."""
    rural, urban = PRESETS["rural"], PRESETS["urban"]
    assert urban.escalation_budget > rural.escalation_budget
    # The rural site sees a quarter of the volume, so its budget is a larger *share*.
    assert rural.escalation_fraction > urban.escalation_fraction


def test_a_small_site_starts_on_rules_only():
    """No local validation cohort and one triage nurse means no model, honestly."""
    assert PRESETS["rural"].model_tier is ModelTier.RULES_ONLY
    assert PRESETS["rural"].integration is IntegrationTier.MANUAL


def test_a_site_without_paediatrics_says_so_rather_than_pretending():
    rural = PRESETS["rural"]
    assert not rural.supports(6.0)
    warning = rural.capability_warning(6.0)
    assert warning and "transfer" in warning
    assert rural.capability_warning(40.0) is None


def test_the_assistant_surfaces_the_capability_warning():
    assistant = TriageAssistant(profile=PRESETS["rural"])
    assessment = assistant.assess(snap(age_years=4, heart_rate=150))
    assert assessment.capability_warning is not None


def test_surge_is_proposed_by_the_system_and_declared_by_a_human():
    """Switching the objective function of a triage department is a rationing
    decision with ethical weight. The system may notice; it may not decide."""
    state = DepartmentState(profile=PRESETS["district"], arrivals_last_hour=250 / 24 * 3)
    assert state.surge_suggested()
    assert not state.in_surge()

    state.surge_declared = True
    assert state.in_surge()


# --- governance ------------------------------------------------------------------------


def test_jurisdictions_differ_where_it_matters():
    assert HIPAA_POLICY.supports_erasure is False
    assert GDPR_POLICY.supports_erasure is True
    assert GDPR_POLICY.requires_dpia is True


def test_an_unknown_jurisdiction_defaults_to_the_stricter_regime():
    """Needlessly strict is an inconvenience. Accidentally lax is a notifiable breach."""
    assert policy_for("Ruritanian Health Act").jurisdiction is Jurisdiction.GDPR


def test_erasure_cannot_reach_the_clinical_audit_trail():
    """GDPR Art. 17(3) carves out legal claims and public-interest health purposes.
    A patient may erase their training data, not the record of a decision made
    about them."""
    erasable = GDPR_POLICY.erasable_categories()
    assert "model_training" in erasable
    assert "audit" not in erasable
    assert "triage_record" not in erasable


def test_audit_outlives_the_operational_record():
    for policy in (HIPAA_POLICY, GDPR_POLICY):
        assert policy.audit_days >= policy.triage_record_days
        assert policy.audit_days > policy.model_training_days


def test_retention_expiry_is_computed_per_category():
    created = datetime(2020, 1, 1, tzinfo=UTC)
    now = datetime(2024, 1, 1, tzinfo=UTC)
    assert HIPAA_POLICY.is_expired("model_training", created, now)
    assert not HIPAA_POLICY.is_expired("audit", created, now)


def test_the_model_never_receives_an_identifier():
    """Minimum-necessary implemented rather than promised — a reviewer can read the
    projection and confirm it themselves."""
    projection = scoring_projection(snap(patient_id="MRN-99887"))
    assert "patient_id" not in projection
    assert "MRN-99887" not in str(projection)
    assert "heart_rate" in projection


def test_pseudonymisation_is_keyed_not_a_bare_hash():
    """A plain hash of a medical record number is reversible by enumeration."""
    original = snap(patient_id="MRN-00042")
    one = pseudonymise(original, b"key-a")
    two = pseudonymise(original, b"key-b")
    assert one.patient_id != original.patient_id
    assert one.patient_id != two.patient_id
    assert pseudonymise(original, b"key-a").patient_id == one.patient_id


def test_data_scientists_cannot_see_identified_patients():
    """The surest way to prevent a re-identification incident is to not grant the
    access in the first place."""
    assert not may(Role.DATA_SCIENTIST, "view_identified")
    assert may(Role.DATA_SCIENTIST, "view_pseudonymised")
    with pytest.raises(AccessDenied):
        require(Role.DATA_SCIENTIST, "view_identified")


def test_only_a_charge_nurse_may_declare_surge():
    assert may(Role.CHARGE_NURSE, "declare_surge")
    assert not may(Role.TRIAGE_NURSE, "declare_surge")


def test_an_incomplete_override_record_is_reported_not_blocked():
    """A validation failure must never stand between a nurse and a patient."""
    problems = validate_override_record(
        {"patient_id": "p1", "assigned_acuity": 2, "recommended_acuity": 3,
         "reason": "clinical_gestalt", "recorded_at": "2026-08-31T09:00:00",
         "clinician_id": ""},
        GDPR_POLICY,
    )
    assert any("Art. 22" in p for p in problems)


def test_a_complete_override_record_validates_clean():
    assert validate_override_record(
        {"patient_id": "p1", "clinician_id": "RN-1", "assigned_acuity": 2,
         "recommended_acuity": 3, "reason": "clinical_gestalt",
         "recorded_at": "2026-08-31T09:00:00"},
        HIPAA_POLICY,
    ) == []
