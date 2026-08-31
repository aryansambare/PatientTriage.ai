"""The rule layer is the safety floor, so its tests are the ones that must never be
loosened to make a build pass. Each case below is a patient the system is not allowed
to leave in the waiting room."""

from __future__ import annotations

import pytest

from patienttriage.features.schema import TriageSnapshot
from patienttriage.rules.engine import RuleEngine

ENGINE = RuleEngine()


def snap(**kwargs) -> TriageSnapshot:
    base = {"patient_id": "t", "arrived_at": "2026-08-31T09:00:00", "age_years": 45.0}
    return TriageSnapshot(**(base | kwargs))


# --- escalation-only invariant --------------------------------------------------------


def test_never_de_escalates_a_human_decision():
    """A well patient with no red flags must not pull a nurse's ESI 1 down to 5."""
    result = ENGINE.evaluate(snap(heart_rate=72, systolic_bp=120, spo2=99, chief_complaint="rash"))
    assert not result.fired
    assert result.apply_to(1) == 1
    assert result.apply_to(3) == 3


def test_escalates_but_never_past_the_floor():
    result = ENGINE.evaluate(snap(spo2=80))
    assert result.acuity_floor == 1
    assert result.apply_to(4) == 1
    assert result.apply_to(1) == 1


# --- the patients who must not be missed ----------------------------------------------


def test_silent_hypotension():
    result = ENGINE.evaluate(snap(systolic_bp=82, heart_rate=104))
    assert result.acuity_floor == 1
    assert any(h.rule_id == "R03" for h in result.hits)


def test_beta_blocker_masks_tachycardia():
    """125 bpm is under the usual 130 cut-off. On a beta-blocker it should still fire —
    a patient who cannot mount a tachycardic response and is at 125 is in trouble."""
    without = ENGINE.evaluate(snap(heart_rate=125))
    with_bb = ENGINE.evaluate(snap(heart_rate=125, on_beta_blocker=True))
    assert not any(h.rule_id == "R15" for h in without.hits)
    assert any(h.rule_id == "R15" for h in with_bb.hits)


def test_well_looking_anticoagulated_head_injury():
    """Normal vitals, walked in, will not look sick for hours."""
    result = ENGINE.evaluate(
        snap(
            age_years=78,
            chief_complaint="fall at home, hit head",
            on_anticoagulant=True,
            heart_rate=76,
            systolic_bp=134,
            spo2=98,
        )
    )
    assert result.acuity_floor == 2
    assert any(h.rule_id == "R13" for h in result.hits)


def test_febrile_infant_regardless_of_appearance():
    result = ENGINE.evaluate(snap(age_years=0.15, temperature_c=38.4, chief_complaint="fussy"))
    assert result.acuity_floor == 2
    assert any(h.rule_id == "R11" for h in result.hits)


def test_paediatric_hypotension_uses_age_band():
    """SBP 78 is fine for a 2-year-old and shock for an adult."""
    child = ENGINE.evaluate(snap(age_years=2, systolic_bp=78))
    adult = ENGINE.evaluate(snap(age_years=40, systolic_bp=78))
    assert not any(h.rule_id == "R03" for h in child.hits)
    assert any(h.rule_id == "R03" for h in adult.hits)


def test_qsofa_sepsis_needs_infection_context():
    """Two qSOFA criteria alone are not sepsis; with a fever complaint they are."""
    vitals = {"respiratory_rate": 24, "systolic_bp": 96}
    no_context = ENGINE.evaluate(snap(chief_complaint="ankle injury", **vitals))
    with_context = ENGINE.evaluate(snap(chief_complaint="fever and cough x3d", **vitals))
    assert not any(h.rule_id == "R07" for h in no_context.hits)
    assert any(h.rule_id == "R07" for h in with_context.hits)


@pytest.mark.parametrize(
    "complaint,rule_id",
    [
        ("CP rad to L arm", "R10"),
        ("SOB, stridor", "R01"),
        ("facial droop, slurred speech", "R09"),
        ("AMS, family reports slurred speech", "R09"),
        ("SI, wants to die", "R19"),
        ("MVC, ejected", "R14"),
        ("throat closing after bee sting", "R04"),
        ("witnessed sz, postictal", "R18"),
    ],
)
def test_abbreviated_complaints_are_understood(complaint: str, rule_id: str):
    """Triage text is clipped and abbreviation-dense. The matcher must cope."""
    result = ENGINE.evaluate(snap(chief_complaint=complaint))
    assert any(h.rule_id == rule_id for h in result.hits), f"{complaint!r} missed {rule_id}"


def test_word_boundaries_prevent_false_fires():
    """'CP' must not fire on 'CPAP'; 'SI' must not fire on 'sinus'."""
    result = ENGINE.evaluate(snap(chief_complaint="CPAP machine broken, sinus congestion"))
    assert not any(h.rule_id in {"R10", "R19"} for h in result.hits)


# --- missing data ---------------------------------------------------------------------


def test_missing_vitals_do_not_crash_or_silently_pass():
    """A patient too combative to get vitals on gets no false reassurance."""
    s = snap(chief_complaint="agitated, unable to obtain vitals")
    result = ENGINE.evaluate(s)
    assert isinstance(result.acuity_floor, int)
    assert set(s.missing_vitals()) == {
        "heart_rate",
        "systolic_bp",
        "respiratory_rate",
        "spo2",
        "temperature_c",
    }
    # Zero recordable vitals is exactly the "no false reassurance" case: the floor
    # rule must fire on its own, with no dependence on the complaint text matching
    # anything.
    assert any(h.rule_id == "R23" for h in result.hits)
    assert result.acuity_floor == 2


def test_one_missing_vital_is_not_escalated():
    """A single gap is routine — a probe not placed yet, nothing to ask for on an
    obviously minor complaint. Firing here would flag most ordinary triage records."""
    result = ENGINE.evaluate(snap(heart_rate=76, systolic_bp=118, respiratory_rate=16))
    assert not any(h.rule_id == "R23" for h in result.hits)


def test_two_missing_core_vitals_together_escalate():
    """Heart rate and oxygen saturation come off one probe. Losing both at once on a
    working monitor usually means the patient, not the equipment, is the problem."""
    result = ENGINE.evaluate(snap(systolic_bp=118, respiratory_rate=16))
    hit = next(h for h in result.hits if h.rule_id == "R23")
    assert "heart rate" in hit.reason and "SpO2" in hit.reason
    assert hit.acuity_floor == 2


def test_every_rule_carries_a_citation():
    """A threshold nobody can trace is a threshold nobody will defend."""
    for rule in ENGINE.rules:
        assert rule.citation.strip(), f"{rule.id} has no citation"
        assert rule.acuity_floor in (1, 2), f"{rule.id} floor must be a red-flag level"
