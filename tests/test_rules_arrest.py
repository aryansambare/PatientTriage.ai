"""Regression tests for the patients the vital-sign rules are blind to.

The rule layer originally scored a cardiac arrest arriving with CPR in progress as
level 5 — the least acute category — because in an arrest there are no vitals to
trigger on and no rule read the complaint text. These tests exist so that cannot
come back.
"""

from __future__ import annotations

import pytest

from patienttriage.features.schema import TriageSnapshot
from patienttriage.rules.engine import RuleEngine

ENGINE = RuleEngine()


def snap(**kwargs) -> TriageSnapshot:
    base = {"patient_id": "t", "arrived_at": "2026-08-31T09:00:00", "age_years": 45.0}
    return TriageSnapshot(**(base | kwargs))


@pytest.mark.parametrize(
    "complaint",
    [
        "CPR in progress",
        "cardiac arrest",
        "VSA, ROSC en route",
        "unresponsive, not breathing",
        "code blue en route",
        "pulseless, compressions ongoing",
        "resp arrest",
        "witnessed v-fib arrest, defib x2",
    ],
)
def test_arrest_is_level_one_with_no_vitals_at_all(complaint: str):
    """The monitor reads nothing and nobody stops compressions to take a pulse, so
    every vital field arrives empty. The complaint text has to carry it alone."""
    result = ENGINE.evaluate(snap(chief_complaint=complaint, arrival_mode="ambulance"))
    assert result.acuity_floor == 1, f"{complaint!r} did not reach level 1"
    assert any(h.rule_id == "R00" for h in result.hits)


def test_peri_arrest_bradycardia():
    result = ENGINE.evaluate(snap(heart_rate=24))
    assert result.acuity_floor == 1
    assert any(h.rule_id == "R00" for h in result.hits)


def test_custody_arrival_is_not_a_cardiac_arrest():
    """Police bring patients in 'under arrest'. A false level 1 on every custody
    arrival would train staff to ignore the layer."""
    result = ENGINE.evaluate(
        snap(chief_complaint="under arrest, minor hand laceration", arrival_mode="police")
    )
    assert not any(h.rule_id == "R00" for h in result.hits)


# --- altered mental status standing alone ---------------------------------------------


def test_unresponsive_without_fever_is_still_a_red_flag():
    """Previously this only escalated if it coincided with an infection complaint,
    so an unresponsive patient with no vitals and no fever scored level 5."""
    result = ENGINE.evaluate(snap(chief_complaint="found unresponsive at home"))
    assert result.acuity_floor <= 2
    assert any(h.rule_id == "R20" for h in result.hits)


def test_low_gcs_escalates_on_the_number_alone():
    result = ENGINE.evaluate(snap(gcs=11, chief_complaint="fall"))
    assert any(h.rule_id == "R20" for h in result.hits)


def test_gcs_eight_reaches_level_one_via_airway():
    result = ENGINE.evaluate(snap(gcs=7))
    assert result.acuity_floor == 1


def test_free_text_beats_a_contradictory_gcs_in_the_safety_layer():
    """GCS 15 recorded but the nurse wrote 'confused'. GCS is often carried forward
    or scored loosely; nobody types 'confused' about a patient who is not."""
    result = ENGINE.evaluate(
        snap(chief_complaint="fever, confused", gcs=15, respiratory_rate=24)
    )
    assert any(h.rule_id == "R07" for h in result.hits)
