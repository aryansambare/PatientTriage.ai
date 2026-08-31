"""Age stratification.

The brief names the failure directly: a single adult-calibrated scoring model applied
across all ages carries silent safety risk. These tests are what "not silent" means —
each one is a patient who is scored wrongly if the system forgets how old they are.
"""

from __future__ import annotations

import pytest

from patienttriage.clinical import agebands
from patienttriage.features.schema import TriageSnapshot
from patienttriage.rules.engine import RuleEngine

ENGINE = RuleEngine()


def snap(**kwargs) -> TriageSnapshot:
    base = {"patient_id": "t", "arrived_at": "2026-08-31T09:00:00", "age_years": 45.0}
    return TriageSnapshot(**(base | kwargs))


# --- the bands themselves --------------------------------------------------------------


@pytest.mark.parametrize(
    "age,expected",
    [
        (10 * agebands.DAY, "neonate"),
        (6 * agebands.MONTH, "infant"),
        (2.0, "toddler"),
        (4.0, "preschool"),
        (9.0, "school_age"),
        (15.0, "adolescent"),
        (40.0, "adult"),
        (70.0, "older_adult"),
        (88.0, "geriatric"),
    ],
)
def test_every_age_lands_in_its_clinical_band(age: float, expected: str):
    assert agebands.band_name(age) == expected


def test_bands_are_contiguous_and_cover_every_age():
    """No age may fall between two bands — a gap is a patient with no thresholds."""
    for younger, older in zip(agebands.BANDS, agebands.BANDS[1:], strict=False):
        assert younger.upper_age == older.lower_age
    assert agebands.BANDS[0].lower_age == 0.0
    assert agebands.band_name(200.0) == "geriatric"


# --- the same number, different meaning ------------------------------------------------


def test_the_briefs_own_example_38_5_at_three_versus_seventy_five():
    """38.5C in a three-year-old and in a seventy-five-year-old are not the same fact."""
    child = agebands.fever_significance(3.0, 38.5)
    elder = agebands.fever_significance(75.0, 38.5)

    assert "usually" in child and "self-limiting" in child
    assert "older patient" in elder
    assert child != elder


def test_absence_of_fever_does_not_reassure_in_the_old():
    """A septic 85-year-old is often normothermic. Saying nothing there is a failure."""
    assert agebands.fever_significance(85.0, 37.2) is not None
    assert "does not exclude" in agebands.fever_significance(85.0, 37.2)
    assert agebands.fever_significance(40.0, 37.2) is None


def test_neonatal_fever_is_never_reassured_by_appearance():
    message = agebands.fever_significance(14 * agebands.DAY, 38.1)
    assert "regardless of how well" in message


# --- thresholds move with age ----------------------------------------------------------


def test_tachycardia_threshold_rises_for_the_young_and_falls_for_the_old():
    assert agebands.tachycardia_threshold(0.5) > agebands.tachycardia_threshold(40.0)
    assert agebands.tachycardia_threshold(88.0) < agebands.tachycardia_threshold(40.0)


def test_beta_blocker_lowers_the_bar_at_every_age():
    for age in (10.0, 40.0, 85.0):
        plain = agebands.tachycardia_threshold(age)
        blocked = agebands.tachycardia_threshold(age, on_beta_blocker=True)
        assert blocked == plain - 20.0


def test_hypotension_threshold_follows_the_paediatric_formula():
    assert agebands.hypotension_threshold(5.0) == pytest.approx(80.0)
    assert agebands.hypotension_threshold(40.0) == 90.0
    # Older patients are usually hypertensive at baseline, so 95 is already a fall.
    assert agebands.hypotension_threshold(82.0) == 100.0


def test_a_pressure_that_is_shock_in_one_patient_is_normal_in_another():
    """SBP 95: unremarkable in an adult, shock in an 82-year-old, fine for a toddler."""
    adult = ENGINE.evaluate(snap(age_years=40, systolic_bp=95))
    elder = ENGINE.evaluate(snap(age_years=82, systolic_bp=95))
    toddler = ENGINE.evaluate(snap(age_years=2, systolic_bp=95))

    assert not any(h.rule_id == "R03" for h in adult.hits)
    assert any(h.rule_id == "R03" for h in elder.hits)
    assert not any(h.rule_id == "R03" for h in toddler.hits)


def test_respiratory_rate_is_judged_against_age():
    """RR 34 is respiratory distress in an adult and entirely normal in an infant."""
    adult = ENGINE.evaluate(snap(age_years=40, respiratory_rate=34))
    infant = ENGINE.evaluate(snap(age_years=0.5, respiratory_rate=34))
    assert any(h.rule_id == "R16" for h in adult.hits)
    assert not any(h.rule_id == "R16" for h in infant.hits)


# --- the two rules that exist only because of age --------------------------------------


def test_occult_sepsis_in_an_older_patient_without_a_fever():
    """37.6C would not register as a fever at any adult threshold. In an 84-year-old
    with a rising respiratory rate it is the presentation."""
    result = ENGINE.evaluate(
        snap(age_years=84, temperature_c=37.6, respiratory_rate=24, heart_rate=104)
    )
    assert any(h.rule_id == "R21" for h in result.hits)
    assert result.acuity_floor <= 2


def test_the_same_temperature_in_a_forty_year_old_does_not_fire():
    result = ENGINE.evaluate(
        snap(age_years=40, temperature_c=37.6, respiratory_rate=24, heart_rate=104)
    )
    assert not any(h.rule_id == "R21" for h in result.hits)


def test_hypothermia_in_an_older_patient_is_read_as_sepsis_not_cold():
    result = ENGINE.evaluate(
        snap(age_years=79, temperature_c=35.2, respiratory_rate=24)
    )
    assert any(h.rule_id == "R21" for h in result.hits)


def test_compensated_paediatric_shock_with_a_textbook_normal_pressure():
    """The child is holding their pressure up by running fast. A system that waits for
    paediatric hypotension is waiting for the pre-arrest sign."""
    result = ENGINE.evaluate(
        snap(age_years=3, heart_rate=165, respiratory_rate=38, systolic_bp=95)
    )
    assert any(h.rule_id == "R22" for h in result.hits)
    assert result.acuity_floor <= 2


def test_compensated_shock_rule_does_not_apply_to_adults():
    """An adult with a normal pressure and a fast heart rate is a different problem."""
    result = ENGINE.evaluate(
        snap(age_years=40, heart_rate=120, respiratory_rate=24, systolic_bp=125)
    )
    assert not any(h.rule_id == "R22" for h in result.hits)


def test_a_calm_child_does_not_trigger_the_shock_rule():
    result = ENGINE.evaluate(
        snap(age_years=4, heart_rate=105, respiratory_rate=22, systolic_bp=100)
    )
    assert not any(h.rule_id == "R22" for h in result.hits)
