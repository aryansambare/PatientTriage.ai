"""NEWS2 — reported, never decided on.

Each test checks either the score against a known worked value, or the boundary the
module's own docstring promises: adults only, non-pregnant, and a lone maxed-out
parameter escalating the band on its own.
"""

from __future__ import annotations

from patienttriage.clinical import news2


def _score(**overrides):
    vitals = {
        "heart_rate": 75.0,
        "respiratory_rate": 16.0,
        "spo2": 98.0,
        "systolic_bp": 120.0,
        "temperature_c": 37.0,
        "gcs": 15,
    }
    return news2.score(**(vitals | overrides))


# --- applicability -----------------------------------------------------------------


def test_not_applicable_under_eighteen():
    """NEWS2 Scale 1 is derived from and validated on adults."""
    assert not news2.is_applicable(12.0, is_pregnant=False)
    assert news2.is_applicable(18.0, is_pregnant=False)


def test_not_applicable_in_pregnancy():
    """NHS guidance points to an obstetric-specific score instead."""
    assert not news2.is_applicable(30.0, is_pregnant=True)


# --- a well patient scores zero -----------------------------------------------------


def test_normal_vitals_score_zero():
    result = _score()
    assert result.total == 0
    assert result.risk_band == "low"
    assert not result.urgent


# --- known worked values -------------------------------------------------------------


def test_textbook_high_risk_patient():
    """RR 28 (3) + SpO2 90 (3) + SBP 85 (3) + HR 135 (3) + GCS 12 (3) + temp 39.5 (2)."""
    result = _score(
        respiratory_rate=28, spo2=90, systolic_bp=85, heart_rate=135,
        gcs=12, temperature_c=39.5,
    )
    assert result.total == 17
    assert result.risk_band == "high"
    assert result.urgent


def test_aggregate_medium_band():
    """SBP 95 (2) + HR 105 (1) + RR 22 (2) = 5, nothing else abnormal."""
    result = _score(systolic_bp=95, heart_rate=105, respiratory_rate=22)
    assert result.total == 5
    assert result.risk_band == "medium"


# --- the single-parameter escalation ------------------------------------------------


def test_isolated_bradycardia_escalates_despite_a_low_total():
    """HR 35 alone scores 3; every other parameter is normal, so the total is only 3.
    NHS guidance still calls this an urgent review, not a low-risk chart."""
    result = _score(heart_rate=35)
    assert result.total == 3
    assert result.max_single_parameter == 3
    assert result.risk_band == "low-medium"
    assert result.urgent


def test_single_parameter_never_downgrades_a_higher_aggregate():
    """A lone maxed-out parameter must only ever pull a low total up — never pull a
    medium or high total back down to 'low-medium'."""
    # SBP 95 (2) + HR 105 (1) + RR 22 (2) + isolated temp 35.0 (3) = total 8, high.
    result = _score(systolic_bp=95, heart_rate=105, respiratory_rate=22, temperature_c=35.0)
    assert result.total == 8
    assert result.risk_band == "high"


# --- missing data --------------------------------------------------------------------


def test_missing_parameters_score_zero_rather_than_raising():
    result = news2.score(
        heart_rate=None, respiratory_rate=None, spo2=None,
        systolic_bp=None, temperature_c=None, gcs=None,
    )
    assert result.total == 0
    assert result.risk_band == "low"
