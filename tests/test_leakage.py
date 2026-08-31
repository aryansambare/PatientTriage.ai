"""The leakage guard is what keeps our reported numbers honest. If these tests are
ever relaxed, the model's metrics stop meaning anything."""

from __future__ import annotations

import pytest

from patienttriage.features.schema import (
    FIELD_AVAILABILITY,
    Availability,
    LeakageError,
    assert_triage_time_only,
    label_fields,
    triage_time_fields,
)


def test_admits_triage_time_columns():
    assert_triage_time_only(["age_years", "heart_rate", "spo2", "arrival_mode"])


def test_admits_engineered_features_derived_from_admissible_sources():
    assert_triage_time_only(["heart_rate__age_zscore", "systolic_bp__shock_index"])


def test_rejects_lab_results():
    with pytest.raises(LeakageError, match="lactate"):
        assert_triage_time_only(["age_years", "lactate"])


def test_rejects_the_label_used_as_an_input():
    with pytest.raises(LeakageError, match="esi_acuity"):
        assert_triage_time_only(["age_years", "esi_acuity"])


def test_rejects_undeclared_columns_by_default():
    """An undeclared column is one nobody has thought about — which is how leaks arrive."""
    with pytest.raises(LeakageError, match="not declared"):
        assert_triage_time_only(["age_years", "mystery_score"])


def test_strict_off_allows_undeclared_but_still_blocks_known_leaks():
    assert_triage_time_only(["mystery_score"], strict=False)
    with pytest.raises(LeakageError):
        assert_triage_time_only(["mystery_score", "troponin"], strict=False)


def test_inputs_and_labels_are_disjoint():
    assert not set(triage_time_fields()) & set(label_fields())


def test_every_declared_field_has_an_availability():
    assert all(isinstance(a, Availability) for a in FIELD_AVAILABILITY.values())
