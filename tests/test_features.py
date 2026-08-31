from __future__ import annotations

import pandas as pd

from patienttriage.features.build import FEATURE_SPEC, build_features, snapshot_to_row
from patienttriage.features.schema import TriageSnapshot


def snap(**kwargs) -> TriageSnapshot:
    base = {"patient_id": "t", "arrived_at": "2026-08-31T09:00:00", "age_years": 45.0}
    return TriageSnapshot(**(base | kwargs))


def test_feature_matrix_passes_its_own_leakage_guard():
    """build_features verifies by default; if the naming convention drifts, this fails."""
    frame = build_features([snap(heart_rate=88, systolic_bp=118)])
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1


# --- train/serve parity ---------------------------------------------------------------


def test_schema_is_identical_for_a_full_batch_and_a_single_sparse_patient():
    """Training sees mostly-complete rows; serving sees one mostly-empty one. If the
    dtypes follow the data, the live patient is scored against a different encoding
    than the model was trained on."""
    full = build_features(
        [
            snap(heart_rate=88, systolic_bp=120, spo2=98, temperature_c=37.0, respiratory_rate=16),
            snap(heart_rate=110, systolic_bp=95, spo2=93, temperature_c=38.6, respiratory_rate=24),
        ]
    )
    sparse = build_features([snap(chief_complaint="rash")])

    assert list(full.columns) == list(sparse.columns) == list(FEATURE_SPEC)
    assert full.dtypes.to_dict() == sparse.dtypes.to_dict()


def test_categorical_codes_are_pinned_not_inferred():
    """A batch with no children must still encode 'child' the way training did."""
    adults = build_features([snap(age_years=40), snap(age_years=50)])
    child = build_features([snap(age_years=6)])
    assert list(adults["age_years__band"].cat.categories) == list(
        child["age_years__band"].cat.categories
    )
    assert child["age_years__band"].iloc[0] == "school_age"


def test_missing_values_stay_nan_and_are_never_imputed():
    frame = build_features([snap(chief_complaint="rash")])
    assert pd.isna(frame["heart_rate"].iloc[0])
    assert pd.isna(frame["systolic_bp"].iloc[0])


# --- zero is a real vital sign --------------------------------------------------------


def test_zero_vitals_are_not_treated_as_missing():
    """HR 0 means cardiac arrest, not 'not measured'. The earlier truthiness check
    silently discarded the sickest patient in the department."""
    row = snapshot_to_row(snap(heart_rate=0, systolic_bp=0, spo2=0))
    assert row["heart_rate"] == 0
    assert row["heart_rate__missing"] == 0.0
    assert row["age_years__missing_vital_count"] == 2.0  # only RR and temp absent


def test_shock_index_guards_division_without_discarding_zero_numerators():
    assert snapshot_to_row(snap(heart_rate=0, systolic_bp=100))["heart_rate__shock_index"] == 0.0
    assert snapshot_to_row(snap(heart_rate=90, systolic_bp=0))["heart_rate__shock_index"] is None


# --- clinical derivations -------------------------------------------------------------


def test_shock_index_catches_the_compensating_patient():
    """SBP 105 alone looks acceptable. HR 120 against it does not."""
    row = snapshot_to_row(snap(heart_rate=120, systolic_bp=105))
    assert row["heart_rate__shock_index"] > 0.9


def test_missingness_is_recorded_rather_than_imputed():
    row = snapshot_to_row(snap(chief_complaint="agitated, vitals refused"))
    assert row["heart_rate"] is None
    assert row["heart_rate__missing"] == 1.0
    assert row["age_years__missing_vital_count"] == 5.0


def test_age_deviation_is_relative_to_the_patients_own_band():
    """HR 150 is normal for an infant and grossly abnormal for an adult. This is the
    whole reason one model can serve a neonate and a pensioner."""
    infant = snapshot_to_row(snap(age_years=0.5, heart_rate=150))
    adult = snapshot_to_row(snap(age_years=45, heart_rate=150))
    assert infant["heart_rate__age_deviation"] == 0.0
    assert adult["heart_rate__age_deviation"] > 0


def test_complaint_flags_become_columns():
    row = snapshot_to_row(snap(chief_complaint="CP rad to L arm, SOB"))
    assert row["chief_complaint__cardiac"] == 1.0
    assert row["chief_complaint__respiratory"] == 1.0
    assert row["chief_complaint__stroke"] == 0.0
    assert row["chief_complaint__flag_count"] == 2.0


def test_unknown_ehr_fields_stay_distinguishable_from_false():
    """No MRN match is not the same as 'confirmed not on a beta-blocker'."""
    unknown = build_features([snap()])
    negative = build_features([snap(on_beta_blocker=False)])
    assert pd.isna(unknown["on_beta_blocker"].iloc[0])
    assert negative["on_beta_blocker"].iloc[0] == 0.0


# --- arrival timing -------------------------------------------------------------------


def test_arrival_hour_is_encoded_cyclically():
    """23:00 and 01:00 are two hours apart, not twenty-two."""
    late = snapshot_to_row(snap(arrived_at="2026-08-31T23:00:00"))
    early = snapshot_to_row(snap(arrived_at="2026-09-01T01:00:00"))
    noon = snapshot_to_row(snap(arrived_at="2026-08-31T12:00:00"))

    def dist(a, b):
        return (a["arrival_hour__sin"] - b["arrival_hour__sin"]) ** 2 + (
            a["arrival_hour__cos"] - b["arrival_hour__cos"]
        ) ** 2

    assert dist(late, early) < dist(late, noon)
    assert late["arrival_hour__is_night"] == 1.0
    assert noon["arrival_hour__is_night"] == 0.0


def test_weekend_arrival_is_flagged():
    saturday = snapshot_to_row(snap(arrived_at="2026-08-29T14:00:00"))
    tuesday = snapshot_to_row(snap(arrived_at="2026-09-01T14:00:00"))
    assert saturday["arrival_dow__is_weekend"] == 1.0
    assert tuesday["arrival_dow__is_weekend"] == 0.0


def test_unparseable_timestamp_degrades_to_missing_not_to_a_crash():
    row = snapshot_to_row(snap(arrived_at="not a timestamp"))
    assert row["arrival_hour__sin"] is None
    assert row["arrival_dow__is_weekend"] is None
