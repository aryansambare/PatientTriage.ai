"""Feature construction from a triage snapshot.

Naming convention: every engineered feature is `<source_field>__<derivation>`, so the
leakage guard in `schema.py` can trace it back to a declared source. A feature that
cannot name its source cannot be built.

The derived features here are the ones emergency clinicians actually reason with —
shock index, pulse pressure, qSOFA and SIRS counts, age-adjusted abnormality. Handing
a gradient-boosted model raw HR and SBP and hoping it rediscovers shock index from
400k rows is possible but wasteful, and it makes the SHAP explanation unreadable to
the nurse who has to act on it.

Every frame this module returns has identical columns, in identical order, with
identical dtypes, regardless of what was in the batch — see `FEATURE_SPEC`. Training
runs on tens of thousands of rows where most values are present; serving runs on one
row where most are absent. If the dtypes were allowed to follow the data, those two
frames would disagree and the model would score the live patient against a different
encoding than it was trained on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime

import pandas as pd

from patienttriage.clinical import agebands
from patienttriage.features.schema import TriageSnapshot, assert_triage_time_only
from patienttriage.rules.lexicon import ComplaintFlag, flags_for
from patienttriage.rules.rvc import RVC_BANDS, band_features

ARRIVAL_MODES = ("ambulance", "walk_in", "police", "transfer", "helicopter", "unknown")
AGE_BANDS = agebands.BAND_NAMES
SEXES = ("female", "male", "other", "unknown")
VITALS = ("heart_rate", "systolic_bp", "respiratory_rate", "spo2", "temperature_c")


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Guarded division.

    Written with explicit None checks rather than truthiness because zero is a real,
    survivable-to-record vital sign: HR 0 means arrest, not "not measured". Treating
    it as missing hides the sickest patient in the department.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _bool_to_float(value: bool | None) -> float | None:
    """Tri-state to numeric, preserving the difference between False and unknown."""
    return None if value is None else float(value)


def snapshot_to_row(s: TriageSnapshot) -> dict[str, object]:
    """One snapshot to one flat feature row."""
    band = agebands.band_name(s.age_years)
    row: dict[str, object] = {}

    # --- raw admissible fields --------------------------------------------------------
    row["age_years"] = s.age_years
    row["heart_rate"] = s.heart_rate
    row["systolic_bp"] = s.systolic_bp
    row["diastolic_bp"] = s.diastolic_bp
    row["respiratory_rate"] = s.respiratory_rate
    row["spo2"] = s.spo2
    row["temperature_c"] = s.temperature_c
    row["pain_score"] = s.pain_score
    row["gcs"] = s.gcs
    row["ed_census"] = s.ed_census
    row["boarding_count"] = s.boarding_count
    row["prior_ed_visits_90d"] = s.prior_ed_visits_90d

    row["age_years__band"] = band
    row["sex"] = s.sex

    # --- haemodynamics ----------------------------------------------------------------
    hr, sbp, dbp = s.heart_rate, s.systolic_bp, s.diastolic_bp

    # Shock index (HR/SBP): rises before blood pressure falls, so it catches the
    # compensating patient who still looks normal. >0.9 is the usual concern threshold.
    row["heart_rate__shock_index"] = _ratio(hr, sbp)

    # Age-adjusted shock index — the elderly decompensate at lower values.
    shock_index = row["heart_rate__shock_index"]
    row["heart_rate__age_shock_index"] = (
        s.age_years * shock_index if shock_index is not None else None
    )

    row["systolic_bp__pulse_pressure"] = (
        sbp - dbp if sbp is not None and dbp is not None else None
    )
    row["systolic_bp__map"] = (
        (sbp + 2 * dbp) / 3 if sbp is not None and dbp is not None else None
    )

    row["heart_rate__age_deviation"] = agebands.heart_rate_deviation(hr, s.age_years)
    row["respiratory_rate__age_deviation"] = agebands.respiratory_rate_deviation(
        s.respiratory_rate, s.age_years
    )

    # --- composite severity scores ----------------------------------------------------
    qsofa = 0
    if s.respiratory_rate is not None and s.respiratory_rate >= 22:
        qsofa += 1
    if s.systolic_bp is not None and s.systolic_bp <= 100:
        qsofa += 1
    if s.gcs is not None and s.gcs < 15:
        qsofa += 1
    row["respiratory_rate__qsofa"] = qsofa

    sirs = 0
    if s.temperature_c is not None and (s.temperature_c > 38.0 or s.temperature_c < 36.0):
        sirs += 1
    if s.heart_rate is not None and s.heart_rate > 90:
        sirs += 1
    if s.respiratory_rate is not None and s.respiratory_rate > 20:
        sirs += 1
    row["temperature_c__sirs"] = sirs

    # --- missingness is signal, not noise ---------------------------------------------
    # A patient too agitated, too young, or too unwell to get a full vital set on is a
    # different patient from one whose vitals are all normal. The model should see that.
    missing = s.missing_vitals()
    for vital in VITALS:
        row[f"{vital}__missing"] = float(vital in missing)
    row["age_years__missing_vital_count"] = float(len(missing))

    # --- presentation -----------------------------------------------------------------
    complaint_flags = flags_for(s.chief_complaint)
    for flag in ComplaintFlag:
        row[f"chief_complaint__{flag.value}"] = float(flag in complaint_flags)
    row["chief_complaint__flag_count"] = float(len(complaint_flags))
    row["chief_complaint__length"] = float(len(s.chief_complaint.split()))
    row["chief_complaint__empty"] = float(not s.chief_complaint.strip())

    # Coded reason for visit, where the department uses a picklist instead of text.
    # These are body-system bands, not diagnoses — coarse enough that the rule layer
    # is not allowed to read them, but useful for the model to weigh. See rules/rvc.py.
    row.update(band_features(s.chief_complaint_codes))
    row["chief_complaint_codes__count"] = float(len(s.chief_complaint_codes))

    # --- arrival context --------------------------------------------------------------
    for mode in ARRIVAL_MODES:
        row[f"arrival_mode__{mode}"] = float(s.arrival_mode == mode)

    # Hour of day is cyclical: 23:00 and 01:00 are adjacent, and a plain integer tells
    # the model they are 22 apart. Overnight matters because staffing is thinnest and
    # the waiting room is least observed exactly when the sickest arrivals come in.
    hour, dow = _arrival_time(s.arrived_at)
    if hour is None:
        row["arrival_hour__sin"] = None
        row["arrival_hour__cos"] = None
        row["arrival_hour__is_night"] = None
    else:
        row["arrival_hour__sin"] = math.sin(2 * math.pi * hour / 24)
        row["arrival_hour__cos"] = math.cos(2 * math.pi * hour / 24)
        row["arrival_hour__is_night"] = float(hour >= 22 or hour < 7)
    row["arrival_dow__is_weekend"] = None if dow is None else float(dow >= 5)

    # Crowding: the same patient is at more risk in a full department, because the
    # waiting time their acuity buys them is longer.
    row["ed_census__boarding_ratio"] = _ratio(s.boarding_count, s.ed_census)

    # --- prior record -----------------------------------------------------------------
    row["on_beta_blocker"] = _bool_to_float(s.on_beta_blocker)
    row["on_anticoagulant"] = _bool_to_float(s.on_anticoagulant)
    row["is_immunosuppressed"] = _bool_to_float(s.is_immunosuppressed)
    row["is_pregnant"] = _bool_to_float(s.is_pregnant)
    row["prior_icu_admission"] = _bool_to_float(s.prior_icu_admission)

    return row


def _arrival_time(arrived_at: str) -> tuple[int | None, int | None]:
    """(hour, weekday) from the arrival timestamp, or (None, None) if unparseable."""
    try:
        moment = datetime.fromisoformat(arrived_at)
    except (ValueError, TypeError):
        return None, None
    return moment.hour, moment.weekday()


# --------------------------------------------------------------------------------------
# The frozen output schema
# --------------------------------------------------------------------------------------

CATEGORICAL_FEATURES: dict[str, tuple[str, ...]] = {
    "sex": SEXES,
    "age_years__band": AGE_BANDS,
}

NUMERIC_FEATURES: tuple[str, ...] = (
    "age_years",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "spo2",
    "temperature_c",
    "pain_score",
    "gcs",
    "ed_census",
    "boarding_count",
    "prior_ed_visits_90d",
    "heart_rate__shock_index",
    "heart_rate__age_shock_index",
    "systolic_bp__pulse_pressure",
    "systolic_bp__map",
    "heart_rate__age_deviation",
    "respiratory_rate__age_deviation",
    "respiratory_rate__qsofa",
    "temperature_c__sirs",
    *(f"{v}__missing" for v in VITALS),
    "age_years__missing_vital_count",
    *(f"chief_complaint__{f.value}" for f in ComplaintFlag),
    "chief_complaint__flag_count",
    "chief_complaint__length",
    "chief_complaint__empty",
    *(f"chief_complaint_codes__{b}" for b in RVC_BANDS),
    "chief_complaint_codes__count",
    *(f"arrival_mode__{m}" for m in ARRIVAL_MODES),
    "arrival_hour__sin",
    "arrival_hour__cos",
    "arrival_hour__is_night",
    "arrival_dow__is_weekend",
    "ed_census__boarding_ratio",
    "on_beta_blocker",
    "on_anticoagulant",
    "is_immunosuppressed",
    "is_pregnant",
    "prior_icu_admission",
)

FEATURE_SPEC: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
"""Every column a feature frame has, in order. Frozen — training and serving share it."""


def build_features(snapshots: Iterable[TriageSnapshot], *, verify: bool = True) -> pd.DataFrame:
    """Feature matrix for a batch of snapshots, leakage-checked before it is returned.

    Missing values stay as NaN throughout. LightGBM treats NaN as a missing value and
    learns a default split direction for it, which is precisely the behaviour we want:
    "this vital was never taken" is a branch the tree can reason about, not a hole to
    be filled with a population median the patient never had.
    """
    rows = [snapshot_to_row(s) for s in snapshots]
    frame = pd.DataFrame(rows, columns=list(FEATURE_SPEC))

    if verify:
        assert_triage_time_only(list(frame.columns))

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    # Categories are pinned rather than inferred, so a batch that happens to contain no
    # paediatric patients still encodes "child" at the same integer code as training did.
    for column, categories in CATEGORICAL_FEATURES.items():
        frame[column] = pd.Categorical(frame[column], categories=list(categories))

    return frame
