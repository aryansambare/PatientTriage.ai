"""The triage-time data contract, and the leakage guard that enforces it.

Most published ED triage models report numbers they could not reproduce in a real
department, because they quietly train on data that does not exist yet when the
prediction has to be made: lab results, imaging, the physician's note, the eventual
diagnosis code. The model looks excellent and is useless at 03:00 on a Saturday.

So availability is a property of the field itself, declared here once, and
`assert_triage_time_only` refuses to let anything else reach a feature matrix.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Availability(str, Enum):
    """When a field becomes knowable, relative to the moment of arrival."""

    AT_TRIAGE = "at_triage"
    """Present within ~5 minutes of arrival. Admissible as a model input."""

    POST_TRIAGE = "post_triage"
    """Labs, imaging, physician assessment. Never an input — this is the leak."""

    OUTCOME = "outcome"
    """Disposition, ICU transfer, mortality. A label, never an input."""


class LeakageError(RuntimeError):
    """Raised when a non-triage-time field is about to be used as a model input."""


# --------------------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------------------

FIELD_AVAILABILITY: dict[str, Availability] = {
    # --- demographics -----------------------------------------------------------------
    "age_years": Availability.AT_TRIAGE,
    "sex": Availability.AT_TRIAGE,
    # --- arrival context --------------------------------------------------------------
    "arrival_mode": Availability.AT_TRIAGE,
    "arrival_hour": Availability.AT_TRIAGE,
    "arrival_dow": Availability.AT_TRIAGE,
    "ed_census": Availability.AT_TRIAGE,
    "boarding_count": Availability.AT_TRIAGE,
    # --- triage vitals ----------------------------------------------------------------
    "heart_rate": Availability.AT_TRIAGE,
    "systolic_bp": Availability.AT_TRIAGE,
    "diastolic_bp": Availability.AT_TRIAGE,
    "respiratory_rate": Availability.AT_TRIAGE,
    "spo2": Availability.AT_TRIAGE,
    "temperature_c": Availability.AT_TRIAGE,
    "pain_score": Availability.AT_TRIAGE,
    "gcs": Availability.AT_TRIAGE,
    # --- presentation -----------------------------------------------------------------
    # Two forms, because departments record this two ways and a deployment may see
    # either: free text typed by the triage nurse, or a code chosen from a picklist.
    # NHAMCS carries only the coded form; MIMIC-IV-ED carries only the text.
    "chief_complaint": Availability.AT_TRIAGE,
    "chief_complaint_codes": Availability.AT_TRIAGE,
    # --- prior record, only if the MRN matched --------------------------------------
    "on_beta_blocker": Availability.AT_TRIAGE,
    "on_anticoagulant": Availability.AT_TRIAGE,
    "is_immunosuppressed": Availability.AT_TRIAGE,
    "is_pregnant": Availability.AT_TRIAGE,
    "prior_ed_visits_90d": Availability.AT_TRIAGE,
    "prior_icu_admission": Availability.AT_TRIAGE,
    # --- the leaks, named explicitly so they cannot be added back by accident ---------
    "wbc": Availability.POST_TRIAGE,
    "lactate": Availability.POST_TRIAGE,
    "troponin": Availability.POST_TRIAGE,
    "creatinine": Availability.POST_TRIAGE,
    "ecg_interpretation": Availability.POST_TRIAGE,
    "ct_result": Availability.POST_TRIAGE,
    "physician_note": Availability.POST_TRIAGE,
    "diagnosis_icd10": Availability.POST_TRIAGE,
    "orders_placed": Availability.POST_TRIAGE,
    "ed_length_of_stay": Availability.POST_TRIAGE,
    # --- labels -----------------------------------------------------------------------
    "esi_acuity": Availability.OUTCOME,
    "disposition": Availability.OUTCOME,
    "icu_transfer_12h": Availability.OUTCOME,
    "died_in_ed": Availability.OUTCOME,
    "critical_intervention_4h": Availability.OUTCOME,
}


def assert_triage_time_only(columns: list[str], *, strict: bool = True) -> None:
    """Refuse a feature matrix that contains anything not knowable at triage.

    `strict` also rejects unregistered columns. Leave it on: an unregistered column is
    a column nobody has thought about, which is exactly how leakage gets in.
    """
    leaks: list[str] = []
    unknown: list[str] = []

    for col in columns:
        base = col.split("__", 1)[0]  # engineered features keep their source prefix
        availability = FIELD_AVAILABILITY.get(base)
        if availability is None:
            unknown.append(col)
        elif availability is not Availability.AT_TRIAGE:
            leaks.append(f"{col} ({availability.value})")

    problems: list[str] = []
    if leaks:
        problems.append(
            "these are not knowable when the prediction must be made: " + ", ".join(sorted(leaks))
        )
    if unknown and strict:
        problems.append(
            "these are not declared in FIELD_AVAILABILITY, so their availability is "
            "unverified: " + ", ".join(sorted(unknown))
        )

    if problems:
        raise LeakageError(
            "Feature matrix rejected — " + "; ".join(problems) + ". "
            "Add the field to FIELD_AVAILABILITY with an honest availability, or drop it."
        )


def triage_time_fields() -> list[str]:
    """Every field admissible as a model input."""
    return [f for f, a in FIELD_AVAILABILITY.items() if a is Availability.AT_TRIAGE]


def label_fields() -> list[str]:
    """Every field usable as a training target."""
    return [f for f, a in FIELD_AVAILABILITY.items() if a is Availability.OUTCOME]


# --------------------------------------------------------------------------------------
# The snapshot
# --------------------------------------------------------------------------------------

ArrivalMode = Literal["ambulance", "walk_in", "police", "transfer", "helicopter", "unknown"]
Sex = Literal["female", "male", "other", "unknown"]


class TriageSnapshot(BaseModel):
    """Everything the system is allowed to know about a patient at arrival.

    Vital-sign bounds match `data.quality.PLAUSIBLE_RANGES`. Anything a source file
    carries outside those bounds is not a measurement and is cleaned to None before
    it reaches here — see that module for why it is not simply passed through.

    Every vital is optional, because in a real department they are genuinely missing:
    the patient is combative, the child will not sit still, the cuff does not fit, the
    interpreter has not arrived. Missingness is a state to be shown to the nurse and
    reasoned about — never quietly filled in with a median.
    """

    model_config = {"extra": "forbid"}

    patient_id: str
    arrived_at: str = Field(description="ISO 8601 timestamp of ED arrival")

    age_years: float = Field(ge=0, le=120)
    sex: Sex = "unknown"

    arrival_mode: ArrivalMode = "unknown"
    ed_census: int | None = Field(default=None, ge=0)
    boarding_count: int | None = Field(default=None, ge=0)

    heart_rate: float | None = Field(default=None, ge=0, le=300)
    systolic_bp: float | None = Field(default=None, ge=0, le=300)
    diastolic_bp: float | None = Field(default=None, ge=0, le=250)
    respiratory_rate: float | None = Field(default=None, ge=0, le=120)
    spo2: float | None = Field(default=None, ge=0, le=100)
    temperature_c: float | None = Field(default=None, ge=25, le=45)
    pain_score: float | None = Field(default=None, ge=0, le=10)
    gcs: int | None = Field(default=None, ge=3, le=15)

    chief_complaint: str = ""
    chief_complaint_codes: tuple[str, ...] = ()
    """Coded reasons for visit, when the department uses a picklist rather than text."""

    on_beta_blocker: bool | None = None
    on_anticoagulant: bool | None = None
    is_immunosuppressed: bool | None = None
    is_pregnant: bool | None = None
    prior_ed_visits_90d: int | None = Field(default=None, ge=0)
    prior_icu_admission: bool | None = None

    def complaint_flags(self) -> set:
        """Clinical categories the free-text complaint touches.

        Coded complaints (`chief_complaint_codes`) deliberately do not contribute
        here — see `rules/rvc.py`. Band-level codes are too coarse to drive red
        flags, so they reach the model as features and the rule layer never sees them.
        """
        from patienttriage.rules.lexicon import flags_for

        return flags_for(self.chief_complaint)

    def missing_vitals(self) -> list[str]:
        """Which vitals were not captured. Drives the 'incomplete' banner in the UI."""
        vitals = [
            "heart_rate",
            "systolic_bp",
            "respiratory_rate",
            "spo2",
            "temperature_c",
        ]
        return [v for v in vitals if getattr(self, v) is None]

    def is_infant(self) -> bool:
        return self.age_years < 0.25  # under ~90 days
