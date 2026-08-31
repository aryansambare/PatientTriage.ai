"""Turning model internals into something a nurse can disagree with.

An explanation exists so a clinician can *reject* the recommendation for a good
reason. That sets a high bar: "shap value of heart_rate__age_shock_index = 0.31" is
not an explanation, it is a number from inside a model. A nurse cannot argue with it,
so it provides no safety and no trust — only the appearance of both.

What a nurse can argue with is a clinical statement naming the observation and why it
matters: "HR 122 against SBP 96 - shock index 1.27, which usually rises before the
blood pressure falls." That can be checked against the patient. If the cuff was on
the wrong arm, the nurse now knows exactly which input to distrust.

So SHAP is used to *select* which few facts to surface, and the phrasing is written
against the feature, not generated from it. Explanations are deterministic: the same
patient always produces the same sentences, because an explanation that varies
between viewings cannot be audited after an incident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from patienttriage.features.schema import TriageSnapshot


@dataclass(frozen=True)
class Driver:
    feature: str
    contribution: float
    sentence: str


def _fmt(value: float | None, digits: int = 0) -> str:
    return "not recorded" if value is None else f"{value:.{digits}f}"


def _phrase(feature: str, s: TriageSnapshot) -> str | None:
    """The clinical sentence for a feature, or None if it has nothing to say."""
    hr, sbp, rr = s.heart_rate, s.systolic_bp, s.respiratory_rate

    if feature in {"heart_rate__shock_index", "heart_rate__age_shock_index"}:
        if hr is None or not sbp:
            return None
        return (
            f"HR {hr:.0f} against SBP {sbp:.0f} - shock index {hr / sbp:.2f}, "
            "which rises before blood pressure falls"
        )
    if feature == "heart_rate":
        return None if hr is None else f"HR {hr:.0f}"
    if feature == "heart_rate__age_deviation":
        return None if hr is None else f"HR {hr:.0f} is outside the normal range for this age"
    if feature in {"systolic_bp", "systolic_bp__map", "systolic_bp__pulse_pressure"}:
        return None if sbp is None else f"SBP {sbp:.0f} mmHg"
    if feature in {"respiratory_rate", "respiratory_rate__age_deviation"}:
        return None if rr is None else f"RR {rr:.0f}"
    if feature == "respiratory_rate__qsofa":
        return "meets qSOFA criteria for organ dysfunction"
    if feature == "temperature_c__sirs":
        return "meets SIRS criteria"
    if feature == "spo2":
        return None if s.spo2 is None else f"SpO2 {s.spo2:.0f}%"
    if feature == "temperature_c":
        return None if s.temperature_c is None else f"temperature {s.temperature_c:.1f}C"
    if feature == "pain_score":
        return None if s.pain_score is None else f"reported pain {s.pain_score:.0f}/10"
    if feature == "gcs":
        return None if s.gcs is None else f"GCS {s.gcs}"
    if feature == "age_years":
        # "age 0" is what a 19-day-old renders as with a naive format, and it is the
        # single most important fact about that patient.
        if s.age_years < 1 / 12:
            return f"age {s.age_years * 365.25:.0f} days"
        if s.age_years < 2:
            return f"age {s.age_years * 12:.0f} months"
        return f"age {s.age_years:.0f}"
    if feature == "arrival_mode__ambulance":
        return (
            "arrived by ambulance - the pre-hospital team already judged this urgent"
            if s.arrival_mode == "ambulance"
            else None
        )
    if feature == "arrival_mode__walk_in":
        return "walked in" if s.arrival_mode == "walk_in" else None
    if feature.endswith("__missing"):
        vital = feature.removesuffix("__missing").replace("_", " ")
        return f"{vital} was not recorded, so the estimate rests on less evidence"
    if feature == "age_years__missing_vital_count":
        missing = s.missing_vitals()
        return f"{len(missing)} vitals not recorded" if missing else None
    if feature.startswith("chief_complaint__") and not feature.startswith(
        "chief_complaint__flag"
    ):
        category = feature.removeprefix("chief_complaint__").replace("_", " ")
        return f"complaint suggests {category}"
    if feature.startswith("chief_complaint_codes__symptom_"):
        system = feature.removeprefix("chief_complaint_codes__symptom_").replace("_", " ")
        return f"reason for visit coded to {system}"
    if feature.startswith("arrival_hour"):
        return "time of arrival"
    return None


def explain(
    snapshot: TriageSnapshot,
    features: pd.DataFrame,
    shap_values: np.ndarray,
    top_n: int = 3,
) -> list[Driver]:
    """The few facts that most pushed this patient toward a more acute level.

    Only positive contributions are surfaced. A nurse deciding whether to escalate
    needs to know what argues *for* escalation; a list padded with reassuring factors
    reads as balanced and is actually harder to act on.
    """
    contributions = np.asarray(shap_values).ravel()
    order = np.argsort(-contributions)

    drivers: list[Driver] = []
    seen: set[str] = set()
    for i in order:
        if contributions[i] <= 0:
            break
        feature = features.columns[i]
        sentence = _phrase(feature, snapshot)
        # Several features describe the same observation - shock index, HR and its age
        # deviation are all "the pulse". Show the observation once.
        if sentence is None or sentence in seen:
            continue
        seen.add(sentence)
        drivers.append(Driver(feature, float(contributions[i]), sentence))
        if len(drivers) == top_n:
            break
    return drivers


def shap_values_for(booster, features: pd.DataFrame) -> np.ndarray:
    """Per-feature contributions from a LightGBM booster.

    `pred_contrib` returns one column per feature plus a trailing bias term, which is
    dropped here: the bias is the model's base rate, not a fact about this patient.
    """
    contributions = booster.predict(features, pred_contrib=True)
    return np.asarray(contributions)[:, :-1]
