"""NEWS2 — the National Early Warning Score, as a second, externally validated opinion.

Every other severity signal in this system is bespoke: the rule engine's thresholds are
age-stratified and cite their own sources, and the ordinal model is trained on this
department's own outcomes. NEWS2 (Royal College of Physicians, 2017) is neither — it is
a single, fixed, widely deployed six-parameter score that a nurse anywhere in a NEWS2
jurisdiction already knows by name and already trusts. Reporting it alongside our own
number costs one small function and buys a sanity check a nurse can verify against a
scoring system she did not have to learn from us.

It is reported, never decided on. Nothing here touches `recommended_acuity` or the rule
floor — see `service.pipeline`. Two honest gaps, both because `TriageSnapshot` does not
carry the field NEWS2 wants:

  * **Consciousness** should be ACVPU (Alert / new Confusion / Voice / Pain /
    Unresponsive). We only have GCS, so `gcs < 15` stands in for "not simply alert" —
    close, but a patient scored GCS 14 for a chronic, longstanding reason reads as an
    acute change under this substitution when they are not.
  * **Inspired oxygen** is not tracked at all, so the "on supplemental oxygen" +2
    parameter is always scored as room air. A patient already on oxygen at triage will
    score lower here than the chart on their bed will show.

And one restriction NEWS2 itself states rather than one this implementation adds:
Scale 1 is derived from, and validated on, adults. It is not applied to anyone under
18, and not to a pregnancy, where NHS guidance points to an obstetric-specific score
(MEOWS) instead of NEWS2 Scale 1's non-pregnant reference ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from patienttriage.clinical import agebands

RiskBand = str  # "low" | "low-medium" | "medium" | "high"


def is_applicable(age_years: float, is_pregnant: bool | None) -> bool:
    """Whether NEWS2 Scale 1 is the right tool for this patient at all."""
    return not agebands.is_paediatric(age_years) and not is_pregnant


def _respiratory_rate_score(rr: float | None) -> int:
    if rr is None:
        return 0
    if rr <= 8 or rr >= 25:
        return 3
    if rr >= 21:
        return 2
    if rr <= 11:
        return 1
    return 0


def _spo2_score(spo2: float | None) -> int:
    if spo2 is None:
        return 0
    if spo2 <= 91:
        return 3
    if spo2 <= 93:
        return 2
    if spo2 <= 95:
        return 1
    return 0


def _systolic_bp_score(sbp: float | None) -> int:
    if sbp is None:
        return 0
    if sbp <= 90 or sbp >= 220:
        return 3
    if sbp <= 100:
        return 2
    if sbp <= 110:
        return 1
    return 0


def _heart_rate_score(hr: float | None) -> int:
    if hr is None:
        return 0
    if hr <= 40 or hr >= 131:
        return 3
    if hr >= 111:
        return 2
    if hr <= 50 or hr >= 91:
        return 1
    return 0


def _consciousness_score(gcs: int | None) -> int:
    """Stands in for ACVPU. See the module docstring for what this substitution costs."""
    return 3 if gcs is not None and gcs < 15 else 0


def _temperature_score(temp_c: float | None) -> int:
    if temp_c is None:
        return 0
    if temp_c <= 35.0:
        return 3
    if temp_c >= 39.1:
        return 2
    if temp_c <= 36.0 or temp_c >= 38.1:
        return 1
    return 0


@dataclass(frozen=True)
class News2Result:
    total: int
    max_single_parameter: int
    """The single highest-scoring parameter. NHS guidance escalates the response for
    any parameter scoring 3 on its own, even when the total looks unremarkable —
    exactly the isolated-derangement failure our own rule engine also guards against
    (R15-R17), so the same principle is honoured here rather than only totalled away."""
    risk_band: RiskBand

    @property
    def urgent(self) -> bool:
        return self.risk_band != "low"


def score(
    *,
    heart_rate: float | None,
    respiratory_rate: float | None,
    spo2: float | None,
    systolic_bp: float | None,
    temperature_c: float | None,
    gcs: int | None,
) -> News2Result:
    """The NEWS2 Scale 1 total and risk band for one set of vitals.

    Every missing parameter scores 0 rather than raising — matching how the score is
    actually used at a bedside, where a chart with gaps still gets a number written on
    it. `Assessment.data_completeness()` is what tells the nurse those gaps exist; this
    function does not repeat that warning, it would just be read as a second one.
    """
    parts = (
        _respiratory_rate_score(respiratory_rate),
        _spo2_score(spo2),
        _systolic_bp_score(systolic_bp),
        _heart_rate_score(heart_rate),
        _consciousness_score(gcs),
        _temperature_score(temperature_c),
    )
    total = sum(parts)
    highest = max(parts)

    # The aggregate tier takes priority whenever it is already the more urgent call;
    # a lone maxed-out parameter only ever pulls an otherwise-low total up to
    # "low-medium", it never pulls a medium or high total back down.
    if total >= 7:
        band: RiskBand = "high"
    elif total >= 5:
        band = "medium"
    elif highest >= 3:
        band = "low-medium"
    else:
        band = "low"

    return News2Result(total=total, max_single_parameter=highest, risk_band=band)
