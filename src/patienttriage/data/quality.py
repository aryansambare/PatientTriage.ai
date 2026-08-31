"""Vital-sign plausibility.

Real triage data contains values that are not measurements. NHAMCS 2022 carries a
recorded pulse of 998 and a respiratory rate of 135. Nobody has a heart rate of 998;
these are transcription artefacts, monitor glitches, or a field used for something
other than its label.

There are three things one can do with such a value, and only one of them is honest:

1. Crash on it. This drops the record entirely — and the records with strange values
   are disproportionately the sick ones, so the model is fitted on a cohort that has
   quietly had its hardest patients removed.
2. Pass it through. The model then learns from a heart rate of 998, and at serving
   time a mistyped vital produces a confident, meaningless prediction.
3. Treat it as not recorded, and count how often that happens. An implausible value
   tells us the measurement failed, which is the same epistemic state as no
   measurement — and the count is reported, so the decision stays visible.

This module does (3). Nothing is repaired or interpolated: a failed measurement stays
a failed measurement, and `missing_vitals()` will show it to the nurse as absent.
"""

from __future__ import annotations

from collections import Counter

# Widest range at which a triage measurement is still believable as a measurement.
# Deliberately generous — the aim is to exclude the impossible, not the alarming. A
# systolic of 40 is survivable and must pass; a systolic of 400 is not a reading.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (0.0, 300.0),
    "systolic_bp": (0.0, 300.0),
    "diastolic_bp": (0.0, 250.0),
    "respiratory_rate": (0.0, 120.0),
    "spo2": (0.0, 100.0),
    "temperature_c": (25.0, 45.0),
    "pain_score": (0.0, 10.0),
    "gcs": (3.0, 15.0),
}


class QualityReport(Counter):
    """How many values each field lost to implausibility."""

    def summary(self, total: int) -> str:
        if not self:
            return "no implausible values"
        lines = [
            f"  {field:18} {count:6} discarded ({count / total:.3%} of {total})"
            for field, count in sorted(self.items(), key=lambda kv: -kv[1])
        ]
        return "\n".join(lines)


def clean(field: str, value: float | None, report: QualityReport | None = None) -> float | None:
    """A vital sign, or None if it is absent or not a believable measurement."""
    if value is None:
        return None
    low, high = PLAUSIBLE_RANGES.get(field, (float("-inf"), float("inf")))
    if low <= value <= high:
        return value
    if report is not None:
        report[field] += 1
    return None
