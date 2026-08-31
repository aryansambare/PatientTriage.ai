"""Age-stratified physiology — the single source of truth for "normal".

A tachycardia threshold of 130 is correct for a 40-year-old, dangerous nonsense for a
toddler whose resting heart rate is 130, and dangerously *lax* for a 90-year-old on a
beta-blocker. These numbers used to be scattered between the rule engine and the
feature builder, in two slightly different forms. That is exactly the "single
adult-calibrated scoring model" failure mode: not a wrong number anybody chose, but a
right number quietly applied to the wrong population.

Everything age-dependent now lives here, once, and both the rules and the features read
it. Ranges follow the PALS/APLS paediatric vital-sign bands and standard adult
emergency criteria. They are a defensible starting set, not a local protocol: a
deployment is expected to review them with its clinical lead, and the structure makes
that a table review rather than a code audit.

Three age effects encoded beyond the ranges themselves, because each one kills people
and none is captured by a threshold table alone:

  * **Children compensate, then crash.** A child maintains blood pressure until they
    are profoundly unwell, so hypotension in a child is a pre-arrest sign rather than
    an early warning. Tachycardia and tachypnoea have to carry the load instead.
  * **The elderly mount a blunted fever response.** A septic 85-year-old is often
    normothermic and not infrequently hypothermic. Absence of fever excludes nothing,
    and low temperature is the more alarming finding.
  * **Neonatal fever is an emergency at any height.** Under 28 days, 38.0C is a full
    sepsis workup regardless of how well the baby looks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

DAY = 1 / 365.25
MONTH = 1 / 12


class AgeBand(str, Enum):
    """Clinically meaningful age strata, not evenly spaced ones."""

    NEONATE = "neonate"
    INFANT = "infant"
    TODDLER = "toddler"
    PRESCHOOL = "preschool"
    SCHOOL_AGE = "school_age"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    OLDER_ADULT = "older_adult"
    GERIATRIC = "geriatric"


@dataclass(frozen=True)
class BandPhysiology:
    """What normal, and what dangerous, look like for one age band."""

    band: AgeBand
    lower_age: float
    upper_age: float

    heart_rate_normal: tuple[float, float]
    respiratory_rate_normal: tuple[float, float]

    tachycardia_danger: float
    """Above this, the rate itself is a red flag."""

    bradycardia_danger: float
    """Below this, the rate itself is a red flag."""

    hypotension: float
    """Systolic pressure defining shock for this band."""

    tachypnoea_danger: float
    bradypnoea_danger: float

    fever: float
    """Temperature counted as febrile."""

    hypothermia: float

    compensates_before_crashing: bool
    """True where hypotension is a late, pre-arrest sign rather than an early warning."""

    blunted_fever_response: bool
    """True where absence of fever does not argue against serious infection."""


# Ordered youngest to oldest; `band_for` walks this list.
BANDS: tuple[BandPhysiology, ...] = (
    BandPhysiology(
        AgeBand.NEONATE, 0.0, 28 * DAY,
        heart_rate_normal=(100, 205), respiratory_rate_normal=(30, 60),
        tachycardia_danger=205, bradycardia_danger=100,
        hypotension=60,
        tachypnoea_danger=60, bradypnoea_danger=30,
        # Any fever under 28 days triggers a full workup, so the bar is the
        # measurement threshold itself rather than a level of concern.
        fever=38.0, hypothermia=36.5,
        compensates_before_crashing=True, blunted_fever_response=False,
    ),
    BandPhysiology(
        AgeBand.INFANT, 28 * DAY, 1.0,
        heart_rate_normal=(100, 180), respiratory_rate_normal=(30, 53),
        tachycardia_danger=180, bradycardia_danger=100,
        hypotension=70,
        tachypnoea_danger=60, bradypnoea_danger=25,
        fever=38.0, hypothermia=36.0,
        compensates_before_crashing=True, blunted_fever_response=False,
    ),
    BandPhysiology(
        AgeBand.TODDLER, 1.0, 3.0,
        heart_rate_normal=(98, 140), respiratory_rate_normal=(22, 37),
        tachycardia_danger=160, bradycardia_danger=80,
        hypotension=72,  # 70 + 2*age, evaluated at the band floor
        tachypnoea_danger=45, bradypnoea_danger=18,
        fever=38.5, hypothermia=36.0,
        compensates_before_crashing=True, blunted_fever_response=False,
    ),
    BandPhysiology(
        AgeBand.PRESCHOOL, 3.0, 6.0,
        heart_rate_normal=(80, 120), respiratory_rate_normal=(20, 28),
        tachycardia_danger=140, bradycardia_danger=70,
        hypotension=76,
        tachypnoea_danger=40, bradypnoea_danger=16,
        fever=38.5, hypothermia=36.0,
        compensates_before_crashing=True, blunted_fever_response=False,
    ),
    BandPhysiology(
        AgeBand.SCHOOL_AGE, 6.0, 12.0,
        heart_rate_normal=(75, 118), respiratory_rate_normal=(18, 25),
        tachycardia_danger=130, bradycardia_danger=60,
        hypotension=82,
        tachypnoea_danger=35, bradypnoea_danger=14,
        fever=38.5, hypothermia=36.0,
        compensates_before_crashing=True, blunted_fever_response=False,
    ),
    BandPhysiology(
        AgeBand.ADOLESCENT, 12.0, 18.0,
        heart_rate_normal=(60, 100), respiratory_rate_normal=(12, 20),
        tachycardia_danger=130, bradycardia_danger=45,
        hypotension=90,
        tachypnoea_danger=30, bradypnoea_danger=10,
        fever=38.5, hypothermia=36.0,
        compensates_before_crashing=True, blunted_fever_response=False,
    ),
    BandPhysiology(
        AgeBand.ADULT, 18.0, 65.0,
        heart_rate_normal=(60, 100), respiratory_rate_normal=(12, 20),
        tachycardia_danger=130, bradycardia_danger=40,
        hypotension=90,
        tachypnoea_danger=30, bradypnoea_danger=8,
        fever=38.0, hypothermia=35.0,
        compensates_before_crashing=False, blunted_fever_response=False,
    ),
    BandPhysiology(
        AgeBand.OLDER_ADULT, 65.0, 80.0,
        heart_rate_normal=(60, 100), respiratory_rate_normal=(12, 22),
        tachycardia_danger=120, bradycardia_danger=40,
        # Older patients are commonly hypertensive at baseline, so a "normal" 100
        # can represent a substantial fall. The bar for concern rises with age.
        hypotension=100,
        tachypnoea_danger=28, bradypnoea_danger=8,
        fever=37.8, hypothermia=35.5,
        compensates_before_crashing=False, blunted_fever_response=True,
    ),
    BandPhysiology(
        AgeBand.GERIATRIC, 80.0, 200.0,
        heart_rate_normal=(60, 100), respiratory_rate_normal=(12, 22),
        tachycardia_danger=110, bradycardia_danger=40,
        hypotension=100,
        tachypnoea_danger=26, bradypnoea_danger=8,
        # A frail 85-year-old with sepsis often never reaches 38.0C. Lowering the
        # bar is the point: the alternative is a septic patient who "has no fever".
        fever=37.5, hypothermia=35.5,
        compensates_before_crashing=False, blunted_fever_response=True,
    ),
)

_BY_BAND: dict[AgeBand, BandPhysiology] = {b.band: b for b in BANDS}
BAND_NAMES: tuple[str, ...] = tuple(b.band.value for b in BANDS)


def band_for(age_years: float) -> BandPhysiology:
    """The physiology that applies to this patient."""
    for physiology in BANDS:
        if age_years < physiology.upper_age:
            return physiology
    return BANDS[-1]


def band_name(age_years: float) -> str:
    return band_for(age_years).band.value


def is_paediatric(age_years: float) -> bool:
    return age_years < 18.0


def is_geriatric(age_years: float) -> bool:
    return age_years >= 65.0


# --------------------------------------------------------------------------------------
# Threshold accessors, with the modifiers that thresholds alone cannot express
# --------------------------------------------------------------------------------------


def tachycardia_threshold(age_years: float, *, on_beta_blocker: bool | None = None) -> float:
    """Heart rate above which the rate itself is a red flag.

    Lowered on a beta-blocker, because such a patient physically cannot mount a
    tachycardic response. A "reassuring" 110 in a shocked patient on metoprolol is
    itself the alarming finding, and a fixed threshold would never fire.
    """
    threshold = band_for(age_years).tachycardia_danger
    return threshold - 20.0 if on_beta_blocker else threshold


def bradycardia_threshold(age_years: float) -> float:
    return band_for(age_years).bradycardia_danger


def hypotension_threshold(age_years: float) -> float:
    """Systolic pressure defining shock.

    Between one and ten years the paediatric formula 70 + 2 x age is finer-grained than
    a band, so it is applied directly and the band value is the fallback.
    """
    if 1.0 <= age_years < 10.0:
        return 70.0 + 2.0 * age_years
    return band_for(age_years).hypotension


def fever_threshold(age_years: float) -> float:
    return band_for(age_years).fever


def hypothermia_threshold(age_years: float) -> float:
    return band_for(age_years).hypothermia


def tachypnoea_threshold(age_years: float) -> float:
    return band_for(age_years).tachypnoea_danger


def bradypnoea_threshold(age_years: float) -> float:
    return band_for(age_years).bradypnoea_danger


def deviation(value: float | None, normal: tuple[float, float]) -> float | None:
    """How far outside the normal band, in band-widths. Zero inside, signed outside.

    Expressing abnormality relative to the patient's own band is what lets one model
    serve a neonate and a pensioner: HR 150 is 0.0 for an infant and strongly positive
    for an adult, so the feature means "abnormal for this patient" rather than
    "abnormal for a notional forty-year-old".
    """
    if value is None:
        return None
    low, high = normal
    width = high - low
    if value < low:
        return (value - low) / width
    if value > high:
        return (value - high) / width
    return 0.0


def heart_rate_deviation(value: float | None, age_years: float) -> float | None:
    return deviation(value, band_for(age_years).heart_rate_normal)


def respiratory_rate_deviation(value: float | None, age_years: float) -> float | None:
    return deviation(value, band_for(age_years).respiratory_rate_normal)


def fever_significance(age_years: float, temperature_c: float | None) -> str | None:
    """Why this temperature matters *for this patient*, in words a nurse would use.

    The brief's own example: 38.5C in a three-year-old versus a seventy-five-year-old.
    The number is identical and the clinical meaning is not, so the system says which
    meaning applies rather than reporting the number twice.
    """
    if temperature_c is None:
        return None
    physiology = band_for(age_years)

    if temperature_c <= physiology.hypothermia:
        if physiology.blunted_fever_response:
            return (
                f"temperature {temperature_c:.1f}C - hypothermia in an older patient is "
                "more suggestive of sepsis than a fever would be"
            )
        return f"temperature {temperature_c:.1f}C - hypothermia"

    if temperature_c < physiology.fever:
        if physiology.blunted_fever_response:
            return (
                f"temperature {temperature_c:.1f}C is not febrile, but older patients "
                "often mount no fever when septic - this does not exclude infection"
            )
        return None

    if physiology.band is AgeBand.NEONATE:
        return (
            f"temperature {temperature_c:.1f}C in a neonate - full sepsis workup "
            "regardless of how well the baby appears"
        )
    if physiology.band is AgeBand.INFANT and age_years < 90 * DAY:
        return (
            f"temperature {temperature_c:.1f}C under 90 days of age - serious "
            "bacterial infection cannot be excluded clinically"
        )
    if physiology.blunted_fever_response:
        return (
            f"temperature {temperature_c:.1f}C in an older patient - a lower bar than "
            "in a younger adult, and more likely to represent serious infection"
        )
    if is_paediatric(age_years):
        return (
            f"temperature {temperature_c:.1f}C - common in this age group and usually "
            "self-limiting; significant in combination with other findings"
        )
    return f"temperature {temperature_c:.1f}C - febrile"
