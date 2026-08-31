"""Reason for Visit Classification (RVC) codes to band features.

NHAMCS does not record a free-text chief complaint. It records up to three RVC codes
chosen by a coder from the NCHS Reason for Visit Classification — five-digit numbers
such as 10501 or 15451. That is a real property of the dataset, not a nuisance to be
papered over, and it has two consequences worth stating plainly.

**First: the text branch cannot be trained on NHAMCS.** Only MIMIC-IV-ED carries free
text. NHAMCS trains the tabular branch and validates the rule layer's vital-sign
rules; the text branch waits for credentialing.

**Second: RVC codes must not drive the rule layer.** What NHAMCS gives reliably is the
module and body-system *band* — the first digit selects the module, the next two the
body system. That is structural and documented, but it is coarse: the injury module
spans a sprained ankle and a gunshot wound; the psychological band spans mild anxiety
and active suicidal intent. Wiring bands into red flags would escalate every sprained
ankle to level 2, and a layer that cries wolf on routine visits is one staff learn to
click past — the alert-fatigue failure mode that makes a triage assistant worse than
no assistant.

So bands feed the *model*, which can learn how much they are worth, and never the
*rules*, which must be right every time. Resolving individual codes needs the
published RVC codebook; codes are not guessed at here, because a guessed mapping that
mislabels a stroke as a general symptom would look like it was working.
"""

from __future__ import annotations

from collections.abc import Iterable

# Module bands (leading digits of the five-digit code).
_MODULES: tuple[tuple[str, range], ...] = (
    ("disease", range(20000, 30000)),
    ("diagnostic", range(30000, 40000)),
    ("treatment", range(40000, 50000)),
    ("injury", range(50000, 58000)),
    ("test_result", range(58000, 59000)),
    ("administrative", range(60000, 70000)),
    ("uncodable", range(89900, 90000)),
)

# Body-system bands within the symptom module (10000-19999).
_SYMPTOM_BANDS: tuple[tuple[str, range], ...] = (
    ("symptom_general", range(10000, 11000)),
    ("symptom_psych", range(11000, 12000)),
    ("symptom_nervous", range(12000, 13000)),
    ("symptom_cardiovascular", range(13000, 14000)),
    ("symptom_eyes_ears", range(14000, 15000)),
    ("symptom_respiratory", range(15000, 16000)),
    ("symptom_digestive", range(16000, 17000)),
    ("symptom_genitourinary", range(17000, 18000)),
    ("symptom_skin", range(18000, 19000)),
    ("symptom_musculoskeletal", range(19000, 20000)),
)

RVC_BANDS: tuple[str, ...] = (
    *(name for name, _ in _SYMPTOM_BANDS),
    *(name for name, _ in _MODULES),
)
"""Every band a code can fall into. Frozen, so the feature columns are stable."""


def _as_int(code: str | int) -> int | None:
    try:
        return int(str(code).strip())
    except (ValueError, TypeError):
        return None


def band_of(code: str | int) -> str | None:
    """The band a single RVC code belongs to, or None if it does not parse."""
    value = _as_int(code)
    if value is None:
        return None
    for name, band in _SYMPTOM_BANDS:
        if value in band:
            return name
    for name, band in _MODULES:
        if value in band:
            return name
    return None


def band_features(codes: Iterable[str | int]) -> dict[str, float]:
    """One indicator per band. A visit can touch several — three codes are recorded."""
    present = {band_of(code) for code in codes}
    return {f"chief_complaint_codes__{band}": float(band in present) for band in RVC_BANDS}
