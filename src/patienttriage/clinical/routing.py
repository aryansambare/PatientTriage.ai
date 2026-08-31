"""Where a patient goes, not just how urgently they need to be seen.

An acuity level answers "how fast." It does not answer "to where" - and those are
different questions with different failure modes. A level-1 patient sent to a chair
because nobody named the resuscitation bay wastes exactly the minutes acuity alone
was meant to protect.

Deliberately NOT modelled here: live bed occupancy. This system has no feed for which
beds are physically free right now, and a recommendation built on a fabricated
occupancy count would be worse than none - it would look authoritative while being
invented. What this module offers instead is the *zone* a patient of this acuity
normally goes to, plus anything about this specific patient or site that should make a
charge nurse look twice before sending them there - a capability the site does not
have, or a target time the zone will not actually meet.
"""

from __future__ import annotations

from dataclasses import dataclass

from patienttriage.config.profile import HospitalProfile
from patienttriage.monitoring.waitingroom import TIME_TO_CLINICIAN_TARGET


@dataclass(frozen=True)
class Zone:
    name: str
    bed_type: str


# The five zones, in the same order and language as the ESI handbook's own
# disposition guidance - not a house style invented for this project.
ZONES: dict[int, Zone] = {
    1: Zone("Resuscitation bay", "resus monitored bed"),
    2: Zone("Acute monitored bay", "monitored acute bed"),
    3: Zone("Monitored waiting area", "monitored chair or bed"),
    4: Zone("Fast-track clinic", "treatment recliner"),
    5: Zone("Non-urgent clinic", "consult room"),
}


@dataclass(frozen=True)
class RoutingRecommendation:
    acuity: int
    zone: str
    bed_type: str
    target_minutes: int
    note: str | None = None


def recommend_destination(
    acuity: int,
    *,
    age_years: float | None = None,
    is_pregnant: bool | None = None,
    profile: HospitalProfile | None = None,
) -> RoutingRecommendation:
    """The zone for this acuity, adjusted for what this specific patient and site need
    said out loud rather than silently routed around.

    Capability gaps are reported, never resolved by this function - the same stance
    `HospitalProfile.capability_warning` already takes for acuity scoring. A site
    without an obstetric service still receives a patient in pregnancy-related
    distress; the useful thing this can do is name that early, not pick a zone that
    pretends the gap does not exist.
    """
    zone = ZONES.get(acuity, ZONES[5])
    note: str | None = None

    if profile is not None:
        if age_years is not None and not profile.supports(age_years):
            note = profile.capability_warning(age_years)
        elif is_pregnant and not profile.has_obstetric_service:
            note = (
                "no obstetric service configured at this site - route to the nearest "
                "obstetric unit if the presentation is pregnancy-related"
            )
        elif acuity == 1 and not profile.is_trauma_centre:
            note = (
                f"{profile.name} is not a designated trauma centre - if this is a "
                "major trauma mechanism, activate the regional transfer pathway "
                "alongside resuscitation here"
            )

    return RoutingRecommendation(
        acuity=acuity,
        zone=zone.name,
        bed_type=zone.bed_type,
        target_minutes=TIME_TO_CLINICIAN_TARGET.get(acuity, 120),
        note=note,
    )
