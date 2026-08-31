"""The waiting-room watcher.

Triage produces a judgement at one instant and the patient then sits in a chair, often
for hours, while the judgement stays frozen. Nothing in a standard department recomputes
it. The patient who deteriorates between being triaged and being seen is invisible to
the process by construction — not because anyone was careless, but because no one is
assigned to keep looking.

This module is the part of the system that keeps looking. It watches every patient who
has been triaged but not yet seen, and raises a re-assessment when any of four things
becomes true:

  * **The clock has run out.** Each acuity level carries a time-to-clinician target;
    once a patient passes it, the acuity they were given is no longer being honoured
    and somebody has to know.
  * **The clock is about to run out.** Warning before the breach is what makes it
    actionable rather than a post-hoc audit statistic.
  * **Their vitals have moved the wrong way.** Not "are they abnormal" — abnormal was
    already known at triage — but *worse than when we last looked*, judged against
    this patient's own age band.
  * **They have been waiting a long time with nothing recorded since triage.** A
    patient nobody has re-measured in three hours is not a stable patient; they are an
    unobserved one, and the distinction matters.

Alerts only ever escalate. Nothing here can move a patient down a queue, shorten a
target, or cancel a red flag raised at triage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from patienttriage.clinical import agebands
from patienttriage.features.schema import TriageSnapshot

# Time-to-clinician targets, in minutes, by acuity level. These follow the Australasian
# Triage Scale, which states them explicitly and publishes performance thresholds
# against them; the ESI does not define times, so a department using ESI still needs
# numbers and these are the defensible default. A deployment overrides them.
TIME_TO_CLINICIAN_TARGET: dict[int, int] = {1: 0, 2: 10, 3: 30, 4: 60, 5: 120}

# How often a waiting patient at each level should physically be re-checked. Distinct
# from the target above: one is when they should be *seen*, this is when somebody
# should *look* at them in the meantime.
REASSESSMENT_INTERVAL: dict[int, int] = {1: 5, 2: 15, 3: 30, 4: 60, 5: 60}

APPROACHING_FRACTION = 0.8
"""Warn at 80% of the target. A breach announced only once it happens is an audit
statistic; announced early it is still a decision somebody can make."""


class AlertReason(str, Enum):
    WAIT_BREACH = "wait_breach"
    WAIT_APPROACHING = "wait_approaching"
    VITALS_WORSENING = "vitals_worsening"
    UNOBSERVED = "unobserved"


class ResolutionReason(str, Enum):
    """Why a patient left the *waiting-room* clock. A short, closed list for the same
    reason `service.override.OverrideReason` is one: it keeps the distribution of
    outcomes something the department can actually count, rather than free text
    nobody aggregates.

    Not every reason here means "gone" — see `MOVES_TO_A_BED` below. Leaving the
    waiting room for a monitored bed is a different event from leaving the department
    entirely, and the two should not collapse into one "resolved" bucket: a patient
    who is still in the building, still deteriorating in principle, is not the same
    as one who has left it.
    """

    SENT_FOR_TREATMENT = "sent_for_treatment"
    """Moved to a bed. The common case — the queue did its job. Still monitored: a
    bed is not immunity from deterioration, only relief from the wait-time clock."""

    STABILISED_ON_RECHECK = "stabilised_on_recheck"
    """Re-observed and no longer needs monitored *waiting* — e.g. the vitals that
    triggered a `VITALS_WORSENING` alert have since returned toward normal. Moves to
    a bed for the same reason `SENT_FOR_TREATMENT` does: improved is not discharged."""

    DISCHARGED = "discharged"
    """Seen, treated or triaged-and-safe, and has left. The record closes."""

    TRANSFERRED = "transferred"
    """Sent to another facility. The record closes here — the receiving department
    now owns their monitoring."""

    OTHER = "other"


MOVES_TO_A_BED = frozenset({ResolutionReason.SENT_FOR_TREATMENT, ResolutionReason.STABILISED_ON_RECHECK})
"""Reasons that end the wait but not the watch — see `WaitingPatient.resolve`."""


@dataclass(frozen=True)
class VitalsReading:
    """One set of observations, at one moment."""

    recorded_at: datetime
    heart_rate: float | None = None
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    respiratory_rate: float | None = None
    spo2: float | None = None
    temperature_c: float | None = None
    pain_score: float | None = None
    gcs: int | None = None

    @classmethod
    def from_snapshot(cls, s: TriageSnapshot, recorded_at: datetime) -> VitalsReading:
        return cls(
            recorded_at=recorded_at,
            heart_rate=s.heart_rate,
            systolic_bp=s.systolic_bp,
            diastolic_bp=s.diastolic_bp,
            respiratory_rate=s.respiratory_rate,
            spo2=s.spo2,
            temperature_c=s.temperature_c,
            pain_score=s.pain_score,
            gcs=s.gcs,
        )


@dataclass
class WaitingPatient:
    """A patient who has been triaged and is not yet in front of a clinician."""

    snapshot: TriageSnapshot
    assigned_acuity: int
    arrived_at: datetime
    readings: list[VitalsReading] = field(default_factory=list)
    seen: bool = False

    in_treatment: bool = False
    """True once a nurse has moved this patient to a bed. The wait-time clock they
    were on stops — they are no longer waiting for a space — but `seen` stays False:
    a monitored bed is not the end of watching them, only the end of waiting for one.
    See `resolve` and `WaitingRoomMonitor.check`."""

    resolution_reason: ResolutionReason | None = None
    resolution_note: str = ""
    resolved_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.readings:
            self.readings = [VitalsReading.from_snapshot(self.snapshot, self.arrived_at)]

    def resolve(
        self, reason: ResolutionReason, now: datetime, note: str = ""
    ) -> None:
        """Record what happened to this patient, and update how they're watched.

        Two different things can happen here, and collapsing them into one "off the
        board" flag was the bug this replaced: `SENT_FOR_TREATMENT` and
        `STABILISED_ON_RECHECK` mean the wait is over but the patient is still in the
        building, so `in_treatment` is set and `check()` keeps watching their vitals —
        it just stops timing a wait that has already ended. Every other reason means
        they have actually left the department, so `seen = True` stops all watching,
        exactly as it always has.
        """
        self.resolution_reason = reason
        self.resolution_note = note
        self.resolved_at = now
        if reason in MOVES_TO_A_BED:
            self.in_treatment = True
        else:
            self.seen = True

    def waited_minutes(self, now: datetime) -> float:
        return (now - self.arrived_at).total_seconds() / 60.0

    def minutes_since_observation(self, now: datetime) -> float:
        return (now - self.readings[-1].recorded_at).total_seconds() / 60.0

    def latest_snapshot(self) -> TriageSnapshot:
        """The patient as they are now, not as they arrived.

        Re-scoring the *triage* vitals of a patient whose observations have since
        changed would defeat the entire purpose of monitoring, so the current reading is
        folded back into a snapshot before anything downstream sees it.
        """
        latest = self.readings[-1]
        return self.snapshot.model_copy(
            update={
                "heart_rate": latest.heart_rate,
                "systolic_bp": latest.systolic_bp,
                "diastolic_bp": latest.diastolic_bp,
                "respiratory_rate": latest.respiratory_rate,
                "spo2": latest.spo2,
                "temperature_c": latest.temperature_c,
                "pain_score": latest.pain_score,
                "gcs": latest.gcs,
            }
        )


@dataclass(frozen=True)
class ReassessmentAlert:
    patient_id: str
    reason: AlertReason
    detail: str
    waited_minutes: float
    current_acuity: int

    suggested_acuity: int
    """Never higher than the current level. The watcher escalates or leaves alone."""

    @property
    def escalates(self) -> bool:
        return self.suggested_acuity < self.current_acuity

    def headline(self) -> str:
        arrow = f" -> level {self.suggested_acuity}" if self.escalates else ""
        return f"[{self.reason.value}] {self.detail}{arrow}"


# --------------------------------------------------------------------------------------
# Detecting a patient who is getting worse
# --------------------------------------------------------------------------------------

# Movements large enough to mean something rather than reflect cuff placement, a crying
# child, or the difference between two nurses counting respirations.
SIGNIFICANT_CHANGE: dict[str, float] = {
    "systolic_bp": -15.0,
    "heart_rate": 15.0,
    "respiratory_rate": 4.0,
    "spo2": -3.0,
    "gcs": -1.0,
}


def compare_readings(
    earlier: VitalsReading, later: VitalsReading, age_years: float
) -> list[str]:
    """What has got worse between two readings, in clinical language.

    Direction matters more than absolute value here. A heart rate of 105 tells you
    little; a heart rate of 105 in a patient who was 78 an hour ago tells you they are
    heading somewhere. Both the absolute crossing of an age-appropriate threshold and
    the trend are checked, because either alone misses cases: a patient can deteriorate
    substantially while staying inside "normal", and a patient can be abnormal on
    arrival and stable since.
    """
    findings: list[str] = []

    def moved(field_name: str) -> float | None:
        before, after = getattr(earlier, field_name), getattr(later, field_name)
        if before is None or after is None:
            return None
        return after - before

    if (delta := moved("systolic_bp")) is not None and delta <= SIGNIFICANT_CHANGE["systolic_bp"]:
        findings.append(
            f"systolic pressure fell {abs(delta):.0f} mmHg "
            f"({earlier.systolic_bp:.0f} -> {later.systolic_bp:.0f})"
        )
    if (delta := moved("heart_rate")) is not None and delta >= SIGNIFICANT_CHANGE["heart_rate"]:
        findings.append(
            f"heart rate rose {delta:.0f} ({earlier.heart_rate:.0f} -> {later.heart_rate:.0f})"
        )
    if (
        delta := moved("respiratory_rate")
    ) is not None and delta >= SIGNIFICANT_CHANGE["respiratory_rate"]:
        findings.append(
            f"respiratory rate rose {delta:.0f} "
            f"({earlier.respiratory_rate:.0f} -> {later.respiratory_rate:.0f})"
        )
    if (delta := moved("spo2")) is not None and delta <= SIGNIFICANT_CHANGE["spo2"]:
        findings.append(
            f"oxygen saturation fell {abs(delta):.0f} points "
            f"({earlier.spo2:.0f}% -> {later.spo2:.0f}%)"
        )
    if (delta := moved("gcs")) is not None and delta <= SIGNIFICANT_CHANGE["gcs"]:
        findings.append(f"GCS fell from {earlier.gcs} to {later.gcs}")

    # Absolute crossings, judged against this patient's own age band. A patient can
    # cross into shock without any single step being large enough to flag as a trend.
    if later.systolic_bp is not None:
        threshold = agebands.hypotension_threshold(age_years)
        was_above = earlier.systolic_bp is None or earlier.systolic_bp >= threshold
        if later.systolic_bp < threshold and was_above:
            findings.append(
                f"systolic pressure has crossed below {threshold:.0f}, the shock "
                "threshold for this age"
            )
    if later.heart_rate is not None:
        threshold = agebands.tachycardia_threshold(
            age_years, on_beta_blocker=None
        )
        was_below = earlier.heart_rate is None or earlier.heart_rate <= threshold
        if later.heart_rate > threshold and was_below:
            findings.append(f"heart rate has crossed above {threshold:.0f} for this age")
    if later.spo2 is not None and later.spo2 < 92:
        was_fine = earlier.spo2 is None or earlier.spo2 >= 92
        if was_fine:
            findings.append(f"oxygen saturation has fallen below 92% ({later.spo2:.0f}%)")

    return findings


# --------------------------------------------------------------------------------------
# The monitor
# --------------------------------------------------------------------------------------


class WaitingRoomMonitor:
    """Re-checks every waiting patient on a fixed cadence."""

    def __init__(
        self,
        targets: dict[int, int] | None = None,
        reassessment_interval: dict[int, int] | None = None,
        approaching_fraction: float = APPROACHING_FRACTION,
    ) -> None:
        self.targets = targets or dict(TIME_TO_CLINICIAN_TARGET)
        self.intervals = reassessment_interval or dict(REASSESSMENT_INTERVAL)
        self.approaching_fraction = approaching_fraction

    # ----------------------------------------------------------------------------------

    def target_for(self, acuity: int) -> int:
        return self.targets.get(acuity, max(self.targets.values()))

    def due_at(self, patient: WaitingPatient) -> datetime:
        """When this patient should have been in front of a clinician."""
        return patient.arrived_at + timedelta(minutes=self.target_for(patient.assigned_acuity))

    # ----------------------------------------------------------------------------------

    def check(self, patient: WaitingPatient, now: datetime) -> list[ReassessmentAlert]:
        """Every reason this patient needs looking at again, right now."""
        if patient.seen:
            return []

        alerts: list[ReassessmentAlert] = []
        waited = patient.waited_minutes(now)
        target = self.target_for(patient.assigned_acuity)
        acuity = patient.assigned_acuity

        # --- the clock ------------------------------------------------------------
        # Only for someone actually waiting for a space. A patient already moved to a
        # bed is not late for anything — the clock this alert times has already ended
        # for them, even though the department keeps watching their vitals below.
        if not patient.in_treatment:
            if waited > target:
                over = waited - target
                alerts.append(
                    ReassessmentAlert(
                        patient_id=patient.snapshot.patient_id,
                        reason=AlertReason.WAIT_BREACH,
                        detail=(
                            f"waiting {waited:.0f} min against a {target} min target for "
                            f"level {acuity} — {over:.0f} min over"
                        ),
                        waited_minutes=waited,
                        current_acuity=acuity,
                        # A breached target does not make the patient sicker, so the
                        # level is not changed on the clock alone. What it does is put
                        # them in front of a human, which is what the alert is for.
                        suggested_acuity=acuity,
                    )
                )
            elif target > 0 and waited >= target * self.approaching_fraction:
                alerts.append(
                    ReassessmentAlert(
                        patient_id=patient.snapshot.patient_id,
                        reason=AlertReason.WAIT_APPROACHING,
                        detail=(
                            f"{waited:.0f} of {target} min used for a level {acuity} patient"
                        ),
                        waited_minutes=waited,
                        current_acuity=acuity,
                        suggested_acuity=acuity,
                    )
                )

        # --- the observations ---------------------------------------------------------
        if len(patient.readings) >= 2:
            worse = compare_readings(
                patient.readings[0], patient.readings[-1], patient.snapshot.age_years
            )
            if worse:
                alerts.append(
                    ReassessmentAlert(
                        patient_id=patient.snapshot.patient_id,
                        reason=AlertReason.VITALS_WORSENING,
                        detail="since triage: " + "; ".join(worse),
                        waited_minutes=waited,
                        current_acuity=acuity,
                        # Deterioration is the one trigger that changes the level. A
                        # patient whose observations are moving the wrong way is a
                        # different patient from the one who was triaged, and level 2
                        # is the floor because they now need a monitored space.
                        suggested_acuity=min(acuity, 2),
                    )
                )

        # --- the silence --------------------------------------------------------------
        interval = self.intervals.get(acuity, 60)
        since = patient.minutes_since_observation(now)
        if since > interval * 2 and waited > interval:
            alerts.append(
                ReassessmentAlert(
                    patient_id=patient.snapshot.patient_id,
                    reason=AlertReason.UNOBSERVED,
                    detail=(
                        f"no observations for {since:.0f} min; a level {acuity} patient "
                        f"should be re-checked every {interval} min"
                    ),
                    waited_minutes=waited,
                    current_acuity=acuity,
                    suggested_acuity=acuity,
                )
            )

        return alerts

    # ----------------------------------------------------------------------------------

    def sweep(
        self, patients: list[WaitingPatient], now: datetime, alert_budget: int | None = None
    ) -> list[ReassessmentAlert]:
        """Every alert across the waiting room, most urgent first.

        `alert_budget` caps the list at what one person can actually act on. A board of
        forty names is a board nobody reads, and an unread board costs attention while
        returning nothing — so the cap is part of the design rather than a display
        preference. What gets cut is always the least urgent.
        """
        alerts: list[ReassessmentAlert] = []
        for patient in patients:
            alerts.extend(self.check(patient, now))

        priority = {
            AlertReason.VITALS_WORSENING: 0,
            AlertReason.WAIT_BREACH: 1,
            AlertReason.UNOBSERVED: 2,
            AlertReason.WAIT_APPROACHING: 3,
        }
        alerts.sort(
            key=lambda a: (a.suggested_acuity, priority[a.reason], -a.waited_minutes)
        )
        return alerts if alert_budget is None else alerts[:alert_budget]

    def breach_rate(self, patients: list[WaitingPatient], now: datetime) -> dict[int, float]:
        """Share of waiting patients past their target, by acuity level.

        The department's own performance measure, computed from the same clock the
        alerts use, so the board and the report cannot disagree. A patient already
        moved to a bed is not counted — they are not waiting for anything, so they
        cannot be late for it.
        """
        by_level: dict[int, list[bool]] = {}
        for patient in patients:
            if patient.seen or patient.in_treatment:
                continue
            breached = patient.waited_minutes(now) > self.target_for(patient.assigned_acuity)
            by_level.setdefault(patient.assigned_acuity, []).append(breached)
        return {
            level: sum(flags) / len(flags) for level, flags in sorted(by_level.items())
        }
