"""Clinician overrides.

An override is the system working, not failing. The whole architecture rests on a
licensed clinician being able to disregard the recommendation instantly and without
argument, so the capture path is built to make that *easy* — one tap, a reason picked
from a short list, no free text required, no confirmation dialogue, no friction that
would tempt anyone to work around the tool instead of through it.

What makes an override worth capturing rather than merely permitting:

  * **Accountability.** Under both HIPAA and the GDPR, a clinical decision influenced
    by an automated system has to be attributable to the human who made it. The record
    is what makes the clinician the decision-maker rather than the model.
  * **It is the best training data in the building.** A disagreement between a
    recommendation and an experienced nurse marks precisely where the model is wrong,
    on a real patient, judged by someone who saw them. Random labels cost money;
    these arrive free and pre-targeted.
  * **It is the trust signal.** An override rate that stays flat says the tool is being
    used. One that collapses toward zero usually means staff have stopped reading the
    recommendation and are clicking through it — the failure mode that looks like
    success on every adoption dashboard.

Nothing here blocks. `capture` records what happened; the nurse's number is already
the number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum


class OverrideReason(str, Enum):
    """A short list, because a long one gets answered with whatever is first.

    Each maps to a distinct model failure, so the distribution is diagnostic on its
    own: a spike in CLINICAL_GESTALT means the model is missing a signal humans can
    see, while a spike in DATA_WRONG means the problem is upstream in the feed.
    """

    CLINICAL_GESTALT = "clinical_gestalt"
    """The patient looks worse, or better, than the numbers do."""

    HISTORY_NOT_IN_SYSTEM = "history_not_in_system"
    """The clinician knows something the record does not contain."""

    DATA_WRONG = "data_wrong"
    """A vital was mis-recorded, the cuff was wrong, the age is wrong."""

    COMPLAINT_MISREAD = "complaint_misread"
    """The system read the presenting complaint incorrectly."""

    DEPARTMENT_CONTEXT = "department_context"
    """Capacity, staffing or resource reality the model cannot see."""

    PATIENT_PREFERENCE = "patient_preference"
    """The patient's own wishes, including ceilings of treatment."""

    OTHER = "other"


@dataclass(frozen=True)
class Override:
    """One clinician disagreeing with one recommendation."""

    patient_id: str
    clinician_id: str
    recommended_acuity: int | None
    """None when the system abstained — overriding an abstention is still an override."""

    assigned_acuity: int
    reason: OverrideReason
    note: str
    recorded_at: str
    system_abstained: bool
    rule_hits: list[str]
    """Red flags standing at the time, so a review can see what was overridden."""

    @property
    def direction(self) -> str:
        if self.recommended_acuity is None:
            return "resolved_abstention"
        if self.assigned_acuity < self.recommended_acuity:
            return "escalated"
        if self.assigned_acuity > self.recommended_acuity:
            return "de_escalated"
        return "agreed"

    @property
    def overrode_a_red_flag(self) -> bool:
        """A clinician setting a level less acute than a deterministic rule demanded.

        Entirely legitimate — a nurse can see that the "arrest" in the handover was the
        patient describing a relative's. But it is the one override worth surfacing for
        routine review, because the rule layer is the system's safety floor.
        """
        return bool(self.rule_hits) and self.direction == "de_escalated"

    def to_payload(self) -> dict[str, object]:
        return asdict(self) | {
            # The enum serialises as its value, not its repr: an audit trail that
            # records "OverrideReason.CLINICAL_GESTALT" is recording a Python detail
            # rather than a clinical fact, and it will not survive a language change.
            "reason": self.reason.value,
            "direction": self.direction,
            "overrode_a_red_flag": self.overrode_a_red_flag,
        }


def capture(
    assessment,
    assigned_acuity: int,
    clinician_id: str,
    reason: OverrideReason = OverrideReason.OTHER,
    note: str = "",
    now: datetime | None = None,
) -> Override:
    """Record a clinician's decision against what the system suggested.

    Called for every triage decision, not only disagreements: `direction` will read
    "agreed" when they match. Recording agreement costs one row and is what makes the
    override rate measurable at all — a numerator with no denominator says nothing.
    """
    moment = now or datetime.now(UTC)
    return Override(
        patient_id=assessment.patient_id,
        clinician_id=clinician_id,
        recommended_acuity=assessment.recommended_acuity,
        assigned_acuity=assigned_acuity,
        reason=reason,
        note=note,
        recorded_at=moment.isoformat(),
        system_abstained=assessment.abstained,
        rule_hits=[h.rule_id for h in assessment.rule_hits],
    )


def override_rate(overrides: list[Override]) -> dict[str, float | int]:
    """Adoption and safety, from the same rows.

    Reported together because they are the same measurement read two ways. A department
    where nobody ever disagrees is not a department with a perfect model.
    """
    if not overrides:
        return {"n": 0}

    total = len(overrides)
    disagreements = [o for o in overrides if o.direction != "agreed"]
    return {
        "n": total,
        "override_rate": len(disagreements) / total,
        "escalated": sum(o.direction == "escalated" for o in overrides) / total,
        "de_escalated": sum(o.direction == "de_escalated" for o in overrides) / total,
        "resolved_abstentions": sum(o.direction == "resolved_abstention" for o in overrides),
        "red_flags_overridden": sum(o.overrode_a_red_flag for o in overrides),
    }


def reasons_breakdown(overrides: list[Override]) -> dict[str, int]:
    """Which failure the model is actually making, ranked."""
    counts: dict[str, int] = {}
    for override in overrides:
        if override.direction != "agreed":
            counts[override.reason.value] = counts.get(override.reason.value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
