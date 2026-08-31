"""Data protection, retention, and what an override must legally record.

**Assumed jurisdiction: HIPAA (United States), with a GDPR profile provided.** The
brief asks for this to be stated, and it is not a formality — the two regimes disagree
about things that change the code. Under the GDPR a patient may demand erasure and may
object to solely automated decision-making; under HIPAA neither right exists in that
form, but the minimum-necessary rule and the six-year audit retention do. A single
"privacy module" that ignores the difference is compliant nowhere.

Three ideas run through this module.

**The audit trail is clinical evidence, not telemetry.** It outlives the patient
record's operational usefulness because it may be needed years later in a review or a
claim. It is therefore retained the longest and erased the most reluctantly.

**The model never needs identity.** Nothing in the feature set is a name, an address or
a medical record number — the model reads vitals, an age, a complaint. So the scoring
path can run on a de-identified projection, and identity stays in the presentation
layer where a nurse needs it. That is minimum-necessary implemented rather than
promised.

**Automated decision-making is why the human is in the loop.** GDPR Article 22 gives a
data subject the right not to be subject to a decision based solely on automated
processing where it significantly affects them. This system never makes such a
decision: a licensed clinician sets every acuity. The architecture's central constraint
is therefore also its legal basis, which is a good sign that the constraint is the
right one.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from patienttriage.features.schema import TriageSnapshot


class Jurisdiction(str, Enum):
    HIPAA = "HIPAA"
    GDPR = "GDPR"


class LawfulBasis(str, Enum):
    """Why we are allowed to process this at all."""

    VITAL_INTERESTS = "vital_interests"
    """GDPR Art. 6(1)(d) / 9(2)(c) — processing necessary to protect life where the
    patient cannot give consent. The unconscious arrival, and the reason consent is the
    wrong model for emergency triage."""

    HEALTHCARE_PROVISION = "healthcare_provision"
    """GDPR Art. 9(2)(h) — medical diagnosis and provision of care."""

    TREATMENT_OPERATIONS = "treatment_operations"
    """HIPAA 45 CFR 164.506 — treatment, payment and healthcare operations, which does
    not require separate authorisation."""


@dataclass(frozen=True)
class RetentionPolicy:
    """How long each class of record is kept, and why.

    Different clocks on purpose. Deleting the audit trail on the same schedule as the
    working record would destroy the evidence exactly when a late-surfacing complaint
    needs it.
    """

    jurisdiction: Jurisdiction
    lawful_basis: LawfulBasis

    audit_days: int
    triage_record_days: int
    model_training_days: int
    waiting_room_telemetry_days: int

    supports_erasure: bool
    """Whether a data subject can compel deletion (GDPR Art. 17)."""

    requires_dpia: bool
    """Whether a data protection impact assessment is required before deployment."""

    def expires_at(self, category: str, created: datetime) -> datetime:
        days = {
            "audit": self.audit_days,
            "triage_record": self.triage_record_days,
            "model_training": self.model_training_days,
            "waiting_room_telemetry": self.waiting_room_telemetry_days,
        }[category]
        return created + timedelta(days=days)

    def is_expired(self, category: str, created: datetime, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at(category, created)

    def erasable_categories(self) -> tuple[str, ...]:
        """What a valid erasure request can actually reach.

        Under the GDPR the audit trail generally survives an erasure request: Art.
        17(3) carves out processing necessary for public-interest health purposes and
        for establishing or defending legal claims. A patient can have their training
        data removed and their operational record redacted; they cannot erase the
        record of a clinical decision made about them. Saying so plainly here is
        better than discovering it during an incident.
        """
        if not self.supports_erasure:
            return ()
        return ("model_training", "waiting_room_telemetry")


HIPAA_POLICY = RetentionPolicy(
    jurisdiction=Jurisdiction.HIPAA,
    lawful_basis=LawfulBasis.TREATMENT_OPERATIONS,
    # 45 CFR 164.316(b)(2)(i): six years for documentation of policies and actions.
    audit_days=6 * 365,
    triage_record_days=6 * 365,
    model_training_days=3 * 365,
    waiting_room_telemetry_days=90,
    supports_erasure=False,
    requires_dpia=False,
)

GDPR_POLICY = RetentionPolicy(
    jurisdiction=Jurisdiction.GDPR,
    lawful_basis=LawfulBasis.HEALTHCARE_PROVISION,
    audit_days=10 * 365,
    triage_record_days=8 * 365,
    # Storage limitation (Art. 5(1)(e)) bites hardest on the secondary use, so the
    # training corpus has the shortest clock of the three durable categories.
    model_training_days=2 * 365,
    waiting_room_telemetry_days=30,
    supports_erasure=True,
    requires_dpia=True,
)

POLICIES: dict[str, RetentionPolicy] = {
    Jurisdiction.HIPAA.value: HIPAA_POLICY,
    Jurisdiction.GDPR.value: GDPR_POLICY,
}


def policy_for(jurisdiction: str) -> RetentionPolicy:
    """The policy for a named jurisdiction, defaulting to the stricter one.

    An unrecognised jurisdiction gets GDPR rules. Being needlessly strict is a
    deployment inconvenience; being accidentally lax is a notifiable breach.
    """
    return POLICIES.get(jurisdiction, GDPR_POLICY)


# --------------------------------------------------------------------------------------
# Minimum necessary: the model never sees identity
# --------------------------------------------------------------------------------------

DIRECT_IDENTIFIERS = ("patient_id",)
"""Fields that identify a person. Everything else in a snapshot is clinical."""


def pseudonymise(snapshot: TriageSnapshot, secret: bytes) -> TriageSnapshot:
    """Replace the identifier with a keyed hash.

    Keyed rather than plain: a bare SHA-256 of a medical record number is trivially
    reversible by anyone who can enumerate the number space, which for an MRN is a
    small and guessable set. With an HMAC, an attacker who obtains the stored value
    still cannot link it back without the key, and the key lives outside the analytics
    store. This is pseudonymisation in the GDPR Art. 4(5) sense — still personal data,
    but no longer attributable without additional information held separately.
    """
    token = hmac.new(secret, snapshot.patient_id.encode("utf-8"), hashlib.sha256)
    return snapshot.model_copy(update={"patient_id": f"psn-{token.hexdigest()[:16]}"})


def scoring_projection(snapshot: TriageSnapshot) -> dict[str, object]:
    """Exactly what the model is given — demonstrably free of identity.

    Useful as a control as much as a transformation: a reviewer can read the output of
    this function and confirm for themselves that no identifier reaches the model,
    rather than taking an architecture diagram's word for it.
    """
    payload = snapshot.model_dump()
    for identifier in DIRECT_IDENTIFIERS:
        payload.pop(identifier, None)
    return payload


# --------------------------------------------------------------------------------------
# Access control
# --------------------------------------------------------------------------------------


class Role(str, Enum):
    TRIAGE_NURSE = "triage_nurse"
    CHARGE_NURSE = "charge_nurse"
    CLINICIAN = "clinician"
    DATA_SCIENTIST = "data_scientist"
    AUDITOR = "auditor"
    SYSTEM = "system"


#: What each role may reach. The data scientist deliberately cannot see identified
#: patients — model work does not require knowing who anyone is, and the surest way to
#: prevent a re-identification incident is to not grant the access in the first place.
PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.TRIAGE_NURSE: frozenset({"assess", "view_identified", "override", "view_watchboard"}),
    Role.CHARGE_NURSE: frozenset(
        {"assess", "view_identified", "override", "view_watchboard",
         "set_escalation_budget", "declare_surge"}
    ),
    Role.CLINICIAN: frozenset({"assess", "view_identified", "override", "view_watchboard"}),
    Role.DATA_SCIENTIST: frozenset({"view_pseudonymised", "train_model"}),
    Role.AUDITOR: frozenset({"read_audit", "view_identified"}),
    Role.SYSTEM: frozenset({"assess", "write_audit"}),
}


def may(role: Role, action: str) -> bool:
    return action in PERMISSIONS.get(role, frozenset())


class AccessDenied(PermissionError):
    """Raised when a role attempts something outside its permissions."""


def require(role: Role, action: str) -> None:
    if not may(role, action):
        raise AccessDenied(f"role {role.value} may not {action}")


# --------------------------------------------------------------------------------------
# What an override must legally record
# --------------------------------------------------------------------------------------

REQUIRED_OVERRIDE_FIELDS: tuple[str, ...] = (
    "patient_id",
    "clinician_id",
    "recommended_acuity",
    "assigned_acuity",
    "reason",
    "recorded_at",
)


def validate_override_record(payload: dict[str, object], policy: RetentionPolicy) -> list[str]:
    """What is missing from an override record, if anything.

    Returns problems rather than raising: an override must never be blocked by a
    validation failure. The clinician's decision stands regardless — the system's job
    is to record it and then complain loudly about its own incompleteness, not to
    stand between a nurse and a patient.
    """
    problems = [
        f"missing required field: {field}"
        for field in REQUIRED_OVERRIDE_FIELDS
        if payload.get(field) is None and field != "recommended_acuity"
    ]

    if "recommended_acuity" not in payload:
        problems.append("missing required field: recommended_acuity")

    if policy.jurisdiction is Jurisdiction.GDPR and not payload.get("clinician_id"):
        problems.append(
            "GDPR Art. 22 requires the decision be attributable to a natural person; "
            "clinician_id cannot be blank"
        )
    return problems


@dataclass(frozen=True)
class GovernanceStatement:
    """The compliance posture of one deployment, in a form that can be printed."""

    profile_name: str
    policy: RetentionPolicy

    def lines(self) -> list[str]:
        p = self.policy
        return [
            f"site                    {self.profile_name}",
            f"jurisdiction            {p.jurisdiction.value}",
            f"lawful basis            {p.lawful_basis.value}",
            f"audit retention         {p.audit_days // 365} years",
            f"triage record           {p.triage_record_days // 365} years",
            f"training corpus         {p.model_training_days // 365} years",
            f"queue telemetry         {p.waiting_room_telemetry_days} days",
            f"right to erasure        {'yes' if p.supports_erasure else 'not applicable'}"
            + (f" (reaches {', '.join(p.erasable_categories())})" if p.supports_erasure else ""),
            f"DPIA required           {'yes' if p.requires_dpia else 'no'}",
            "automated decisions     none — a licensed clinician sets every acuity",
            "model inputs            de-identified; no name, address or record number",
        ]
