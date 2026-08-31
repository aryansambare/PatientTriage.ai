"""The red-flag rule layer (L0).

This layer runs before the model and is the reason the system can be trusted at all.
Three properties, none negotiable:

1. **It can only escalate.** Every rule declares an acuity *floor*. The engine takes
   the most acute floor any rule fired and never returns anything less acute than the
   input. In ESI, 1 is most acute, so escalation means taking a minimum.
2. **It is deterministic and readable.** A clinician can audit these rules line by
   line and argue with them. No weights, no embeddings, no training data.
3. **Every rule cites its source.** A threshold nobody can trace is a threshold
   nobody will defend at 03:00 when it fires on a patient who looks fine.

Every age-dependent threshold is read from `clinical/agebands.py` rather than written
here, so a neonate, a toddler and an eighty-five-year-old are each judged against their
own physiology. Applying one adult-calibrated scale across all ages is not a rounding
error — it is a silent failure that reads as normal on every dashboard.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from patienttriage.clinical import agebands
from patienttriage.features.schema import TriageSnapshot
from patienttriage.rules.lexicon import ComplaintFlag, has_flag

MOST_ACUTE = 1
LEAST_ACUTE = 5


@dataclass(frozen=True)
class Rule:
    """A single red flag."""

    id: str
    name: str
    acuity_floor: int
    citation: str
    check: Callable[[TriageSnapshot], str | None]
    """Returns the human-readable reason it fired, or None."""


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    name: str
    acuity_floor: int
    reason: str
    citation: str


@dataclass
class RuleResult:
    acuity_floor: int
    """Most acute floor demanded by any fired rule. LEAST_ACUTE if none fired."""

    hits: list[RuleHit] = field(default_factory=list)

    @property
    def fired(self) -> bool:
        return bool(self.hits)

    def apply_to(self, proposed_acuity: int) -> int:
        """Ratchet: escalate the proposal if a rule demands it, never de-escalate."""
        return min(proposed_acuity, self.acuity_floor)


# --------------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------------


def _r_arrest(s: TriageSnapshot) -> str | None:
    """Cardiac or respiratory arrest, or peri-arrest bradycardia.

    This rule exists because every *other* rule fails on this patient. In an arrest
    there are no vitals to trigger on: the monitor reads nothing, nobody stops
    compressions to take a manual pulse, and the triage fields come through empty. A
    system that scores acuity from vital signs therefore scores the sickest patient in
    the department as the least sick. The complaint text has to carry it alone.
    """
    if has_flag(s.chief_complaint, ComplaintFlag.ARREST):
        return "arrest or peri-arrest reported at handover — resuscitation bay now"
    if s.heart_rate is not None and s.heart_rate < 30:
        return f"HR {s.heart_rate:.0f} — peri-arrest bradycardia"
    if s.spo2 is not None and s.spo2 == 0 and s.heart_rate == 0:
        return "no recordable pulse or saturation"
    return None


def _r_altered_mental_status(s: TriageSnapshot) -> str | None:
    """AMS with no other abnormality still cannot wait.

    Previously this only escalated when it happened to coincide with an infection
    complaint (via qSOFA). An unresponsive patient with no vitals recorded and no
    fever scored as level 5.
    """
    if s.gcs is not None and s.gcs < 13:
        return f"GCS {s.gcs} — significantly depressed consciousness"
    if has_flag(s.chief_complaint, ComplaintFlag.ALTERED_MENTAL_STATUS):
        return (
            "altered consciousness reported — cause unknown, cannot be safely "
            "observed from a waiting room"
        )
    return None


def _r_airway(s: TriageSnapshot) -> str | None:
    if s.gcs is not None and s.gcs <= 8:
        return f"GCS {s.gcs} — unable to protect airway"
    if has_flag(s.chief_complaint, ComplaintFlag.RESPIRATORY) and s.chief_complaint:
        low = s.chief_complaint.lower()
        for token in ("stridor", "choking", "airway", "apne", "cyanos"):
            if token in low:
                return f"chief complaint indicates airway compromise ({token})"
    return None


def _r_hypoxia(s: TriageSnapshot) -> str | None:
    if s.spo2 is not None and s.spo2 < 85:
        return f"SpO2 {s.spo2:.0f}% — critical hypoxia"
    return None


def _r_hypoxia_moderate(s: TriageSnapshot) -> str | None:
    if s.spo2 is not None and 85 <= s.spo2 < 92:
        return f"SpO2 {s.spo2:.0f}% — hypoxia"
    return None


def _r_shock(s: TriageSnapshot) -> str | None:
    if s.systolic_bp is None:
        return None
    threshold = agebands.hypotension_threshold(s.age_years)
    if s.systolic_bp < threshold:
        return f"SBP {s.systolic_bp:.0f} mmHg (below {threshold:.0f} for age) — shock"
    return None


def _r_qsofa_sepsis(s: TriageSnapshot) -> str | None:
    """qSOFA >= 2 with an infection-suggestive complaint."""
    if not has_flag(s.chief_complaint, ComplaintFlag.INFECTION):
        return None
    criteria: list[str] = []
    if s.respiratory_rate is not None and s.respiratory_rate >= 22:
        criteria.append(f"RR {s.respiratory_rate:.0f}")
    if s.systolic_bp is not None and s.systolic_bp <= 100:
        criteria.append(f"SBP {s.systolic_bp:.0f}")
    # When the structured GCS and the free text disagree — GCS 15 recorded but the
    # nurse wrote "confused" — the safety layer deliberately takes the more acute
    # reading. GCS is often carried forward from an earlier observation or scored
    # loosely at triage, whereas nobody types "confused" about a patient who is not.
    if s.gcs is not None and s.gcs < 15:
        criteria.append(f"GCS {s.gcs}")
    elif has_flag(s.chief_complaint, ComplaintFlag.ALTERED_MENTAL_STATUS):
        criteria.append("altered mental status noted in complaint")
    if len(criteria) >= 2:
        return "possible sepsis — infection-suggestive complaint with qSOFA " + ", ".join(criteria)
    return None


def _r_neutropenic_fever(s: TriageSnapshot) -> str | None:
    if (
        s.is_immunosuppressed
        and s.temperature_c is not None
        and s.temperature_c >= 38.0
    ):
        return f"immunosuppressed with temperature {s.temperature_c:.1f}C — neutropenic fever risk"
    return None


def _r_stroke(s: TriageSnapshot) -> str | None:
    if has_flag(s.chief_complaint, ComplaintFlag.STROKE):
        return "stroke-suggestive complaint — time-critical, establish last-known-well now"
    return None


def _r_acs(s: TriageSnapshot) -> str | None:
    if has_flag(s.chief_complaint, ComplaintFlag.CARDIAC) and s.age_years >= 30:
        return "cardiac-suggestive complaint in an adult — ECG within 10 minutes of arrival"
    return None


def _r_anaphylaxis(s: TriageSnapshot) -> str | None:
    if has_flag(s.chief_complaint, ComplaintFlag.ANAPHYLAXIS):
        return "possible anaphylaxis — airway may close without warning"
    return None


def _r_infant_fever(s: TriageSnapshot) -> str | None:
    """Fever in the very young, where "how well do they look" is not evidence."""
    if s.temperature_c is None or s.age_years >= 90 * agebands.DAY:
        return None
    if s.temperature_c >= agebands.fever_threshold(s.age_years):
        band = agebands.band_for(s.age_years).band.value
        return (
            f"{band} at {s.age_years * 365.25:.0f} days with temperature "
            f"{s.temperature_c:.1f}C — full sepsis workup regardless of appearance"
        )
    return None


def _r_occult_sepsis_in_the_old(s: TriageSnapshot) -> str | None:
    """The septic older patient who never developed a fever.

    Older patients mount a blunted febrile response; a substantial minority are
    normothermic or frankly hypothermic when septic. A rule that waits for 38.0C
    finds them late or not at all, so this one triggers on the *combination* of a
    lowered temperature bar, or hypothermia, with another sign of instability.
    """
    physiology = agebands.band_for(s.age_years)
    if not physiology.blunted_fever_response or s.temperature_c is None:
        return None

    hypothermic = s.temperature_c <= physiology.hypothermia
    febrile = s.temperature_c >= physiology.fever
    if not (hypothermic or febrile):
        return None

    unstable: list[str] = []
    if s.respiratory_rate is not None and s.respiratory_rate >= 22:
        unstable.append(f"RR {s.respiratory_rate:.0f}")
    if s.systolic_bp is not None and s.systolic_bp <= physiology.hypotension:
        unstable.append(f"SBP {s.systolic_bp:.0f}")
    if s.heart_rate is not None and s.heart_rate > 100:
        unstable.append(f"HR {s.heart_rate:.0f}")
    if not unstable:
        return None

    descriptor = "hypothermia" if hypothermic else f"temperature {s.temperature_c:.1f}C"
    return (
        f"{descriptor} with {', '.join(unstable)} in an older patient — older patients "
        "often mount no fever when septic, so this is a lower bar on purpose"
    )


def _r_paediatric_compensated_shock(s: TriageSnapshot) -> str | None:
    """A child holding their blood pressure up by running fast.

    Children compensate for hypovolaemia by raising heart rate and vascular tone, and
    keep a textbook-normal blood pressure until they are close to arrest. Waiting for
    paediatric hypotension means waiting for the pre-arrest sign. So marked tachycardia
    plus delayed perfusion cues, *with* a normal pressure, is the finding — the normal
    pressure is what makes it dangerous rather than reassuring.
    """
    if not agebands.is_paediatric(s.age_years) or s.heart_rate is None:
        return None
    physiology = agebands.band_for(s.age_years)
    if s.heart_rate <= physiology.heart_rate_normal[1]:
        return None
    pressure_still_normal = (
        s.systolic_bp is not None
        and s.systolic_bp >= agebands.hypotension_threshold(s.age_years)
    )
    tachypnoeic = (
        s.respiratory_rate is not None
        and s.respiratory_rate > physiology.respiratory_rate_normal[1]
    )
    if pressure_still_normal and tachypnoeic:
        return (
            f"HR {s.heart_rate:.0f} and RR {s.respiratory_rate:.0f} with a normal blood "
            "pressure for age — a compensating child; the normal pressure is not "
            "reassurance, it is the last thing to fail"
        )
    return None


def _r_ob_emergency(s: TriageSnapshot) -> str | None:
    pregnant = s.is_pregnant or has_flag(s.chief_complaint, ComplaintFlag.OB_EMERGENCY)
    if pregnant and has_flag(s.chief_complaint, ComplaintFlag.BLEEDING):
        return "pregnancy with bleeding — ectopic or placental emergency until excluded"
    if pregnant and s.systolic_bp is not None and s.systolic_bp >= 160:
        return f"pregnancy with SBP {s.systolic_bp:.0f} — pre-eclampsia range"
    return None


def _r_anticoag_head_injury(s: TriageSnapshot) -> str | None:
    if s.on_anticoagulant and has_flag(s.chief_complaint, ComplaintFlag.HEAD_INJURY):
        return (
            "head injury while anticoagulated — intracranial bleed can be delayed and "
            "the patient may look entirely well now"
        )
    return None


def _r_major_trauma(s: TriageSnapshot) -> str | None:
    if has_flag(s.chief_complaint, ComplaintFlag.MAJOR_TRAUMA):
        return "high-energy mechanism — trauma team activation criteria may apply"
    return None


def _r_tachycardia(s: TriageSnapshot) -> str | None:
    if s.heart_rate is None:
        return None
    threshold = agebands.tachycardia_threshold(s.age_years, on_beta_blocker=s.on_beta_blocker)
    if s.heart_rate > threshold:
        note = " (threshold lowered: beta-blocker masks tachycardia)" if s.on_beta_blocker else ""
        return f"HR {s.heart_rate:.0f} above {threshold:.0f} for age{note}"
    if s.heart_rate < 40:
        return f"HR {s.heart_rate:.0f} — bradycardia"
    return None


def _r_respiratory_rate(s: TriageSnapshot) -> str | None:
    if s.respiratory_rate is None:
        return None
    upper = agebands.tachypnoea_threshold(s.age_years)
    lower = agebands.bradypnoea_threshold(s.age_years)
    if s.respiratory_rate > upper:
        return f"RR {s.respiratory_rate:.0f} (above {upper:.0f} for age) — respiratory distress"
    if s.respiratory_rate < lower:
        return f"RR {s.respiratory_rate:.0f} (below {lower:.0f} for age) — inadequate ventilation"
    return None


def _r_temperature_extreme(s: TriageSnapshot) -> str | None:
    if s.temperature_c is None:
        return None
    if s.temperature_c >= 41.0:
        return f"temperature {s.temperature_c:.1f}C — hyperthermia"
    lower = agebands.hypothermia_threshold(s.age_years)
    if s.temperature_c <= lower:
        return f"temperature {s.temperature_c:.1f}C (below {lower:.1f} for age) — hypothermia"
    return None


def _r_seizure(s: TriageSnapshot) -> str | None:
    if has_flag(s.chief_complaint, ComplaintFlag.SEIZURE):
        low = s.chief_complaint.lower()
        if "status" in low or "ongoing" in low or "continuous" in low:
            return "possible status epilepticus"
        return "recent seizure — requires monitored space"
    return None


def _r_self_harm(s: TriageSnapshot) -> str | None:
    if has_flag(s.chief_complaint, ComplaintFlag.SELF_HARM):
        return "self-harm or suicidal ideation — requires continuous observation, not a waiting room"
    return None


def _r_arrival_by_air(s: TriageSnapshot) -> str | None:
    if s.arrival_mode == "helicopter":
        return "helicopter transfer — pre-hospital team judged this time-critical"
    return None


def _r_vitals_unobtainable(s: TriageSnapshot) -> str | None:
    """Two or more of the vitals that should take seconds to obtain, missing together.

    A single missing reading is routine — a probe not placed yet, a value nobody
    needed for an obviously minor presentation — and firing on it alone would flag a
    large share of ordinary triage records. Two or more of heart rate, systolic blood
    pressure, respiratory rate and oxygen saturation missing *together* is a different
    finding: on a working monitor these come off one cuff and one probe in the time
    triage takes, so losing several at once usually means the patient could not be
    measured — combative, peri-arrest, or the equipment failed on someone in
    distress — not that nobody asked. Absence of data is exactly the case this system
    is designed for, not an edge case around it: a scorer that reads missing vitals as
    unremarkable scores the least-known patient in the department as the least sick
    one.
    """
    core = {
        "heart rate": s.heart_rate,
        "systolic BP": s.systolic_bp,
        "respiratory rate": s.respiratory_rate,
        "SpO2": s.spo2,
    }
    absent = [name for name, value in core.items() if value is None]
    if len(absent) < 2:
        return None
    return (
        f"{len(absent)} of 4 core vitals unmeasurable ({', '.join(absent)}) — inability "
        "to obtain vitals is treated as a finding, not as a normal reading"
    )


RULES: list[Rule] = [
    Rule("R00", "Cardiac or respiratory arrest", 1,
         "AHA 2020 Guidelines for CPR and ECC; ESI Handbook v4 level-1 criteria",
         _r_arrest),
    Rule("R01", "Airway compromise", 1,
         "ATLS 10th ed., primary survey; ESI Handbook v4 level-1 criteria", _r_airway),
    Rule("R02", "Critical hypoxia", 1,
         "ESI Handbook v4: SpO2 <90% is a level-1/2 discriminator", _r_hypoxia),
    Rule("R03", "Shock", 1,
         "Surviving Sepsis Campaign 2021; PALS age-adjusted hypotension", _r_shock),
    Rule("R04", "Anaphylaxis", 1,
         "WAO Anaphylaxis Guidance 2020", _r_anaphylaxis),
    Rule("R05", "Helicopter arrival", 2,
         "Pre-hospital triage decision is itself evidence of acuity", _r_arrival_by_air),
    Rule("R06", "Hypoxia", 2,
         "ESI Handbook v4 level-2 discriminator", _r_hypoxia_moderate),
    Rule("R07", "Suspected sepsis (qSOFA)", 2,
         "Singer et al., Sepsis-3, JAMA 2016; Surviving Sepsis Campaign 2021",
         _r_qsofa_sepsis),
    Rule("R08", "Neutropenic fever", 2,
         "IDSA Guideline for Febrile Neutropenia 2010", _r_neutropenic_fever),
    Rule("R09", "Suspected stroke", 2,
         "AHA/ASA 2019 Acute Ischemic Stroke Guidelines — door-to-needle target",
         _r_stroke),
    Rule("R10", "Suspected ACS", 2,
         "AHA/ACC 2021 Chest Pain Guideline — ECG within 10 min of arrival", _r_acs),
    Rule("R11", "Febrile infant", 2,
         "AAP Clinical Practice Guideline, Febrile Infants 8-60 Days, 2021",
         _r_infant_fever),
    Rule("R21", "Occult sepsis in an older patient", 2,
         "Norman, Fever in the Elderly, Clin Infect Dis 2000; Sepsis-3 (JAMA 2016)",
         _r_occult_sepsis_in_the_old),
    Rule("R22", "Compensated paediatric shock", 2,
         "PALS 2020: paediatric hypotension is a late sign of shock",
         _r_paediatric_compensated_shock),
    Rule("R12", "Obstetric emergency", 2,
         "ACOG Committee Opinion 767; ectopic pregnancy is a leading first-trimester "
         "cause of maternal death", _r_ob_emergency),
    Rule("R13", "Anticoagulated head injury", 2,
         "NICE CG176 Head Injury — imaging within 1 hour if anticoagulated",
         _r_anticoag_head_injury),
    Rule("R14", "Major trauma mechanism", 2,
         "CDC Field Triage Guidelines 2021", _r_major_trauma),
    Rule("R15", "Dangerous heart rate", 2,
         "ESI Handbook v4 danger-zone vital signs; PALS age bands", _r_tachycardia),
    Rule("R16", "Dangerous respiratory rate", 2,
         "ESI Handbook v4 danger-zone vital signs", _r_respiratory_rate),
    Rule("R17", "Temperature extreme", 2,
         "ESI Handbook v4 danger-zone vital signs", _r_temperature_extreme),
    Rule("R18", "Seizure", 2,
         "NCS Guidelines for Evaluation and Management of Status Epilepticus 2012",
         _r_seizure),
    Rule("R19", "Self-harm risk", 2,
         "Joint Commission NPSG 15.01.01 — suicide risk requires continuous observation",
         _r_self_harm),
    Rule("R20", "Altered mental status", 2,
         "ESI Handbook v4 level-2 discriminator; Teasdale & Jennett GCS",
         _r_altered_mental_status),
    Rule("R23", "Vitals unobtainable", 2,
         "ESI Handbook v4: vital signs cannot be safely assumed normal when they "
         "cannot be measured — unobtainable readings are a danger-zone finding, not "
         "an absence of one", _r_vitals_unobtainable),
]


class RuleEngine:
    """Evaluates every red flag against a snapshot. Escalation only."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules if rules is not None else RULES

    def evaluate(self, snapshot: TriageSnapshot) -> RuleResult:
        hits = [
            RuleHit(r.id, r.name, r.acuity_floor, reason, r.citation)
            for r in self.rules
            if (reason := r.check(snapshot)) is not None
        ]
        hits.sort(key=lambda h: (h.acuity_floor, h.rule_id))
        floor = min((h.acuity_floor for h in hits), default=LEAST_ACUTE)
        return RuleResult(acuity_floor=floor, hits=hits)
