"""A simulated arrival cohort for demonstration.

Twenty patients, each one written to exercise a specific behaviour rather than to fill
a table. Real NHAMCS visits train and validate the models; this cohort exists because a
national survey cannot show you a *named failure mode* — it has no free-text complaint,
no serial observations, and no way to say "this is the patient the design is for".

Composition:

  * **Ambiguous presentations** — cases 3, 13 and 14. Borderline numbers, a complaint
    that reads two ways, or pain that the scale cannot adjudicate.
  * **Paediatric** — cases 4, 5 and 17, spanning neonate, toddler and adolescent, since
    "paediatric" is three different physiologies, not one.
  * **Geriatric** — cases 6, 8 and 16, including the septic patient with no fever.
  * **Zero-history first-time patients** — cases 9, 11 and 20 carry no prior record at
    all: every history field is None, not False.
  * **Prior record availability** — half the cohort arrives with no usable prior
    record, matching the stated assumption in docs/SOLUTION.md. Their history fields
    are None rather than False, because an absent record is not a negative finding.

Every case carries `expectation`: what a competent triage nurse would do, written
before the system was run against it. That is the point of the field — it is a
pre-registered answer, so the demo cannot be quietly tuned until it agrees with itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from patienttriage.features.schema import TriageSnapshot
from patienttriage.monitoring.waitingroom import VitalsReading, WaitingPatient

SHIFT_START = datetime(2026, 8, 31, 19, 0, 0, tzinfo=UTC)


@dataclass
class DemoCase:
    """One patient, plus what we said about them before running anything."""

    snapshot: TriageSnapshot
    label: str
    expectation: str
    nurse_acuity: int
    """What the triage nurse assigned. The system's recommendation sits beside this."""

    expected_system_acuity: int | None = None
    """What we said the system should produce, written before it was ever run.

    A recommendation *more* acute than this is not scored as a failure: escalation is
    the safe direction and the whole system is tuned toward it. A recommendation less
    acute is a miss, and the demo says so rather than quietly moving the goalposts.
    """

    expects_abstention: bool = False
    """True where the honest answer is that the data cannot decide."""

    minutes_after_shift_start: int = 0
    has_prior_record: bool = True
    later_readings: list[tuple[int, dict[str, float | int | None]]] = field(
        default_factory=list
    )
    """(minutes after arrival, observations) — repeat vitals taken while waiting."""

    def arrival_time(self) -> datetime:
        return SHIFT_START + timedelta(minutes=self.minutes_after_shift_start)

    def to_waiting_patient(self) -> WaitingPatient:
        arrived = self.arrival_time()
        patient = WaitingPatient(
            snapshot=self.snapshot,
            assigned_acuity=self.nurse_acuity,
            arrived_at=arrived,
        )
        for offset, observations in self.later_readings:
            patient.readings.append(
                VitalsReading(recorded_at=arrived + timedelta(minutes=offset), **observations)
            )
        return patient


def _snap(patient_id: str, minutes: int, **kwargs) -> TriageSnapshot:
    arrived = (SHIFT_START + timedelta(minutes=minutes)).isoformat()
    return TriageSnapshot(patient_id=patient_id, arrived_at=arrived, **kwargs)


def build_cohort() -> list[DemoCase]:
    """The twenty arrivals, in the order they come through the door."""
    return [
        # ------------------------------------------------------------------ 1
        DemoCase(
            label="Cardiac arrest, no obtainable vitals",
            expectation=(
                "Level 1 on the handover alone. Every vital-sign rule is blind here — "
                "nobody stops compressions to take a pulse — so the complaint text has "
                "to carry it. A vitals-driven scorer rates the sickest patient in the "
                "department as the least sick."
            ),
            nurse_acuity=1,
            minutes_after_shift_start=0,
            snapshot=_snap(
                "ED-001", 0, age_years=61, sex="male", arrival_mode="ambulance",
                chief_complaint="VSA, CPR in progress, ROSC en route",
        ),
            has_prior_record=False,
            expected_system_acuity=1,
        ),
        # ------------------------------------------------------------------ 2
        DemoCase(
            label="Classic acute coronary syndrome",
            expectation="Level 2. Cardiac complaint in an adult — ECG within 10 minutes.",
            nurse_acuity=2,
            minutes_after_shift_start=4,
            snapshot=_snap(
                "ED-002", 4, age_years=58, sex="male", arrival_mode="ambulance",
                chief_complaint="CP rad to L arm, diaphoretic, onset 40 min",
                heart_rate=98, systolic_bp=148, diastolic_bp=88,
                respiratory_rate=20, spo2=96, temperature_c=36.8, pain_score=8,
                on_beta_blocker=False, on_anticoagulant=False, prior_ed_visits_90d=0,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 3
        DemoCase(
            label="AMBIGUOUS — 'just feels unwell', everything borderline",
            expectation=(
                "Genuinely undecidable from the data. Every vital sits just inside "
                "normal, the complaint maps to nothing, and she is either coming down "
                "with something or in early sepsis. This is the case the system should "
                "ABSTAIN on rather than guess — a confident number here would be the "
                "system inventing certainty it does not have."
            ),
            nurse_acuity=3,
            minutes_after_shift_start=11,
            snapshot=_snap(
                "ED-003", 11, age_years=34, sex="female", arrival_mode="walk_in",
                chief_complaint="generally unwell, off food x2d",
                heart_rate=98, systolic_bp=112, diastolic_bp=70,
                respiratory_rate=20, spo2=96, temperature_c=37.4, pain_score=3,
            ),
            later_readings=[
                # She declares herself while still in the waiting room.
                (35, {"heart_rate": 121, "systolic_bp": 94, "respiratory_rate": 26,
                      "spo2": 93, "temperature_c": 38.6}),
            ],
            has_prior_record=False,
            expects_abstention=True,
        ),
        # ------------------------------------------------------------------ 4
        DemoCase(
            label="PAEDIATRIC — febrile neonate, 19 days old",
            expectation=(
                "Level 2 on temperature alone. Under 28 days, 38.1C is a full sepsis "
                "workup no matter how settled the baby looks. An adult-calibrated "
                "scorer sees a mild fever and normal-looking observations."
            ),
            nurse_acuity=2,
            minutes_after_shift_start=18,
            snapshot=_snap(
                "ED-004", 18, age_years=19 / 365.25, sex="female", arrival_mode="walk_in",
                chief_complaint="fever, feeding less than usual",
                heart_rate=168, systolic_bp=72, respiratory_rate=48,
                spo2=97, temperature_c=38.1,
        ),
            has_prior_record=False,
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 5
        DemoCase(
            label="PAEDIATRIC — compensated shock, textbook-normal pressure",
            expectation=(
                "Level 2. HR 168 and RR 40 in a three-year-old whose blood pressure is "
                "still perfect. The normal pressure is not reassurance — children hold "
                "it up until they are nearly arrested. Waiting for paediatric "
                "hypotension means waiting for the pre-arrest sign."
            ),
            nurse_acuity=3,
            minutes_after_shift_start=26,
            snapshot=_snap(
                "ED-005", 26, age_years=3, sex="male", arrival_mode="walk_in",
                chief_complaint="vomiting and diarrhoea 3 days, not drinking",
                heart_rate=168, systolic_bp=96, diastolic_bp=58,
                respiratory_rate=40, spo2=97, temperature_c=38.4,
            ),
            has_prior_record=False,
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 6
        DemoCase(
            label="GERIATRIC — septic with no fever",
            expectation=(
                "Level 2. 37.6C is not a fever by any adult threshold, and this is "
                "what sepsis looks like at 84. A scorer waiting for 38.0C finds her "
                "late or not at all."
            ),
            nurse_acuity=3,
            minutes_after_shift_start=33,
            snapshot=_snap(
                "ED-006", 33, age_years=84, sex="female", arrival_mode="ambulance",
                chief_complaint="confused since this morning, family concerned, cough",
                heart_rate=104, systolic_bp=108, diastolic_bp=62,
                respiratory_rate=24, spo2=94, temperature_c=37.6,
                on_beta_blocker=False, prior_ed_visits_90d=2, prior_icu_admission=False,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 7
        DemoCase(
            label="Beta-blocker masking a shocked patient",
            expectation=(
                "Level 2. HR 118 would sit under a fixed 130 threshold. On metoprolol "
                "he physically cannot mount a tachycardia, so 118 with a soft pressure "
                "is the alarming finding, not the reassuring one."
            ),
            nurse_acuity=2,
            minutes_after_shift_start=41,
            snapshot=_snap(
                "ED-007", 41, age_years=67, sex="male", arrival_mode="ambulance",
                chief_complaint="black stools x2d, dizzy on standing",
                heart_rate=118, systolic_bp=98, diastolic_bp=58,
                respiratory_rate=22, spo2=96, temperature_c=36.4, pain_score=2,
                on_beta_blocker=True, on_anticoagulant=True, prior_ed_visits_90d=1,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 8
        DemoCase(
            label="GERIATRIC — anticoagulated head injury, looks completely well",
            expectation=(
                "Level 2 on mechanism, not appearance. Every observation is normal and "
                "she will look fine for hours. An intracranial bleed on apixaban "
                "declares itself late."
            ),
            nurse_acuity=3,
            minutes_after_shift_start=48,
            snapshot=_snap(
                "ED-008", 48, age_years=78, sex="female", arrival_mode="walk_in",
                chief_complaint="tripped on kerb, hit head, no LOC",
                heart_rate=76, systolic_bp=142, diastolic_bp=80,
                respiratory_rate=16, spo2=98, temperature_c=36.7, pain_score=3, gcs=15,
                on_anticoagulant=True, on_beta_blocker=False, prior_ed_visits_90d=0,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 9
        DemoCase(
            label="ZERO HISTORY — first presentation, nothing on file",
            expectation=(
                "No MRN match. Every history field is None rather than False, so the "
                "system must not read 'not on anticoagulants' from an absent record. "
                "It should score on what is in front of it and say what it is missing."
            ),
            nurse_acuity=3,
            minutes_after_shift_start=55,
            snapshot=_snap(
                "ED-009", 55, age_years=41, sex="male", arrival_mode="walk_in",
                chief_complaint="abdominal pain RLQ since last night",
                heart_rate=96, systolic_bp=132, diastolic_bp=78,
                respiratory_rate=18, spo2=98, temperature_c=37.9, pain_score=7,
        ),
            has_prior_record=False,
            expected_system_acuity=3,
        ),
        # ------------------------------------------------------------------ 10
        DemoCase(
            label="Walk-in stroke",
            expectation=(
                "Level 2, time-critical. He drove himself here, which is exactly the "
                "group our own evaluation showed the model is weakest on."
            ),
            nurse_acuity=2,
            minutes_after_shift_start=62,
            snapshot=_snap(
                "ED-010", 62, age_years=66, sex="male", arrival_mode="walk_in",
                chief_complaint="facial droop and slurred speech since 18:30",
                heart_rate=88, systolic_bp=168, diastolic_bp=94,
                respiratory_rate=18, spo2=97, temperature_c=36.9, gcs=15,
                on_anticoagulant=False, prior_ed_visits_90d=0,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 11
        DemoCase(
            label="ZERO HISTORY — combative, no vitals obtainable",
            expectation=(
                "Almost no data. Missingness is the signal: a patient too agitated to "
                "measure is not a stable patient. The system must not read absent "
                "observations as normal ones."
            ),
            nurse_acuity=2,
            minutes_after_shift_start=70,
            snapshot=_snap(
                "ED-011", 70, age_years=29, sex="male", arrival_mode="police",
                chief_complaint="agitated, aggressive, unable to obtain observations",
        ),
            has_prior_record=False,
            expects_abstention=True,
        ),
        # ------------------------------------------------------------------ 12
        DemoCase(
            label="Minor injury — the overtriage control",
            expectation=(
                "Level 4 and it should stay there. If the system pulls this forward, "
                "the escalation budget is being spent on a sprained ankle and the "
                "priority band is diluted for everyone who needs it."
            ),
            nurse_acuity=4,
            minutes_after_shift_start=77,
            snapshot=_snap(
                "ED-012", 77, age_years=24, sex="female", arrival_mode="walk_in",
                chief_complaint="twisted ankle playing netball, can weight bear",
                heart_rate=78, systolic_bp=118, diastolic_bp=72,
                respiratory_rate=16, spo2=99, temperature_c=36.6, pain_score=5,
            ),
            has_prior_record=False,
            expected_system_acuity=4,
        ),
        # ------------------------------------------------------------------ 13
        DemoCase(
            label="AMBIGUOUS — panic attack or cardiac event",
            expectation=(
                "Reads both ways and the data cannot separate them. A 27-year-old with "
                "chest tightness and a documented anxiety history is usually panic — "
                "and 'usually' is how young women with cardiac disease get missed. "
                "Abstain or escalate; do not confidently call it a 4."
            ),
            nurse_acuity=3,
            minutes_after_shift_start=84,
            snapshot=_snap(
                "ED-013", 84, age_years=27, sex="female", arrival_mode="walk_in",
                chief_complaint="chest tightness, SOB, tingling hands, feels panicky",
                heart_rate=112, systolic_bp=128, diastolic_bp=80,
                respiratory_rate=26, spo2=99, temperature_c=36.8, pain_score=6,
                prior_ed_visits_90d=3,
            ),
            expects_abstention=True,
        ),
        # ------------------------------------------------------------------ 14
        DemoCase(
            label="AMBIGUOUS — sickle cell crisis, 10/10 pain, normal observations",
            expectation=(
                "Severe pain with entirely normal vitals. The scale has nothing to grip "
                "and this is where documented bias lives: pain is recorded lower for "
                "Black patients in published ED data, and pain score is a model input. "
                "The system should defer to the patient's own report, not average it away."
            ),
            nurse_acuity=2,
            minutes_after_shift_start=91,
            snapshot=_snap(
                "ED-014", 91, age_years=23, sex="male", arrival_mode="walk_in",
                chief_complaint="sickle cell pain crisis, back and legs, usual pattern",
                heart_rate=102, systolic_bp=124, diastolic_bp=74,
                respiratory_rate=20, spo2=98, temperature_c=37.1, pain_score=10,
                prior_ed_visits_90d=4, prior_icu_admission=True,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 15
        DemoCase(
            label="Obstetric — pregnancy with bleeding",
            expectation="Level 2. Ectopic until excluded.",
            nurse_acuity=2,
            minutes_after_shift_start=98,
            snapshot=_snap(
                "ED-015", 98, age_years=31, sex="female", arrival_mode="walk_in",
                chief_complaint="9 weeks pregnant, vaginal bleeding and cramping",
                heart_rate=104, systolic_bp=106, diastolic_bp=64,
                respiratory_rate=18, spo2=99, temperature_c=36.9, pain_score=7,
                is_pregnant=True,
            ),
            has_prior_record=False,
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 16
        DemoCase(
            label="GERIATRIC — fall with hip pain and a soft pressure",
            expectation=(
                "SBP 96 would be unremarkable in a 40-year-old. At 88, on a background "
                "of hypertension, it represents a substantial fall."
            ),
            nurse_acuity=3,
            minutes_after_shift_start=105,
            snapshot=_snap(
                "ED-016", 105, age_years=88, sex="female", arrival_mode="ambulance",
                chief_complaint="found on floor at home, hip pain, unable to weight bear",
                heart_rate=96, systolic_bp=96, diastolic_bp=58,
                respiratory_rate=20, spo2=95, temperature_c=36.2, pain_score=8, gcs=15,
                on_beta_blocker=True, prior_ed_visits_90d=1,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 17
        DemoCase(
            label="PAEDIATRIC — adolescent asthma",
            expectation=(
                "Level 2. RR 32 is above the adolescent threshold and he is using "
                "accessory muscles. Note the band matters: 32 would be normal at 18 "
                "months."
            ),
            nurse_acuity=2,
            minutes_after_shift_start=112,
            snapshot=_snap(
                "ED-017", 112, age_years=14, sex="male", arrival_mode="walk_in",
                chief_complaint="asthma attack, SOB, wheeze, inhaler not helping",
                heart_rate=124, systolic_bp=118, diastolic_bp=70,
                respiratory_rate=32, spo2=93, temperature_c=36.9,
            ),
            has_prior_record=False,
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 18
        DemoCase(
            label="Diabetic ketoacidosis",
            expectation=(
                "Level 2. Tachypnoea plus vomiting plus a known diabetic — the "
                "respiratory rate here is compensation, not a chest problem."
            ),
            nurse_acuity=2,
            minutes_after_shift_start=119,
            snapshot=_snap(
                "ED-018", 119, age_years=22, sex="female", arrival_mode="ambulance",
                chief_complaint="vomiting, abdo pain, type 1 diabetic, deep breathing",
                heart_rate=128, systolic_bp=102, diastolic_bp=60,
                respiratory_rate=30, spo2=99, temperature_c=37.2, pain_score=6,
                prior_ed_visits_90d=1,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 19
        DemoCase(
            label="Self-harm risk",
            expectation=(
                "Level 2, and specifically not a waiting-room patient. Continuous "
                "observation is the requirement regardless of how the vitals read."
            ),
            nurse_acuity=2,
            minutes_after_shift_start=126,
            snapshot=_snap(
                "ED-019", 126, age_years=19, sex="female", arrival_mode="police",
                chief_complaint="SI, took 20 paracetamol 2h ago, wants to die",
                heart_rate=88, systolic_bp=114, diastolic_bp=70,
                respiratory_rate=16, spo2=99, temperature_c=36.6, gcs=15,
                prior_ed_visits_90d=2,
            ),
            expected_system_acuity=2,
        ),
        # ------------------------------------------------------------------ 20
        DemoCase(
            label="ZERO HISTORY — language barrier, pain not assessable",
            expectation=(
                "No record, no shared language, no pain score. The system has vitals "
                "and an age. It should say clearly how much it is missing rather than "
                "producing a confident number from a third of the inputs."
            ),
            nurse_acuity=3,
            minutes_after_shift_start=133,
            snapshot=_snap(
                "ED-020", 133, age_years=52, sex="female", arrival_mode="walk_in",
                chief_complaint="abdominal pain, interpreter not yet available",
                heart_rate=102, systolic_bp=126, diastolic_bp=76,
                respiratory_rate=20, spo2=97, temperature_c=37.3,
        ),
            has_prior_record=False,
            expected_system_acuity=3,
        ),
    ]


def surge_cohort(multiplier: int = 3) -> list[DemoCase]:
    """The same shift at `multiplier` times normal volume.

    Built by compressing arrival times rather than by inventing new patients: the case
    mix of a surge is roughly the case mix of a normal shift, arriving faster. Inventing
    extra trauma to make the surge look dramatic would be measuring a scenario we made
    up rather than the one departments actually face.
    """
    base = build_cohort()
    surged: list[DemoCase] = []
    for repeat in range(multiplier):
        for case in base:
            clone = DemoCase(
                snapshot=case.snapshot.model_copy(
                    update={"patient_id": f"{case.snapshot.patient_id}-s{repeat}"}
                ),
                label=case.label,
                expectation=case.expectation,
                nurse_acuity=case.nurse_acuity,
                minutes_after_shift_start=case.minutes_after_shift_start // multiplier
                + repeat * 2,
                has_prior_record=case.has_prior_record,
                later_readings=list(case.later_readings),
            )
            surged.append(clone)
    return sorted(surged, key=lambda c: c.minutes_after_shift_start)


def cohort_summary(cases: list[DemoCase]) -> dict[str, int]:
    """What the cohort covers, so the demo can assert it rather than claim it."""
    from patienttriage.clinical import agebands

    return {
        "total": len(cases),
        "paediatric": sum(agebands.is_paediatric(c.snapshot.age_years) for c in cases),
        "geriatric": sum(agebands.is_geriatric(c.snapshot.age_years) for c in cases),
        "zero_history": sum(not c.has_prior_record for c in cases),
        "ambiguous": sum("AMBIGUOUS" in c.label for c in cases),
        "with_serial_observations": sum(bool(c.later_readings) for c in cases),
        "no_vitals_at_all": sum(len(c.snapshot.missing_vitals()) == 5 for c in cases),
    }
