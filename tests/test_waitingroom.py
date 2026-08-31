"""The waiting-room watcher.

The brief requires the system to monitor patients already in the queue and trigger
re-assessment when the wait exceeds a safe threshold for their severity, or when vitals
are re-recorded as worsening. These tests are that requirement, stated executably.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from patienttriage.features.schema import TriageSnapshot
from patienttriage.monitoring.waitingroom import (
    AlertReason,
    ResolutionReason,
    VitalsReading,
    WaitingPatient,
    WaitingRoomMonitor,
    compare_readings,
)

T0 = datetime(2026, 8, 31, 9, 0, 0, tzinfo=UTC)
MONITOR = WaitingRoomMonitor()


def snap(**kwargs) -> TriageSnapshot:
    base = {"patient_id": "p1", "arrived_at": T0.isoformat(), "age_years": 45.0}
    return TriageSnapshot(**(base | kwargs))


def waiting(acuity: int, minutes_ago: int = 0, **kwargs) -> WaitingPatient:
    return WaitingPatient(
        snapshot=snap(**kwargs),
        assigned_acuity=acuity,
        arrived_at=T0 - timedelta(minutes=minutes_ago),
    )


# --- trigger 1: the clock --------------------------------------------------------------


def test_breach_fires_once_the_target_is_passed():
    """A level 3 patient has a 30 minute target. At 45 minutes it has been missed."""
    alerts = MONITOR.check(waiting(3, minutes_ago=45), T0)
    breaches = [a for a in alerts if a.reason is AlertReason.WAIT_BREACH]
    assert len(breaches) == 1
    assert "15 min over" in breaches[0].detail


def test_warning_fires_before_the_breach_not_after():
    """Announced only once it has happened, a breach is an audit statistic."""
    alerts = MONITOR.check(waiting(3, minutes_ago=25), T0)
    assert any(a.reason is AlertReason.WAIT_APPROACHING for a in alerts)
    assert not any(a.reason is AlertReason.WAIT_BREACH for a in alerts)


def test_targets_differ_by_severity():
    """Forty minutes is a breach for a level 2 patient and fine for a level 4."""
    level_two = MONITOR.check(waiting(2, minutes_ago=40), T0)
    level_four = MONITOR.check(waiting(4, minutes_ago=40), T0)
    assert any(a.reason is AlertReason.WAIT_BREACH for a in level_two)
    assert not any(a.reason is AlertReason.WAIT_BREACH for a in level_four)


def test_a_patient_who_has_been_seen_generates_nothing():
    patient = waiting(3, minutes_ago=200)
    patient.seen = True
    assert MONITOR.check(patient, T0) == []


def test_the_clock_alone_does_not_change_the_acuity():
    """A missed target does not make a patient sicker. It puts them in front of a
    human, which is what the alert is for."""
    alerts = MONITOR.check(waiting(3, minutes_ago=90), T0)
    breach = next(a for a in alerts if a.reason is AlertReason.WAIT_BREACH)
    assert breach.suggested_acuity == 3
    assert not breach.escalates


# --- trigger 2: vitals re-recorded as worsening ----------------------------------------


def test_worsening_vitals_escalate_the_patient():
    patient = waiting(4, minutes_ago=40, heart_rate=78, systolic_bp=128, spo2=98)
    patient.readings.append(
        VitalsReading(recorded_at=T0, heart_rate=112, systolic_bp=104, spo2=93)
    )
    alerts = MONITOR.check(patient, T0)
    worsening = next(a for a in alerts if a.reason is AlertReason.VITALS_WORSENING)

    assert worsening.escalates
    assert worsening.suggested_acuity == 2
    assert "heart rate rose" in worsening.detail
    assert "systolic pressure fell" in worsening.detail


def test_stable_repeat_observations_do_not_alert():
    patient = waiting(4, minutes_ago=20, heart_rate=78, systolic_bp=128, spo2=98)
    patient.readings.append(
        VitalsReading(recorded_at=T0, heart_rate=80, systolic_bp=126, spo2=98)
    )
    assert not any(
        a.reason is AlertReason.VITALS_WORSENING for a in MONITOR.check(patient, T0)
    )


def test_improvement_never_de_escalates():
    """The watcher may escalate or leave alone. It has no path to moving anyone down."""
    patient = waiting(2, minutes_ago=20, heart_rate=130, systolic_bp=88, spo2=90)
    patient.readings.append(
        VitalsReading(recorded_at=T0, heart_rate=82, systolic_bp=124, spo2=99)
    )
    for alert in MONITOR.check(patient, T0):
        assert alert.suggested_acuity <= alert.current_acuity


def test_deterioration_inside_the_normal_range_is_still_caught():
    """A patient can deteriorate substantially without any value looking abnormal.
    Trend catches what a threshold cannot."""
    findings = compare_readings(
        VitalsReading(recorded_at=T0, heart_rate=62, systolic_bp=138),
        VitalsReading(recorded_at=T0, heart_rate=94, systolic_bp=118),
        age_years=40,
    )
    assert any("heart rate rose" in f for f in findings)
    assert any("systolic pressure fell" in f for f in findings)


def test_crossing_an_age_threshold_is_caught_even_in_small_steps():
    """No single step is large enough to read as a trend, but the patient is now in
    shock for their age."""
    findings = compare_readings(
        VitalsReading(recorded_at=T0, systolic_bp=103),
        VitalsReading(recorded_at=T0, systolic_bp=96),
        age_years=82,
    )
    assert any("shock threshold for this age" in f for f in findings)


def test_the_same_fall_is_not_a_crossing_for_a_younger_patient():
    findings = compare_readings(
        VitalsReading(recorded_at=T0, systolic_bp=103),
        VitalsReading(recorded_at=T0, systolic_bp=96),
        age_years=35,
    )
    assert not any("shock threshold" in f for f in findings)


def test_latest_snapshot_reflects_the_new_observations():
    """Re-scoring a patient's triage vitals would defeat the point of monitoring."""
    patient = waiting(3, minutes_ago=60, heart_rate=80)
    patient.readings.append(VitalsReading(recorded_at=T0, heart_rate=128))
    assert patient.latest_snapshot().heart_rate == 128
    assert patient.snapshot.heart_rate == 80


# --- trigger 3: nobody has looked ------------------------------------------------------


def test_a_long_unobserved_wait_is_itself_an_alert():
    """A patient nobody has re-measured in three hours is not a stable patient,
    they are an unobserved one."""
    alerts = MONITOR.check(waiting(4, minutes_ago=190), T0)
    assert any(a.reason is AlertReason.UNOBSERVED for a in alerts)


# --- the board -------------------------------------------------------------------------


def test_sweep_ranks_deterioration_above_a_clock_breach():
    deteriorating = waiting(4, minutes_ago=30, heart_rate=80, systolic_bp=130)
    deteriorating.snapshot = deteriorating.snapshot.model_copy(
        update={"patient_id": "worse"}
    )
    deteriorating.readings.append(
        VitalsReading(recorded_at=T0, heart_rate=125, systolic_bp=100)
    )
    breached = waiting(3, minutes_ago=200)
    breached.snapshot = breached.snapshot.model_copy(update={"patient_id": "late"})

    ranked = MONITOR.sweep([breached, deteriorating], T0)
    assert ranked[0].patient_id == "worse"


def test_alert_budget_caps_the_board_and_cuts_the_least_urgent():
    patients = [
        WaitingPatient(
            snapshot=snap(patient_id=f"p{i}"),
            assigned_acuity=4,
            arrived_at=T0 - timedelta(minutes=70 + i),
        )
        for i in range(20)
    ]
    capped = MONITOR.sweep(patients, T0, alert_budget=5)
    assert len(capped) == 5


def test_breach_rate_reports_by_level():
    patients = [
        waiting(3, minutes_ago=45),
        waiting(3, minutes_ago=10),
        waiting(2, minutes_ago=40),
    ]
    rates = MONITOR.breach_rate(patients, T0)
    assert rates[3] == pytest.approx(0.5)
    assert rates[2] == pytest.approx(1.0)


# --- resolving a patient: leaving the wait, versus leaving the department --------------


def test_sent_for_treatment_stops_the_wait_time_clock_but_not_the_watch():
    """Moved to a bed: the wait-time breach clears immediately, but a repeat
    observation taken since arrival still gets checked — a bed is not immunity."""
    patient = waiting(2, minutes_ago=90, heart_rate=78, systolic_bp=128, spo2=98)
    patient.readings.append(
        VitalsReading(recorded_at=T0, heart_rate=112, systolic_bp=104, spo2=93)
    )
    patient.resolve(ResolutionReason.SENT_FOR_TREATMENT, T0, note="bed 4")

    assert patient.in_treatment
    assert not patient.seen
    assert MONITOR.breach_rate([patient], T0) == {}
    alerts = MONITOR.check(patient, T0)
    assert not any(a.reason is AlertReason.WAIT_BREACH for a in alerts)
    assert any(a.reason is AlertReason.VITALS_WORSENING for a in alerts)


def test_stabilised_on_recheck_also_moves_to_a_bed_rather_than_closing_the_record():
    patient = waiting(3, minutes_ago=90)
    patient.resolve(ResolutionReason.STABILISED_ON_RECHECK, T0, note="repeat obs normal")

    assert patient.in_treatment
    assert not patient.seen
    assert MONITOR.breach_rate([patient], T0) == {}


def test_discharged_and_transferred_close_the_record_entirely():
    """Unlike a bed move, these mean the patient has left — nothing further to watch."""
    for reason in (ResolutionReason.DISCHARGED, ResolutionReason.TRANSFERRED):
        patient = waiting(2, minutes_ago=90)
        patient.resolve(reason, T0)
        assert patient.seen
        assert not patient.in_treatment
        assert MONITOR.check(patient, T0) == []


def test_resolve_records_why_and_when():
    patient = waiting(3)
    patient.resolve(ResolutionReason.STABILISED_ON_RECHECK, T0, note="repeat obs normal")

    assert patient.resolution_reason is ResolutionReason.STABILISED_ON_RECHECK
    assert patient.resolution_note == "repeat obs normal"
    assert patient.resolved_at == T0
