"""A browser front end for the triage assistant.

    python -m patienttriage.ui.server
    python -m patienttriage.ui.server --profile urban --port 8010

Then open http://127.0.0.1:8000 (or whatever port was printed).

This file does no clinical reasoning of its own. It builds a `TriageAssistant` per
hospital profile exactly as `demo.run` does, hands it whatever the form on screen
submits, and serialises the `Assessment` it gets back. Every invariant that matters —
the model may only escalate, an assessment never carries a bare number with no
confidence, a fired red flag is never shown as "uncertain" — lives in
`service.pipeline` and is enforced there, not re-implemented here.
"""

from __future__ import annotations

import argparse
import dataclasses
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from patienttriage.clinical.routing import recommend_destination
from patienttriage.compliance.governance import GovernanceStatement, policy_for
from patienttriage.config.profile import PRESETS, HospitalProfile, ModelTier
from patienttriage.data.cohort import SHIFT_START, build_cohort
from patienttriage.demo.run import load_models
from patienttriage.features.schema import TriageSnapshot
from patienttriage.monitoring.waitingroom import (
    ResolutionReason,
    WaitingPatient,
    WaitingRoomMonitor,
)
from patienttriage.service.audit import AuditLog
from patienttriage.service.override import (
    Override,
    OverrideReason,
    capture,
    override_rate,
    reasons_breakdown,
)
from patienttriage.service.pipeline import Assessment, TriageAssistant

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_AUDIT_PATH = Path("artifacts/ui_audit.jsonl")

app = FastAPI(title="PatientTriage.ai")

# --------------------------------------------------------------------------------------
# Process-lifetime state. This is a single-department demo console, not a multi-tenant
# service — one assistant per profile, built once and reused, because retraining on
# every request would make the form feel broken rather than responsive.
# --------------------------------------------------------------------------------------

_lock = Lock()
_assistants: dict[str, TriageAssistant] = {}
_preparing: set[str] = set()
_executor = ThreadPoolExecutor(max_workers=1)
_last_assessment: dict[str, Assessment] = {}
"""patient_id -> the assessment shown on screen for it, so an override can be checked
against what the nurse actually saw rather than re-derived after the fact."""

_last_snapshot: dict[str, TriageSnapshot] = {}
"""patient_id -> what was actually entered, so the same patient can be placed on the
waiting-room board without asking the nurse to type the vitals a second time."""

_overrides: list[Override] = []
"""Every decision captured this session, kept as typed records rather than just the
audit log's flattened payloads, so the real `override_rate`/`reasons_breakdown`
functions can run over them instead of a re-parsed approximation."""

_waiting_room: dict[str, WaitingPatient] = {}
"""The live board. A patient is placed here by a nurse after triage — this console
does not put anyone on it by itself — and stays until a nurse resolves them, exactly
as `service.pipeline` never assigns a final acuity and never removes one from a queue
on its own."""

_resolved_log: list[dict[str, Any]] = []
"""A short recent history of who left the board and why, purely for the confirmation
a nurse sees right after resolving someone — the audit log is the record of truth."""

_data_dir = DEFAULT_DATA_DIR
_audit = AuditLog(DEFAULT_AUDIT_PATH)


def _assistant_for(profile_name: str) -> TriageAssistant:
    if profile_name not in PRESETS:
        raise HTTPException(404, f"unknown profile '{profile_name}'")
    with _lock:
        if profile_name not in _assistants:
            profile = PRESETS[profile_name]
            acuity, deterioration, abstainer, costs = load_models(_data_dir, profile)
            _assistants[profile_name] = TriageAssistant(
                acuity_model=acuity,
                deterioration_model=deterioration,
                abstainer=abstainer,
                costs=costs,
                profile=profile,
            )
        return _assistants[profile_name]


def _serialize(
    assessment: Assessment,
    snapshot: TriageSnapshot | None = None,
    profile: HospitalProfile | None = None,
) -> dict[str, Any]:
    """Everything the screen needs, in the language a nurse reads rather than a
    model's internals — the same discipline `models.explain` applies to drivers."""
    payload = assessment.to_payload()
    payload["patient_id"] = assessment.patient_id
    payload["headline"] = assessment.headline()
    payload["source"] = assessment.source
    payload["rule_hits"] = [asdict(h) for h in assessment.rule_hits]
    payload["drivers"] = [
        {"feature": d.feature, "contribution": d.contribution, "sentence": d.sentence}
        for d in assessment.drivers
    ]

    destination = None
    if assessment.effective_acuity is not None:
        rec = recommend_destination(
            assessment.effective_acuity,
            age_years=snapshot.age_years if snapshot else None,
            is_pregnant=snapshot.is_pregnant if snapshot else None,
            profile=profile,
        )
        destination = {
            "zone": rec.zone,
            "bed_type": rec.bed_type,
            "target_minutes": rec.target_minutes,
            "note": rec.note,
        }
    payload["destination"] = destination
    return payload


# --------------------------------------------------------------------------------------
# Site
# --------------------------------------------------------------------------------------


@app.get("/api/profiles")
def list_profiles() -> list[dict[str, Any]]:
    out = []
    for key, profile in PRESETS.items():
        policy = policy_for(profile.jurisdiction)
        out.append(
            {
                "id": key,
                "name": profile.name,
                "describe": profile.describe(),
                "model_tier": profile.model_tier.value,
                "has_paediatric_service": profile.has_paediatric_service,
                "escalation_budget": profile.escalation_budget,
                "alert_budget": profile.alert_budget,
                "governance": GovernanceStatement(profile.name, policy).lines(),
            }
        )
    return out


@app.get("/api/routing")
def routing_overview(profile: str = "district") -> dict[str, Any]:
    """The five zones, for a floor view — not a per-bed map.

    No bed occupancy is tracked or invented here; `/api/waitingroom` already reports
    how many real patients this session has at each acuity level, and the frontend
    pairs that live count with the static zone this endpoint describes. Site-level
    capability gaps (no paediatric service, not a trauma centre) are named once here
    rather than repeated on every patient in `recommend_destination`.
    """
    if profile not in PRESETS:
        raise HTTPException(404, f"unknown profile '{profile}'")
    site = PRESETS[profile]
    zones = [
        {
            "acuity": level,
            "zone": rec.zone,
            "bed_type": rec.bed_type,
            "target_minutes": rec.target_minutes,
        }
        for level in (1, 2, 3, 4, 5)
        for rec in [recommend_destination(level)]
    ]
    site_notes = []
    if not site.has_paediatric_service:
        site_notes.append(
            "no paediatric service configured — children route through any zone for "
            "stabilisation and transfer, not routine care"
        )
    if not site.is_trauma_centre:
        site_notes.append(
            f"{site.name} is not a designated trauma centre — major trauma activates "
            "the regional transfer pathway alongside resuscitation here"
        )
    if not site.has_obstetric_service:
        site_notes.append("no obstetric service configured — route pregnancy-related presentations onward")
    return {"zones": zones, "site_notes": site_notes}


def _prepare_worker(profile_id: str) -> None:
    try:
        _assistant_for(profile_id)
    finally:
        with _lock:
            _preparing.discard(profile_id)


@app.post("/api/profiles/{profile_id}/prepare")
def prepare_profile(profile_id: str) -> dict[str, Any]:
    """Kick off training in the background so switching sites in the console does
    not freeze the page for the several seconds a fit takes."""
    if profile_id not in PRESETS:
        raise HTTPException(404, f"unknown profile '{profile_id}'")
    with _lock:
        already_going = profile_id in _assistants or profile_id in _preparing
        if not already_going:
            _preparing.add(profile_id)
            _executor.submit(_prepare_worker, profile_id)
    return {"id": profile_id, "started": not already_going}


@app.get("/api/profiles/{profile_id}/status")
def profile_status(profile_id: str) -> dict[str, Any]:
    """Whether the model is ready yet, without forcing training to happen on a page
    load — the console shows a 'preparing' state instead of hanging the form."""
    if profile_id not in PRESETS:
        raise HTTPException(404, f"unknown profile '{profile_id}'")
    profile = PRESETS[profile_id]
    tier = profile.model_tier
    with _lock:
        ready = profile_id in _assistants
        preparing = profile_id in _preparing
    return {
        "id": profile_id,
        "ready": ready,
        "preparing": preparing,
        "model_tier": tier.value,
        "rules_only": tier is ModelTier.RULES_ONLY,
        "data_available": (_data_dir / "ed2021.zip").exists(),
    }


# --------------------------------------------------------------------------------------
# Example cohort, for populating the form without typing twenty numbers by hand
# --------------------------------------------------------------------------------------


@app.get("/api/cohort")
def cohort_index() -> list[dict[str, Any]]:
    return [
        {
            "id": case.snapshot.patient_id,
            "label": case.label,
            "nurse_acuity": case.nurse_acuity,
            "age_years": case.snapshot.age_years,
        }
        for case in build_cohort()
    ]


@app.get("/api/cohort/{patient_id}")
def cohort_case(patient_id: str) -> dict[str, Any]:
    for case in build_cohort():
        if case.snapshot.patient_id == patient_id:
            return {
                "label": case.label,
                "expectation": case.expectation,
                "nurse_acuity": case.nurse_acuity,
                "snapshot": case.snapshot.model_dump(mode="json"),
            }
    raise HTTPException(404, f"no such case '{patient_id}'")


# --------------------------------------------------------------------------------------
# Assessing one patient
# --------------------------------------------------------------------------------------


@app.post("/api/assess")
def assess(body: dict[str, Any]) -> dict[str, Any]:
    profile_name = body.get("profile", "district")
    snapshot_fields = {k: v for k, v in body.get("snapshot", {}).items() if v not in (None, "")}
    snapshot_fields.setdefault("patient_id", "UI-" + uuid.uuid4().hex[:8])
    # The form asks the nurse for what she knows about the patient, not for the
    # timestamp the screen itself can supply.
    snapshot_fields.setdefault("arrived_at", datetime.now(UTC).isoformat())

    try:
        snapshot = TriageSnapshot(**snapshot_fields)
    except ValidationError as error:
        raise HTTPException(422, error.errors()) from error

    assistant = _assistant_for(profile_name)
    assessment = assistant.assess(snapshot)
    _last_assessment[snapshot.patient_id] = assessment
    _last_snapshot[snapshot.patient_id] = snapshot
    payload = _serialize(assessment, snapshot, PRESETS.get(profile_name))
    _audit.append("assessment", snapshot.patient_id, payload)
    return payload


# --------------------------------------------------------------------------------------
# Overrides — a clinician disagreeing with, or confirming, the recommendation
# --------------------------------------------------------------------------------------


class OverrideRequest(BaseModel):
    patient_id: str
    assigned_acuity: int
    clinician_id: str
    reason: OverrideReason = OverrideReason.OTHER
    note: str = ""


@app.post("/api/override")
def record_override(body: OverrideRequest) -> dict[str, Any]:
    assessment = _last_assessment.get(body.patient_id)
    if assessment is None:
        raise HTTPException(
            409, "no assessment on record for this patient — assess before overriding"
        )
    override = capture(
        assessment, body.assigned_acuity, body.clinician_id, body.reason, body.note
    )
    _overrides.append(override)
    payload = override.to_payload()
    _audit.append("override", body.patient_id, payload)
    return payload


# --------------------------------------------------------------------------------------
# Waiting room — live, nurse-managed: a patient is on this board because a nurse put
# them there, and stays until a nurse takes them off. Nothing here happens on its own.
# --------------------------------------------------------------------------------------

_monitor = WaitingRoomMonitor()


class AddToWaitingRoomRequest(BaseModel):
    patient_id: str
    assigned_acuity: int


@app.post("/api/waitingroom/add")
def add_to_waiting_room(body: AddToWaitingRoomRequest) -> dict[str, Any]:
    """Place a just-triaged patient on the board.

    The acuity recorded is whatever the nurse assigns here — never read back from the
    model's recommendation — the same rule `service.override.capture` follows: the
    number that governs a real patient's care is always the human one.
    """
    snapshot = _last_snapshot.get(body.patient_id)
    if snapshot is None:
        raise HTTPException(
            409, "no assessment on record for this patient — assess before adding them"
        )
    now = datetime.now(UTC)
    patient = WaitingPatient(
        snapshot=snapshot, assigned_acuity=body.assigned_acuity, arrived_at=now
    )
    _waiting_room[body.patient_id] = patient
    _audit.append(
        "added_to_waiting_room", body.patient_id, {"assigned_acuity": body.assigned_acuity}
    )
    return {"patient_id": body.patient_id, "arrived_at": now.isoformat()}


class ResolveRequest(BaseModel):
    patient_id: str
    reason: ResolutionReason = ResolutionReason.SENT_FOR_TREATMENT
    note: str = ""


@app.post("/api/waitingroom/resolve")
def resolve_patient(body: ResolveRequest) -> dict[str, Any]:
    """End a patient's wait — which is not always the same as ending their record.

    `SENT_FOR_TREATMENT` and `STABILISED_ON_RECHECK` move them to a bed: the
    wait-time clock stops but the monitor keeps watching their vitals, per
    `WaitingPatient.resolve`. `DISCHARGED`, `TRANSFERRED` and `OTHER` mean they have
    actually left, so all watching stops. Only the latter appear in `recently_resolved`
    — someone still `in_treatment` is still on the live board, just past the wait.
    """
    patient = _waiting_room.get(body.patient_id)
    if patient is None:
        raise HTTPException(404, f"'{body.patient_id}' is not on the waiting-room board")
    now = datetime.now(UTC)
    waited = patient.waited_minutes(now)
    patient.resolve(body.reason, now, body.note)

    result = {
        "patient_id": body.patient_id,
        "reason": body.reason.value,
        "note": body.note,
        "waited_minutes": round(waited, 1),
        "resolved_at": now.isoformat(),
        "in_treatment": patient.in_treatment,
    }
    if not patient.in_treatment:
        _resolved_log.insert(0, result)
        del _resolved_log[15:]
    _audit.append("resolved", body.patient_id, result)
    return result


@app.post("/api/waitingroom/seed_demo")
def seed_demo_waiting_room() -> dict[str, Any]:
    """Load the 20-case demo cohort onto the board, for showing the feature off.

    Every timestamp in the script is translated by one constant so the last scripted
    arrival lands on real "now" — the whole shift keeps its original pacing and the
    ED-003 storyline, it just happens to end at the moment this button was pressed
    instead of a fixed date in 2026. Merges with whatever a nurse has already added;
    it does not clear the board.
    """
    anchor = datetime.now(UTC)
    cases = build_cohort()
    last_scripted = SHIFT_START + timedelta(
        minutes=max(c.minutes_after_shift_start for c in cases)
    )
    delta = anchor - last_scripted

    added = 0
    for case in cases:
        patient = case.to_waiting_patient()
        patient.arrived_at += delta
        patient.readings = [
            dataclasses.replace(r, recorded_at=r.recorded_at + delta)
            for r in patient.readings
        ]
        _waiting_room[patient.snapshot.patient_id] = patient
        _last_snapshot[patient.snapshot.patient_id] = patient.snapshot
        added += 1
    return {"added": added}


def _board_row(patient: WaitingPatient, now: datetime, alerts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "patient_id": patient.snapshot.patient_id,
        "assigned_acuity": patient.assigned_acuity,
        "waited_minutes": round(patient.waited_minutes(now), 1),
        "target_minutes": _monitor.target_for(patient.assigned_acuity),
        "in_treatment": patient.in_treatment,
        "alerts": alerts,
    }


def _sort_board(board: list[dict[str, Any]]) -> None:
    board.sort(
        key=lambda row: (
            0 if row["alerts"] else 1,
            -max((a["escalates"] for a in row["alerts"]), default=False),
            row["assigned_acuity"],
            -row["waited_minutes"],
        )
    )


@app.get("/api/waitingroom")
def waiting_room(profile: str = "district") -> dict[str, Any]:
    """The board right now — a live view, not a replay. Nobody is marked `seen` by
    this endpoint; that only ever happens through `/api/waitingroom/resolve`, at a
    nurse's action.

    Two lists, not one: `board` is who is still waiting for a space, against the
    wait-time clock. `in_treatment` is who a nurse has already moved to a bed —
    off that clock, but still swept for the same deterioration alerts, because a bed
    is monitoring, not discharge.
    """
    if profile not in PRESETS:
        raise HTTPException(404, f"unknown profile '{profile}'")
    site = PRESETS[profile]
    now = datetime.now(UTC)

    present = [p for p in _waiting_room.values() if not p.seen]
    alerts = _monitor.sweep(present, now, alert_budget=site.alert_budget)
    breach = _monitor.breach_rate(present, now)

    alerts_by_patient: dict[str, list[dict[str, Any]]] = {}
    for alert in alerts:
        alerts_by_patient.setdefault(alert.patient_id, []).append(
            {
                "reason": alert.reason.value,
                "detail": alert.detail,
                "suggested_acuity": alert.suggested_acuity,
                "escalates": alert.escalates,
            }
        )

    waiting = [
        _board_row(p, now, alerts_by_patient.get(p.snapshot.patient_id, []))
        for p in present
        if not p.in_treatment
    ]
    in_treatment = [
        _board_row(p, now, alerts_by_patient.get(p.snapshot.patient_id, []))
        for p in present
        if p.in_treatment
    ]
    _sort_board(waiting)
    _sort_board(in_treatment)

    return {
        "as_of": now.isoformat(),
        "total_waiting": len(waiting),
        "total_in_treatment": len(in_treatment),
        "breach_rate": breach,
        "board": waiting,
        "in_treatment": in_treatment,
        "recently_resolved": _resolved_log[:5],
    }


# --------------------------------------------------------------------------------------
# Audit — the same hash-chained log every assessment and override is written to
# --------------------------------------------------------------------------------------


@app.get("/api/audit")
def audit_log(limit: int = 50) -> dict[str, Any]:
    """The chain, plus real numbers computed over what actually happened this
    session — not a fabricated census. `override_rate` and `reasons_breakdown` are the
    same functions `demo.run` reports at the end of a shift; this just runs them over
    whatever this console has captured instead of the 20-case cohort."""
    intact, problem = _audit.verify()
    all_records = _audit.records()
    assessments = [r.payload for r in all_records if r.event == "assessment"]

    acuity_distribution: dict[str, int] = {}
    for payload in assessments:
        level = payload.get("recommended_acuity")
        key = str(level) if level is not None else "abstained"
        acuity_distribution[key] = acuity_distribution.get(key, 0) + 1

    stats = {
        "assessment_count": len(assessments),
        "red_flag_rate": (
            sum(1 for p in assessments if p.get("source") == "rule") / len(assessments)
            if assessments
            else None
        ),
        "abstention_rate": (
            sum(1 for p in assessments if p.get("abstained")) / len(assessments)
            if assessments
            else None
        ),
        "acuity_distribution": acuity_distribution,
        "overrides": override_rate(_overrides),
        "override_reasons": reasons_breakdown(_overrides),
    }

    records = all_records[-limit:]
    return {
        "intact": intact,
        "problem": problem,
        "count": len(all_records),
        "stats": stats,
        "records": [
            {
                "event": r.event,
                "patient_id": r.patient_id,
                "timestamp": r.timestamp,
                "record_hash": r.record_hash[:12],
            }
            for r in reversed(records)
        ],
    }


# --------------------------------------------------------------------------------------
# Static front end
# --------------------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument(
        "--profile",
        default="district",
        choices=sorted(PRESETS),
        help="site to have trained and ready before the first request",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    global _data_dir, _audit
    _data_dir = args.data_dir
    _audit = AuditLog(args.audit)

    print(f"preparing '{args.profile}' ...")
    _assistant_for(args.profile)

    import uvicorn

    if not args.no_browser:
        webbrowser.open(f"http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
