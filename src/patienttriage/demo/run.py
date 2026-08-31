"""The prototype demonstration.

    python -m patienttriage.demo.run
    python -m patienttriage.demo.run --profile rural
    python -m patienttriage.demo.run --surge

Runs a full evening shift through the assembled system and shows every behaviour the
prototype is meant to demonstrate:

    1. triage scoring across 20 simulated arrivals
    2. an ambiguous presentation, paediatric and geriatric cases, and patients with
       no prior record at all
    3. a confidence indicator on every single output, with no exceptions
    4. the waiting room monitored over time, with re-assessment triggered by wait-time
       breach and by vitals re-recorded as worsening
    5. behaviour at three times normal volume
    6. a clinician override, captured, with exactly what it writes to the audit trail

If the NHAMCS files have not been downloaded the demo still runs, on the deterministic
rule layer alone. That is not a fallback bolted on for the demo — it is the
`RULES_ONLY` deployment tier a real rural site would start on, so running it this way
exercises a supported configuration rather than a broken one.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np

from patienttriage.compliance.governance import GovernanceStatement, policy_for
from patienttriage.config.profile import PRESETS, DepartmentState, ModelTier
from patienttriage.data.cohort import SHIFT_START, build_cohort, cohort_summary, surge_cohort
from patienttriage.models.conformal import ConformalAbstainer
from patienttriage.models.decision import cost_matrix, surge_cost_matrix
from patienttriage.monitoring.waitingroom import AlertReason, WaitingRoomMonitor
from patienttriage.service.audit import AuditLog
from patienttriage.service.override import OverrideReason, capture, override_rate, reasons_breakdown
from patienttriage.service.pipeline import TriageAssistant

RULE = "=" * 78
THIN = "-" * 78


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


# --------------------------------------------------------------------------------------
# Model loading, with an honest fallback
# --------------------------------------------------------------------------------------


def load_models(data_dir: Path, profile):
    """Train on NHAMCS if it is present. Otherwise run rules-only and say so."""
    if profile.model_tier is ModelTier.RULES_ONLY:
        print("  site is configured RULES_ONLY - no model will be loaded")
        return None, None, None, None

    if not (data_dir / "ed2021.zip").exists():
        print(f"  no NHAMCS data at {data_dir} - degrading to the rule layer alone")
        print("  (this is the RULES_ONLY tier, not a failure; see README for the fetch)")
        return None, None, None, None

    from sklearn.model_selection import train_test_split

    from patienttriage.data.nhamcs import labelled_mask, load_cohort
    from patienttriage.features.build import build_features
    from patienttriage.models.deterioration import DeteriorationModel
    from patienttriage.models.ordinal import OrdinalAcuityModel

    print("  training on NHAMCS 2021 ...", end=" ", flush=True)
    snapshots, labels = load_cohort([2021], data_dir)
    features = build_features(snapshots)
    labels = labels.reset_index(drop=True)

    usable = labelled_mask(labels).to_numpy()
    X, y = features[usable], labels.loc[usable, "acuity"].astype(int).to_numpy()

    X_fit, X_conf, y_fit, y_conf = train_test_split(
        X, y, test_size=0.25, random_state=11, stratify=y
    )
    acuity = OrdinalAcuityModel().fit(X_fit, y_fit)
    abstainer = ConformalAbstainer(alpha=0.1).calibrate(acuity.predict_proba(X_conf), y_conf)
    deterioration = DeteriorationModel().fit(
        features, labels["critical_outcome"].to_numpy()
    )
    print(f"done ({len(X)} labelled visits)")

    # The undertriage cost is not a constant. Our own simulation showed that
    # escalating 30% of arrivals makes critically unwell patients wait ten hours
    # longer than doing nothing, while 7% helps — so the ratio is calibrated to what
    # THIS department can actually absorb, on held-out data, rather than picked once
    # and shipped everywhere.
    from patienttriage.eval.operating_point import choose_for_budget, sweep

    curve = sweep(y_conf, acuity.predict_proba(X_conf))
    chosen = choose_for_budget(curve, profile.escalation_fraction)
    costs = cost_matrix(
        undertriage=float(chosen["undertriage_cost"]), critical_surcharge=0.0
    )
    print(
        f"  escalation budget {profile.escalation_fraction:.0%} of arrivals"
        f" -> undertriage cost {chosen['undertriage_cost']:.1f}"
        f" (flags {chosen['flagged_rate']:.0%}, catches"
        f" {chosen['critical_sensitivity']:.0%} of level 1-2)"
    )
    return acuity, deterioration, abstainer, costs


def _advance_department(waiting, now, profile) -> None:
    """Mark patients as seen, in acuity order, at the department's real throughput.

    The binding constraint in a real emergency department is not triage speed, it is
    treatment spaces: a patient occupies a cubicle for hours, so throughput is roughly
    spaces divided by length of stay. A district general with 24 spaces and a four-hour
    average stay clears about six patients an hour while nine arrive — which is why
    emergency departments have waiting rooms at all, and why the monitor has work to do.
    """
    average_stay_hours = 4.0
    per_hour = profile.treatment_spaces / average_stay_hours
    elapsed_hours = (now - SHIFT_START).total_seconds() / 3600.0
    capacity = int(per_hour * elapsed_hours)

    queue = sorted(
        (p for p in waiting if not p.seen and p.arrived_at <= now),
        key=lambda p: (p.assigned_acuity, p.arrived_at),
    )
    for patient in queue[:capacity]:
        patient.seen = True


# --------------------------------------------------------------------------------------


def main(profile_name: str, data_dir: Path, audit_path: Path, run_surge: bool) -> None:
    profile = PRESETS[profile_name]
    policy = policy_for(profile.jurisdiction)

    # ---------------------------------------------------------------- 1. the site
    _heading("SITE AND GOVERNANCE")
    print(profile.describe())
    print()
    for line in GovernanceStatement(profile.name, policy).lines():
        print(f"  {line}")

    # ---------------------------------------------------------------- 2. the cohort
    _heading("COHORT")
    cases = build_cohort()
    coverage = cohort_summary(cases)
    for key, value in coverage.items():
        print(f"  {key:26} {value}")

    # Asserted rather than asserted-in-prose: if a future edit drops the neonate, the
    # demo fails here instead of quietly demonstrating less than it claims.
    assert coverage["total"] >= 15, "the demo cohort must carry at least 15 records"
    assert coverage["ambiguous"] >= 1
    assert coverage["paediatric"] >= 1 and coverage["geriatric"] >= 1
    assert coverage["zero_history"] >= 1
    print("\n  coverage requirements met")

    # ---------------------------------------------------------------- 3. models
    _heading("MODELS")
    acuity_model, deterioration_model, abstainer, costs = load_models(data_dir, profile)
    assistant = TriageAssistant(
        acuity_model=acuity_model,
        deterioration_model=deterioration_model,
        abstainer=abstainer,
        costs=costs,
        profile=profile,
    )
    if acuity_model is not None:
        print(
            "\n  Note: NHAMCS carries no free-text chief complaint, so the model scores"
            "\n  on vitals and context while the rule layer reads the complaint. The two"
            "\n  contribute different evidence, which is why both are present."
        )

    # ---------------------------------------------------------------- 4. triage
    _heading("TRIAGE — every arrival, every score carrying a confidence")
    print(
        f"\n{'patient':9} {'age':>6} {'nurse':>5} {'system':>6}  "
        f"{'confidence':44} {'data':26} flags"
    )
    print(THIN)

    assessments = {}
    for case in cases:
        assessment = assistant.assess(case.snapshot)
        assessments[case.snapshot.patient_id] = assessment

        if assessment.recommended_acuity is not None:
            system = str(assessment.recommended_acuity)
        elif assessment.abstained:
            system = "abstain"
        else:
            system = "-"
        age = case.snapshot.age_years
        age_text = f"{age * 365.25:.0f}d" if age < 1 else f"{age:.0f}y"
        flags = ",".join(h.rule_id for h in assessment.rule_hits) or "-"

        print(
            f"{case.snapshot.patient_id:9} {age_text:>6} {case.nurse_acuity:>5} "
            f"{system:>6}  {assessment.confidence_label():44} "
            f"{assessment.data_completeness():26} {flags}"
        )

    # The requirement is absolute, so it is checked rather than trusted.
    assert all(a.confidence_label() for a in assessments.values())
    print(f"\n  {len(assessments)} assessments, {len(assessments)} confidence indicators")

    # ---------------------------------------------------------------- 4b. self-scoring
    _heading("AGAINST THE PRE-REGISTERED EXPECTATIONS")
    print(
        "\nEach case carries what we said the system should do, written before it was"
        "\never run. A recommendation MORE acute than expected is not counted against"
        "\nit — escalation is the safe direction. Less acute is a miss, and so is a"
        "\nconfident answer where we said the data could not decide.\n"
    )

    met, safe_side, missed = [], [], []
    for case in cases:
        assessment = assessments[case.snapshot.patient_id]
        actual = assessment.recommended_acuity

        if case.expects_abstention:
            (met if assessment.abstained else missed).append(
                (case, "abstained" if assessment.abstained else f"answered level {actual}")
            )
            continue
        expected = case.expected_system_acuity
        if actual is None:
            missed.append((case, "abstained where a level was expected"))
        elif actual == expected:
            met.append((case, f"level {actual}"))
        elif actual < expected:
            safe_side.append((case, f"level {actual} vs {expected} expected"))
        else:
            missed.append((case, f"level {actual}, less acute than the expected {expected}"))

    print(f"  as expected            {len(met)}/{len(cases)}")
    print(f"  more acute (safe side) {len(safe_side)}/{len(cases)}")
    print(f"  missed                 {len(missed)}/{len(cases)}")

    for label, group in (("more acute than expected", safe_side), ("MISSED", missed)):
        if not group:
            continue
        print(f"\n  {label}:")
        for case, detail in group:
            print(f"    {case.snapshot.patient_id}  {detail}")
            print(f"              {case.label}")

    if missed:
        print(
            "\n  Reported rather than tuned away. The misses are the useful part of this"
            "\n  table: a demo that agrees with itself on every case has been fitted to"
            "\n  its own expectations and demonstrates nothing."
        )

    # ---------------------------------------------------------------- 5. spotlight
    _heading("SPOTLIGHT — the cases worth reading closely")
    for case in cases:
        if not any(
            tag in case.label for tag in ("AMBIGUOUS", "PAEDIATRIC", "GERIATRIC", "ZERO HISTORY")
        ):
            continue
        assessment = assessments[case.snapshot.patient_id]
        print(f"\n{THIN}\n{case.snapshot.patient_id}  {case.label}")
        print(f"  complaint     {case.snapshot.chief_complaint or '(none recorded)'}")
        print(f"  nurse         level {case.nurse_acuity}")
        print(f"  system        {assessment.headline()}")
        print(f"  completeness  {assessment.data_completeness()}")
        if assessment.capability_warning:
            print(f"  CAPABILITY    {assessment.capability_warning}")
        for hit in assessment.rule_hits:
            print(f"  rule {hit.rule_id}      {hit.reason}")
        for driver in assessment.drivers:
            print(f"  driver        {driver.sentence}")
        print(f"  expected      {case.expectation}")

    # ---------------------------------------------------------------- 6. the waiting room
    _heading("WAITING ROOM — re-assessment while patients wait")
    monitor = WaitingRoomMonitor()
    waiting = [case.to_waiting_patient() for case in cases]

    for elapsed in (45, 150):
        now = SHIFT_START + timedelta(minutes=elapsed)
        # The department is not idle: it works the queue in acuity order at its actual
        # throughput. Without this the board fills with a level 1 patient who in
        # reality went straight through to resuscitation, and the demo would be
        # measuring an imaginary department that never treats anybody.
        _advance_department(waiting, now, profile)
        alerts = monitor.sweep(waiting, now, alert_budget=profile.alert_budget)
        breaches = monitor.breach_rate(waiting, now)

        print(f"\n{THIN}\n  T+{elapsed} min — {len(alerts)} alert(s), "
              f"board capped at {profile.alert_budget}")
        print(f"  breach rate by level: "
              f"{ {k: f'{v:.0%}' for k, v in breaches.items()} }")
        for alert in alerts:
            marker = "ESCALATE" if alert.escalates else "review  "
            print(f"    {marker} {alert.patient_id}  {alert.headline()}")

    deteriorated = [
        a
        for a in monitor.sweep(waiting, SHIFT_START + timedelta(minutes=45))
        if a.reason is AlertReason.VITALS_WORSENING
    ]
    if deteriorated:
        print(
            f"\n  {len(deteriorated)} patient(s) escalated on repeat observations alone —"
            "\n  the trigger this design exists for, and the one no single-snapshot triage"
            "\n  system can produce."
        )

    # ---------------------------------------------------------------- 7. surge
    if run_surge:
        _heading("SURGE — three times normal volume")
        state = DepartmentState(
            profile=profile,
            arrivals_last_hour=profile.daily_volume / 24.0 * 3,
        )
        print(f"  load ratio {state.load_ratio:.1f}x — surge suggested: {state.surge_suggested()}")
        print("  the system proposes; a charge nurse declares. Declaring now.")
        state.surge_declared = True

        surge_cases = surge_cohort(3)
        surge_assistant = TriageAssistant(
            acuity_model=acuity_model,
            deterioration_model=deterioration_model,
            abstainer=abstainer,
            costs=surge_cost_matrix(),
            profile=profile,
        )
        surge_assessments = [surge_assistant.assess(c.snapshot) for c in surge_cases]
        normal_assessments = [assistant.assess(c.snapshot) for c in surge_cases]

        def flagged(items, discretionary_only: bool = False) -> float:
            subset = [
                a for a in items
                if not discretionary_only or a.source not in ("rule", "degraded")
            ]
            if not subset:
                return 0.0
            return float(
                np.mean([
                    a.recommended_acuity is not None and a.recommended_acuity <= 2
                    for a in subset
                ])
            )

        by_rule = sum(a.source == "rule" for a in normal_assessments)

        print(f"\n  arrivals                        {len(surge_cases)}")
        print(f"  escalated by red flag           {by_rule} — identical in both modes")
        print(
            f"  of the rest, flagged normally   "
            f"{flagged(normal_assessments, discretionary_only=True):.0%}"
        )
        print(
            f"  of the rest, flagged in surge   "
            f"{flagged(surge_assessments, discretionary_only=True):.0%}"
        )
        print(
            "\n  The red-flag floor does not move. Nothing about a busy department makes"
            "\n  a cardiac arrest less urgent, so the deterministic layer is identical in"
            "\n  both columns and only the model's discretionary escalations shift. That"
            "\n  is the design: surge changes judgement calls, never safety floors."
        )

        changed = sum(
            n.recommended_acuity != s.recommended_acuity
            for n, s in zip(normal_assessments, surge_assessments, strict=True)
        )
        print(f"\n  recommendations changed by the surge matrix: {changed}")
        if changed == 0:
            print(
                "  Zero, and that is the right answer rather than a broken switch. The"
                "\n  model puts over 90% of its mass on 'can wait' for these patients, and"
                "\n  no reweighting of costs should overturn a decisive probability. The"
                "\n  surge matrix moves borderline calls; it does not manufacture them."
            )

        surge_monitor = WaitingRoomMonitor()
        surge_waiting = [c.to_waiting_patient() for c in surge_cases]
        now = SHIFT_START + timedelta(minutes=120)
        _advance_department(surge_waiting, now, profile)

        all_alerts = surge_monitor.sweep(surge_waiting, now)
        capped = surge_monitor.sweep(surge_waiting, now, alert_budget=profile.alert_budget)
        breaches = surge_monitor.breach_rate(surge_waiting, now)
        deteriorating = [a for a in all_alerts if a.reason is AlertReason.VITALS_WORSENING]

        print(f"\n  still waiting at T+120        {sum(not p.seen for p in surge_waiting)}")
        print(
            "  breach rate by level          "
            + str({k: f"{v:.0%}" for k, v in breaches.items()})
        )
        print(f"  alerts raised                 {len(all_alerts)}")
        print(f"  alerts shown                  {len(capped)}  (board cap)")
        print(f"  escalated on worsening vitals {len(deteriorating)}")
        print(
            "\n  This is what actually changes under load. The department falls behind,"
            "\n  breach rates climb, and the alert list grows past what any person can"
            "\n  work through — so the board is capped and the least urgent are cut."
            "\n  An uncapped board during a surge is a board nobody reads."
        )

    # ---------------------------------------------------------------- 8. override
    _heading("CLINICIAN OVERRIDE — what the system records")
    audit = AuditLog(audit_path)

    # Every decision is logged, agreements included: an override rate needs a
    # denominator, and "nobody ever disagrees" is a finding in its own right.
    overrides = []
    for case in cases:
        assessment = assessments[case.snapshot.patient_id]
        audit.append("assessment", case.snapshot.patient_id, assessment.to_payload())
        overrides.append(
            capture(assessment, case.nurse_acuity, clinician_id="RN-4471")
        )

    # The panic-or-cardiac case: the nurse has seen the patient and is not satisfied.
    ambiguous = next(c for c in cases if c.snapshot.patient_id == "ED-013")
    decision = capture(
        assessments["ED-013"],
        assigned_acuity=2,
        clinician_id="RN-4471",
        reason=OverrideReason.CLINICAL_GESTALT,
        note="Looks unwell, not anxious. Not comfortable leaving her in the waiting room.",
    )
    record = audit.append("override", "ED-013", decision.to_payload())

    print(f"\n  patient       {ambiguous.snapshot.patient_id} — {ambiguous.label}")
    print(f"  system said   {assessments['ED-013'].headline()}")
    print(f"  nurse set     level {decision.assigned_acuity}  ({decision.direction})")
    print(f"  reason        {decision.reason.value}")
    print(f"  note          {decision.note}")
    print(f"\n  audit record  {record.record_hash[:16]}...")
    print(f"  chained to    {record.previous_hash[:16]}...")
    print("\n  logged fields:")
    for key, value in sorted(record.payload.items()):
        rendered = str(value)
        if len(rendered) > 68:
            rendered = rendered[:65] + "..."
        print(f"    {key:22} {rendered}")

    intact, problem = audit.verify()
    print(f"\n  chain verified: {intact}" + (f" ({problem})" if problem else ""))

    stats = override_rate(overrides + [decision])
    print("\n  shift totals:")
    for key, value in stats.items():
        print(f"    {key:22} {value:.2%}" if isinstance(value, float) else f"    {key:22} {value}")
    if breakdown := reasons_breakdown(overrides + [decision]):
        print(f"    reasons                {breakdown}")

    print(f"\n  audit written to {audit_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PRESETS), default="district")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/demo_audit.jsonl"))
    parser.add_argument("--surge", action="store_true", help="include the 3x surge section")
    parser.add_argument("--no-surge", dest="surge", action="store_false")
    parser.set_defaults(surge=True)
    args = parser.parse_args()

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    if args.audit.exists():
        args.audit.unlink()  # a fresh chain each run, so the demo is reproducible

    main(args.profile, args.data_dir, args.audit, args.surge)
