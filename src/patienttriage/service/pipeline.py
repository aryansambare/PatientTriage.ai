"""The assembled assistant: L0 rules through L5 watch list.

This is where the architecture's central promise is actually enforced rather than
described. Three invariants, each implemented here and tested in `tests/test_pipeline.py`:

1. **`assess` never raises.** A triage screen that throws an exception at 03:00 is
   worse than no triage screen, because the nurse is now debugging software instead of
   seeing a patient. Every failure path degrades to the deterministic rule layer and
   says so on the face of the result.

2. **The model can only escalate.** The rule floor and the model recommendation are
   combined with a minimum, and the nurse's own acuity is never an input to either.

3. **Uncertainty is shown, not hidden.** When the conformal layer abstains, the
   assessment carries no acuity at all — not a greyed-out number, not a low-confidence
   number. There is nothing for a busy nurse to accidentally accept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from patienttriage.clinical import news2
from patienttriage.features.build import build_features
from patienttriage.features.schema import TriageSnapshot
from patienttriage.models.conformal import ConformalAbstainer
from patienttriage.models.decision import cost_matrix, recommend
from patienttriage.models.deterioration import DeteriorationModel
from patienttriage.models.explain import Driver, explain, shap_values_for
from patienttriage.models.ordinal import OrdinalAcuityModel
from patienttriage.rules.engine import RuleEngine, RuleHit

LATENCY_BUDGET_MS = 500.0
"""Past this, the assessment is late enough that the nurse has moved on."""


@dataclass
class Assessment:
    """What the triage screen shows beside the nurse's own acuity field."""

    patient_id: str

    recommended_acuity: int | None
    """None when the system abstains. Deliberately not a hedged number."""

    abstained: bool
    abstention_reason: str | None

    rule_hits: list[RuleHit]
    rule_floor: int

    critical_probability: float | None
    deterioration_risk: float | None
    drivers: list[Driver] = field(default_factory=list)

    confidence: float | None = None
    """Probability mass on the recommended level. None when no model contributed."""

    prediction_set: list[int] = field(default_factory=list)
    """Acuity levels the conformal layer considers plausible for this patient."""

    degraded: bool = False
    degraded_reason: str | None = None
    latency_ms: float = 0.0
    missing_vitals: list[str] = field(default_factory=list)
    capability_warning: str | None = None
    """Set when the patient falls outside what this site is configured to handle."""

    news2_score: int | None = None
    news2_band: str | None = None
    """NEWS2 (RCP 2017) reported alongside our own number, never in place of it. None
    for anyone the score was not derived for — under 18, or pregnant. See
    `clinical.news2` for exactly what it does and does not cover."""

    @property
    def source(self) -> str:
        """Which layer produced the number on screen.

        This exists because of a bug worth remembering. When the conformal layer
        abstained on a patient who had *also* tripped a red flag, the screen showed
        "uncertain" — model uncertainty was masking a deterministic level-1 rule, and
        a cardiac arrest displayed as an unanswered question.

        The layers answer different questions and the rule layer's is not in doubt.
        R00 firing is not a probabilistic claim that can be uncertain; it is a fact
        about the handover. So a fired rule always owns the display, and abstention
        describes only what the model declined to add on top of it.
        """
        if self.rule_hits and self.recommended_acuity == self.rule_floor:
            return "rule"
        if self.degraded:
            return "degraded"
        if self.abstained:
            return "abstained"
        return "model"

    @property
    def effective_acuity(self) -> int | None:
        """The acuity to act on for anything that needs *a* number rather than an
        exact model prediction — routing a patient to a zone, for instance.

        `recommended_acuity` when the pipeline produced one. The rule floor when it
        did not — no model was available, or the model abstained — because a red-flag
        or default floor still applies even then. `None` only when there is truly
        nothing to go on: an abstention with no rule behind it, which is exactly the
        case a nurse's own judgement has to fill rather than anything downstream.
        """
        if self.recommended_acuity is not None:
            return self.recommended_acuity
        return self.rule_floor if self.degraded else None

    def confidence_label(self) -> str:
        """A confidence statement for every assessment, without exception.

        No number is ever shown on its own. A recommendation with no indication of how
        firm it is invites a busy nurse to read all recommendations as equally solid,
        and the ones that matter are exactly the ones that are not.

        The number reported is confidence in the *band* — "needs to be seen now" versus
        "can wait" — not in the exact level. That is deliberate: the decision rule
        intentionally recommends a more acute level than the single most likely one,
        because undertriage costs more, so probability-of-the-recommended-level reads
        as absurdly low precisely when the system is being appropriately cautious. The
        band is also the only distinction that changes what happens to the patient next.
        """
        if self.source == "rule":
            rules = ", ".join(h.rule_id for h in self.rule_hits)
            return f"certain — deterministic rule ({rules})"
        if self.degraded:
            return "no model available — rule layer only"
        if self.abstained:
            return "insufficient — nurse judgement required"
        if self.confidence is None:
            return "not scored"

        span = f", plausible {self.prediction_set}" if len(self.prediction_set) > 1 else ""
        band = "urgent" if (self.recommended_acuity or 5) <= 2 else "can wait"
        if self.confidence >= 0.75:
            return f"high — {band} {self.confidence:.0%}{span}"
        if self.confidence >= 0.5:
            return f"moderate — {band} {self.confidence:.0%}{span}"
        return f"low — {band} {self.confidence:.0%}{span}"

    def data_completeness(self) -> str:
        """How much of the input was actually present. Shown beside the confidence.

        A model can be confident on thin data, and that is precisely when a nurse most
        needs to know the data was thin. Confidence and completeness answer different
        questions and are both reported.
        """
        missing = len(self.missing_vitals)
        if missing == 0:
            return "complete"
        if missing >= 4:
            return f"sparse — {missing}/5 vitals absent"
        return f"partial — {missing}/5 vitals absent"

    def news2_label(self) -> str:
        """A second opinion in a scale the nurse already trusts, or an honest note on
        why there isn't one — never a blank with no explanation."""
        if self.news2_score is None:
            return "NEWS2 not applicable — under 18 or pregnant"
        band = self.news2_band or "low"
        return f"NEWS2 {self.news2_score} — {band} clinical risk"

    def to_payload(self) -> dict[str, Any]:
        """Flat dictionary for the audit log."""
        return {
            "recommended_acuity": self.recommended_acuity,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "rule_hits": [
                {"id": h.rule_id, "name": h.name, "floor": h.acuity_floor, "reason": h.reason}
                for h in self.rule_hits
            ],
            "rule_floor": self.rule_floor,
            "critical_probability": self.critical_probability,
            "deterioration_risk": self.deterioration_risk,
            "drivers": [d.sentence for d in self.drivers],
            "confidence": self.confidence,
            "confidence_label": self.confidence_label(),
            "source": self.source,
            "prediction_set": self.prediction_set,
            "data_completeness": self.data_completeness(),
            "news2_score": self.news2_score,
            "news2_band": self.news2_band,
            "news2_label": self.news2_label(),
            "capability_warning": self.capability_warning,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "latency_ms": round(self.latency_ms, 2),
            "missing_vitals": self.missing_vitals,
        }

    def headline(self) -> str:
        """One line, as it would read on the screen."""
        if self.source == "rule":
            base = f"RED FLAG - level {self.rule_floor}"
        elif self.degraded:
            base = f"RULES ONLY - suggest at least level {self.rule_floor}"
        elif self.abstained:
            base = "UNCERTAIN - nurse judgement required"
        else:
            base = f"suggest level {self.recommended_acuity}"
        base += f"  ({self.confidence_label()})"
        if self.rule_hits:
            base += f"  [{', '.join(h.name for h in self.rule_hits)}]"
        return base


class TriageAssistant:
    """The whole stack, wired together."""

    def __init__(
        self,
        acuity_model: OrdinalAcuityModel | None = None,
        deterioration_model: DeteriorationModel | None = None,
        abstainer: ConformalAbstainer | None = None,
        rule_engine: RuleEngine | None = None,
        costs: np.ndarray | None = None,
        latency_budget_ms: float = LATENCY_BUDGET_MS,
        profile=None,
    ) -> None:
        self.acuity_model = acuity_model
        self.deterioration_model = deterioration_model
        self.abstainer = abstainer
        self.rules = rule_engine or RuleEngine()
        self.costs = cost_matrix() if costs is None else costs
        self.latency_budget_ms = latency_budget_ms
        # The site's own configuration. A department with no paediatric service
        # still receives children; the useful thing is to say so, not to refuse.
        self.profile = profile

    # ----------------------------------------------------------------------------------

    def assess(self, snapshot: TriageSnapshot) -> Assessment:
        """Never raises. Degrades to rules on any failure, and says so."""
        started = time.perf_counter()

        # The rule layer runs first and runs unconditionally. It has no dependencies
        # beyond the snapshot itself, so it is the one part that cannot be unavailable.
        rule_result = self.rules.evaluate(snapshot)

        # Pure vitals arithmetic, no model and no training data required — so it is
        # computed unconditionally, on the RULES_ONLY tier and the degraded path alike,
        # rather than only when the ML stack happens to be available.
        news2_score = news2_band = None
        if news2.is_applicable(snapshot.age_years, snapshot.is_pregnant):
            result = news2.score(
                heart_rate=snapshot.heart_rate,
                respiratory_rate=snapshot.respiratory_rate,
                spo2=snapshot.spo2,
                systolic_bp=snapshot.systolic_bp,
                temperature_c=snapshot.temperature_c,
                gcs=snapshot.gcs,
            )
            news2_score, news2_band = result.total, result.risk_band

        assessment = Assessment(
            patient_id=snapshot.patient_id,
            recommended_acuity=rule_result.acuity_floor if rule_result.fired else None,
            abstained=False,
            abstention_reason=None,
            rule_hits=rule_result.hits,
            rule_floor=rule_result.acuity_floor,
            critical_probability=None,
            deterioration_risk=None,
            missing_vitals=snapshot.missing_vitals(),
            news2_score=news2_score,
            news2_band=news2_band,
            capability_warning=(
                self.profile.capability_warning(snapshot.age_years)
                if self.profile is not None
                else None
            ),
        )

        if self.acuity_model is None:
            assessment.degraded = True
            assessment.degraded_reason = "no model loaded"
            assessment.latency_ms = (time.perf_counter() - started) * 1000
            return assessment

        try:
            self._score(snapshot, rule_result.acuity_floor, assessment)
        except Exception as error:  # noqa: BLE001 - a triage screen must not crash
            assessment.degraded = True
            assessment.degraded_reason = f"{type(error).__name__}: {error}"
            assessment.recommended_acuity = (
                rule_result.acuity_floor if rule_result.fired else None
            )

        assessment.latency_ms = (time.perf_counter() - started) * 1000
        if assessment.latency_ms > self.latency_budget_ms and not assessment.degraded:
            # The answer arrived, but late enough that a real screen would already have
            # fallen back. Flagged rather than discarded so the delay is measurable.
            assessment.degraded = True
            assessment.degraded_reason = (
                f"exceeded {self.latency_budget_ms:.0f} ms latency budget"
            )
        return assessment

    # ----------------------------------------------------------------------------------

    def _score(self, snapshot: TriageSnapshot, floor: int, assessment: Assessment) -> None:
        features = build_features([snapshot])
        probabilities = self.acuity_model.predict_proba(features)
        assessment.critical_probability = float(probabilities[0, :2].sum())

        if self.abstainer is not None:
            assessment.prediction_set = sorted(
                self.abstainer.prediction_sets(probabilities)[0]
            )

        if self.abstainer is not None and self.abstainer.should_abstain(probabilities)[0]:
            levels = assessment.prediction_set
            assessment.abstained = True
            assessment.abstention_reason = (
                "no level is supportable at the required confidence"
                if not levels
                else f"cannot separate levels {levels} - this patient sits on the boundary"
            )
            # An abstention still respects the rule layer. Declining to refine a
            # judgement is not the same as withdrawing a hard red flag.
            assessment.recommended_acuity = floor if floor < 5 else None
        else:
            proposed = recommend(probabilities, self.costs)[0].acuity
            assessment.recommended_acuity = min(proposed, floor)
            urgent_mass = float(probabilities[0, :2].sum())
            assessment.confidence = (
                urgent_mass if assessment.recommended_acuity <= 2 else 1.0 - urgent_mass
            )

        if self.deterioration_model is not None:
            risk = float(self.deterioration_model.predict_proba(features)[0])
            assessment.deterioration_risk = risk
            contributions = shap_values_for(self.deterioration_model.booster, features)
            assessment.drivers = explain(snapshot, features, contributions[0])

    # ----------------------------------------------------------------------------------

    def watch_list(
        self, snapshots: list[TriageSnapshot], alerts: int = 10
    ) -> pd.DataFrame:
        """The waiting room, ranked by who to look at next.

        Capped at what a person can actually act on in a shift. A board of forty names
        is a board nobody reads, and an unread board is indistinguishable from no board
        while still costing the attention it took to build.
        """
        if self.deterioration_model is None or not snapshots:
            return pd.DataFrame(columns=["rank", "patient_id", "risk", "drivers"])

        features = build_features(snapshots)
        risks = self.deterioration_model.predict_proba(features)
        contributions = shap_values_for(self.deterioration_model.booster, features)

        order = np.argsort(-risks)[:alerts]
        return pd.DataFrame(
            {
                "rank": np.arange(1, len(order) + 1),
                "patient_id": [snapshots[i].patient_id for i in order],
                "risk": risks[order],
                "drivers": [
                    "; ".join(
                        d.sentence for d in explain(snapshots[i], features, contributions[i])
                    )
                    for i in order
                ],
            }
        )
