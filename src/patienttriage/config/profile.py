"""Hospital profiles — the same assistant, flexed to very different departments.

A 90-bed rural emergency department seeing 100 patients a day and a metropolitan
trauma centre seeing 600 are not the same customer, and the usual response to that —
build for the big one, tell the small one to grow into it — puts the tool in exactly
the departments least able to absorb its failure modes.

Three axes actually differ, and each changes the system's behaviour rather than its
branding:

**Scale** sets the alert budget. The escalation budget is not a percentage that
transfers between departments: our own simulation showed that escalating 30% of
arrivals makes critically unwell patients wait *longer* than no system at all, while
7% helps. What a department can absorb depends on its staffing, so the budget is
expressed in patients per shift and derived from real capacity.

**Technical maturity** sets what the system may read and write. Most departments
cannot offer a bidirectional EHR integration on day one, and a design that requires
one simply does not deploy. The tiers below degrade to a nurse typing six numbers into
a web form, which works everywhere and is where most sites will start.

**Specialty mix** sets which capabilities are safe to switch on. A department with no
paediatric service should not be quietly running paediatric acuity recommendations —
it should be told, loudly, that the child in front of them is outside the profile.

The default for anything unconfigured is always the most conservative option, because
a profile that has not been filled in is a department nobody has assessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IntegrationTier(str, Enum):
    """How much of the hospital's own systems the assistant can reach."""

    MANUAL = "manual"
    """No integration. The nurse types the vitals into our screen. Works anywhere."""

    READ_ONLY = "read_only"
    """We read demographics, problem list and medications. We write nothing back."""

    BIDIRECTIONAL = "bidirectional"
    """We read the record and write the recommendation and audit trail back to it."""


class ModelTier(str, Enum):
    """What the site is licensed and calibrated to run."""

    RULES_ONLY = "rules_only"
    """The deterministic red-flag layer alone. No local data needed, no drift to
    monitor, and it is the honest starting point for a site that has not yet been
    able to validate a model against its own population."""

    RULES_PLUS_MODEL = "rules_plus_model"
    """Full stack. Requires a local validation cohort and ongoing monitoring."""


@dataclass(frozen=True)
class HospitalProfile:
    """One department's configuration."""

    name: str
    daily_volume: int
    treatment_spaces: int
    triage_nurses_per_shift: int

    integration: IntegrationTier = IntegrationTier.MANUAL
    model_tier: ModelTier = ModelTier.RULES_ONLY

    has_paediatric_service: bool = False
    has_obstetric_service: bool = False
    is_trauma_centre: bool = False

    jurisdiction: str = "HIPAA"

    escalations_per_shift: int | None = None
    """How many patients the department can pull forward per shift. None derives it."""

    time_to_clinician_target: dict[int, int] | None = None
    """Local targets, if the department publishes its own."""

    prior_record_availability: float = 0.5
    """Share of arrivals expected to have a usable prior record. Drives how much the
    system may lean on history features, and how loudly it says when it cannot."""

    surge_multiplier: float = 3.0
    """Volume, relative to normal, at which the department declares surge."""

    # ----------------------------------------------------------------------------------

    @property
    def arrivals_per_shift(self) -> float:
        return self.daily_volume / 3.0  # three eight-hour shifts

    @property
    def escalation_budget(self) -> int:
        """Patients per shift the system may pull forward.

        Derived from the number of triage nurses rather than from volume, because the
        constraint is who does the re-assessing, not who arrives. A department with one
        triage nurse cannot act on forty alerts however many patients it sees.
        """
        if self.escalations_per_shift is not None:
            return self.escalations_per_shift
        return max(3, self.triage_nurses_per_shift * 8)

    @property
    def escalation_fraction(self) -> float:
        """The budget as a share of arrivals — the number the decision rule needs."""
        return min(1.0, self.escalation_budget / max(1.0, self.arrivals_per_shift))

    @property
    def alert_budget(self) -> int:
        """How many names may appear on the watch board at once."""
        return max(5, self.triage_nurses_per_shift * 5)

    def supports(self, age_years: float) -> bool:
        """Whether this patient is inside the department's declared capability."""
        return not (age_years < 18.0 and not self.has_paediatric_service)

    def capability_warning(self, age_years: float) -> str | None:
        """What to say when a patient falls outside the profile.

        Said out loud rather than handled silently. A department without a paediatric
        service still receives children — it stabilises and transfers them — and the
        useful thing the system can do is name that early, not decline to help.
        """
        if age_years < 18.0 and not self.has_paediatric_service:
            return (
                "no paediatric service is configured at this site — recommendations for "
                "this patient are advisory only, and transfer criteria should be "
                "considered early"
            )
        return None

    def describe(self) -> str:
        return (
            f"{self.name}: {self.daily_volume}/day, {self.treatment_spaces} spaces, "
            f"{self.triage_nurses_per_shift} triage nurse(s)/shift · "
            f"{self.integration.value} integration · {self.model_tier.value} · "
            f"{self.jurisdiction} · escalation budget {self.escalation_budget}/shift "
            f"({self.escalation_fraction:.0%} of arrivals)"
        )


# --------------------------------------------------------------------------------------
# Presets, spanning a realistic range of departments
# --------------------------------------------------------------------------------------

RURAL_GENERAL = HospitalProfile(
    name="Rural general ED",
    daily_volume=100,
    treatment_spaces=8,
    triage_nurses_per_shift=1,
    integration=IntegrationTier.MANUAL,
    # Starts on rules alone: no local validation cohort exists yet, and a site with one
    # triage nurse has no capacity to monitor a model for drift. This is not a lesser
    # product, it is the honest one for the setting.
    model_tier=ModelTier.RULES_ONLY,
    has_paediatric_service=False,
    prior_record_availability=0.65,
)

DISTRICT_GENERAL = HospitalProfile(
    name="District general hospital",
    daily_volume=250,
    treatment_spaces=24,
    triage_nurses_per_shift=2,
    integration=IntegrationTier.READ_ONLY,
    model_tier=ModelTier.RULES_PLUS_MODEL,
    has_paediatric_service=True,
    has_obstetric_service=True,
    prior_record_availability=0.5,
)

URBAN_TRAUMA_CENTRE = HospitalProfile(
    name="Urban major trauma centre",
    daily_volume=550,
    treatment_spaces=60,
    triage_nurses_per_shift=4,
    integration=IntegrationTier.BIDIRECTIONAL,
    model_tier=ModelTier.RULES_PLUS_MODEL,
    has_paediatric_service=True,
    has_obstetric_service=True,
    is_trauma_centre=True,
    prior_record_availability=0.4,
)

PRESETS: dict[str, HospitalProfile] = {
    "rural": RURAL_GENERAL,
    "district": DISTRICT_GENERAL,
    "urban": URBAN_TRAUMA_CENTRE,
}


@dataclass
class DepartmentState:
    """What is true in the department right now, as opposed to in general."""

    profile: HospitalProfile
    waiting_count: int = 0
    arrivals_last_hour: float = 0.0
    staff_present: int | None = None
    surge_declared: bool = False
    """Set by a human in charge. Never inferred from a census number."""

    notes: list[str] = field(default_factory=list)

    @property
    def expected_arrivals_per_hour(self) -> float:
        return self.profile.daily_volume / 24.0

    @property
    def load_ratio(self) -> float:
        expected = self.expected_arrivals_per_hour
        return self.arrivals_last_hour / expected if expected else 0.0

    def surge_suggested(self) -> bool:
        """Whether the numbers *suggest* surge. Distinct from declaring it.

        The system can notice that volume has tripled. It must not switch the objective
        function of a triage department on its own — that is a resource-rationing
        decision with clinical and ethical weight, and it belongs to the person in
        charge. So this proposes, and `surge_declared` disposes.
        """
        return self.load_ratio >= self.profile.surge_multiplier

    def in_surge(self) -> bool:
        return self.surge_declared
