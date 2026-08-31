"""Turning a probability distribution into a recommendation.

This module holds the project's central value judgement, on purpose, in one place a
clinician can read in a minute and argue with: **undertriage costs far more than
overtriage**. Sending a level 4 patient to a resuscitation bay wastes a bed for an
hour. Leaving a level 2 patient in the waiting room can kill them.

Keeping that judgement here rather than inside the model's loss function has three
consequences worth the separation:

  * the number is legible — a department can set it, not tune it;
  * the probability model stays calibrated, so its outputs still mean what they say;
  * changing the department's risk appetite, or switching to surge mode, is a
    configuration change rather than a retraining job.

The rule is standard decision theory: given a calibrated distribution over acuity and
a cost for each (truth, action) pair, recommend the action with the lowest expected
cost. Because the cost matrix is asymmetric, this systematically recommends a more
acute level than the single most likely one — which is the intended behaviour, not a
bias to be corrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from patienttriage.models.ordinal import ACUITY_LEVELS

UNDERTRIAGE_COST = 8.0
"""Cost per level of under-acuity. Eight times the overtriage cost.

Chosen to sit in the range implied by published ED undertriage studies rather than
picked to make a metric look good, and intended to be set by the department's
clinical lead. It is the single most consequential number in the system, so it lives
alone, named, at the top of a file — not buried in a training config.
"""

OVERTRIAGE_COST = 1.0
"""Cost per level of over-acuity: a wasted resource, and a longer wait for someone."""

CRITICAL_MISS_SURCHARGE = 12.0
"""Extra penalty for calling a level 1 or 2 patient non-urgent (level 3 or worse).

Distance alone understates this error. Missing level 2 by one level means the patient
is seen a bit late; missing it by two means they join the general waiting room, where
nobody is watching them. That is a difference in kind, so it gets its own term.
"""


def cost_matrix(
    undertriage: float = UNDERTRIAGE_COST,
    overtriage: float = OVERTRIAGE_COST,
    critical_surcharge: float = CRITICAL_MISS_SURCHARGE,
) -> np.ndarray:
    """`C[i, j]` — the cost of recommending level j when the truth is level i."""
    levels = np.asarray(ACUITY_LEVELS, dtype=float)
    truth = levels[:, None]
    action = levels[None, :]

    difference = action - truth
    costs = np.where(
        difference > 0,
        undertriage * difference,  # recommended a less acute level than the truth
        overtriage * -difference,  # recommended a more acute level
    )

    # The cliff: a genuinely urgent patient routed to the general waiting room.
    critical_truth = truth <= 2
    sent_to_waiting_room = action >= 3
    costs = costs + np.where(critical_truth & sent_to_waiting_room, critical_surcharge, 0.0)

    return costs


@dataclass(frozen=True)
class Recommendation:
    """What the system suggests, and how sure it is. Never what it decides."""

    acuity: int
    probabilities: np.ndarray
    expected_costs: np.ndarray
    confidence: float
    """Probability mass on the recommended level."""

    @property
    def critical_probability(self) -> float:
        """P(this patient is really level 1 or 2) — the number that drives the alert."""
        return float(self.probabilities[:2].sum())


def recommend(
    probabilities: np.ndarray, costs: np.ndarray | None = None
) -> list[Recommendation]:
    """Lowest-expected-cost acuity for each row of a predicted distribution."""
    costs = cost_matrix() if costs is None else costs
    probabilities = np.atleast_2d(probabilities)

    # expected_cost[n, j] = sum_i P(truth = i | x_n) * C[i, j]
    expected = probabilities @ costs
    chosen = expected.argmin(axis=1)

    return [
        Recommendation(
            acuity=int(ACUITY_LEVELS[j]),
            probabilities=probabilities[n],
            expected_costs=expected[n],
            confidence=float(probabilities[n, j]),
        )
        for n, j in enumerate(chosen)
    ]


def surge_cost_matrix() -> np.ndarray:
    """Cost matrix for a department in surge or mass-casualty mode.

    When there are more patients than the department can see, the objective changes.
    In normal running, overtriage costs a bed for an hour. In surge, that bed is the
    scarce resource the whole department is rationing, so overtriage starts to harm
    the people who are not getting it — and the gap between the two costs narrows.

    This is deliberately *not* automatic. A department entering surge mode is a
    declaration made by a human in charge, and the system follows that declaration
    rather than inferring it from a census number.
    """
    return cost_matrix(undertriage=6.0, overtriage=3.0, critical_surcharge=10.0)
