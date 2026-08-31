"""Choosing where to sit on the undertriage/overtriage trade-off.

Setting the undertriage cost to 8 and calling it principled produces a system that
recommends level 2 for 78% of arrivals. Its undertriage rate looks excellent, and it
is operationally worthless: a department cannot resuscitate four fifths of its
waiting room, so staff would learn within a shift to ignore the recommendation
entirely. An assistant that is ignored is worse than no assistant, because it still
consumed the nurse's attention on the way to being dismissed.

The mistake is choosing the cost ratio in the abstract. A department has a real,
finite capacity for high-acuity patients, and the honest way to set the operating
point is against that capacity:

    "We can hold about N patients per shift in the level 1-2 pathway.
     Within that budget, catch as many genuinely urgent patients as possible."

That turns an unanswerable question ("how many times worse is undertriage?") into one
a charge nurse can answer in a sentence. This module sweeps the cost ratio, reports
what each point buys, and selects the one that fits the stated budget.

The sweep itself is the deliverable, not just the chosen point. Handing a department a
single tuned number invites them to trust it; handing them the curve shows them the
trade they are making and leaves the choice where it belongs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from patienttriage.eval.metrics import CRITICAL_LEVELS, evaluate
from patienttriage.models.decision import cost_matrix, recommend


@dataclass(frozen=True)
class OperatingPoint:
    undertriage_cost: float
    critical_surcharge: float
    flagged_rate: float
    """Share of arrivals recommended level 1 or 2 — what the department must absorb."""

    undertriage_rate: float
    severe_undertriage_rate: float
    overtriage_rate: float
    critical_sensitivity: float
    critical_precision: float
    mean_cost: float


def sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    rule_floors: np.ndarray | None = None,
    undertriage_costs: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0),
    critical_surcharge: float = 0.0,
) -> pd.DataFrame:
    """What each undertriage cost buys, and what it costs to buy it."""
    y_true = np.asarray(y_true, dtype=int)
    rows: list[OperatingPoint] = []

    for undertriage in undertriage_costs:
        costs = cost_matrix(undertriage=undertriage, critical_surcharge=critical_surcharge)
        predicted = np.array([r.acuity for r in recommend(probabilities, costs)])
        if rule_floors is not None:
            predicted = np.minimum(predicted, rule_floors)

        performance = evaluate(y_true, predicted)
        rows.append(
            OperatingPoint(
                undertriage_cost=undertriage,
                critical_surcharge=critical_surcharge,
                flagged_rate=float(np.isin(predicted, CRITICAL_LEVELS).mean()),
                undertriage_rate=performance.undertriage_rate,
                severe_undertriage_rate=performance.severe_undertriage_rate,
                overtriage_rate=performance.overtriage_rate,
                critical_sensitivity=performance.critical_sensitivity,
                critical_precision=performance.critical_precision,
                mean_cost=performance.mean_cost,
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])


def choose_for_budget(curve: pd.DataFrame, capacity: float) -> pd.Series:
    """The most sensitive operating point that still fits the department's capacity.

    `capacity` is the share of arrivals the level 1-2 pathway can actually absorb.
    Among the points within it, take the one catching the most urgent patients; if
    none fit, fall back to the most conservative available and let the caller see
    that the budget could not be met rather than silently exceeding it.
    """
    affordable = curve[curve["flagged_rate"] <= capacity]
    if affordable.empty:
        return curve.nsmallest(1, "flagged_rate").iloc[0]
    return affordable.nlargest(1, "critical_sensitivity").iloc[0]


def observed_critical_rate(y_true: np.ndarray) -> float:
    """Share of arrivals the nurses themselves judged level 1-2.

    A natural anchor for capacity: the department is already staffed for roughly this
    many, so it is the volume the system can recommend without asking for resources
    that do not exist.
    """
    return float(np.isin(np.asarray(y_true, dtype=int), CRITICAL_LEVELS).mean())
