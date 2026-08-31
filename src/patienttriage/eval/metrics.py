"""Evaluation built around the error that actually hurts.

Accuracy is close to meaningless for triage. A model that assigns everyone level 3
scores about 52% on this cohort and would be dangerous in a department. The numbers
below are the ones a clinical lead would ask for.

The headline is **undertriage**: how often a patient who was genuinely level 1 or 2
gets recommended level 3 or worse — sent to the general waiting room. Everything else
is secondary, including overtriage, which is reported honestly alongside it because
it is the price being paid and somebody has to see the bill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from patienttriage.models.decision import cost_matrix
from patienttriage.models.ordinal import ACUITY_LEVELS

CRITICAL_LEVELS = (1, 2)


@dataclass(frozen=True)
class TriagePerformance:
    undertriage_rate: float
    """P(recommended >= 3 | truth in {1, 2}). The number the project optimises."""

    severe_undertriage_rate: float
    """Truth level 1, recommended 3 or worse. The catastrophic cell."""

    overtriage_rate: float
    """P(recommended <= 2 | truth in {3, 4, 5}). The price of the above."""

    exact_agreement: float
    within_one_level: float
    mean_cost: float
    critical_sensitivity: float
    critical_precision: float
    n: int

    def summary(self) -> str:
        return "\n".join(
            [
                f"  undertriage (level 1-2 sent to waiting room)  {self.undertriage_rate:7.2%}",
                f"  severe undertriage (level 1 sent to waiting)  {self.severe_undertriage_rate:7.2%}",
                f"  overtriage (level 3-5 pulled forward)         {self.overtriage_rate:7.2%}",
                f"  sensitivity for level 1-2                     {self.critical_sensitivity:7.2%}",
                f"  precision for level 1-2                       {self.critical_precision:7.2%}",
                f"  exact agreement with the nurse                {self.exact_agreement:7.2%}",
                f"  within one level                              {self.within_one_level:7.2%}",
                f"  mean asymmetric cost                          {self.mean_cost:7.3f}",
                f"  n                                             {self.n:7d}",
            ]
        )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> TriagePerformance:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    critical_truth = np.isin(y_true, CRITICAL_LEVELS)
    critical_pred = np.isin(y_pred, CRITICAL_LEVELS)
    costs = cost_matrix()

    level_one = y_true == 1

    return TriagePerformance(
        undertriage_rate=_safe_mean(y_pred[critical_truth] >= 3),
        severe_undertriage_rate=_safe_mean(y_pred[level_one] >= 3),
        overtriage_rate=_safe_mean(critical_pred[~critical_truth]),
        exact_agreement=float((y_true == y_pred).mean()),
        within_one_level=float((np.abs(y_true - y_pred) <= 1).mean()),
        mean_cost=float(costs[y_true - 1, y_pred - 1].mean()),
        critical_sensitivity=_safe_mean(critical_pred[critical_truth]),
        critical_precision=_safe_mean(critical_truth[critical_pred]),
        n=len(y_true),
    )


def _safe_mean(values: np.ndarray) -> float:
    """Mean of a possibly empty selection — an empty stratum is not a zero score."""
    return float(np.mean(values)) if len(values) else float("nan")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Rows are the nurse's assigned acuity, columns the model's recommendation."""
    table = pd.crosstab(
        pd.Series(np.asarray(y_true), name="assigned"),
        pd.Series(np.asarray(y_pred), name="recommended"),
    )
    return table.reindex(index=ACUITY_LEVELS, columns=ACUITY_LEVELS, fill_value=0)


# --------------------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------------------


def ranking_quality(y_binary: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """How well the score ranks the positives above the negatives.

    Reported separately from the threshold metrics because they answer different
    questions. A threshold metric asks "was this decision right"; ranking asks "does
    the score know who is sicker". For a watch board, ranking is the whole job — the
    board shows the top of the list, and a model with good ranking and bad calibration
    still puts the right people at the top.

    Average precision is the one to read here. With a ~2% positive rate, AUROC of 0.85
    can coexist with a watch list that is mostly false alarms; average precision does
    not let that pass unnoticed, since its no-skill baseline is the prevalence itself.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prevalence = float(y_binary.mean())

    if y_binary.min() == y_binary.max():
        return {"auroc": float("nan"), "average_precision": float("nan"),
                "prevalence": prevalence, "lift": float("nan")}

    average_precision = float(average_precision_score(y_binary, scores))
    return {
        "auroc": float(roc_auc_score(y_binary, scores)),
        "average_precision": average_precision,
        "prevalence": prevalence,
        # How many times better than guessing at the base rate. A lift near 1 means
        # the model has learned nothing useful, whatever its AUROC says.
        "lift": average_precision / prevalence if prevalence > 0 else float("nan"),
    }


def sensitivity_at_budget(
    y_binary: np.ndarray, scores: np.ndarray, budgets: tuple[float, ...] = (0.02, 0.05, 0.10, 0.17)
) -> pd.DataFrame:
    """Positives caught when only the top `budget` share of patients can be flagged.

    This is the question a charge nurse actually asks: "if I can watch 5% of the
    waiting room, how many of the people who go on to crash am I watching?"
    """
    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    total_positive = int(y_binary.sum())

    rows = []
    for budget in budgets:
        k = max(1, round(budget * len(scores)))
        caught = int(y_binary[order[:k]].sum())
        rows.append(
            {
                "budget": budget,
                "n_flagged": k,
                "caught": caught,
                "sensitivity": caught / total_positive if total_positive else float("nan"),
                "precision": caught / k,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def expected_calibration_error(
    y_binary: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Mean gap between predicted probability and observed frequency.

    Reported because the decision rule computes an expected cost from these
    probabilities. If a predicted 0.30 does not happen about 30% of the time, that
    expectation is arithmetic on numbers that do not mean what they claim.
    """
    y_binary = np.asarray(y_binary, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)

    error = 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        error += mask.mean() * abs(probabilities[mask].mean() - y_binary[mask].mean())
    return float(error)


def reliability_table(
    y_binary: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    y_binary = np.asarray(y_binary, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(mask.sum()),
                "predicted": float(probabilities[mask].mean()),
                "observed": float(y_binary[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Fairness
# --------------------------------------------------------------------------------------


def stratified_undertriage(
    y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series, min_count: int = 30
) -> pd.DataFrame:
    """Undertriage rate within each subgroup.

    A release gate, not an appendix. Pain is documented lower for Black patients and
    for women in published ED data, and pain score is a model input — so a model
    fitted on that data will reproduce the pattern unless somebody looks. Strata
    thinner than `min_count` report NaN rather than a number that would be noise.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)

    rows = []
    for value in sorted(groups.dropna().unique(), key=str):
        mask = (groups == value).to_numpy()
        critical = mask & np.isin(y_true, CRITICAL_LEVELS)
        n_critical = int(critical.sum())
        rows.append(
            {
                "group": value,
                "n": int(mask.sum()),
                "n_critical": n_critical,
                "undertriage_rate": (
                    float((y_pred[critical] >= 3).mean()) if n_critical >= min_count else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
