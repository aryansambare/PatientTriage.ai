"""Ordinal acuity model — a cumulative-link ensemble over LightGBM.

Acuity is ordered, and flat multiclass classification throws that away. To a softmax
head, calling a level 2 patient a 3 and calling them a 5 are both simply "wrong", one
mistake each. Clinically they are not remotely the same mistake.

So the model predicts a *cumulative* distribution instead. For each threshold t it
learns P(acuity <= t) — "is this patient at least this acute?" — and the five-level
distribution falls out of the differences. Ordering is then structural: a prediction
cannot be confident about level 2 without also being confident about "at most 3".

Calibration matters more here than accuracy. The decision rule downstream weighs an
expected cost against these probabilities, and an expected cost computed from
uncalibrated scores is arithmetic performed on numbers that do not mean what they
say. Each threshold model therefore gets its own isotonic calibrator, fitted on data
the gradient booster never saw.

The probability model is deliberately fitted *without* asymmetric class weights. Its
only job is to say what is likely. The judgement that undertriage is far worse than
overtriage lives in `decision.py`, where it is one readable number a clinician can
argue with, rather than being smeared through the loss function of four boosters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

ACUITY_LEVELS = (1, 2, 3, 4, 5)
THRESHOLDS = (1, 2, 3, 4)  # P(y <= t); P(y <= 5) is 1 by construction

DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
}


@dataclass
class OrdinalAcuityModel:
    """P(acuity = k) for k in 1..5, via calibrated cumulative thresholds."""

    params: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 400
    calibration_fraction: float = 0.2
    random_state: int = 7

    boosters: dict[int, lgb.Booster] = field(default_factory=dict, init=False)
    calibrators: dict[int, IsotonicRegression] = field(default_factory=dict, init=False)
    feature_names: list[str] = field(default_factory=list, init=False)

    # ----------------------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> OrdinalAcuityModel:
        y = np.asarray(y, dtype=int)
        if not np.isin(y, ACUITY_LEVELS).all():
            raise ValueError("acuity labels must all be in 1..5")

        self.feature_names = list(X.columns)

        # The calibration split is stratified on the label, because level 1 is ~1.5% of
        # the data and a random split can easily leave the calibrator with none of it.
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X,
            y,
            test_size=self.calibration_fraction,
            random_state=self.random_state,
            stratify=y,
        )

        for threshold in THRESHOLDS:
            z_fit = (y_fit <= threshold).astype(int)
            z_cal = (y_cal <= threshold).astype(int)

            booster = lgb.train(
                self.params,
                lgb.Dataset(X_fit, label=z_fit, free_raw_data=False),
                num_boost_round=self.num_boost_round,
            )
            self.boosters[threshold] = booster

            raw_cal = booster.predict(X_cal)
            # out_of_bounds="clip" so a test-time score beyond the calibration range
            # returns the nearest fitted probability instead of NaN.
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
            )
            calibrator.fit(raw_cal, z_cal)
            self.calibrators[threshold] = calibrator

        return self

    # ----------------------------------------------------------------------------------

    def cumulative_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(acuity <= t) for each threshold, as an (n, 4) array, monotone in t."""
        self._check_features(X)
        columns = [
            self.calibrators[t].predict(self.boosters[t].predict(X)) for t in THRESHOLDS
        ]
        cumulative = np.column_stack(columns)
        # Four independently fitted models can disagree about ordering on an individual
        # patient. Accumulating a maximum restores it without discarding information:
        # once a patient is judged "at most level 2", they are at most level 3 as well.
        return np.maximum.accumulate(cumulative, axis=1).clip(0.0, 1.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(acuity = k) as an (n, 5) array, columns ordered level 1 to 5."""
        cumulative = self.cumulative_proba(X)
        padded = np.column_stack([np.zeros(len(cumulative)), cumulative, np.ones(len(cumulative))])
        probabilities = np.diff(padded, axis=1).clip(min=0.0)
        totals = probabilities.sum(axis=1, keepdims=True)
        # A row can only sum to zero if every threshold returned an identical value;
        # fall back to a uniform distribution rather than dividing by zero.
        uniform = np.full_like(probabilities, 1 / len(ACUITY_LEVELS))
        return np.where(totals > 0, probabilities / np.where(totals > 0, totals, 1), uniform)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely acuity. Not the recommendation — see `decision.py` for that."""
        return np.asarray(ACUITY_LEVELS)[self.predict_proba(X).argmax(axis=1)]

    # ----------------------------------------------------------------------------------

    def _check_features(self, X: pd.DataFrame) -> None:
        if list(X.columns) != self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names)
            raise ValueError(
                "feature frame does not match training layout. "
                f"missing={sorted(missing)} unexpected={sorted(extra)}"
            )

    def feature_importance(self) -> pd.DataFrame:
        """Gain per feature, summed across thresholds."""
        totals = np.zeros(len(self.feature_names))
        for booster in self.boosters.values():
            totals += booster.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self.feature_names, "gain": totals})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
