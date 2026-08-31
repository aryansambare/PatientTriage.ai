"""Deterioration model — P(this patient turns out to be critically unwell).

The acuity model in `ordinal.py` predicts *the nurse's judgement*. That is the right
target for a second opinion at triage, but it inherits everything the judgement
inherits: it learns the department's habits, including its blind spots, and it is
capped by how consistently humans apply a five-level scale under pressure.

This model predicts something the nurse's opinion cannot bias — what actually
happened. The label is the composite outcome: died in the department, or was admitted
to critical care. A patient can be assigned level 4 and still be in this group, and
those are exactly the patients the project exists to find.

Two consequences of using an outcome rather than a judgement:

  * it is not capped by inter-rater agreement, so it can in principle beat the label;
  * it is rare — around 2% of arrivals — so ranking quality matters far more than any
    accuracy figure, and average precision is the number to read.

This is the model behind the waiting-room watch board. It does not assign an acuity
and it never contradicts a nurse; it produces a ranking of who to look at next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

PROBABILITY_FLOOR = 0.001
"""No prediction is reported as certain. See `predict_proba`."""

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
class DeteriorationModel:
    """Calibrated P(critical outcome) from triage-time features alone."""

    params: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 400
    calibration_fraction: float = 0.2
    random_state: int = 7

    booster: lgb.Booster | None = field(default=None, init=False)
    calibrator: IsotonicRegression | None = field(default=None, init=False)
    feature_names: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> DeteriorationModel:
        y = np.asarray(y, dtype=int)
        self.feature_names = list(X.columns)

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X,
            y,
            test_size=self.calibration_fraction,
            random_state=self.random_state,
            stratify=y,
        )

        # `is_unbalance` rather than a hand-set weight: the positive class is ~2%, and
        # without it the booster spends its capacity on the easy negative majority.
        # The calibrator afterwards undoes the resulting probability distortion, so the
        # output still means what it says.
        params = dict(self.params) | {"is_unbalance": True}
        self.booster = lgb.train(
            params,
            lgb.Dataset(X_fit, label=y_fit, free_raw_data=False),
            num_boost_round=self.num_boost_round,
        )

        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
        )
        calibrator.fit(self.booster.predict(X_cal), y_cal)
        self.calibrator = calibrator
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None or self.calibrator is None:
            raise RuntimeError("model is not fitted")
        if list(X.columns) != self.feature_names:
            raise ValueError("feature frame does not match training layout")
        raw = self.calibrator.predict(self.booster.predict(X))
        # Isotonic regression saturates: its top bin maps to exactly 1.0, so the model
        # would report "100% risk" for a patient on the strength of a few dozen
        # calibration examples. No finite sample justifies certainty, and a nurse shown
        # 100% has been told something false. Clipped to the resolution the calibration
        # set can actually support.
        return np.clip(raw, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)

    def feature_importance(self) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "gain": self.booster.feature_importance(importance_type="gain"),
                }
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
