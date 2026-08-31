"""Tests for the model layer's invariants — the properties that must hold whatever
the data does, as distinct from the accuracy numbers, which are what they are."""

from __future__ import annotations

import numpy as np
import pytest

from patienttriage.models.conformal import ConformalAbstainer
from patienttriage.models.decision import (
    cost_matrix,
    recommend,
    surge_cost_matrix,
)
from patienttriage.models.ordinal import ACUITY_LEVELS, OrdinalAcuityModel

# --- the cost matrix encodes the project's central judgement --------------------------


def test_undertriage_costs_more_than_overtriage_at_equal_distance():
    costs = cost_matrix()
    # truth level 2 (index 1): recommending 3 versus recommending 1
    under_by_one = costs[1, 2]
    over_by_one = costs[1, 0]
    assert under_by_one > over_by_one


def test_being_right_is_free():
    assert np.allclose(np.diag(cost_matrix()), 0.0)


def test_sending_an_urgent_patient_to_the_waiting_room_carries_the_surcharge():
    costs = cost_matrix()
    # Truth level 2. Going 2 -> 3 crosses into the waiting room; 1 -> 2 does not.
    crossing = costs[1, 2]
    not_crossing = costs[0, 1]
    assert crossing > not_crossing + 1


def test_surge_mode_narrows_the_gap_between_the_two_errors():
    """In surge the scarce bed is what is being rationed, so overtriage starts to
    harm the people not getting it."""
    normal, surge = cost_matrix(), surge_cost_matrix()
    normal_ratio = normal[1, 2] / normal[1, 0]
    surge_ratio = surge[1, 2] / surge[1, 0]
    assert surge_ratio < normal_ratio


# --- the decision rule ----------------------------------------------------------------


def test_confident_certainty_recommends_that_level():
    certain_level_four = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
    assert recommend(certain_level_four)[0].acuity == 4


def test_asymmetry_pulls_the_recommendation_toward_acuity():
    """Most likely is level 3, but a real tail on level 1 should escalate the action.
    That is the cost matrix working, not a bug."""
    probabilities = np.array([[0.15, 0.10, 0.75, 0.0, 0.0]])
    assert probabilities.argmax() == 2  # most likely is level 3
    assert recommend(probabilities)[0].acuity < 3


def test_recommendation_reports_critical_probability():
    r = recommend(np.array([[0.2, 0.3, 0.5, 0.0, 0.0]]))[0]
    assert r.critical_probability == pytest.approx(0.5)


# --- the ordinal model ----------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model():
    rng = np.random.default_rng(0)
    n = 1500
    import pandas as pd

    severity = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": severity + rng.normal(scale=0.5, size=n),
            "b": rng.normal(size=n),
            "c": severity * 2 + rng.normal(scale=0.5, size=n),
        }
    )
    y = np.clip(np.digitize(severity, [-1.2, -0.4, 0.4, 1.2]) + 1, 1, 5)
    return OrdinalAcuityModel(num_boost_round=60).fit(X, y), X, y


def test_probabilities_are_a_distribution(fitted_model):
    model, X, _ = fitted_model
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), len(ACUITY_LEVELS))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_cumulative_probabilities_are_monotone(fitted_model):
    """Four independently fitted boosters can disagree on an individual patient;
    the ordinal structure has to survive that."""
    model, X, _ = fitted_model
    cumulative = model.cumulative_proba(X)
    assert (np.diff(cumulative, axis=1) >= -1e-9).all()


def test_predicting_with_the_wrong_columns_is_refused(fitted_model):
    model, X, _ = fitted_model
    with pytest.raises(ValueError, match="feature frame"):
        model.predict_proba(X.rename(columns={"a": "z"}))


def test_rejects_labels_outside_the_scale():
    import pandas as pd

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="1..5"):
        OrdinalAcuityModel().fit(X, np.array([1, 2, 3, 9]))


# --- conformal abstention -------------------------------------------------------------


def _synthetic_probabilities(rng, y, sharpness=3.0):
    """Probabilities loosely centred on the truth, so coverage is testable."""
    logits = -sharpness * np.abs(np.arange(1, 6)[None, :] - y[:, None])
    logits = logits + rng.normal(scale=1.0, size=logits.shape)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def test_coverage_guarantee_holds_on_unseen_data():
    rng = np.random.default_rng(3)
    y_cal = rng.integers(1, 6, size=4000)
    y_test = rng.integers(1, 6, size=4000)

    abstainer = ConformalAbstainer(alpha=0.1).calibrate(
        _synthetic_probabilities(rng, y_cal), y_cal
    )
    coverage = abstainer.coverage(_synthetic_probabilities(rng, y_test), y_test)
    # Finite-sample slack: the guarantee is 1 - alpha in expectation.
    assert coverage["empirical_coverage"] >= 0.87


def test_tighter_alpha_produces_larger_sets():
    rng = np.random.default_rng(4)
    y = rng.integers(1, 6, size=3000)
    probabilities = _synthetic_probabilities(rng, y)

    loose = ConformalAbstainer(alpha=0.2).calibrate(probabilities, y)
    strict = ConformalAbstainer(alpha=0.01).calibrate(probabilities, y)
    assert strict.coverage(probabilities, y)["mean_set_size"] > (
        loose.coverage(probabilities, y)["mean_set_size"]
    )


def test_an_empty_set_means_abstain_not_a_confident_guess():
    """The tempting fallback — return the single most likely level so something is
    always produced — hands the narrowest output to the patient the model understands
    least. An empty set must stay empty and route to the nurse."""
    rng = np.random.default_rng(5)
    y = rng.integers(1, 6, size=2000)
    abstainer = ConformalAbstainer(alpha=0.3).calibrate(_synthetic_probabilities(rng, y), y)

    flat = np.full((1, 5), 0.2)  # maximally uncertain
    assert abstainer.prediction_sets(flat)[0] == set()
    assert abstainer.should_abstain(flat)[0]


def _abstainer_with_threshold(threshold: float) -> ConformalAbstainer:
    """An abstainer with the threshold pinned, so the boundary logic is tested on its
    own rather than through whatever quantile a random calibration happened to pick."""
    abstainer = ConformalAbstainer(alpha=0.1)
    abstainer.quantile = 1.0 - threshold
    return abstainer


def test_no_abstention_when_the_set_stays_on_one_side_of_the_boundary():
    """Levels 3 and 4 both mean 'can wait'. Abstaining there would spend a nurse's
    attention on a distinction that changes nothing that happens next."""
    abstainer = _abstainer_with_threshold(0.4)
    can_wait = np.array([[0.0, 0.0, 0.55, 0.45, 0.0]])
    assert abstainer.prediction_sets(can_wait)[0] == {3, 4}
    assert not abstainer.should_abstain(can_wait)[0]


def test_abstains_when_the_set_straddles_the_urgent_boundary():
    """Level 2 and level 3 are not a near-miss: one is seen now, the other joins the
    waiting room. A single number here would be doing unearned work."""
    abstainer = _abstainer_with_threshold(0.4)
    straddling = np.array([[0.0, 0.45, 0.45, 0.10, 0.0]])
    decided = np.array([[0.0, 0.0, 0.02, 0.96, 0.02]])

    assert abstainer.prediction_sets(straddling)[0] == {2, 3}
    assert abstainer.should_abstain(straddling)[0]
    assert not abstainer.should_abstain(decided)[0]


def test_uncalibrated_abstainer_refuses_to_guess():
    with pytest.raises(RuntimeError, match="calibrate"):
        ConformalAbstainer().prediction_sets(np.full((2, 5), 0.2))
