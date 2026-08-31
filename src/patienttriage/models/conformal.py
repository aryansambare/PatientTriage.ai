"""Knowing when not to answer.

A triage assistant that always produces a number is lying about part of its output.
Some patients genuinely sit on a boundary — an elderly patient with a soft blood
pressure and a vague complaint might be a 2 or might be a 4, and the honest answer is
that the available data does not separate those. Printing "3" there does real harm:
it looks like the same kind of statement as the confident ones, and automation bias
means it will be treated like one.

Split conformal prediction turns that into a guarantee rather than a vibe. Given a
held-out calibration set and a target coverage of 1 - alpha, it returns for each new
patient a *set* of acuity levels that contains the truth at least (1 - alpha) of the
time, over the long run, with no assumptions about the model being well specified.
The set is small when the model is sure and large when it is not.

The system then abstains when the set spans a decision boundary — when it contains
both an urgent level (1-2) and a non-urgent one (3-5) — because that is precisely the
case where a single recommended number would be doing unearned work. Abstention is
routed to the nurse as "uncertain, your judgement required", which is not a failure
of the system: it is the system declining to add noise to a hard call.

Coverage is validated empirically on the test set, because a guarantee that is stated
but never checked is just a citation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from patienttriage.models.ordinal import ACUITY_LEVELS

URGENT_LEVELS = frozenset({1, 2})


@dataclass
class ConformalAbstainer:
    """Split-conformal prediction sets over acuity levels."""

    alpha: float = 0.1
    """Miscoverage rate. 0.1 targets prediction sets containing the truth 90% of time."""

    quantile: float = field(default=float("nan"), init=False)

    def calibrate(self, probabilities: np.ndarray, y_true: np.ndarray) -> ConformalAbstainer:
        """Fit the threshold on data the model has never seen.

        The nonconformity score is 1 - P(true label): high when the model assigned
        little mass to what actually happened. The threshold is the (1 - alpha)
        quantile of those scores, with the finite-sample correction that makes the
        coverage guarantee exact rather than asymptotic.
        """
        probabilities = np.atleast_2d(probabilities)
        y_true = np.asarray(y_true, dtype=int)
        if len(probabilities) != len(y_true):
            raise ValueError("probabilities and labels must have the same length")

        index = np.searchsorted(np.asarray(ACUITY_LEVELS), y_true)
        scores = 1.0 - probabilities[np.arange(len(y_true)), index]

        n = len(scores)
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = float(np.quantile(scores, level, method="higher"))
        return self

    def prediction_sets(self, probabilities: np.ndarray) -> list[set[int]]:
        """Levels plausible enough to keep, per patient.

        A set can come back **empty**, and that is a meaningful answer rather than a
        failure: it says no acuity level carries enough support to be asserted at the
        required confidence. That is the strongest abstention signal the method can
        produce, and `should_abstain` treats it as such.

        The tempting alternative — falling back to the single most likely level so
        something is always returned — inverts the whole point. It hands the narrowest,
        most confident-looking output to the patient the model understands least.
        """
        if np.isnan(self.quantile):
            raise RuntimeError("call calibrate() before requesting prediction sets")
        probabilities = np.atleast_2d(probabilities)
        keep = probabilities >= (1.0 - self.quantile)
        return [
            {int(ACUITY_LEVELS[i]) for i, k in enumerate(mask) if k} for mask in keep
        ]

    def should_abstain(self, probabilities: np.ndarray) -> np.ndarray:
        """True where the system should show "uncertain" instead of a level.

        Two cases qualify. The set may be **empty** — nothing is supportable at this
        confidence. Or it may **straddle the urgent boundary**, containing both a
        level 1-2 and a level 3-5: the model cannot tell whether this patient can
        safely wait, which is the only question the recommendation exists to answer.

        A set spanning 3 and 4 is not abstention-worthy. Both mean "can wait", the
        recommendation is actionable, and abstaining there would spend the nurse's
        attention on a distinction that does not change what happens next.
        """
        return np.array(
            [
                not s or (bool(s & URGENT_LEVELS) and bool(s - URGENT_LEVELS))
                for s in self.prediction_sets(probabilities)
            ]
        )

    def coverage(self, probabilities: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Empirical check that the guarantee holds, plus what it cost in set size."""
        sets = self.prediction_sets(probabilities)
        y_true = np.asarray(y_true, dtype=int)
        covered = np.array([int(truth) in s for s, truth in zip(sets, y_true, strict=True)])
        sizes = np.array([len(s) for s in sets])
        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "abstention_rate": float(self.should_abstain(probabilities).mean()),
        }
