"""Run the queue simulation on the NHAMCS test cohort.

    python -m patienttriage.sim.run

Two things this script gets right that the obvious version of it gets wrong.

**The assisted ordering escalates the nurse; it does not replace her.** The tempting
setup is to queue by the model's recommended acuity and compare against the nurse's.
That measures a system nobody would deploy and this architecture explicitly forbids:
the model is not allowed to move a patient *down*. So the assisted priority here is
`min(nurse_acuity, model_signal)` - the nurse's judgement stands, and the model may
only pull someone forward. Anything else is measuring the wrong system.

**Capacity is derived from the load, not guessed.** A department where arrivals times
service time exceed capacity has an unbounded queue, and every wait statistic from it
is an artefact of how long the simulation ran. Capacity is therefore set from a target
utilisation, and utilisation is what gets swept.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from patienttriage.data.nhamcs import labelled_mask, load_cohort
from patienttriage.eval.operating_point import choose_for_budget, observed_critical_rate, sweep
from patienttriage.features.build import build_features
from patienttriage.models.decision import cost_matrix, recommend
from patienttriage.models.deterioration import DeteriorationModel
from patienttriage.models.ordinal import OrdinalAcuityModel
from patienttriage.rules.engine import RuleEngine
from patienttriage.sim.queue import build_arrival_stream, compare, summarise

TRAIN_YEARS = [2021]
TEST_YEARS = [2022]

ARRIVALS_PER_HOUR = 9.0
HOURS = 24 * 7
REPLICATIONS = 25
TARGET_UTILISATION = 0.85
WATCH_BUDGET = 0.05
"""Share of arrivals the waiting-room watcher may escalate. An alert budget."""


def _capacity(arrivals_per_hour: float, mean_service_minutes: float, utilisation: float) -> int:
    """Treatment spaces needed to run at the given utilisation. Must exceed the load."""
    return max(1, math.ceil(arrivals_per_hour * mean_service_minutes / 60 / utilisation))


def _run(arrivals, capacity: int) -> pd.DataFrame:
    nurse, assisted = compare(arrivals, capacity)
    return summarise(nurse, assisted)


def main(data_dir: Path, output_dir: Path) -> None:
    snapshots, labels = load_cohort(TRAIN_YEARS + TEST_YEARS, data_dir)
    features = build_features(snapshots)
    labels = labels.reset_index(drop=True)

    usable = labelled_mask(labels).to_numpy()
    is_train = labels["survey_year"].isin(TRAIN_YEARS).to_numpy()
    is_test = labels["survey_year"].isin(TEST_YEARS).to_numpy()
    train_rows, test_rows = usable & is_train, usable & is_test

    X_train = features[train_rows]
    y_train = labels.loc[train_rows, "acuity"].astype(int).to_numpy()
    X_test = features[test_rows]
    y_test = labels.loc[test_rows, "acuity"].astype(int).to_numpy()

    X_fit, _, y_fit, _ = train_test_split(
        X_train, y_train, test_size=0.25, random_state=11, stratify=y_train
    )
    acuity_model = OrdinalAcuityModel().fit(X_fit, y_fit)
    probabilities = acuity_model.predict_proba(X_test)

    deterioration = DeteriorationModel().fit(
        features[is_train], labels.loc[is_train, "critical_outcome"].to_numpy()
    )
    risk = deterioration.predict_proba(X_test)

    engine = RuleEngine()
    rule_floors = np.array(
        [
            engine.evaluate(s).acuity_floor
            for s, keep in zip(snapshots, test_rows, strict=True)
            if keep
        ]
    )

    curve = sweep(y_test, probabilities, rule_floors)
    chosen = choose_for_budget(curve, observed_critical_rate(y_test))
    costs = cost_matrix(undertriage=float(chosen["undertriage_cost"]), critical_surcharge=0.0)
    model_acuity = np.minimum(
        np.array([r.acuity for r in recommend(probabilities, costs)]), rule_floors
    )

    # Escalation only: the nurse's level stands unless the model argues for sooner.
    assisted_acuity = np.minimum(y_test, model_acuity)

    # The watch-board intervention: within the alert budget, the highest-risk patients
    # the nurse placed in the waiting room get pulled up to level 2. Nobody moves down.
    threshold = float(np.quantile(risk, 1 - WATCH_BUDGET))
    assisted_watch = np.where((risk >= threshold) & (y_test >= 3), 2, y_test)

    test_labels = labels[test_rows].reset_index(drop=True)
    critical = test_labels["critical_outcome"].to_numpy().astype(bool)

    # NHAMCS LOV is the whole length of visit, which is what occupies a treatment space.
    # Missing values take the cohort median: a visit with no recorded duration still
    # occupies a bed, so dropping it would understate the load.
    length_of_visit = test_labels["length_of_visit_minutes"].to_numpy(dtype=float)
    median_lov = float(np.nanmedian(length_of_visit))
    service = np.clip(
        np.where(np.isnan(length_of_visit), median_lov, length_of_visit), 5.0, 600.0
    )
    mean_service = float(service.mean())
    capacity = _capacity(ARRIVALS_PER_HOUR, mean_service, TARGET_UTILISATION)

    print("=" * 78)
    print("QUEUE SIMULATION")
    print("=" * 78)
    print(
        f"\ncohort            {len(y_test)} test-year visits"
        f"\nmean service time {mean_service:.0f} min (NHAMCS length of visit)"
        f"\narrival rate      {ARRIVALS_PER_HOUR:.0f}/hour over {HOURS // 24} days"
        f"\ncapacity          {capacity} treatment spaces"
        f" (sized for {TARGET_UTILISATION:.0%} utilisation)"
        f"\nreplications      {REPLICATIONS}"
    )

    scenarios = {
        "acuity model escalating the nurse": assisted_acuity,
        f"watch board escalating the nurse (top {WATCH_BUDGET:.0%} risk)": assisted_watch,
    }

    # Two regimes, because they give opposite answers and only reporting the flattering
    # one would be dishonest. At 85% utilisation a department with this many spaces
    # barely queues at all, so there is nothing for any ordering to improve. The regime
    # this design actually needs to handle - the worst case, not the average - is the
    # overloaded one, where arrivals exceed the rate the department can clear and the
    # queue builds through the week.
    for utilisation, regime in ((TARGET_UTILISATION, "normal running"), (1.15, "surge")):
        seats = _capacity(ARRIVALS_PER_HOUR, mean_service, utilisation)
        print("\n" + "=" * 78)
        print(f"{regime.upper()} — utilisation {utilisation:.0%}, {seats} treatment spaces")
        print("=" * 78)

        for name, assisted_priority in scenarios.items():
            moved = int((assisted_priority < y_test).sum())
            print("\n" + "-" * 78)
            print(f"{name}")
            print(f"  escalates {moved} of {len(y_test)} patients ({moved / len(y_test):.1%})")
            print("-" * 78)

            frames = []
            for seed in range(REPLICATIONS):
                stream = build_arrival_stream(
                    y_test, assisted_priority, critical, service,
                    ARRIVALS_PER_HOUR, HOURS, np.random.default_rng(100 + seed),
                )
                frames.append(_run(stream, seats))
            pooled = pd.concat(frames).groupby("group", sort=False).mean(numeric_only=True)
            print(pooled.to_string(float_format=lambda v: f"{v:.1f}"))

    print(
        "\nRe-ordering a queue is zero-sum at fixed capacity. A minute saved for a"
        "\ncritically unwell patient is a minute somebody else waits, so both halves of"
        "\nthe trade are shown. Negative change means a shorter wait."
    )

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("LOAD SWEEP — how much the department has to be struggling to benefit")
    print("=" * 78)
    print(
        "\nAn empty department has nothing to reorder; a saturated one cannot rescue"
        "\nanybody. Whatever benefit exists lives in between.\n"
    )

    rows = []
    for utilisation in (0.80, 0.90, 0.95, 1.05, 1.15, 1.30):
        seats = _capacity(ARRIVALS_PER_HOUR, mean_service, utilisation)
        per_seed = []
        for seed in range(REPLICATIONS):
            stream = build_arrival_stream(
                y_test, assisted_watch, critical, service,
                ARRIVALS_PER_HOUR, HOURS, np.random.default_rng(200 + seed),
            )
            nurse_run, assisted_run = compare(stream, seats)
            per_seed.append(
                {
                    "critical_p90_nurse": nurse_run[nurse_run["critical"]][
                        "wait_minutes"
                    ].quantile(0.9),
                    "critical_p90_assisted": assisted_run[assisted_run["critical"]][
                        "wait_minutes"
                    ].quantile(0.9),
                    "others_median_nurse": nurse_run[~nurse_run["critical"]][
                        "wait_minutes"
                    ].median(),
                    "others_median_assisted": assisted_run[~assisted_run["critical"]][
                        "wait_minutes"
                    ].median(),
                }
            )
        averaged = pd.DataFrame(per_seed).mean()
        rows.append(
            {
                "utilisation": utilisation,
                "spaces": seats,
                "crit_p90_nurse": averaged["critical_p90_nurse"],
                "crit_p90_assisted": averaged["critical_p90_assisted"],
                "crit_p90_change": averaged["critical_p90_assisted"]
                - averaged["critical_p90_nurse"],
                "others_med_change": averaged["others_median_assisted"]
                - averaged["others_median_nurse"],
            }
        )
    load = pd.DataFrame(rows)
    print(load.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    output_dir.mkdir(parents=True, exist_ok=True)
    load.to_csv(output_dir / "simulation_load_sweep.csv", index=False)
    print(f"\nartifacts written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
