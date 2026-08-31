"""Train and evaluate the acuity model on NHAMCS.

Run with:  python -m patienttriage.models.train

Validation is a **temporal** split: fit on the 2021 survey year, test on 2022. A
random split would let the model see visits from the same weeks, the same seasonal
respiratory wave and the same departments it is then tested on, and would report a
number the department will never reproduce. Temporal splitting is how the model will
actually be used — trained on the past, run on today.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from patienttriage.data.nhamcs import labelled_mask, load_cohort
from patienttriage.data.quality import QualityReport
from patienttriage.eval.metrics import (
    confusion,
    evaluate,
    expected_calibration_error,
    ranking_quality,
    reliability_table,
    sensitivity_at_budget,
    stratified_undertriage,
)
from patienttriage.eval.operating_point import choose_for_budget, observed_critical_rate, sweep
from patienttriage.features.build import build_features
from patienttriage.models.conformal import ConformalAbstainer
from patienttriage.models.decision import cost_matrix, recommend, surge_cost_matrix
from patienttriage.models.deterioration import DeteriorationModel
from patienttriage.models.explain import explain, shap_values_for
from patienttriage.models.ordinal import OrdinalAcuityModel
from patienttriage.rules.engine import RuleEngine

TRAIN_YEARS = [2021]
TEST_YEARS = [2022]


def main(data_dir: Path, output_dir: Path) -> None:
    report = QualityReport()

    print("=" * 78)
    print("LOADING")
    print("=" * 78)
    snapshots, labels = load_cohort(TRAIN_YEARS + TEST_YEARS, data_dir, report)
    print(f"visits: {len(snapshots)}")
    print("implausible values discarded (treated as not recorded):")
    print(report.summary(len(labels)))

    features = build_features(snapshots)
    labels = labels.reset_index(drop=True)

    usable = labelled_mask(labels).to_numpy()
    is_train = labels["survey_year"].isin(TRAIN_YEARS).to_numpy()
    is_test = labels["survey_year"].isin(TEST_YEARS).to_numpy()

    train_rows = usable & is_train
    test_rows = usable & is_test

    X_train, y_train = features[train_rows], labels.loc[train_rows, "acuity"].astype(int)
    X_test, y_test = features[test_rows], labels.loc[test_rows, "acuity"].astype(int)

    # A slice of the training year held back from fitting entirely. Conformal coverage
    # is only valid on data exchangeable with the test set, and a model is systematically
    # overconfident on rows it was fitted on — calibrating there produces nonconformity
    # scores that are too small, prediction sets that are too narrow, and a guarantee
    # that silently does not hold.
    X_fit, X_conformal, y_fit, y_conformal = train_test_split(
        X_train, y_train.to_numpy(), test_size=0.25, random_state=11, stratify=y_train
    )

    print(f"\ntrain {TRAIN_YEARS}: {len(X_train)} labelled visits")
    print(f"test  {TEST_YEARS}: {len(X_test)} labelled visits")
    print("\ntrain acuity distribution:")
    print(y_train.value_counts().sort_index().to_string())

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BASELINES")
    print("=" * 78)
    y_test_array = y_test.to_numpy()

    print("\n[everyone level 3] — what accuracy looks like when it means nothing")
    print(evaluate(y_test_array, np.full_like(y_test_array, 3)).summary())

    engine = RuleEngine()
    test_snapshots = [s for s, keep in zip(snapshots, test_rows, strict=True) if keep]
    rule_floors = np.array([engine.evaluate(s).acuity_floor for s in test_snapshots])
    print("\n[rule layer alone] — deterministic red flags, no model")
    print(evaluate(y_test_array, rule_floors).summary())

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ORDINAL MODEL")
    print("=" * 78)
    model = OrdinalAcuityModel().fit(X_fit, y_fit)
    probabilities = model.predict_proba(X_test)

    print("\n[most likely level] — no cost asymmetry applied")
    argmax = model.predict(X_test)
    print(evaluate(y_test_array, argmax).summary())

    print("\n[expected-cost recommendation] — undertriage weighted 8x")
    recommended = np.array([r.acuity for r in recommend(probabilities)])
    print(evaluate(y_test_array, recommended).summary())

    print("\n[with the rule layer ratcheting on top]")
    print(evaluate(y_test_array, np.minimum(recommended, rule_floors)).summary())

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("OPERATING POINT — what each undertriage cost actually buys")
    print("=" * 78)
    print(
        "\nA cost ratio picked in the abstract recommends level 1-2 for most of the"
        "\ndepartment, which no department can staff. The choice belongs against a real"
        "\ncapacity, so here is the whole curve.\n"
    )
    curve = sweep(y_test_array, probabilities, rule_floors)
    print(
        curve.drop(columns=["critical_surcharge"])
        .rename(
            columns={
                "undertriage_cost": "cost",
                "flagged_rate": "flagged",
                "undertriage_rate": "under",
                "severe_undertriage_rate": "under_L1",
                "overtriage_rate": "over",
                "critical_sensitivity": "sens",
                "critical_precision": "prec",
                "mean_cost": "cost_mean",
            }
        )
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    capacity = observed_critical_rate(y_test_array)
    print(
        f"\nnurses assigned level 1-2 to {capacity:.1%} of arrivals - the department is"
        f"\nalready staffed for roughly that volume, so we take it as the capacity."
    )
    chosen = choose_for_budget(curve, capacity)
    print(
        f"\nchosen: undertriage cost {chosen['undertriage_cost']:.1f}"
        f" -> flags {chosen['flagged_rate']:.1%} of arrivals,"
        f" catches {chosen['critical_sensitivity']:.1%} of level 1-2"
    )

    final_costs = cost_matrix(
        undertriage=float(chosen["undertriage_cost"]), critical_surcharge=0.0
    )
    final = np.minimum(
        np.array([r.acuity for r in recommend(probabilities, final_costs)]), rule_floors
    )
    print("\n[deployed configuration — budgeted cost matrix plus rule layer]")
    print(evaluate(y_test_array, final).summary())

    print("\n[surge mode] — same probabilities, department-declared cost matrix")
    surge = np.array([r.acuity for r in recommend(probabilities, surge_cost_matrix())])
    print(evaluate(y_test_array, np.minimum(surge, rule_floors)).summary())

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DISCRIMINATION — is the ceiling the model, or the label?")
    print("=" * 78)
    print(
        "\nThe acuity model predicts the nurse's judgement. The deterioration model"
        "\npredicts what actually happened. If the second ranks better than the first,"
        "\nthe limit is the label, not the features - and the watch board is where the"
        "\nvalue is.\n"
    )

    critical_probability = probabilities[:, :2].sum(axis=1)
    critical_truth = np.isin(y_test_array, (1, 2)).astype(int)
    acuity_ranking = ranking_quality(critical_truth, critical_probability)
    print("acuity model, predicting the nurse's level 1-2:")
    for key, value in acuity_ranking.items():
        print(f"  {key:20} {value:.4f}")

    # The deterioration model trains on every visit, labelled or not: its target is the
    # outcome, which exists whether or not the department recorded a triage level.
    outcome_train = is_train
    outcome_test = is_test
    deterioration = DeteriorationModel().fit(
        features[outcome_train], labels.loc[outcome_train, "critical_outcome"].to_numpy()
    )
    outcome_truth = labels.loc[outcome_test, "critical_outcome"].to_numpy()
    outcome_score = deterioration.predict_proba(features[outcome_test])

    print("\ndeterioration model, predicting death in ED or critical care admission:")
    for key, value in ranking_quality(outcome_truth, outcome_score).items():
        print(f"  {key:20} {value:.4f}")

    print("\nnurse's own acuity, as a predictor of the same outcome (the human baseline):")
    nurse_rows = usable & is_test
    nurse_score = -labels.loc[nurse_rows, "acuity"].to_numpy().astype(float)
    for key, value in ranking_quality(
        labels.loc[nurse_rows, "critical_outcome"].to_numpy(), nurse_score
    ).items():
        print(f"  {key:20} {value:.4f}")

    print("\nwatch board: critical outcomes caught within an alert budget")
    print(sensitivity_at_budget(outcome_truth, outcome_score).to_string(index=False))

    print("\nsame budget, using the nurse's acuity to rank instead:")
    print(
        sensitivity_at_budget(
            labels.loc[nurse_rows, "critical_outcome"].to_numpy(), nurse_score
        ).to_string(index=False)
    )

    print("\nwhat the deterioration model leans on:")
    print(deterioration.feature_importance().head(12).to_string(index=False))

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ABSTENTION — the patients the system declines to score")
    print("=" * 78)
    abstainer = ConformalAbstainer(alpha=0.1).calibrate(
        model.predict_proba(X_conformal), y_conformal
    )
    print(f"\ncalibrated on {len(X_conformal)} held-out visits the model never saw")
    print("coverage on the test year:")
    for key, value in abstainer.coverage(probabilities, y_test_array).items():
        print(f"  {key:22} {value:.4f}")

    abstains = abstainer.should_abstain(probabilities)
    if abstains.any():
        confident = ~abstains
        answered = evaluate(y_test_array[confident], final[confident])
        declined = evaluate(y_test_array[abstains], final[abstains])
        print(
            f"\n  exact agreement where the system answered   {answered.exact_agreement:6.2%}"
            f"  (n={confident.sum()})"
        )
        print(
            f"  exact agreement where it abstained          {declined.exact_agreement:6.2%}"
            f"  (n={abstains.sum()})"
        )
        print(
            "\nAbstention is doing its job when the second number is the worse one - the"
            "\nsystem declining precisely the cases it would have got wrong."
        )

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPLANATIONS — what a nurse would see, for the highest-risk arrivals")
    print("=" * 78)
    test_snapshot_list = [s for s, keep in zip(snapshots, is_test, strict=True) if keep]
    top = np.argsort(-outcome_score)[:4]
    contributions = shap_values_for(deterioration.booster, features[outcome_test])
    for rank, i in enumerate(top, start=1):
        s = test_snapshot_list[i]
        print(
            f"\n{rank}. risk {outcome_score[i]:.1%}  |  age {s.age_years:.0f} {s.sex}, "
            f"{s.arrival_mode.replace('_', ' ')}  |  outcome: "
            f"{'critical' if outcome_truth[i] else 'not critical'}"
        )
        for driver in explain(s, features[outcome_test], contributions[i]):
            print(f"     - {driver.sentence}")

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CONFUSION — rows: nurse assigned, columns: system recommended")
    print("=" * 78)
    print(confusion(y_test_array, final).to_string())

    print("\n" + "=" * 78)
    print("CALIBRATION — P(level 1 or 2)")
    print("=" * 78)
    critical_probability = probabilities[:, :2].sum(axis=1)
    critical_truth = np.isin(y_test_array, (1, 2)).astype(int)
    ece = expected_calibration_error(critical_truth, critical_probability)
    print(f"expected calibration error: {ece:.4f}")
    print(reliability_table(critical_truth, critical_probability).to_string(index=False))

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FAIRNESS — undertriage by subgroup (release gate, not an appendix)")
    print("=" * 78)
    test_labels = labels[test_rows].reset_index(drop=True)
    test_features = X_test.reset_index(drop=True)
    for name, series in (
        ("sex", test_features["sex"].astype(str)),
        ("age band", test_features["age_years__band"].astype(str)),
        ("arrival by ambulance", test_features["arrival_mode__ambulance"].map({1.0: "yes", 0.0: "no"})),
    ):
        print(f"\nby {name}:")
        print(stratified_undertriage(y_test_array, final, series).to_string(index=False))

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("WHAT THE MODEL LEANS ON")
    print("=" * 78)
    print(model.feature_importance().head(20).to_string(index=False))

    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("OUTCOME CHECK — does the recommendation track what happened?")
    print("=" * 78)
    outcome = test_labels["critical_outcome"].to_numpy()
    print("critical outcome rate (died in ED or admitted to critical care):")
    for label, assignment in (("nurse assigned", y_test_array), ("system recommended", final)):
        rates = pd.DataFrame({"level": assignment, "critical": outcome})
        print(f"\n  by {label}:")
        print(
            rates.groupby("level")["critical"]
            .agg(["size", "sum", "mean"])
            .to_string()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output_dir / "operating_points.csv", index=False)
    pd.DataFrame(final_costs).to_csv(output_dir / "cost_matrix.csv", index=False)
    model.feature_importance().to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"\nartifacts written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
