"""Measure whether state-level Census indicators add predictive value."""

import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    get_features,
    load_and_prepare,
    merge_census_state_features,
    scale_features,
    split_data,
)


def evaluate(df, include_census):
    """Train and evaluate a logistic-regression model on the validation split."""
    X, y = get_features(
        df, include_category=True, include_census=include_census
    )
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)
    X_train_s, X_val_s, _ = scale_features(X_train, X_val)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    predictions = model.predict(X_val_s)
    probabilities = model.predict_proba(X_val_s)[:, 1]
    return f1_score(y_val, predictions), roc_auc_score(y_val, probabilities)


def main():
    """Run the baseline and optional Census-feature comparison."""
    df = load_and_prepare()

    print("=" * 60)
    print("EXTERNAL CENSUS FEATURE INTEGRATION")
    print("=" * 60)
    print(
        "ACS indicators: median household income, population, and "
        "unemployment rate (state level)"
    )
    print()

    baseline_f1, baseline_auc = evaluate(df, include_census=False)
    print(
        f"Transaction features only: F1={baseline_f1:.3f}, "
        f"AUC-ROC={baseline_auc:.3f}"
    )

    try:
        df_census = merge_census_state_features(df)
    except RuntimeError as exc:
        print("\nCensus comparison skipped:")
        print(f"  {exc}")
        print(
            "\nThe baseline model ran successfully. Set CENSUS_API_KEY and "
            "rerun this script to include the external features."
        )
    else:
        print(f"Transactions matched: {len(df_census):,}")
        census_f1, census_auc = evaluate(df_census, include_census=True)
        print(
            f"Transaction + Census features: F1={census_f1:.3f}, "
            f"AUC-ROC={census_auc:.3f}"
        )


if __name__ == "__main__":
    main()
