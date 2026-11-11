# 08_discount_thresholds.py 
# Uses the per-category Random Forest models to find the discount level where P(unprofitable) crosses 50% for each category. Prints a recommendation as well.

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (load_and_prepare, get_features_ordinal,
                         split_data, PROJECT_ROOT, RANDOM_STATE)

# ── 1. Load and prepare ──────────────────────────────────────
df = load_and_prepare()
X, y, encoder = get_features_ordinal(df, include_category=True)
(X_train, X_val, X_test,
 y_train, y_val, y_test,
 cat_train, cat_val, cat_test) = split_data(X, y, df)

print("=" * 60)
print("DISCOUNT THRESHOLD ANALYSIS")
print("=" * 60)

# ── 2. Train per-category models ─────────────────────────────
categories = ["Furniture", "Office Supplies", "Technology"]
models = {}

for cat in categories:
    mask = (cat_train == cat)
    X_tr = X_train[mask].drop(columns=["Category"])
    y_tr = y_train[mask]

    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                random_state=RANDOM_STATE)
    rf.fit(X_tr, y_tr)
    models[cat] = rf

# ── 3. Find threshold for each category ──────────────────────
# We sweep discount from 0% to 50% and compute average P(unprofitable) using median values for all other features.

discount_levels = np.arange(0, 0.51, 0.05)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, cat in enumerate(categories):
    # Get the training data for this category to compute medians
    mask = (cat_train == cat)
    X_tr = X_train[mask].drop(columns=["Category"])

    # Create a "template" row with median values
    template = X_tr.median().values

    # Sweep discount and compute predicted probability
    probs = []
    for d in discount_levels:
        row = template.copy()
        row[1] = d  # Discount is column index 1
        prob = models[cat].predict_proba(
            pd.DataFrame([row], columns=X_tr.columns)
        )[0][1]
        probs.append(prob * 100)

    # Find the threshold where P(unprof) > 50%
    threshold = None
    for d, p in zip(discount_levels, probs):
        if p > 50:
            threshold = d
            break

    # Print recommendation
    print(f"\n{cat}:")
    if threshold is not None:
        print(f"  Recommended max discount: {int(threshold*100)}%")
        print(f"  At {int(threshold*100)}% discount, "
              f"P(unprofitable) = {probs[list(discount_levels).index(threshold)]:.1f}%")
    else:
        print(f"  P(unprofitable) stays below 50% up to 50% discount")

    # Plot
    axes[i].plot(discount_levels * 100, probs,
                 marker="o", linewidth=2,
                 color=["#e74c3c", "#3498db", "#2ecc71"][i])
    axes[i].axhline(y=50, color="gray", linestyle="--", alpha=0.7,
                    label="50% threshold")
    if threshold is not None:
        axes[i].axvline(x=threshold * 100, color="red",
                        linestyle=":", alpha=0.7)
    axes[i].set_title(f"{cat}")
    axes[i].set_xlabel("Discount (%)")
    axes[i].set_ylabel("P(Unprofitable) %")
    axes[i].set_ylim(0, 100)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle("Predicted Unprofitability by Discount Level", fontsize=14)
plt.tight_layout()
fig_dir = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "discount_thresholds.png")
plt.savefig(fig_path, dpi=150)
print(f"\nFigure saved in {fig_path}")
