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

from data_loader import apply_plot_style, CATEGORY_COLORS
apply_plot_style(16)
fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))

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

    # Plot: model prediction (line), observed rate in the data (hollow squares, only where n >= 20)
    color = CATEGORY_COLORS[cat]
    axes[i].plot(discount_levels * 100, probs, marker="o", linewidth=2.5,
                 color=color)
    obs = df[df["Category"] == cat].groupby("Discount")["Unprofitable"].agg(["mean", "count"])
    obs = obs[obs["count"] >= 20]
    axes[i].scatter(obs.index * 100, obs["mean"] * 100, marker="s", s=90,
                    facecolors="none", edgecolors="black", linewidths=1.8, zorder=5)
    max_d = df[df["Category"] == cat]["Discount"].max() * 100
    if max_d < 50:
        axes[i].axvspan(max_d, 50, color="gray", alpha=0.15)
    axes[i].axhline(y=50, color="gray", linestyle="--", alpha=0.8)
    if threshold is not None:
        axes[i].axvline(x=threshold * 100, color="red", linestyle=":", alpha=0.8)
        axes[i].text(threshold * 100 + 1, 4, f"{int(threshold*100)}%", color="red", fontsize=14, fontweight="bold")
    axes[i].set_title(f"{cat}")
    axes[i].set_xlabel("Discount (%)")
    axes[i].set_ylabel("Probability of loss (%)")
    axes[i].set_ylim(0, 108)
    axes[i].set_yticks([0, 20, 40, 60, 80, 100])
    axes[i].grid(True, alpha=0.3)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_items = [
    Line2D([0], [0], color="black", marker="o", linewidth=2.5, label="Model prediction"),
    Line2D([0], [0], marker="s", linestyle="", markerfacecolor="none", markeredgecolor="black",
           markeredgewidth=1.8, markersize=11, label="Observed rate (levels with n \u2265 20)"),
    Line2D([0], [0], color="gray", linestyle="--", label="50% risk"),
    Line2D([0], [0], color="red", linestyle=":", label="Model crosses 50%"),
    Patch(facecolor="gray", alpha=0.25, label="No transactions at this discount"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=3, fontsize=13, frameon=False)
plt.tight_layout(rect=[0, 0.14, 1, 0.97])
fig_dir = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "discount_thresholds.png")
plt.savefig(fig_path, dpi=200)
print(f"\nFigure saved in {fig_path}")