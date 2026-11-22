# 07_rf_per_category.py — Random Forest per Category
# Train 3 separate Random Forests (one per category) and use ordinal encoding instead of one-hot (trees handle it natively, less sparse data). Also prints feature importance from each forest.

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (load_and_prepare, get_features_ordinal,
                         split_data, PROJECT_ROOT, RANDOM_STATE)

# ── 1. Load and split ────────────────────────────────────────
df = load_and_prepare()
X, y, encoder = get_features_ordinal(df, include_category=True)
(X_train, X_val, X_test,
 y_train, y_val, y_test,
 cat_train, cat_val, cat_test) = split_data(X, y, df)

print("=" * 60)
print("RANDOM FOREST PER CATEGORY (Ordinal Encoding)")
print("=" * 60)

# ── 2. Train one RF per category ─────────────────────────────
categories = ["Furniture", "Office Supplies", "Technology"]
results = {}

for cat in categories:
    # Filter to this category
    train_mask = (cat_train == cat)
    val_mask = (cat_val == cat)

    X_tr = X_train[train_mask].drop(columns=["Category"])
    y_tr = y_train[train_mask]
    X_va = X_val[val_mask].drop(columns=["Category"])
    y_va = y_val[val_mask]

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE
    )
    rf.fit(X_tr, y_tr)

    # Evaluate
    y_pred = rf.predict(X_va)
    y_prob = rf.predict_proba(X_va)[:, 1]
    f1 = f1_score(y_va, y_pred)
    auc = roc_auc_score(y_va, y_prob) if len(np.unique(y_va)) > 1 else 0

    # Feature importance (built into Random Forest)
    importances = dict(zip(X_tr.columns, rf.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

    n_unprof = y_va.sum()
    results[cat] = {"f1": f1, "auc": auc, "n": len(y_va),
                    "unprof": n_unprof, "top5": top5, "model": rf,
                    "features": X_tr.columns.tolist()}

    print(f"\n--- {cat} ---")
    print(f"  Train: {len(X_tr)} rows | Val: {len(X_va)} rows "
          f"({n_unprof} unprofitable, {n_unprof/len(y_va):.1%})")
    print(f"  F1 (unprofitable): {f1:.3f}")
    print(f"  AUC-ROC:           {auc:.3f}")
    print(f"  Top 5 features:")
    for feat, imp in top5:
        print(f"    {feat:30s}: {imp:.3f}")

# ── 3. Summary table ─────────────────────────────────────────
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"{'Category':25s} {'n':>4s} {'Unprof%':>7s} {'F1':>6s} {'AUC':>6s}")
print("-" * 50)
for cat in categories:
    r = results[cat]
    print(f"{cat:25s} {r['n']:4d} {r['unprof']/r['n']:7.1%} "
          f"{r['f1']:6.3f} {r['auc']:6.3f}")

# ── 4. Feature importance figure ──────────────────
from data_loader import apply_plot_style, CATEGORY_COLORS
apply_plot_style(16)
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))

for i, cat in enumerate(categories):
    top5 = results[cat]["top5"]
    names = [x[0].replace("_", " ") for x in top5]
    values = [x[1] for x in top5]

    axes[i].barh(names[::-1], values[::-1], color=CATEGORY_COLORS[cat])
    for yy, v in enumerate(values[::-1]):
        axes[i].text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=13)
    axes[i].set_title(f"{cat}")
    axes[i].set_xlabel("Importance (share of total)")
    axes[i].set_xlim(0, max(values) * 1.25)

plt.tight_layout()
fig_dir = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "rf_feature_importance_per_category.png")
plt.savefig(fig_path, dpi=200)
print(f"\nFigure saved in {fig_path}")