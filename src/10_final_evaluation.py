# 10_final_evaluation.py
# This script is run ONCE, after all model selection is done. It trains on train+validation combined, then evaluates on the test set that has never been seen.
# Reports F1 and AUC-ROC with 95% bootstrap confidence intervals.

import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score, roc_auc_score,
                             classification_report)

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (load_and_prepare, get_features_ordinal,
                         split_data, PROJECT_ROOT, RANDOM_STATE)

# ── 1. Load and split ────────────────────────────────────────
df = load_and_prepare()
X, y, encoder = get_features_ordinal(df, include_category=True)
(X_train, X_val, X_test,
 y_train, y_val, y_test,
 cat_train, cat_val, cat_test) = split_data(X, y, df)

# ── 2. Combine train + validation for final training ─────────
X_final_train = pd.concat([X_train, X_val])
y_final_train = np.concatenate([y_train, y_val])

print("=" * 60)
print("FINAL EVALUATION ON HELD-OUT TEST SET")
print("=" * 60)
print(f"\nFinal training size: {len(X_final_train)}")
print(f"Test size:           {len(X_test)}")
print(f"Test unprofitable:   {y_test.sum()} ({y_test.mean():.1%})")

# ── 3. Train final model ─────────────────────────────────────
gbt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1,
    max_depth=4, random_state=RANDOM_STATE
)
gbt.fit(X_final_train, y_final_train)

y_pred = gbt.predict(X_test)
y_prob = gbt.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# ── 4. Bootstrap confidence intervals ────────────────────────
n_bootstrap = 1000
f1_scores = []
auc_scores = []
rng = np.random.RandomState(RANDOM_STATE)

for _ in range(n_bootstrap):
    idx = rng.choice(len(y_test), size=len(y_test), replace=True)
    y_b = y_test[idx]
    pred_b = y_pred[idx]
    prob_b = y_prob[idx]

    # Skip if bootstrap sample has only one class
    if len(np.unique(y_b)) < 2:
        continue

    f1_scores.append(f1_score(y_b, pred_b))
    auc_scores.append(roc_auc_score(y_b, prob_b))

f1_lo, f1_hi = np.percentile(f1_scores, [2.5, 97.5])
auc_lo, auc_hi = np.percentile(auc_scores, [2.5, 97.5])

# ── 5. Print results ─────────────────────────────────────────
print(f"\n--- Global Gradient Boosting (TEST SET) ---")
print(f"  Accuracy:  {(y_pred == y_test).mean():.3f}")
print(f"  F1:        {f1:.3f}  (95% CI: [{f1_lo:.3f}, {f1_hi:.3f}])")
print(f"  AUC-ROC:   {auc:.3f}  (95% CI: [{auc_lo:.3f}, {auc_hi:.3f}])")

print(f"\n  Full classification report:")
print(classification_report(y_test, y_pred,
                            target_names=["Profitable", "Unprofitable"]))

# ── 6. Per-category test performance ─────────────────────────
print(f"{'-' * 60}")
print("PER-CATEGORY TEST PERFORMANCE")
print(f"{'-' * 60}")

for cat in ["Furniture", "Office Supplies", "Technology"]:
    mask = (cat_test == cat)
    y_t = y_test[mask]
    y_p = y_pred[mask]
    y_pr = y_prob[mask]

    n_unprof = y_t.sum()
    cat_f1 = f1_score(y_t, y_p) if n_unprof > 0 else 0
    cat_auc = roc_auc_score(y_t, y_pr) if len(np.unique(y_t)) > 1 else 0

    print(f"\n  {cat} ({len(y_t)} rows, {n_unprof} unprofitable = {n_unprof/len(y_t):.1%}):")
    print(f"    F1:      {cat_f1:.3f}")
    print(f"    AUC-ROC: {cat_auc:.3f}")