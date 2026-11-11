# 09_error_analysis.py
# Examines the false positives and false negatives from the Gradient Boosting model to understand where the model fails.
# - False positives: flagged as unprofitable but actually profitable
# - False negatives: missed unprofitable transactions

import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (load_and_prepare, get_features_ordinal,
                         split_data, PROJECT_ROOT, RANDOM_STATE)

# ── 1. Load, split, train ────────────────────────────────────
df = load_and_prepare()
X, y, encoder = get_features_ordinal(df, include_category=True)
(X_train, X_val, X_test,
 y_train, y_val, y_test,
 cat_train, cat_val, cat_test) = split_data(X, y, df)

gbt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1,
    max_depth=4, random_state=RANDOM_STATE
)
gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_val)

print("=" * 60)
print("ERROR ANALYSIS (Gradient Boosting on Validation Set)")
print("=" * 60)

# ── 2. Confusion matrix ─────────────────────────────────────
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
f1 = f1_score(y_val, y_pred)

print(f"\nConfusion Matrix:")
print(f"  True Negatives  (correct profitable): {tn}")
print(f"  True Positives  (correct unprofitable): {tp}")
print(f"  False Positives (flagged but profitable): {fp}")
print(f"  False Negatives (missed losses): {fn}")
print(f"\n  F1 (unprofitable): {f1:.3f}")

# ── 3. Build a DataFrame for analysis ────────────────────────
val_df = df.loc[X_val.index].copy()
val_df["Predicted"] = y_pred
val_df["Actual"] = y_val

# ── 4. False Positives ───────────────────────────────────────
fp_df = val_df[(val_df["Predicted"] == 1) & (val_df["Actual"] == 0)]

print(f"\n{'-' * 60}")
print(f"FALSE POSITIVES: {len(fp_df)} transactions")
print(f"(Model said unprofitable, but they were actually profitable)")
print(f"{'-' * 60}")

print(f"\n  By Category:")
for cat, count in fp_df["Category"].value_counts().items():
    print(f"    {cat}: {count} ({count/len(fp_df):.0%})")

print(f"\n  By Sub-Category (top 5):")
for subcat, count in fp_df["Sub-Category"].value_counts().head(5).items():
    print(f"    {subcat}: {count}")

print(f"\n  Average Discount: {fp_df['Discount'].mean():.1%}")
print(f"  Average Profit:   ${fp_df['Profit'].mean():.2f}")
print(f"    These are low-profit but positive transactions")
print(f"    the model treats as risky due to high discounts.")

# ── 5. False Negatives ───────────────────────────────────────
fn_df = val_df[(val_df["Predicted"] == 0) & (val_df["Actual"] == 1)]

print(f"\n{'-' * 60}")
print(f"FALSE NEGATIVES: {len(fn_df)} transactions")
print(f"(Model said profitable, but they actually lost money)")
print(f"{'-' * 60}")

print(f"\n  By Category:")
for cat, count in fn_df["Category"].value_counts().items():
    print(f"    {cat}: {count} ({count/len(fn_df):.0%})")

print(f"\n  By Sub-Category (top 5):")
for subcat, count in fn_df["Sub-Category"].value_counts().head(5).items():
    print(f"    {subcat}: {count}")

print(f"\n  Average Discount: {fn_df['Discount'].mean():.1%}")
print(f"  Average Profit:   ${fn_df['Profit'].mean():.2f}")
print(f"    These are transactions with moderate discounts")
print(f"    that lost money for other reasons (shipping, low-margin items).")

# ── 6. Key takeaway ──────────────────────────────────────────
total_missed_loss = fn_df["Profit"].sum()
print(f"\n{'=' * 60}")
print(f"TAKEAWAY")
print(f"{'=' * 60}")
print(f"  The model is more conservative than wrong:")
print(f"  - It over-flags {fp} profitable transactions")
print(f"  - It misses {fn} unprofitable ones")
print(f"  - Total missed losses: ${total_missed_loss:.2f}")
print(f"  - Average missed loss: ${total_missed_loss/fn:.2f} per transaction")