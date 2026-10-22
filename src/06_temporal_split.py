# 06_temporal_split.py
# Test if our model generalizes across time: train on 2014-2016, test on 2017.


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from data_loader import load_and_prepare, get_features

# ── 1. Load data ──────────────────────────────────────────────
df = load_and_prepare()

# Parse dates to extract the year
df["Year"] = pd.to_datetime(df["Order Date"]).dt.year

print("=" * 50)
print("TEMPORAL SPLIT: Train 2014-2016, Test 2017")
print("=" * 50)
print()

# ── 2. Split by time ─────────────────────────────────────────
train_mask = df["Year"] <= 2016
test_mask = df["Year"] == 2017

df_train = df[train_mask].reset_index(drop=True)
df_test = df[test_mask].reset_index(drop=True)

print(f"Train (2014-2016): {len(df_train)} rows, "
      f"unprof rate = {df_train['Unprofitable'].mean():.1%}")
print(f"Test  (2017):      {len(df_test)} rows, "
      f"unprof rate = {df_test['Unprofitable'].mean():.1%}")
print()

# ── 3. Prepare features ──────────────────────────────────────
X_train, y_train = get_features(df_train, include_category=True)
X_test, y_test = get_features(df_test, include_category=True)

# Make sure both have the same columns
# (some one-hot columns might be missing in one split)
X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── 4. Train and evaluate ────────────────────────────────────
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

y_pred = lr.predict(X_test_s)
y_prob = lr.predict_proba(X_test_s)[:, 1]

print("--- Logistic Regression (temporal split) ---")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"  F1:       {f1_score(y_test, y_pred):.3f}")
print(f"  AUC-ROC:  {roc_auc_score(y_test, y_prob):.3f}")
print()

# ── 5. Compare with random split ─────────────────────────────
# (we already know these from 02_global_model.py)
print("--- Comparison ---")
print("  Random split:   F1 ≈ 0.705, AUC ≈ 0.978")
print(f"  Temporal split:  F1 = {f1_score(y_test, y_pred):.3f}, "
      f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
print()

diff_f1 = abs(f1_score(y_test, y_pred) - 0.705)
if diff_f1 < 0.05:
    print("  Performance is stable across time (< 5% difference).")
else:
    print("  Performance differs across time — investigate further.")
