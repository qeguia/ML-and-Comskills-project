# 02_global_model.py
# Train ONE logistic regression on ALL categories combined.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_loader import load_and_prepare, get_features, split_data, scale_features

# ── 1. Load and prepare ──────────────────────────────────────
df = load_and_prepare()
X, y = get_features(df, include_category=True)
X_train, X_val, X_test, y_train, y_val, y_test, cat_train, cat_val, cat_test = split_data(X, y, df)
X_train_s, X_val_s, scaler = scale_features(X_train, X_val)

print("=" * 50)
print("GLOBAL MODEL")
print("=" * 50)
print(f"Train size: {len(X_train)}")
print(f"Val size:   {len(X_val)}")
print(f"Features:   {X_train.shape[1]}")
print(f"Unprofitable rate: {y_train.mean():.1%}")
print()

# ── 2. Majority baseline ────────────────────────────────────
# A "dumb" model that always predicts "profitable" (the majority class)
y_pred_majority = [0] * len(y_val)  # always predict profitable
print("--- Majority Baseline ---")
print(f"  Accuracy: {accuracy_score(y_val, y_pred_majority):.3f}")
print(f"  F1 (unprofitable): 0.000  (never predicts unprofitable)")
print()

# ── 3. Logistic Regression ──────────────────────────────────
# C=1.0 is the default: L2 regularization strength
# max_iter=1000 ensures convergence
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

y_pred = lr.predict(X_val_s)
y_prob = lr.predict_proba(X_val_s)[:, 1]  # probability of being unprofitable

print("--- Logistic Regression (L2, C=1.0) ---")
print(f"  Accuracy:          {accuracy_score(y_val, y_pred):.3f}")
print(f"  F1 (unprofitable): {f1_score(y_val, y_pred):.3f}")
print(f"  AUC-ROC:           {roc_auc_score(y_val, y_prob):.3f}")
print()

# ── 4. Top coefficients ──────────────────────────────────────
# Which features matter most?
feature_names = X_train.columns.tolist()
coefs = lr.coef_[0]

print("--- Top 5 Coefficients (strongest predictors) ---")
# Sort by absolute value
ranked = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
for name, coef in ranked[:5]:
    direction = "→ more unprofitable" if coef > 0 else "→ more profitable"
    print(f"  {name:30s}: {coef:+.3f}  {direction}")
print()

# ── 5. Per-category breakdown of the global model ────────────
# Professor asked: how does the global model perform on each category?
print("--- Global Model: Per-Category Performance ---")
for cat in ["Furniture", "Office Supplies", "Technology"]:
    # Find which validation rows belong to this category
    mask = (cat_val == cat)
    y_cat = y_val[mask]
    yp_cat = y_pred[mask]
    yprob_cat = y_prob[mask]

    f1 = f1_score(y_cat, yp_cat) if y_cat.sum() > 0 else 0
    auc = roc_auc_score(y_cat, yprob_cat) if len(set(y_cat)) > 1 else 0

    print(f"  {cat:20s}: F1={f1:.3f}, AUC={auc:.3f}, "
          f"n={len(y_cat)}, unprof={y_cat.mean():.1%}")