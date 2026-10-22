# 03_category_models.py
# Train 3 SEPARATE logistic regressions — one per category.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from data_loader import load_and_prepare, get_features, split_data, scale_features

# ── 1. Load data ──────────────────────────────────────────────
df = load_and_prepare()

print("=" * 50)
print("PER-CATEGORY MODELS")
print("=" * 50)
print()

# ── 2. Train one model per category ──────────────────────────
for cat in ["Furniture", "Office Supplies", "Technology"]:

    # Filter to this category only
    df_cat = df[df["Category"] == cat].copy().reset_index(drop=True)

    # Get features WITHOUT Category column (it's always the same here)
    X, y = get_features(df_cat, include_category=False)

    # Split and scale
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_s, X_val_s, scaler = scale_features(X_train, X_val)

    # Train logistic regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    # Evaluate
    y_pred = lr.predict(X_val_s)
    y_prob = lr.predict_proba(X_val_s)[:, 1]

    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"--- {cat} ---")
    print(f"  Samples: {len(df_cat)}")
    print(f"  Unprofitable rate: {y.mean():.1%}")
    print(f"  F1 (unprofitable): {f1:.3f}")
    print(f"  AUC-ROC:           {auc:.3f}")
    print()

    # Top 3 coefficients for this category
    feature_names = X_train.columns.tolist()
    coefs = lr.coef_[0]
    ranked = sorted(zip(feature_names, coefs),
                    key=lambda x: abs(x[1]), reverse=True)

    print(f"  Top 3 loss drivers for {cat}:")
    for name, coef in ranked[:3]:
        direction = "↑ risk" if coef > 0 else "↓ risk"
        print(f"    {name:30s}: {coef:+.3f}  ({direction})")
    print()