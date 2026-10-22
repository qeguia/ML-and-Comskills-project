# 04_model_comparison.py
# Compare Logistic Regression vs Random Forest vs Gradient Boosting. Runs globally AND per-category.

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from data_loader import load_and_prepare, get_features, split_data, scale_features

# ── 1. Load data ──────────────────────────────────────────────
df = load_and_prepare()

# ── 2. Define our 3 models ───────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }


def evaluate_model(model, X_train, X_val, y_train, y_val, needs_scaling=False):
    """Train a model and return its metrics."""
    if needs_scaling:
        X_train_s, X_val_s, _ = scale_features(X_train, X_val)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_val_s)
        y_prob = model.predict_proba(X_val_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_prob),
    }


# ── 3. Global comparison ─────────────────────────────────────
print("=" * 60)
print("GLOBAL MODEL COMPARISON (all categories together)")
print("=" * 60)

X, y = get_features(df, include_category=True)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
print("-" * 60)

for name, model in get_models().items():
    needs_scaling = ("Logistic" in name)
    results = evaluate_model(model, X_train, X_val, y_train, y_val, needs_scaling)
    print(f"{name:<25} {results['accuracy']:>10.3f} {results['f1']:>10.3f} {results['auc']:>10.3f}")

print()

# ── 4. Per-category comparison ────────────────────────────────
print("=" * 60)
print("PER-CATEGORY MODEL COMPARISON")
print("=" * 60)

for cat in ["Furniture", "Office Supplies", "Technology"]:
    df_cat = df[df["Category"] == cat].copy().reset_index(drop=True)
    X_cat, y_cat = get_features(df_cat, include_category=False)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X_cat, y_cat)

    print(f"\n--- {cat} (n={len(df_cat)}, unprof={y_cat.mean():.1%}) ---")
    print(f"{'Model':<25} {'F1':>10} {'AUC':>10}")
    print("-" * 50)

    for name, model in get_models().items():
        needs_scaling = ("Logistic" in name)
        results = evaluate_model(model, X_tr, X_v, y_tr, y_v, needs_scaling)
        print(f"{name:<25} {results['f1']:>10.3f} {results['auc']:>10.3f}")
