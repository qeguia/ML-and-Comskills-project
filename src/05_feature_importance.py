# 05_feature_importance.py
# Use SHAP to understand which features drive predictions for each category. This produces the interpretability results.

import os
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from data_loader import load_and_prepare, get_features, split_data, PROJECT_ROOT

# ── 1. Load data ──────────────────────────────────────────────
df = load_and_prepare()

print("=" * 50)
print("SHAP FEATURE IMPORTANCE")
print("=" * 50)
print()

# ── 2. SHAP per category ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, cat in enumerate(["Furniture", "Office Supplies", "Technology"]):

    df_cat = df[df["Category"] == cat].copy().reset_index(drop=True)
    X, y = get_features(df_cat, include_category=False)
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    # Train a Gradient Boosting model (works best with SHAP)
    gbt = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    gbt.fit(X_train, y_train)

    # Compute SHAP values on validation set
    explainer = shap.TreeExplainer(gbt)
    shap_values = explainer.shap_values(X_val)

    # Print top 5 features
    print(f"--- {cat} ---")
    feature_importance = abs(shap_values).mean(axis=0)
    feature_names = X_val.columns.tolist()
    ranked = sorted(zip(feature_names, feature_importance),
                    key=lambda x: x[1], reverse=True)

    for name, imp in ranked[:5]:
        print(f"  {name:30s}: {imp:.4f}")
    print()

    # SHAP summary plot for this category
    plt.sca(axes[idx])
    shap.summary_plot(shap_values, X_val, show=False,
                      max_display=8, plot_size=None)
    axes[idx].set_title(f"{cat}")

plt.tight_layout()
fig_dir = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "shap_per_category.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to {fig_path}")
