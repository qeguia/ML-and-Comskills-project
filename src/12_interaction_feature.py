# 12_interaction_features.py
# Logistic regression with Discount x Category and Discount x Region interaction features. Same split, scaling and hyperparameters as 02_global_model.py, so the two are directly comparable.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_loader import load_and_prepare, get_features, split_data, scale_features

df = load_and_prepare()
X, y = get_features(df, include_category=True)

# Discount (unscaled) multiplied by each Category and Region dummy column
dummy_cols = [c for c in X.columns if c.startswith("Category_") or c.startswith("Region_")]
for c in dummy_cols:
    X[f"Discount_x_{c}"] = X["Discount"] * X[c]

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train_s, X_val_s, _ = scale_features(X_train, X_val)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_val_s)
y_prob = lr.predict_proba(X_val_s)[:, 1]

print("LOGISTIC REGRESSION WITH INTERACTIONS (validation set)")
print(f"Features: {X_train.shape[1]}  (added {len(dummy_cols)} interaction columns)")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.3f}")
print(f"F1:       {f1_score(y_val, y_pred):.3f}")
print(f"AUC-ROC:  {roc_auc_score(y_val, y_prob):.3f}")

print("\nInteraction coefficients (standardized):")
for name, coef in zip(X_train.columns, lr.coef_[0]):
    if name.startswith("Discount_x_"):
        print(f"  {name:35s}: {coef:+.3f}")