# data_loader.py
# Loads the Superstore CSV, creates features, and splits into train / validation / test sets.

# We exclude Sales: it is computed simultaneously with Profit (Profit = Sales - Cost), using it would be data leakage.

# We filter Discount > 50%: those transactions are 100% unprofitable across all categories, so no model needed.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# Project root = one folder above src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_prepare(filepath=None):
    """Read the CSV and create the features we need."""

    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "data.csv")

    df = pd.read_csv(filepath, encoding="latin-1")

    # Drop discounts above 50%
    df = df[df["Discount"] <= 0.5].copy()

    # Target: 1 = unprofitable, 0 = profitable
    df["Unprofitable"] = (df["Profit"] < 0).astype(int)

    # Engineered feature: days between order and shipment
    df["Shipping_Days"] = (
        pd.to_datetime(df["Ship Date"]) - pd.to_datetime(df["Order Date"])
    ).dt.days

    return df


def get_features(df, include_category=True):
    """
    Build the feature matrix X and target vector y.

    Parameters
    ----------
    df : DataFrame from load_and_prepare()
    include_category : bool
        True for the global model (need to distinguish categories).
        False for per-category models (category is redundant).

    Returns
    -------
    X : DataFrame of features (one-hot encoded)
    y : array of 0/1 labels
    """

    numeric_cols = ["Quantity", "Discount", "Shipping_Days"]

    if include_category:
        cat_cols = ["Category", "Sub-Category", "Segment", "Ship Mode", "Region"]
    else:
        cat_cols = ["Sub-Category", "Segment", "Ship Mode", "Region"]

    # One-hot encode categoricals; drop_first avoids collinearity
    X = pd.get_dummies(
        df[numeric_cols + cat_cols],
        columns=cat_cols,
        drop_first=True,
    )
    y = df["Unprofitable"].values

    return X, y


def split_data(X, y, df=None):
    """
    Stratified split: 60% train, 20% validation, 20% test.
    Stratified means each split keeps the same unprofitable rate.
    
    If df is provided, also return category labels aligned with splits.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    if df is not None:
        cat_train = df.loc[X_train.index, "Category"].values
        cat_val = df.loc[X_val.index, "Category"].values
        cat_test = df.loc[X_test.index, "Category"].values
        return X_train, X_val, X_test, y_train, y_val, y_test, cat_train, cat_val, cat_test

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test=None):
    """
    Standardize to zero mean & unit variance.
    Important: fit ONLY on train, then apply to val/test.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    if X_test is not None:
        X_test_s = scaler.transform(X_test)
        return X_train_s, X_val_s, X_test_s, scaler

    return X_train_s, X_val_s, scaler