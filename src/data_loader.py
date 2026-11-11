# data_loader.py
# Loads the Superstore CSV, creates features, and splits into train / validation / test sets.

# We exclude Sales: it is computed simultaneously with Profit (Profit = Sales - Cost), using it would be data leakage.

# We filter Discount > 50%: those transactions are 100% unprofitable across all categories, so no model needed.

import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

RANDOM_STATE = 42

# Project root = one folder above src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENSUS_YEAR = 2023
STATE_FIPS = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
    "California": "06", "Colorado": "08", "Connecticut": "09",
    "Delaware": "10", "District of Columbia": "11", "Florida": "12",
    "Georgia": "13", "Hawaii": "15", "Idaho": "16", "Illinois": "17",
    "Indiana": "18", "Iowa": "19", "Kansas": "20", "Kentucky": "21",
    "Louisiana": "22", "Maine": "23", "Maryland": "24",
    "Massachusetts": "25", "Michigan": "26", "Minnesota": "27",
    "Mississippi": "28", "Missouri": "29", "Montana": "30",
    "Nebraska": "31", "Nevada": "32", "New Hampshire": "33",
    "New Jersey": "34", "New Mexico": "35", "New York": "36",
    "North Carolina": "37", "North Dakota": "38", "Ohio": "39",
    "Oklahoma": "40", "Oregon": "41", "Pennsylvania": "42",
    "Rhode Island": "44", "South Carolina": "45", "South Dakota": "46",
    "Tennessee": "47", "Texas": "48", "Utah": "49", "Vermont": "50",
    "Virginia": "51", "Washington": "53", "West Virginia": "54",
    "Wisconsin": "55", "Wyoming": "56",
}


def _load_census_reporter_features(states):
    """Fetch equivalent ACS indicators from the public Census Reporter API."""
    unknown_states = sorted(set(states).difference(STATE_FIPS))
    if unknown_states:
        raise ValueError(f"Unknown transaction states: {unknown_states}")

    geo_ids = ",".join(
        f"04000US{STATE_FIPS[state]}" for state in sorted(set(states))
    )
    url = (
        "https://api.censusreporter.org/1.0/data/show/latest?"
        f"table_ids=B19013,B01003,B23025&geo_ids={geo_ids}"
    )
    request = Request(url, headers={"User-Agent": "DRIP-ML-project/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Could not download public ACS data from Census Reporter."
        ) from exc

    rows = []
    for state in sorted(set(states)):
        geo_id = f"04000US{STATE_FIPS[state]}"
        try:
            data = payload["data"][geo_id]
            income = data["B19013"]["estimate"]["B19013001"]
            population = data["B01003"]["estimate"]["B01003001"]
            labor_force = data["B23025"]["estimate"]["B23025003"]
            unemployed = data["B23025"]["estimate"]["B23025005"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Census Reporter returned incomplete data for {state}."
            ) from exc
        rows.append({
            "state_name": state,
            "median_household_income": income,
            "population": population,
            "unemployment_rate": unemployed / labor_force * 100,
        })
    return pd.DataFrame(rows)


def load_census_state_features(year=CENSUS_YEAR):
    """Fetch state-level ACS indicators used as external model features."""
    variables = {
        "NAME": "state_name",
        "B19013_001E": "median_household_income",
        "B01003_001E": "population",
        "B23025_003E": "labor_force",
        "B23025_005E": "unemployed",
    }
    query_params = {
        "get": ",".join(variables),
        "for": "state:*",
    }
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        return _load_census_reporter_features(STATE_FIPS)
    if api_key:
        query_params["key"] = api_key
    query = urlencode(query_params)
    url = f"https://api.census.gov/data/{year}/acs/acs5?{query}"
    request = Request(url, headers={"User-Agent": "DRIP-ML-project/1.0"})

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"Census API request failed with HTTP {exc.code}. Check that "
            "CENSUS_API_KEY is valid and enabled."
        ) from exc
    except (OSError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download ACS Census data for {year} from {url}"
        ) from exc
    except json.JSONDecodeError as exc:
        message = (
            "The Census API did not return JSON. Set CENSUS_API_KEY to a "
            "valid Census API key and try again."
        )
        raise RuntimeError(message) from exc

    if len(payload) < 2 or not isinstance(payload[0], list):
        raise RuntimeError("The Census API returned no state records.")

    census = pd.DataFrame(payload[1:], columns=payload[0])
    census = census.rename(columns=variables)
    numeric_columns = [
        "median_household_income", "population", "labor_force", "unemployed"
    ]
    for column in numeric_columns:
        census[column] = pd.to_numeric(census[column], errors="coerce")

    census["unemployment_rate"] = (
        census["unemployed"] / census["labor_force"].replace(0, pd.NA) * 100
    )
    return census[
        ["state_name", "median_household_income", "population",
         "unemployment_rate"]
    ]


def merge_census_state_features(df, census=None, year=CENSUS_YEAR):
    """Merge ACS state indicators into transaction data using State."""
    if "State" not in df.columns:
        raise ValueError("Transaction data must contain a 'State' column.")

    if census is None:
        census = load_census_state_features(year=year)

    required = {
        "state_name", "median_household_income", "population",
        "unemployment_rate",
    }
    missing = required.difference(census.columns)
    if missing:
        raise ValueError(
            f"Census data is missing required columns: {sorted(missing)}"
        )

    merged = df.merge(
        census,
        left_on="State",
        right_on="state_name",
        how="left",
        validate="many_to_one",
    ).drop(columns=["state_name"])

    feature_columns = [
        "median_household_income", "population", "unemployment_rate"
    ]
    if merged[feature_columns].isna().any().any():
        missing_states = sorted(
            merged.loc[
                merged[feature_columns].isna().any(axis=1), "State"
            ].unique()
        )
        raise ValueError(
            f"Census data did not match transaction states: {missing_states}"
        )
    return merged


def load_and_prepare(filepath=None):
    """Read the CSV and create the features we need."""

    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "data.csv")

    df = pd.read_csv(filepath, encoding="latin-1")

    # Drop discounts above 50% 
    df = df[df["Discount"] <= 0.5].copy()
    df = df.reset_index(drop=True)

    # Target: 1 = unprofitable, 0 = profitable
    df["Unprofitable"] = (df["Profit"] < 0).astype(int)

    # Engineered feature: days between order and shipment
    df["Shipping_Days"] = (
        pd.to_datetime(df["Ship Date"]) - pd.to_datetime(df["Order Date"])
    ).dt.days

    return df


def get_features(df, include_category=True, include_census=False):
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
    if include_census:
        numeric_cols += [
            "median_household_income", "population", "unemployment_rate"
        ]

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

def get_features_ordinal(df, include_category=True):
    """
    Build the feature matrix X using ORDINAL encoding for categoricals.
    Used for Random Forest — trees don't need one-hot encoding,
    and ordinal keeps the data less sparse (professor's feedback).

    Returns X as a DataFrame, y as an array, and the encoder.
    """

    numeric_cols = ["Quantity", "Discount", "Shipping_Days"]

    if include_category:
        cat_cols = ["Category", "Sub-Category", "Segment",
                    "Ship Mode", "Region"]
    else:
        cat_cols = ["Sub-Category", "Segment",
                    "Ship Mode", "Region"]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1)

    # Encode categoricals as integers (0, 1, 2, ...)
    cat_encoded = encoder.fit_transform(df[cat_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=df.index)

    X = pd.concat([df[numeric_cols].reset_index(drop=True),
                    cat_df.reset_index(drop=True)], axis=1)
    y = df["Unprofitable"].values

    return X, y, encoder


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
    Fit ONLY on train, then apply to val/test.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    if X_test is not None:
        X_test_s = scaler.transform(X_test)
        return X_train_s, X_val_s, X_test_s, scaler

    return X_train_s, X_val_s, scaler