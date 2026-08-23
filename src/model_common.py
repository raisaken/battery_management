from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import TRAIN_BATTERIES, TEST_BATTERY


# Exclude target-derived variables such as capacity_ah, soh, and RUL.
# This reduces target leakage for SOH/RUL estimation.
BASE_FEATURES = [
    "cycle_number",
    "ambient_temperature",
    "avg_voltage",
    "min_voltage",
    "max_voltage",
    "std_voltage",
    "voltage_drop",
    "avg_current",
    "min_current",
    "max_current",
    "avg_temperature",
    "min_temperature",
    "max_temperature",
    "temperature_rise",
    "avg_current_load",
    "avg_voltage_load",
    "discharge_duration",
]


def load_model_data(path):
    df = pd.read_csv(path)
    df = df[df["before_or_at_eol"].astype(str).str.lower().isin(["true", "1"])].copy()

    available_features = [c for c in BASE_FEATURES if c in df.columns]
    if not available_features:
        raise RuntimeError("No modelling features were found in the processed CSV.")

    train = df[df["battery_id"].isin(TRAIN_BATTERIES)].copy()
    test = df[df["battery_id"].eq(TEST_BATTERY)].copy()

    if train.empty:
        raise RuntimeError(
            f"No training rows found for configured batteries: {TRAIN_BATTERIES}"
        )
    if test.empty:
        raise RuntimeError(
            f"No test rows found for configured battery: {TEST_BATTERY}"
        )

    # Fill missing values using training medians only.
    medians = train[available_features].median(numeric_only=True)
    train.loc[:, available_features] = train[available_features].fillna(medians)
    test.loc[:, available_features] = test[available_features].fillna(medians)

    # Remove features that remain entirely missing.
    usable = [
        c for c in available_features
        if train[c].notna().any() and test[c].notna().any()
    ]
    if not usable:
        raise RuntimeError("All model features are missing after preprocessing.")

    return train, test, usable


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }
