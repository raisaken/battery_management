\
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    METADATA_FILE,
    EXPERIMENT_DATA_DIR,
    PROCESSED_FILE,
    PROCESSED_DIR,
    SELECTED_BATTERIES,
    EOL_SOH_THRESHOLD,
)
from utils import (
    ensure_directories,
    normalise_filename,
    parse_start_time,
    numeric_series,
    safe_stat,
)


def extract_discharge_features(csv_path):
    df = pd.read_csv(csv_path)

    voltage = numeric_series(df, ["Voltage_measured", "voltage_measured", "Voltage"])
    current = numeric_series(df, ["Current_measured", "current_measured", "Current"])
    temp = numeric_series(df, ["Temperature_measured", "temperature_measured", "Temperature"])
    current_load = numeric_series(df, ["Current_load", "current_load", "Current_charge"])
    voltage_load = numeric_series(df, ["Voltage_load", "voltage_load", "Voltage_charge"])
    time = numeric_series(df, ["Time", "time"])

    duration = np.nan
    if time is not None:
        clean_time = time.dropna()
        if not clean_time.empty:
            duration = float(clean_time.max() - clean_time.min())

    voltage_drop = np.nan
    if voltage is not None:
        clean = voltage.dropna()
        if len(clean) >= 2:
            voltage_drop = float(clean.iloc[0] - clean.iloc[-1])

    temp_rise = np.nan
    if temp is not None:
        clean = temp.dropna()
        if len(clean) >= 2:
            temp_rise = float(clean.iloc[-1] - clean.iloc[0])

    return {
        "sample_count": int(len(df)),
        "avg_voltage": safe_stat(voltage, "mean"),
        "min_voltage": safe_stat(voltage, "min"),
        "max_voltage": safe_stat(voltage, "max"),
        "std_voltage": safe_stat(voltage, "std"),
        "voltage_drop": voltage_drop,
        "avg_current": safe_stat(current, "mean"),
        "min_current": safe_stat(current, "min"),
        "max_current": safe_stat(current, "max"),
        "avg_temperature": safe_stat(temp, "mean"),
        "min_temperature": safe_stat(temp, "min"),
        "max_temperature": safe_stat(temp, "max"),
        "temperature_rise": temp_rise,
        "avg_current_load": safe_stat(current_load, "mean"),
        "avg_voltage_load": safe_stat(voltage_load, "mean"),
        "discharge_duration": duration,
    }


def prepare_metadata():
    meta = pd.read_csv(METADATA_FILE).copy()

    required = {"type", "battery_id", "filename", "Capacity"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            "metadata.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    meta["type"] = meta["type"].astype(str).str.lower().str.strip()
    meta["battery_id"] = meta["battery_id"].astype(str).str.strip()
    meta["filename"] = meta["filename"].apply(normalise_filename)
    meta["Capacity"] = pd.to_numeric(meta["Capacity"], errors="coerce")

    if "ambient_temperature" in meta.columns:
        meta["ambient_temperature"] = pd.to_numeric(
            meta["ambient_temperature"], errors="coerce"
        )
    else:
        meta["ambient_temperature"] = np.nan

    if "start_time" in meta.columns:
        meta["parsed_start_time"] = meta["start_time"].apply(parse_start_time)
    else:
        meta["parsed_start_time"] = pd.NaT

    meta = meta[meta["type"].eq("discharge")].copy()
    meta = meta[meta["Capacity"].notna() & (meta["Capacity"] > 0)].copy()

    if SELECTED_BATTERIES:
        meta = meta[meta["battery_id"].isin(SELECTED_BATTERIES)].copy()

    # Use time when available. Keep metadata order as a stable fallback.
    meta["_original_order"] = np.arange(len(meta))
    meta = meta.sort_values(
        ["battery_id", "parsed_start_time", "_original_order"],
        na_position="last"
    ).reset_index(drop=True)

    meta["cycle_number"] = meta.groupby("battery_id").cumcount() + 1

    return meta


def create_labels(df):
    frames = []

    for battery_id, group in df.groupby("battery_id", sort=False):
        group = group.sort_values("cycle_number").copy()

        initial_capacity = float(group["capacity_ah"].iloc[0])
        group["initial_capacity_ah"] = initial_capacity
        group["soh"] = group["capacity_ah"] / initial_capacity
        group["soh_percent"] = group["soh"] * 100.0
        group["capacity_fade"] = 1.0 - group["soh"]

        # Find the first cycle at or below the configured EOL threshold.
        reached_eol = group[group["soh"] <= EOL_SOH_THRESHOLD]

        if not reached_eol.empty:
            eol_cycle = int(reached_eol["cycle_number"].iloc[0])
            eol_source = f"SOH<={EOL_SOH_THRESHOLD:.2f}"
        else:
            # Some NASA series end before a mathematically exact crossing.
            # We use the final observed discharge as a transparent fallback.
            eol_cycle = int(group["cycle_number"].max())
            eol_source = "last_observed_cycle_fallback"

        group["eol_cycle"] = eol_cycle
        group["eol_source"] = eol_source
        group["rul_cycles"] = (eol_cycle - group["cycle_number"]).clip(lower=0)

        # Cycles recorded after the threshold crossing are not useful RUL targets.
        group["before_or_at_eol"] = group["cycle_number"] <= eol_cycle

        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def main():
    ensure_directories([PROCESSED_DIR])

    meta = prepare_metadata()

    if meta.empty:
        raise RuntimeError(
            "No valid discharge records found. Check battery IDs and metadata."
        )

    rows = []
    missing_files = []

    print(f"Processing {len(meta):,} discharge experiments...")

    for idx, row in meta.iterrows():
        csv_path = EXPERIMENT_DATA_DIR / row["filename"]

        if not csv_path.exists():
            missing_files.append(row["filename"])
            continue

        features = extract_discharge_features(csv_path)

        result = {
            "battery_id": row["battery_id"],
            "cycle_number": int(row["cycle_number"]),
            "filename": row["filename"],
            "start_time": row.get("start_time", np.nan),
            "ambient_temperature": row["ambient_temperature"],
            "capacity_ah": float(row["Capacity"]),
            "Re": pd.to_numeric(row.get("Re", np.nan), errors="coerce"),
            "Rct": pd.to_numeric(row.get("Rct", np.nan), errors="coerce"),
        }
        result.update(features)
        rows.append(result)

        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1:,}/{len(meta):,}")

    if not rows:
        raise RuntimeError("No discharge experiment files could be processed.")

    processed = pd.DataFrame(rows)
    processed = processed.sort_values(
        ["battery_id", "cycle_number"]
    ).reset_index(drop=True)

    processed = create_labels(processed)
    processed.to_csv(PROCESSED_FILE, index=False)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Saved: {PROCESSED_FILE}")
    print(f"Rows: {len(processed):,}")
    print(f"Batteries: {processed['battery_id'].nunique()}")

    print("\nRows by battery:")
    print(processed.groupby("battery_id").size().to_string())

    print("\nSOH ranges:")
    print(
        processed.groupby("battery_id")["soh"]
        .agg(["min", "max"])
        .round(4)
        .to_string()
    )

    print("\nEOL definition used:")
    print(
        processed.groupby("battery_id")[["eol_cycle", "eol_source"]]
        .first()
        .to_string()
    )

    if missing_files:
        print(f"\nWarning: {len(missing_files)} referenced CSV files were missing.")
        print("First missing files:", missing_files[:10])


if __name__ == "__main__":
    main()
