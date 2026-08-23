import pandas as pd

from config import METADATA_FILE, EXPERIMENT_DATA_DIR


def main():
    metadata = pd.read_csv(METADATA_FILE)

    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    print(f"Shape: {metadata.shape}")
    print(f"Columns: {metadata.columns.tolist()}")
    print("\nFirst 10 rows:")
    print(metadata.head(10).to_string(index=False))
    print("\nMissing values:")
    print(metadata.isna().sum().to_string())

    if "battery_id" in metadata.columns:
        print("\nBattery IDs:")
        print(metadata["battery_id"].unique())

    if "type" in metadata.columns:
        print("\nExperiment types:")
        print(metadata["type"].value_counts())

    csv_files = sorted(EXPERIMENT_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("\nNo experiment CSV files found.")
        return

    # Prefer a discharge file because it is central to SOH/RUL.
    first_file = None
    if {"filename", "type"}.issubset(metadata.columns):
        discharge = metadata[
            metadata["type"].astype(str).str.lower().eq("discharge")
        ]
        if not discharge.empty:
            candidate = str(discharge.iloc[0]["filename"])
            if not candidate.lower().endswith(".csv"):
                candidate += ".csv"
            path = EXPERIMENT_DATA_DIR / candidate
            if path.exists():
                first_file = path

    if first_file is None:
        first_file = csv_files[0]

    df = pd.read_csv(first_file)

    print("\n" + "=" * 70)
    print("SAMPLE EXPERIMENT FILE")
    print("=" * 70)
    print(f"File: {first_file.name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nData types:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
