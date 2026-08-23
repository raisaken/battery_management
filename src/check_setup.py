from pathlib import Path
import pandas as pd

from config import DATASET_DIR, METADATA_FILE, EXPERIMENT_DATA_DIR


REQUIRED_METADATA_COLUMNS = {
    "type",
    "battery_id",
    "filename",
}


def main():
    print("=" * 70)
    print("NASA BATTERY PROJECT - SETUP CHECK")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"\n[ERROR] Dataset folder not found:\n{DATASET_DIR}")
        print("\nCopy `cleaned_dataset` to:")
        print(DATASET_DIR.parent)
        return 1

    if not METADATA_FILE.exists():
        print(f"\n[ERROR] metadata.csv not found:\n{METADATA_FILE}")
        return 1

    if not EXPERIMENT_DATA_DIR.exists():
        print(f"\n[ERROR] data/ folder not found:\n{EXPERIMENT_DATA_DIR}")
        return 1

    metadata = pd.read_csv(METADATA_FILE)
    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)

    print(f"\nMetadata rows: {len(metadata):,}")
    print(f"Metadata columns: {list(metadata.columns)}")

    if missing:
        print(f"\n[ERROR] Required metadata columns missing: {sorted(missing)}")
        return 1

    csv_count = len(list(EXPERIMENT_DATA_DIR.glob("*.csv")))
    print(f"Experiment CSV files: {csv_count:,}")

    print("\nExperiment types:")
    print(metadata["type"].astype(str).str.lower().value_counts())

    print(f"\nBattery count: {metadata['battery_id'].nunique()}")
    print("Battery IDs:")
    print(sorted(metadata["battery_id"].astype(str).unique().tolist()))

    discharge = metadata[metadata["type"].astype(str).str.lower().eq("discharge")]
    print(f"\nDischarge experiments: {len(discharge):,}")

    print("\n[OK] Dataset structure is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
