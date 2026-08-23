from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "cleaned_dataset"
METADATA_FILE = DATASET_DIR / "metadata.csv"
EXPERIMENT_DATA_DIR = DATASET_DIR / "data"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "battery_cycles.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Classic NASA batteries used in many SOH/RUL studies.
# Change this to None to process every battery in metadata.csv.
SELECTED_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

# Battery-wise generalisation experiment:
TRAIN_BATTERIES = ["B0005", "B0006", "B0007"]
TEST_BATTERY = "B0018"

# NASA experiments commonly use 30% capacity fade as EOL.
EOL_SOH_THRESHOLD = 0.70

RANDOM_STATE = 42
