# Intelligent Battery Management System
## NASA Battery SOH and RUL Estimation

This starter project implements the first complete experimental pipeline for a final-year/dissertation project on:

**Development of an Intelligent Battery Management System Featuring Machine Learning-Based SOH and RUL Estimation**

The project expects the cleaned CSV version of Patrick Fleith's NASA Battery Dataset.

Dataset page:
https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset

The Kaggle version contains one CSV per charge, discharge, or impedance experiment. The project initially uses **discharge experiments** because they contain the battery capacity information needed to derive State of Health (SOH) and Remaining Useful Life (RUL).

---

## 1. Project structure

```text
intelligent-battery-management/
├── data/
│   ├── raw/
│   │   └── cleaned_dataset/       <- YOU COPY THIS HERE
│   │       ├── metadata.csv
│   │       ├── data/
│   │       │   ├── 00001.csv
│   │       │   └── ...
│   │       └── extra_infos/
│   └── processed/
├── models/
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── check_setup.py
│   ├── inspect_dataset.py
│   ├── preprocess.py
│   ├── plot_eda.py
│   ├── model_common.py
│   ├── train_soh.py
│   └── train_rul.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 2. Setup on macOS / VS Code

Open the project folder in VS Code and open a terminal.

### Create a virtual environment

```bash
python3 -m venv .venv
```

### Activate it

```bash
source .venv/bin/activate
```

### Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Install packages

```bash
pip install -r requirements.txt
```

---

## 3. Add the dataset

Download and unzip the Kaggle dataset.

Copy the whole folder:

```text
cleaned_dataset
```

into:

```text
data/raw/
```

The important paths should therefore be:

```text
data/raw/cleaned_dataset/metadata.csv
data/raw/cleaned_dataset/data/00001.csv
data/raw/cleaned_dataset/data/00002.csv
...
```

Do not rename the experiment CSV files.

---

## 4. Validate the dataset

Run:

```bash
python src/check_setup.py
```

The known cleaned metadata commonly contains fields such as:

```text
type
start_time
ambient_temperature
battery_id
test_id
uid
filename
Capacity
Re
Rct
```

The script only requires the essential columns needed for the initial pipeline.

---

## 5. Inspect the dataset

Run:

```bash
python src/inspect_dataset.py
```

This prints:

- metadata shape and columns
- battery identifiers
- counts of charge/discharge/impedance experiments
- one sample discharge CSV
- the sensor columns inside that experiment

Typical discharge signals include:

- Voltage_measured
- Current_measured
- Temperature_measured
- Current_load
- Voltage_load
- Time

---

## 6. Preprocess the NASA data

Run:

```bash
python src/preprocess.py
```

The preprocessing script:

1. reads `metadata.csv`
2. keeps discharge experiments
3. initially keeps B0005, B0006, B0007 and B0018
4. opens the corresponding experiment CSV
5. converts each discharge experiment into one cycle-level row
6. extracts voltage/current/temperature/time features
7. calculates SOH
8. defines an EOL cycle
9. creates RUL labels
10. saves the result to:

```text
data/processed/battery_cycles.csv
```

Important derived variables:

### SOH

```text
SOH = current discharge capacity / initial discharge capacity
```

The CSV contains both:

- `soh` as a 0-1 ratio
- `soh_percent` as 0-100%

### RUL

The default EOL threshold is configured as:

```text
SOH <= 0.70
```

RUL is:

```text
RUL = EOL cycle - current cycle
```

If a battery does not cross the configured threshold in the available data, the script transparently uses its final observed discharge cycle as a fallback and records this in `eol_source`.

---

## 7. Create exploratory graphs

Run:

```bash
python src/plot_eda.py
```

This creates:

```text
results/figures/capacity_vs_cycle.png
results/figures/soh_vs_cycle.png
results/figures/temperature_vs_cycle.png
```

These are useful for the dissertation's dataset analysis and results chapters.

---

## 8. Train SOH models

Run:

```bash
python src/train_soh.py
```

The first baseline experiment compares:

- Random Forest
- XGBoost

The default battery-wise split is:

```text
Train:
B0005
B0006
B0007

Test:
B0018
```

This is deliberately battery-wise rather than a random row split. It tests whether the model generalises to an unseen battery.

Metrics produced:

- MAE
- RMSE
- R²

Outputs are written to `models/`, `results/metrics/`, and `results/figures/`.

---

## 9. Train RUL models

Run:

```bash
python src/train_rul.py
```

This compares Random Forest and XGBoost for cycle-level RUL prediction using the same unseen-battery experiment.

---

## 10. Run everything

After the dataset has been copied into the right folder, run:

```bash
python main.py
```

That executes:

```text
dataset check
     ↓
preprocessing
     ↓
EDA
     ↓
SOH models
     ↓
RUL models
```

---

# Important academic design decision

The first model features intentionally exclude:

- capacity_ah
- soh
- soh_percent
- rul_cycles

from model inputs.

Those values are targets or directly target-derived variables. Feeding them into the model would cause target leakage and could produce unrealistically high results.

The initial predictors instead use sensor-derived information such as voltage, current, temperature, discharge duration, ambient temperature, and cycle number.

---

# Next development stages

Once this baseline runs correctly, develop the dissertation in this order:

### Stage 1 — Verify the baseline
Confirm that `battery_cycles.csv`, graphs, and model metrics are produced without errors.

### Stage 2 — Feature engineering
Add stronger battery health indicators, for example:

- voltage curve area
- time to voltage threshold
- temperature rise rate
- voltage decline slope
- discharge energy
- rolling degradation features

### Stage 3 — Improve validation
Run leave-one-battery-out experiments rather than testing only B0018.

Example:

```text
Experiment 1: test B0005
Experiment 2: test B0006
Experiment 3: test B0007
Experiment 4: test B0018
```

Report mean and standard deviation across experiments.

### Stage 4 — Sequence model
Add an LSTM model using previous battery cycles as a time sequence.

This gives a useful comparison:

```text
Random Forest
vs
XGBoost
vs
LSTM
```

### Stage 5 — Hyperparameter tuning
Use training data only for model selection. Do not tune directly against the held-out battery.

### Stage 6 — Final evaluation
Prepare:

- Actual vs predicted SOH
- Actual vs predicted RUL
- MAE comparison
- RMSE comparison
- R² comparison
- model training time
- feature importance
- limitations and generalisation discussion

### Stage 7 — Intelligent BMS demo
After the prediction pipeline is scientifically sound, add a simple dashboard that takes battery measurements and displays:

- estimated SOH
- estimated RUL
- health status
- warning when SOH approaches EOL
- degradation graph

Do the dashboard last. The dissertation's core contribution should first be the data pipeline, modelling methodology, and evaluation.

---

## Recommended first commands

After copying the dataset:

```bash
source .venv/bin/activate
python src/check_setup.py
python src/inspect_dataset.py
python src/preprocess.py
```

If those three commands work, inspect:

```text
data/processed/battery_cycles.csv
```

before training any model.
