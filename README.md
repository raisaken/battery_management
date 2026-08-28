# Intelligent Battery Management System — Complete Dissertation Project

This is the **complete standalone project**. You do **not** need the earlier ZIPs.

Project topic:

> **Development of an Intelligent Battery Management System Featuring Machine Learning-Based SOH and RUL Estimation**

The code uses the cleaned CSV version of the NASA Battery Dataset and implements the full progression from preprocessing and baseline models to ablation studies, engineered features, physical EOL/RUL labelling, LSTM comparison, feature importance, deployment models, and a Streamlit BMS demo.

## 1. Dataset placement

Download the NASA Battery Dataset from Kaggle and copy the entire `cleaned_dataset` folder into:

```text
data/raw/cleaned_dataset/
```

Expected structure:

```text
intelligent-battery-management-complete/
├── data/
│   ├── raw/
│   │   └── cleaned_dataset/
│   │       ├── metadata.csv
│   │       ├── data/
│   │       │   ├── 00001.csv
│   │       │   ├── 00002.csv
│   │       │   └── ...
│   │       └── extra_infos/
│   └── processed/
├── models/
├── results/
│   ├── figures/
│   └── metrics/
├── src/
├── dashboard.py
├── main.py
├── run_all.py
└── requirements.txt
```

The main experiments use B0005, B0006, B0007 and B0018.

## 2. Environment setup

On macOS / VS Code:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Check the dataset

```bash
python src/check_setup.py
```

Optional inspection:

```bash
python src/inspect_dataset.py
```

## 4. Fast end-to-end test

Run the complete workflow with only 15 LSTM epochs per fold:

```bash
python run_all.py --quick
```

If that finishes successfully, run the full workflow:

```bash
python run_all.py
```

## 5. What `run_all.py` executes

### Baseline stage

```text
check_setup.py
preprocess.py
plot_eda.py
train_soh.py
train_rul.py
cross_validate.py
cross_validate_no_cycle.py
```

This creates the first two experiments:

- Experiment A — basic features + cycle number
- Experiment B — basic features without cycle number

SOH is defined using one fixed rated capacity:

```text
SOH = measured capacity / 2.0 Ah
```

The early baseline RUL uses remaining recorded cycles. This is retained only as a baseline/ablation study.

### Engineered-feature stage

```text
preprocess_engineered.py
cross_validate_engineered_no_cycle.py
cross_validate_engineered_with_cycle.py
```

Engineered health indicators include:

- discharge energy
- voltage curve area
- temperature curve area
- voltage slope
- temperature slope
- time to 4.0 V
- time to 3.8 V
- time to 3.6 V
- voltage at 25%, 50%, and 75% elapsed discharge time
- temperature at 50% elapsed discharge time
- relative discharge duration
- relative discharge energy

This produces:

- Experiment C — engineered features without cycle number
- Experiment D — engineered features + cycle number

The two relative features use the **first observed discharge cycle only** as their reference, avoiding future-cycle look-ahead.

### Final physical RUL stage

```text
prepare_final_dataset.py
```

This creates:

```text
data/processed/battery_cycles_final.csv
data/processed/physical_eol_summary.csv
```

The final RUL target uses a physical end-of-life threshold:

```text
rated capacity = 2.0 Ah
EOL capacity = 1.4 Ah
```

EOL is the first point at which **three consecutive discharge cycles** are at or below 1.4 Ah. If a battery never produces a sustained crossing, the final observed cycle is retained as a clearly documented fallback.

### Final classical evaluation

```text
evaluate_final_classical.py
```

Performs Leave-One-Battery-Out evaluation for:

- Random Forest
- XGBoost

with:

- engineered features without cycle number
- engineered features with cycle number

for:

- SOH
- physical RUL

It reports both `all_cycles` results and `sequence_aligned` results. Sequence-aligned evaluation begins at cycle 10 so the classical models can later be compared fairly with an LSTM using a 10-cycle sequence.

### Feature importance

```text
feature_importance.py
```

Uses held-out-battery **permutation importance** rather than relying only on tree training importance.

Outputs include ranked CSV files and top-15 plots for SOH and RUL.

### LSTM stage

```text
train_lstm.py
```

Default configuration:

```text
sequence length = 10 cycles
epochs = 60
hidden size = 64
layers = 2
```

It evaluates:

- LSTM + engineered features without cycle number
- LSTM + engineered features with cycle number

for both SOH and physical RUL under Leave-One-Battery-Out validation.

Quick direct test:

```bash
python src/train_lstm.py --epochs 15
```

### Final RF vs XGBoost vs LSTM comparison

```text
final_comparison.py
```

Uses the sequence-aligned classical results and LSTM results to produce a fair final comparison.

### Deployment models

```text
train_deployment_models.py
```

Selects the strongest final classical model for each target from the all-cycle cross-validation results and trains deployment bundles:

```text
models/deployment_soh.joblib
models/deployment_rul.joblib
```

The LSTM remains part of the academic model comparison. Classical models are used for the dashboard because they can estimate health from one processed discharge cycle.

## 6. Run the BMS dashboard

After the full pipeline completes:

```bash
streamlit run dashboard.py
```

The dashboard provides:

- battery selection
- discharge-cycle selection
- measured capacity
- predicted SOH
- predicted physical RUL
- battery health status
- SOH trajectory
- RUL trajectory
- selected model information

This is a dissertation demonstration, not a certified real-world safety system.

## 7. Important output files

### Baseline and ablation

```text
results/metrics/leave_one_battery_out_summary.csv
results/metrics/leave_one_battery_out_no_cycle_summary.csv
results/metrics/engineered_no_cycle_summary.csv
results/metrics/engineered_with_cycle_summary.csv
results/metrics/all_experiments_comparison.csv
```

### Final methodology

```text
data/processed/physical_eol_summary.csv
results/metrics/final_classical_summary.csv
results/metrics/permutation_importance_summary.csv
results/metrics/lstm_summary.csv
results/metrics/final_model_comparison.csv
results/metrics/best_models.csv
results/metrics/deployment_model_selection.csv
```

### Dissertation figures

```text
results/figures/capacity_vs_cycle.png
results/figures/soh_vs_cycle.png
results/figures/temperature_vs_cycle.png
results/figures/importance_*.png
results/figures/final_r2_comparison_*.png
results/figures/final_mae_comparison_*.png
```

## 8. Manual execution order

If you prefer to run every step individually:

```bash
python src/check_setup.py
python src/inspect_dataset.py

python src/preprocess.py
python src/plot_eda.py
python src/train_soh.py
python src/train_rul.py
python src/cross_validate.py
python src/cross_validate_no_cycle.py

python src/preprocess_engineered.py
python src/cross_validate_engineered_no_cycle.py
python src/cross_validate_engineered_with_cycle.py

python src/prepare_final_dataset.py
python src/evaluate_final_classical.py
python src/feature_importance.py
python src/train_lstm.py
python src/final_comparison.py
python src/train_deployment_models.py

streamlit run dashboard.py
```

## 9. Recommended dissertation interpretation

Keep the baseline/ablation stage in the report because it demonstrates the research progression:

```text
basic + cycle
basic - cycle
engineered - cycle
engineered + cycle
```

Then treat the physical 1.4 Ah EOL experiment, feature-importance study, and RF/XGBoost/LSTM comparison as the final methodological stage.

Do not select a model purely because one fold gives the largest R². Report mean and standard deviation across held-out batteries and discuss B0018 separately if it remains substantially harder than the other cells.
