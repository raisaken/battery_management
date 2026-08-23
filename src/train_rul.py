import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import (
    PROCESSED_FILE,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
)
from model_common import load_model_data, regression_metrics
from utils import ensure_directories


def build_models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "xgboost": XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
        ),
    }


def main():
    ensure_directories([MODELS_DIR, METRICS_DIR, FIGURES_DIR])
    train, test, features = load_model_data(PROCESSED_FILE)

    X_train = train[features]
    y_train = train["rul_cycles"]
    X_test = test[features]
    y_test = test["rul_cycles"]

    rows = []

    print("RUL features:")
    print(features)
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = pred.clip(min=0)

        metrics = regression_metrics(y_test, pred)
        metrics["model"] = name
        rows.append(metrics)

        joblib.dump(model, MODELS_DIR / f"rul_{name}.joblib")

        pred_df = test[["battery_id", "cycle_number", "rul_cycles"]].copy()
        pred_df["predicted_rul_cycles"] = pred
        pred_df.to_csv(
            METRICS_DIR / f"rul_predictions_{name}.csv",
            index=False,
        )

        plt.figure(figsize=(9, 6))
        plt.plot(test["cycle_number"], y_test, label="Actual RUL")
        plt.plot(test["cycle_number"], pred, label="Predicted RUL")
        plt.xlabel("Discharge cycle")
        plt.ylabel("Remaining useful life (cycles)")
        plt.title(f"RUL Prediction - {name.replace('_', ' ').title()}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"rul_prediction_{name}.png",
            dpi=180,
        )
        plt.close()

        print(metrics)

    metrics_df = pd.DataFrame(rows)[["model", "MAE", "RMSE", "R2"]]
    metrics_df.to_csv(METRICS_DIR / "rul_model_comparison.csv", index=False)

    print("\nRUL model comparison:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
