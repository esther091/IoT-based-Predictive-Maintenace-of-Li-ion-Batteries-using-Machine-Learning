"""
train_models.py
Loads the NASA Battery Dataset, extracts discharge cycles, and trains
Random Forest and XGBoost models for SOH and RUL prediction.

  1. Hyperparameters are tuned by leave-one-battery-out cross-validation
     on the training batteries (B0005, B0006, B0007).
  2. Models trained on those batteries are tested on B0018, which they
     never saw. These evaluation models are saved to models/eval/ and give
     the reported accuracy.
  3. The final models, retrained on every battery with the same
     hyperparameters, are saved to models/ for the live backend.
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import joblib

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from utils import compute_soh, compute_rul, mean_absolute_percentage_error

# ============================================================
# CONSTANTS
# ============================================================
DATA_DIR = "data"
MODELS_DIR = "models"
RATED_CAPACITY = 2.0
EOL_CAPACITY = 1.4  # 70% of rated -- Table 3.4
FEATURE_COLS = ["mean_voltage", "mean_current", "mean_power", "energy_wh"]
# Same rule the backend uses to decide a sample belongs to a discharge
# (backend/main.py). Features are computed only over these samples so
# training and live inference see identically-defined inputs.
DISCHARGE_THRESHOLD = -0.5

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
TRAIN_BATTERIES = ["B0005", "B0006", "B0007"]
TEST_BATTERY = "B0018"
EVAL_DIR = os.path.join(MODELS_DIR, "eval")

PARAM_GRID_RF = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "bootstrap": [True],
}

PARAM_DIST_XGB = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
}


# ============================================================
# FUNCTIONS
# ============================================================
def extract_discharge_cycles(battery_name):
    """
    Loads a NASA .mat file and extracts every discharge cycle.
    Returns a list of dicts, one per discharge cycle, containing
    mean voltage, mean current, mean power, energy, and capacity.
    """
    filepath = os.path.join(DATA_DIR, f"{battery_name}.mat")
    mat = loadmat(filepath)
    cycles = mat[battery_name][0, 0]['cycle'][0]

    rows = []
    discharge_number = 0

    for entry in cycles:
        if entry['type'][0] != 'discharge':
            continue

        discharge_number += 1
        data = entry['data'][0, 0]

        voltage = data['Voltage_measured'][0]
        current = data['Current_measured'][0]
        time = data['Time'][0]
        capacity = data['Capacity'][0][0]

        # Drop the rest samples (current ~ 0) at the start/end of each
        # record -- the live backend never sees them, because a reading
        # above DISCHARGE_THRESHOLD closes the cycle.
        active = current < DISCHARGE_THRESHOLD
        if active.sum() < 3:
            continue
        voltage, current, time = voltage[active], current[active], time[active]

        mean_voltage = float(np.mean(voltage))
        mean_current = float(np.mean(np.abs(current)))
        mean_power = float(np.mean(voltage * np.abs(current)))
        energy_wh = float(np.trapezoid(voltage * np.abs(current), time) / 3600.0)

        rows.append({
            "battery": battery_name,
            "cycle_number": discharge_number,
            "mean_voltage": mean_voltage,
            "mean_current": mean_current,
            "mean_power": mean_power,
            "energy_wh": energy_wh,
            "capacity": capacity,
        })

    print(f"{battery_name}: {discharge_number} discharge cycles extracted")
    return rows


def compute_targets(df):
    """Adds SOH and RUL columns to the dataframe -- Equations 3.8, 3.9."""
    df["soh"] = df["capacity"].apply(lambda c: compute_soh(c, RATED_CAPACITY))

    rul_values = []
    for battery_name in df["battery"].unique():
        battery_df = df[df["battery"] == battery_name]
        eol_rows = battery_df[battery_df["soh"] < 70.0]

        if len(eol_rows) > 0:
            eol_cycle = eol_rows["cycle_number"].iloc[0]
        else:
            eol_cycle = battery_df["cycle_number"].max()

        for cycle_num in battery_df["cycle_number"]:
            rul_values.append(compute_rul(eol_cycle, cycle_num))

    df["rul"] = rul_values
    return df


def remove_outliers(df):
    """Removes rows with implausible SOH readings."""
    before = len(df)
    df = df[(df["soh"] >= 65.0) & (df["soh"] <= 103.0)].reset_index(drop=True)
    after = len(df)
    print(f"\nOutlier removal: {before} -> {after} rows ({before - after} removed)")
    return df


def split_data(df):
    """Splits into the training batteries and the unseen test battery."""
    train_df = df[df["battery"].isin(TRAIN_BATTERIES)].reset_index(drop=True)
    test_df = df[df["battery"] == TEST_BATTERY].reset_index(drop=True)

    print(f"\nTrain set ({'+'.join(TRAIN_BATTERIES)}): {len(train_df)} rows")
    print(f"Test set ({TEST_BATTERY}):  {len(test_df)} rows")

    return train_df, test_df


def tune(target, X, y, groups):
    """Leave-one-battery-out tuning of both models for one target."""
    cv = list(LeaveOneGroupOut().split(X, y, groups))
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    print(f"\nTuning Random Forest ({target.upper()}) via GridSearchCV, leave-one-battery-out...")
    rf = GridSearchCV(RandomForestRegressor(random_state=42), PARAM_GRID_RF,
                      cv=cv, scoring=scoring, refit="r2", n_jobs=-1).fit(X, y)
    print(f"Best params (RF-{target.upper()}): {rf.best_params_}")

    print(f"Tuning XGBoost ({target.upper()}) via RandomizedSearchCV, leave-one-battery-out...")
    xgb = RandomizedSearchCV(XGBRegressor(objective="reg:squarederror", random_state=42),
                             PARAM_DIST_XGB, n_iter=50, cv=cv, scoring=scoring, refit="r2",
                             n_jobs=-1, random_state=42).fit(X, y)
    print(f"Best params (XGB-{target.upper()}): {xgb.best_params_}")
    return rf, xgb


def cv_scores(search):
    """Mean leave-one-battery-out R2 and MAE of the selected configuration."""
    i = search.best_index_
    return {"r2": float(search.cv_results_["mean_test_r2"][i]),
            "mae": float(-search.cv_results_["mean_test_mae"][i])}


def evaluate(y_pred, y_true, model_name, target_name, battery):
    """Evaluates predictions using MAE, RMSE, MAPE, R2 -- Equations 3.17-3.20."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} -- {target_name} (on unseen battery {battery}):")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAPE = {mape:.2f}%")
    print(f"  R2   = {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":

    # --- Step 1: Extract discharge cycles ---
    all_rows = []
    for battery in BATTERIES:
        all_rows.extend(extract_discharge_cycles(battery))

    df = pd.DataFrame(all_rows)
    print(f"\nTotal discharge cycles across all batteries: {len(df)}")

    # --- Step 2: Compute SOH and RUL ---
    df = compute_targets(df)
    print("\nSample with SOH and RUL computed:")
    print(df[["battery", "cycle_number", "capacity", "soh", "rul"]].head(10))
    print("\nSOH range per battery:")
    print(df.groupby("battery")["soh"].agg(["min", "max"]))

    # --- Step 3: Clean outliers ---
    df = remove_outliers(df)

    # --- Step 4: Split by battery ---
    train_df, test_df = split_data(df)

    # The raw features are used directly. Tree models split each feature at a
    # threshold, so rescaling would not change what they learn.
    X_train = train_df[FEATURE_COLS].values
    groups = train_df["battery"].values
    os.makedirs(EVAL_DIR, exist_ok=True)

    # --- Step 5: Tune and train on the training batteries (Tables 3.6, 3.7) ---
    searches = {}
    for target in ("soh", "rul"):
        searches[target] = tune(target, X_train, train_df[target].values, groups)

    # --- Step 6: Test on the unseen battery ---
    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON THE UNSEEN BATTERY")
    print("=" * 50)
    X_test = test_df[FEATURE_COLS].values
    for target in ("soh", "rul"):
        rf, xgb = (s.best_estimator_ for s in searches[target])
        y = test_df[target].values
        p_rf, p_xgb = rf.predict(X_test), xgb.predict(X_test)
        evaluate(p_rf, y, "Random Forest", target.upper(), TEST_BATTERY)
        evaluate(p_xgb, y, "XGBoost", target.upper(), TEST_BATTERY)
        evaluate((p_rf + p_xgb) / 2, y, "Ensemble", target.upper(), TEST_BATTERY)
        joblib.dump(rf, os.path.join(EVAL_DIR, f"rf_{target}.pkl"))
        joblib.dump(xgb, os.path.join(EVAL_DIR, f"xgb_{target}.pkl"))
        joblib.dump({"rf": cv_scores(searches[target][0]), "xgb": cv_scores(searches[target][1])},
                    os.path.join(EVAL_DIR, f"cv_{target}.pkl"))
    print(f"\nEvaluation models saved to {EVAL_DIR}/")

    # --- Step 7: Final models -- same hyperparameters, retrained on every battery ---
    X_all = df[FEATURE_COLS].values
    for target in ("soh", "rul"):
        for name, search in zip(("rf", "xgb"), searches[target]):
            model = clone(search.best_estimator_).fit(X_all, df[target].values)
            joblib.dump(model, os.path.join(MODELS_DIR, f"{name}_{target}.pkl"))
    print(f"\nFinal models retrained on all batteries: {', '.join(BATTERIES)}")

    print("\n" + "=" * 50)
    print("Final models saved to models/ for the backend:")
    print("  rf_soh.pkl, rf_rul.pkl, xgb_soh.pkl, xgb_rul.pkl")
    print("=" * 50)