"""
train_models.py
Loads the NASA Battery Dataset, extracts discharge cycles,
trains Random Forest and XGBoost models for SOH and RUL prediction,
evaluates them, and saves the trained models.
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from utils import compute_soh, compute_rul, clip_normalized, mean_absolute_percentage_error

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
TRAIN_BATTERIES = ["B0005", "B0006"]
VAL_BATTERY = "B0007"
TEST_BATTERY = "B0018"

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
    """Splits into train/validation/test sets -- Table 3.4."""
    train_df = df[df["battery"].isin(TRAIN_BATTERIES)].reset_index(drop=True)
    val_df = df[df["battery"] == VAL_BATTERY].reset_index(drop=True)
    test_df = df[df["battery"] == TEST_BATTERY].reset_index(drop=True)

    print(f"\nTrain set ({'+'.join(TRAIN_BATTERIES)}): {len(train_df)} rows")
    print(f"Validation set ({VAL_BATTERY}):  {len(val_df)} rows")
    print(f"Test set ({TEST_BATTERY}):        {len(test_df)} rows")

    return train_df, val_df, test_df


def evaluate(model, X, y_true, model_name, target_name):
    """Evaluates a model using MAE, RMSE, MAPE, R2 -- Equations 3.17-3.20."""
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} -- {target_name} (on {TEST_BATTERY} test set):")
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

    # --- Step 4: Split into train/val/test ---
    train_df, val_df, test_df = split_data(df)

    # --- Step 5: Min-Max normalization -- Equation 3.10 ---
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLS])  # fit ONLY on training data

    X_train = scaler.transform(train_df[FEATURE_COLS])
    X_val = scaler.transform(val_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])

    y_train_soh = train_df["soh"].values
    y_val_soh = val_df["soh"].values
    y_test_soh = test_df["soh"].values

    y_train_rul = train_df["rul"].values
    y_val_rul = val_df["rul"].values
    y_test_rul = test_df["rul"].values

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"\nScaler fitted and saved to {MODELS_DIR}/scaler.pkl")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")

    # --- Step 6: Train Random Forest -- tuned via GridSearchCV (Table 3.5) ---
    print("\nTuning Random Forest (SOH) via GridSearchCV...")
    grid_rf_soh = GridSearchCV(
        RandomForestRegressor(random_state=42),
        PARAM_GRID_RF, cv=5, scoring="r2", n_jobs=-1,
    )
    grid_rf_soh.fit(X_train, y_train_soh)
    rf_soh = grid_rf_soh.best_estimator_
    print(f"Best params (RF-SOH): {grid_rf_soh.best_params_}")

    print("\nTuning Random Forest (RUL) via GridSearchCV...")
    grid_rf_rul = GridSearchCV(
        RandomForestRegressor(random_state=42),
        PARAM_GRID_RF, cv=5, scoring="r2", n_jobs=-1,
    )
    grid_rf_rul.fit(X_train, y_train_rul)
    rf_rul = grid_rf_rul.best_estimator_
    print(f"Best params (RF-RUL): {grid_rf_rul.best_params_}")

    # --- Step 7: Train XGBoost -- tuned via RandomizedSearchCV (Table 3.6) ---
    print("\nTuning XGBoost (SOH) via RandomizedSearchCV...")
    search_xgb_soh = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=42),
        PARAM_DIST_XGB, n_iter=50, cv=5, scoring="r2", n_jobs=-1, random_state=42,
    )
    search_xgb_soh.fit(X_train, y_train_soh)
    xgb_soh = search_xgb_soh.best_estimator_
    print(f"Best params (XGB-SOH): {search_xgb_soh.best_params_}")

    print("\nTuning XGBoost (RUL) via RandomizedSearchCV...")
    search_xgb_rul = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=42),
        PARAM_DIST_XGB, n_iter=50, cv=5, scoring="r2", n_jobs=-1, random_state=42,
    )
    search_xgb_rul.fit(X_train, y_train_rul)
    xgb_rul = search_xgb_rul.best_estimator_
    print(f"Best params (XGB-RUL): {search_xgb_rul.best_params_}")

    print("\nAll four models tuned and trained successfully.")

    # --- Step 8: Evaluate on held-out test set (B0018) ---
    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 50)

    evaluate(rf_soh, X_test, y_test_soh, "Random Forest", "SOH")
    evaluate(rf_rul, X_test, y_test_rul, "Random Forest", "RUL")
    evaluate(xgb_soh, X_test, y_test_soh, "XGBoost", "SOH")
    evaluate(xgb_rul, X_test, y_test_rul, "XGBoost", "RUL")

    # --- Step 9: Save trained models ---
    joblib.dump(rf_soh, os.path.join(MODELS_DIR, "rf_soh.pkl"))
    joblib.dump(rf_rul, os.path.join(MODELS_DIR, "rf_rul.pkl"))
    joblib.dump(xgb_soh, os.path.join(MODELS_DIR, "xgb_soh.pkl"))
    joblib.dump(xgb_rul, os.path.join(MODELS_DIR, "xgb_rul.pkl"))

    print("\n" + "=" * 50)
    print("All models saved to models/ folder:")
    print("  scaler.pkl, rf_soh.pkl, rf_rul.pkl, xgb_soh.pkl, xgb_rul.pkl")
    print("=" * 50)