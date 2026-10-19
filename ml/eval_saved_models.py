import joblib
import pandas as pd

from utils import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Reuse the training pipeline's own feature extraction and labelling so the
# evaluation always matches exactly what the models were trained on.
from train_models import (
    BATTERIES, TEST_BATTERY, FEATURE_COLS,
    extract_discharge_cycles, compute_targets, remove_outliers,
)

# Load saved models
# Evaluation models (trained without the test battery); models/ holds the
# final models retrained on every battery, which have seen B0018.
rf_soh = joblib.load('models/eval/rf_soh.pkl')
rf_rul = joblib.load('models/eval/rf_rul.pkl')
xgb_soh = joblib.load('models/eval/xgb_soh.pkl')
xgb_rul = joblib.load('models/eval/xgb_rul.pkl')

print("Loaded models successfully.")

# Build the labelled dataset exactly as train_models.py does, then keep the test cell
all_rows = []
for battery in BATTERIES:
    all_rows.extend(extract_discharge_cycles(battery))
df = remove_outliers(compute_targets(pd.DataFrame(all_rows)))
df = df[df['battery'] == TEST_BATTERY].reset_index(drop=True)

X_scaled = df[FEATURE_COLS].values

y_soh = df['soh']
y_rul = df['rul']

pred_rf_soh = rf_soh.predict(X_scaled)
pred_xgb_soh = xgb_soh.predict(X_scaled)
pred_ens_soh = (pred_rf_soh + pred_xgb_soh) / 2.0

pred_rf_rul = rf_rul.predict(X_scaled)
pred_xgb_rul = xgb_rul.predict(X_scaled)
pred_ens_rul = (pred_rf_rul + pred_xgb_rul) / 2.0

print(f"\n--- SOH RESULTS ({TEST_BATTERY} Test Battery) ---")
for name, p in [('Random Forest', pred_rf_soh), ('XGBoost', pred_xgb_soh), ('Ensemble', pred_ens_soh)]:
    mae = mean_absolute_error(y_soh, p)
    rmse = root_mean_squared_error(y_soh, p)
    mape = mean_absolute_percentage_error(y_soh, p)
    r2 = r2_score(y_soh, p)
    print(f"{name:15s} | MAE: {mae:.3f}% | RMSE: {rmse:.3f}% | MAPE: {mape:.3f}% | R2: {r2:.4f}")

print(f"\n--- RUL RESULTS ({TEST_BATTERY} Test Battery) ---")
for name, p in [('Random Forest', pred_rf_rul), ('XGBoost', pred_xgb_rul), ('Ensemble', pred_ens_rul)]:
    mae = mean_absolute_error(y_rul, p)
    rmse = root_mean_squared_error(y_rul, p)
    mape = mean_absolute_percentage_error(y_rul, p)
    r2 = r2_score(y_rul, p)
    print(f"{name:15s} | MAE: {mae:.3f} days | RMSE: {rmse:.3f} days | MAPE: {mape:.3f}% | R2: {r2:.4f}")
