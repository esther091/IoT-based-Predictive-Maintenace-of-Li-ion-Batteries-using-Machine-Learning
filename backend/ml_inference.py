"""
ml_inference.py
Loads the trained models, and exposes a predict() method
used by the backend to turn live cycle features into SOH/RUL predictions.
"""

import os
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "models")


class MLInference:
    def __init__(self):
        self.loaded = False
        self.rf_soh = None
        self.rf_rul = None
        self.xgb_soh = None
        self.xgb_rul = None

    def load_models(self):
        try:
            self.rf_soh = joblib.load(os.path.join(MODELS_DIR, "rf_soh.pkl"))
            self.rf_rul = joblib.load(os.path.join(MODELS_DIR, "rf_rul.pkl"))
            self.xgb_soh = joblib.load(os.path.join(MODELS_DIR, "xgb_soh.pkl"))
            self.xgb_rul = joblib.load(os.path.join(MODELS_DIR, "xgb_rul.pkl"))
            self.loaded = True
            print("All ML models loaded successfully.")
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            print("Run ml/train_models.py first to generate the .pkl files.")
            self.loaded = False

    def predict(self, mean_voltage, mean_current, mean_power, energy_delta):
        if not self.loaded:
            return {
                "soh_rf": None, "soh_xgb": None, "soh_final": None,
                "rul_rf": None, "rul_xgb": None, "rul_final": None,
                "alert_level": "MODELS_NOT_LOADED"
            }

        # Raw cycle features, in the training column order:
        # mean_voltage, mean_current, mean_power, energy_wh
        features = np.array([[float(mean_voltage), abs(float(mean_current)),
                              abs(float(mean_power)), float(energy_delta)]])

        soh_rf = float(self.rf_soh.predict(features)[0])
        soh_xgb = float(self.xgb_soh.predict(features)[0])
        soh_final = (soh_rf + soh_xgb) / 2.0

        soh_rf = max(0.0, min(100.0, soh_rf))
        soh_xgb = max(0.0, min(100.0, soh_xgb))
        soh_final = max(0.0, min(100.0, soh_final))

        rul_rf = max(0.0, float(self.rf_rul.predict(features)[0]))
        rul_xgb = max(0.0, float(self.xgb_rul.predict(features)[0]))
        rul_final = (rul_rf + rul_xgb) / 2.0

        if rul_final <= 7:
            alert = "URGENT"
        elif rul_final <= 15:
            alert = "CRITICAL"
        elif rul_final <= 30:
            alert = "WARNING"
        else:
            alert = "NORMAL"

        return {
            "soh_rf": round(soh_rf, 2),
            "soh_xgb": round(soh_xgb, 2),
            "soh_final": round(soh_final, 2),
            "rul_rf": round(rul_rf, 1),
            "rul_xgb": round(rul_xgb, 1),
            "rul_final": round(rul_final, 1),
            "alert_level": alert,
        }


inference = MLInference()