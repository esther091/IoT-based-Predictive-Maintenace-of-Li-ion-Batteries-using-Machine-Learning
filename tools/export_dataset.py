"""
export_dataset.py
---------------------------------------------------------------
Writes the processed training dataset to ml/data/processed_cycles.csv:
one row per NASA discharge cycle kept after outlier removal (602 rows),
with the four cycle features, the measured capacity and the SOH and
RUL targets, exactly as ml/train_models.py computes them.

USAGE (from the project root, venv active):
  python tools/export_dataset.py
---------------------------------------------------------------
"""

import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ML_DIR = os.path.join(ROOT, "ml")
sys.path.insert(0, ML_DIR)
os.chdir(ML_DIR)  # train_models uses paths relative to ml/

from train_models import (
    BATTERIES, TRAIN_BATTERIES, TEST_BATTERY,
    extract_discharge_cycles, compute_targets, remove_outliers,
)


def main():
    rows = []
    for battery in BATTERIES:
        rows.extend(extract_discharge_cycles(battery))
    df = remove_outliers(compute_targets(pd.DataFrame(rows)))

    df["role"] = df["battery"].map(lambda b: "test" if b == TEST_BATTERY else
                                   "training" if b in TRAIN_BATTERIES else "")
    out = df[["battery", "role", "cycle_number", "mean_voltage", "mean_current",
              "mean_power", "energy_wh", "capacity", "soh", "rul"]].rename(
        columns={"mean_voltage": "mean_voltage_V", "mean_current": "mean_current_A",
                 "mean_power": "mean_power_W", "capacity": "capacity_Ah",
                 "soh": "soh_percent", "rul": "rul_days"})
    path = os.path.join("data", "processed_cycles.csv")
    out.round(4).to_csv(path, index=False)
    print(f"Wrote {len(out)} cycles to ml/{path}")


if __name__ == "__main__":
    main()
