# Report updates needed (COMPLETE_PROJECT_REPORT_FINAL_v2)

The code was corrected and the models retrained, so the report must be updated
to match. New tables are in `report_results/results.md`, and Figures 4.3 to 4.6
are in `report_results/`. Regenerate both at any time with
`python tools/report_results.py`.

## 1. Scope: Li-ion validation (agree with your supervisor first)

- Keep the title, but state in Sections 1.6, 4.3 and 5.3 that the system is
  **validated on Li-ion**. VRLA support is future work: it needs per-unit
  (chemistry-referenced) normalisation and VRLA capacity data.
- Update the test rig in Sections 3.1.1 and 4.1.3 and in Table 4.8:
  - one 18650 cell of about 2 Ah
  - a 2.0 A constant-current load
  - a 2.7 V cut-off
  - the PZEM-017 powered through USB
  - the 12 V VRLA battery, 12 V charger and 21 W load replaced
- Remove Figure 3.3's 12 V `BAT1`, or relabel it as the cell under test.

## 2. Chapter 3

- **Section 3.1.3, step 4, and Section 3.2.2:** the four features are computed
  over the readings in which the current is below −0.5 A, in training and in
  the backend alike.
- **Section 3.2.2, last paragraph:** replace "the backend uses the change in the
  PZEM-017 cumulative energy register" with "the backend integrates V·|I| over
  the device timestamps with the trapezoidal rule, the same method as training,
  because the PZEM-017 energy register resolves only 1 Wh".
- **Section 3.2.4:** the training ranges are now: mean voltage 3.38 to 3.58 V,
  mean current 2.010 to 2.013 A, mean power 6.79 to 7.20 W, energy 4.42 to
  7.24 Wh.
- **Section 3.1.5 and Figure 3.9:** already correct. The dashboard now has all
  four charts: voltage, current, power and energy.

## 3. Chapter 4 numbers (copy from `report_results/results.md`)

- **Table 4.4:** new hyperparameters and cross-validation scores. Rewrite the
  sentence "Both XGBoost models selected shallow trees of depth 3 with 100
  rounds and a learning rate of 0.2".
- **Tables 4.5 and 4.6 and Figures 4.3 to 4.5:** new values and images.
- **Section 4.2.3:** XGBoost SOH MAE is 0.57% (R² 0.992). Random Forest is
  1.76% (0.944). The ensemble is 1.16% (0.975).
- **Section 4.2.4:** XGBoost RUL MAE is 4.71 days (R² 0.883). Random Forest is
  4.82 days (0.876). The ensemble is 4.73 days (0.887).
- **Section 4.2.5:** XGBoost is best for SOH. For RUL the **ensemble** has the
  best R² (0.887) and RMSE (5.38 days).
- **Section 4.2.6 and Figure 4.6:** importances are now energy 0.50, mean
  voltage 0.34, mean power 0.14 and mean current 0.02 (the SOH model).
  **Mean voltage is now informative**, which agrees with Sedighfar and Moniri
  (2018). Mean current carries almost nothing because the NASA cells were
  discharged at a constant 2 A. Rewrite this paragraph and the matching
  sentences in Sections 4.3 and 5.2.
- **Table 4.7 (live replay through the backend):**

| Quantity | Cycle 1 (healthy) | Cycle 132 (end of life) |
|---|---|---|
| True SOH / true RUL | 92.8% / 48.0 days | 67.1% / 0 days |
| Offline prediction, full-resolution features: SOH / RUL | 91.7% / 35.0 days | 68.5% / 2.3 days |
| Live pipeline, ensemble: SOH / RUL | 90.5% / 32.7 days | 68.2% / 2.7 days |
| Alert raised on dashboard | NORMAL | URGENT |

  Other replayed cycles:

| Cycle | True SOH | Live SOH |
|---|---|---|
| B0005 cycle 51 | 87.9% | 86.7% |
| B0005 cycle 161 | 65.2% | 65.5% |
| B0007 cycle 81 | 80.8% | 80.8% |
| B0018 cycle 61 | 79.0% | 78.8% |

- **Section 4.2.7 text:** delete the explanation of the 8.6-point gap (60
  readings and the energy register). Say instead that the live SOH was within
  about 1 to 2 points of the true value for every replayed cycle, because the
  live features are computed exactly as in training. Retake the Figure 4.7
  screenshot, because the current one shows empty boxes from the old dashboard.

## 4. Abstract and Chapter 5

- **Abstract and Section 5.1:** "XGBoost achieved an SOH MAE of 0.57%
  (R² = 0.992) and an RUL MAE of 4.7 days (R² = 0.883); the ensemble achieved
  R² = 0.975 and 0.887."
- **Section 5.2, fourth bullet, and Section 5.3, third bullet:** replace the
  "faster sampling" idea with this: live features must be computed over the
  same discharge-only readings, and with the same energy integration, as
  training. The project did this and it gave close live predictions.

## 5. Appendices: paste the current files

| Appendix | File | What changed |
|---|---|---|
| A | `firmware/esp32_firmware_v2/esp32_firmware_v2.ino` | A rejected packet no longer blocks the SPIFFS cache; single-cell notes added |
| B | `ml/train_models.py` | Features computed over discharge samples only (current < −0.5 A) |
| C | `backend/ml_inference.py` | Clipping to [0, 1], matching the report |
| D | `backend/main.py` | Absolute current, V·I·dt energy integration, readings sorted by timestamp |
| F | `dashboard/dashboard.py` | 127.0.0.1, bordered panels, power and energy charts, "models not loaded" banner |
| G | `tools/simulate_discharge.py` | 127.0.0.1, dataset-time timestamps |

In Table 4.1, the libraries also include matplotlib 3.11.2 (used for the figures).
