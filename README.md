# IoT-Based Predictive Maintenance of Li-ion Batteries using ML

A PZEM-017 DC meter read by an ESP32 streams voltage, current, power and energy
every 5 s to a FastAPI backend. The backend detects each discharge cycle, turns
it into four features, and predicts State of Health (SOH) and Remaining Useful
Life (RUL) with a Random Forest + XGBoost ensemble trained on the NASA PCoE
Li-ion dataset. A Streamlit dashboard shows the live readings, predictions and
staged 30 / 15 / 7-day alerts.

## Layout

| Folder | Contents |
|---|---|
| `ml/` | `train_models.py` (training), `eval_saved_models.py`, `utils.py`, `data/` (NASA `.mat` files), `models/` (trained `.pkl` files) |
| `backend/` | `main.py` (API + discharge-cycle detection), `ml_inference.py`, `database.py` |
| `dashboard/` | `dashboard.py` (Streamlit) |
| `tools/` | `simulate_discharge.py` (replays a NASA cycle in place of the ESP32), `report_results.py` (regenerates Chapter 4 tables and figures) |
| `firmware/` | `esp32_firmware_v2/esp32_firmware_v2.ino` (Arduino IDE) |
| `report_results/` | Chapter 4 tables (`results.md`) and Figures 4.3 to 4.6 |
| `docs/` | `SETUP_AND_RUN_GUIDE.md` (hardware wiring + running, **start here**), `PROJECT_SPEC.md`, `REPORT_UPDATES.md` |
| `archive/` | Old files kept for reference only; nothing uses them (see `archive/README.txt`) |
| `start_system.bat` | Double-click to start the backend and dashboard together |

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the models (only needed after changing the training code)

```powershell
cd ml
python train_models.py      # ~5 min; writes ml/models/*.pkl
python eval_saved_models.py # prints test-set metrics
cd ..
```

## Quick start

Double-click `start_system.bat`. It opens the backend and the dashboard in two
windows and prints your laptop's IP address for the firmware. For the full
hardware wiring and test procedure, see `docs/SETUP_AND_RUN_GUIDE.md`.

## Run the demo manually (three terminals, venv active in each)

```powershell
# 1 - backend
cd backend; python main.py        # wait for "All ML models loaded successfully."

# 2 - dashboard
cd dashboard; python -m streamlit run dashboard.py

# 3 - simulator, from the project root
python tools\simulate_discharge.py --stage late    # worn cell  -> red/amber, URGENT
python tools\simulate_discharge.py --stage early   # healthy cell -> green, NORMAL
```

To start with an empty history, stop the backend and delete `backend/battery.db`.

## Regenerate the report's results

```powershell
python tools\report_results.py    # writes report_results/results.md and Figures 4.3-4.6
```

## Hardware test rig (3S 18650 Li-ion pack)

The rig as built (circuit: `docs/Schematic_diagram.pdf`):

- **Three 18650 Li-ion cells in series (3S, 8.1-12.6 V)** in single holders,
  protected by a **3S 20 A BMS**. A pack is needed because the PZEM-017 powers
  its measuring side from the battery and needs at least 7 V.
- Each cell is charged **off the rig** on its own TP4056 module to 4.2 V.
- Load: **two 3.3 ohm 25 W resistors in series (6.6 ohm)** through a 3-pin
  switch; the current is about 1.9 A falling to 1.3 A.
- The discharge ends when the **BMS cuts off**; the backend recognises the
  cut-off when the readings stop after the pack reached 3.2 V per cell.
- Firmware settings: `CELLS_IN_SERIES 3` (voltage, power and energy are sent
  per cell, matching the single-cell training data), `DISCHARGE_ONLY_RIG true`
  (current always sent as negative; GPIO27 not used), `SHUNT_RANGE_CODE` to
  match your shunt, and `WIFI_SSID`, `WIFI_PASSWORD`, `SERVER_URL` (your
  laptop's LAN IP, port 8000).
- A cycle is predicted only if the resting voltage before the load was
  3.99-4.5 V per cell (charged), it reached 3.2 V per cell or the BMS cut-off,
  and the readings did not stop for 3 minutes. Otherwise it is stored as
  `NOT_STARTED_FULL`, `PARTIAL_DISCHARGE` or `INTERRUPTED` with no prediction.

The models were trained on single cells discharged at a constant 2 A. The
resistive load draws less (1.3-1.9 A), so rig features fall below the training
ranges and are clipped; for accurate rig predictions use a 2 A constant-current
load or retrain on rig data.

The backend computes cycle energy by integrating V x I over the ESP32
timestamps (the same trapezoidal method as training), because the PZEM-017
energy register only resolves 1 Wh.
