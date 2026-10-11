# IoT-Based Predictive Maintenance of VRLA and Li-ion Batteries using ML
## Comprehensive Software Build Specification

> **Purpose of this document.** This is a complete, self-contained blueprint for building the *software* of this final-year project from start to finish. It is written to be handed directly to **Claude Code** (or any capable engineer) so the entire system can be reconstructed, file by file, without needing any other document. It describes *what* each component does, *why* it is designed that way, the *exact* data contracts between components, and a *phased build plan* with acceptance criteria you can check off one at a time. Full, corrected reference implementations of every source file are provided in the appendices.

---

## 0. How to use this document (read first)

**If you are Claude Code:** Build the project by following **Section 14 — Phased Build Plan** in order. Each phase has a goal, the files to create, and an explicit *acceptance test* that must pass before moving on. The appendices (A–J) contain the exact, corrected source for every file — treat them as the source of truth for implementation details, and use Sections 1–13 for the *reasoning* behind each choice. Do not skip the acceptance tests; they are how we prove the real-time pipeline works without any hardware attached.

**If you are the student (project owner):** This doubles as your technical reference and viva-defence companion. Section 17 lists the design decisions you should be ready to justify, each with the reasoning already written out.

**One deliberate correction is baked into this blueprint.** The original code computed the discharge-cycle *mean current* feature with a different sign convention than the training pipeline used. Training used the **absolute value** of current (a positive magnitude), while the backend averaged the **raw signed** current (negative during discharge). This blueprint standardises on **absolute-value current magnitude everywhere**, so the feature fed to the models at inference time matches the feature the models were trained on. See **Section 12** for the full explanation. Everything else mirrors the working system you already built.

---

## 1. Project overview

### 1.1 What the system does

The system continuously watches a battery's electrical behaviour and predicts, ahead of time, **how healthy it is** and **how long it has left**, raising an alert before the battery fails. It is a classic *predictive-maintenance* (PdM) loop applied to energy-storage batteries:

1. **Sense** the battery's terminal quantities (voltage, current, power, cumulative energy) at a fixed cadence.
2. **Detect** each discharge cycle from the current sign and aggregate that cycle into a small set of features.
3. **Predict** two quantities with a machine-learning ensemble: **State of Health (SOH)** and **Remaining Useful Life (RUL)**.
4. **Present** the live readings, the latest SOH/RUL, and a colour-coded alert on a real-time dashboard.

### 1.2 Key definitions

- **State of Health (SOH):** the battery's present usable capacity as a percentage of its rated (new) capacity. `SOH = (Q_actual / Q_rated) × 100`. A brand-new battery is ~100%; end-of-life is conventionally 70–80%.
- **Remaining Useful Life (RUL):** how much longer the battery will remain above its end-of-life threshold, expressed here in **days** (converted from charge/discharge cycles via a documented assumption).
- **End of Life (EOL):** the point at which SOH drops below **70%** of rated capacity. For the NASA cells used here (2.0 Ah rated), that is **1.4 Ah**.
- **Predictive maintenance:** replacing/servicing based on the *predicted* condition of the asset, rather than on a fixed calendar schedule (preventive) or after it breaks (reactive).

### 1.3 Why it matters (motivation)

The project is framed around Nigeria's unstable grid: frequent outages force batteries (in inverters, UPS systems, telecom base stations, solar installations) through many more charge/discharge cycles than they were rated for, accelerating degradation. Unplanned battery failure is costly and disruptive. A low-cost IoT monitor that forecasts failure gives operators time to act. The approach is deliberately **chemistry-general**: it senses only terminal electrical quantities that any battery exposes, so the same pipeline is meant to serve both **VRLA** (lead-acid) and **Li-ion** batteries.

### 1.4 The chemistry-transfer idea

The ML models are trained on the **NASA PCoE Li-ion 18650** ageing dataset (the best public, labelled capacity-fade data available), then applied to VRLA batteries in deployment. The bridge between the two chemistries is **Min-Max feature normalisation with clipping**: features are scaled into the `[0, 1]` range learned from training data, and out-of-range live values are clipped, so the models operate on a chemistry-neutral normalised feature space. This is a documented modelling assumption, not a claim that the chemistries are identical — see Section 17.

---

## 2. Scope of this build

### 2.1 In scope (this document)

The complete **software** stack, end to end:

- The **ML pipeline**: dataset ingestion, feature extraction, SOH/RUL labelling, train/validation/test split, Min-Max scaling, Random Forest + XGBoost training with hyper-parameter search, evaluation, and model persistence.
- The **backend**: a FastAPI service that ingests readings, detects discharge cycles, runs ensemble inference, persists everything to SQLite, and serves the dashboard.
- The **dashboard**: a Streamlit app that polls the backend and renders live metrics, an SOH gauge, an RUL countdown, a model-comparison table, time-series charts, and colour-coded alerts.
- The **simulator**: a tool that replays a real NASA discharge cycle into the backend so the whole pipeline can be demonstrated producing a live prediction **with no hardware attached**.

### 2.2 Out of scope (this document)

- **ESP32 firmware** (Arduino C++) and the **PZEM-017 / MAX485 wiring**. In deployment the ESP32 is the data source that POSTs readings to the backend; for this software build the **simulator plays that role exactly** (same HTTP contract). The firmware is listed under future work (Section 18) and can be specified separately.

### 2.3 How the hardware boundary is represented in software

The backend neither knows nor cares whether readings come from a real ESP32 or the simulator — both send identical JSON to `POST /readings`. This clean seam is what makes the system fully testable in software. Anywhere this document says "the ESP32," read "the ESP32 *or* the simulator."

---

## 3. System architecture

### 3.1 Four conceptual layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — SENSING            (hardware; out of scope here)            │
│  Battery ─▶ PZEM-017 DC meter (200A shunt) ─▶ ESP32                    │
│  Measures: voltage, current, power, cumulative energy                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  JSON over local WiFi (HTTP POST)
                                 │  ****  simulator stands in here  ****
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — TRANSPORT / EDGE                                            │
│  HTTP POST /readings  (one reading every ~5 s)                         │
└───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — APPLICATION / INFERENCE   (FastAPI backend)                 │
│  • Persist every reading to SQLite                                     │
│  • Detect discharge cycles (current < −0.5 A)                          │
│  • Buffer a cycle, aggregate to 4 features when the cycle ends         │
│  • Run RF + XGBoost ensemble ─▶ SOH, RUL, alert level                  │
│  • Persist cycle + prediction; expose REST endpoints                   │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  REST (GET /latest, /history, ...)
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — PRESENTATION   (Streamlit dashboard)                        │
│  Polls every 5 s ─▶ live metrics, SOH gauge, RUL, alerts, charts       │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 End-to-end data flow (the two clocks)

The single most important behaviour to understand — and the thing most likely to be questioned at a viva — is that the system runs on **two different clocks**:

1. **The 5-second monitoring clock.** Every ~5 seconds a new reading arrives and is stored; the dashboard's live metric cards and time-series charts update at this cadence. This is what makes the system feel "real-time."
2. **The per-cycle prediction clock.** SOH and RUL are *cycle-level* quantities — they only have meaning once a whole discharge cycle has completed. So a *new prediction* is produced only when a discharge cycle **ends**, not on every reading. Between cycles, the dashboard holds and displays the **last** prediction.

This is physically correct: a battery's health cannot meaningfully change *within* a single 30-minute discharge, so predicting on every 5-second reading would be noise. The system predicts once per completed cycle and monitors continuously in between. Section 9.4 specifies the cycle-detection state machine that implements this.

---

## 4. Technology stack & prerequisites

### 4.1 Runtime

| Component | Choice | Notes |
|---|---|---|
| Language | **Python 3.10+** | 3.10 or 3.11 recommended (XGBoost/scikit-learn wheels are well-supported). |
| Backend framework | **FastAPI** + **Uvicorn** | ASGI server, lifespan startup hook. |
| Database | **SQLite** via **SQLAlchemy** ORM | Single-file DB (`battery.db`), zero-config. |
| ML | **scikit-learn** (Random Forest, scaling, metrics) + **XGBoost** | Ensemble = average of the two. |
| Numerics / data | **NumPy**, **pandas**, **SciPy** (`scipy.io.loadmat`) | SciPy reads the NASA `.mat` files. |
| Persistence of models | **joblib** | `.pkl` artifacts. |
| Dashboard | **Streamlit** + **Plotly** | Gauge + line charts. |
| HTTP client (sim/dashboard) | **requests** | |

### 4.2 `requirements.txt`

Pin at least these packages (see Appendix I for the full file):

```
fastapi
uvicorn[standard]
sqlalchemy
pydantic>=2
scikit-learn
xgboost
numpy
pandas
scipy
joblib
streamlit
plotly
requests
```

### 4.3 Prerequisites checklist

- Python 3.10+ installed and on `PATH`.
- Ability to create a virtual environment (`python -m venv venv`).
- The four NASA battery `.mat` files (Section 6.2) placed in `ml/data/`.
- ~500 MB free disk for dependencies; training runs comfortably on a laptop CPU (minutes, not hours).

---

## 5. Repository structure (target)

Build the project to this exact layout. Paths in this document are relative to the project root.

```
Battery-predictive-maintenance/
├── PROJECT_SPEC.md            ← this document
├── README.md                  ← quick-start (Appendix J)
├── requirements.txt           ← pinned dependencies (Appendix I)
├── .gitignore                 ← Appendix I
│
├── ml/                        ← the machine-learning pipeline
│   ├── train_models.py        ← end-to-end training script (Appendix C)
│   ├── utils.py               ← SOH/RUL/metric helpers (Appendix B)
│   ├── data/                  ← NASA .mat dataset files (you provide)
│   │   ├── B0005.mat
│   │   ├── B0006.mat
│   │   ├── B0007.mat
│   │   └── B0018.mat
│   └── models/                ← trained artifacts (generated by training)
│       ├── scaler.pkl
│       ├── rf_soh.pkl
│       ├── rf_rul.pkl
│       ├── xgb_soh.pkl
│       └── xgb_rul.pkl
│
├── backend/                   ← FastAPI application
│   ├── main.py                ← app, endpoints, cycle state machine (Appendix F)
│   ├── database.py            ← SQLAlchemy models + session (Appendix D)
│   ├── ml_inference.py        ← model loading + predict() (Appendix E)
│   └── battery.db             ← SQLite file (generated at runtime)
│
├── dashboard/                 ← Streamlit UI
│   └── dashboard.py           ← the live dashboard (Appendix G)
│
└── tools/                     ← developer/demo tooling
    ├── simulate_discharge.py  ← replays a NASA cycle into the backend (Appendix H)
    └── RUN_DEMO.md            ← 3-terminal demo walkthrough
```

**Import-path note.** `backend/main.py` imports `from database import ...` and `from ml_inference import ...` (flat imports, not `backend.database`). This works because the backend is launched **from inside the `backend/` directory** (`cd backend && python main.py`). `ml_inference.py` locates the models with a path relative to its own file (`../ml/models`), so it resolves correctly regardless of the current working directory. Keep these conventions; they are what the run commands in Section 15 rely on.

---

## 6. Data layer — the NASA battery dataset

### 6.1 Source and cells

Training data is the **NASA Ames Prognostics Center of Excellence (PCoE) Battery Data Set**: commercial **Li-ion 18650** cells (2.0 Ah rated) cycled to failure at room temperature. Four cells are used:

| Cell | Role | Why |
|---|---|---|
| **B0005** | Train | Full run-to-failure capacity fade. |
| **B0006** | Train | Full run-to-failure capacity fade. |
| **B0007** | Validation | Held out for tuning sanity checks. |
| **B0018** | Test | Fully held out; all reported metrics come from this cell. |

Across all four, ~636 discharge cycles are available after cleaning. Splitting **by cell** (not by random rows) is deliberate: it tests whether the model generalises to a battery it has *never seen*, which is the real deployment condition.

### 6.2 Obtaining the files

Place `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` in `ml/data/`. They are distributed by NASA PCoE (and widely mirrored on Kaggle as the "NASA Battery Dataset"). This build assumes they are already present — the training script reads them directly.

### 6.3 Structure of a `.mat` file (exact access pattern)

Each file is a MATLAB struct. The training script and simulator both navigate it identically — replicate this exactly:

```python
mat = loadmat("B0005.mat")           # scipy.io.loadmat
cycles = mat["B0005"][0, 0]["cycle"][0]   # array of cycle records

for entry in cycles:
    if entry["type"][0] != "discharge":   # keep only discharge cycles
        continue
    data     = entry["data"][0, 0]
    voltage  = data["Voltage_measured"][0]   # 1-D array over the cycle
    current  = data["Current_measured"][0]   # 1-D array (negative during discharge)
    time     = data["Time"][0]               # seconds within the cycle
    capacity = data["Capacity"][0][0]         # scalar: measured Ah for this cycle
```

`Capacity` is the ground-truth measured discharge capacity for that cycle — it is what SOH labels are derived from. `Current_measured` is **negative during discharge** in the raw dataset (discharge draws current out of the cell).

---

## 7. Domain model & equations

All equations below are the ones referenced in Chapter 3 of the report; keep the numbering so the code comments and the write-up stay aligned.

### 7.1 Features (exactly four)

Each completed discharge cycle is reduced to **four features**, computed over the samples in that cycle:

| Feature | Definition | Sign convention |
|---|---|---|
| `mean_voltage` | mean of `Voltage_measured` over the cycle | positive |
| `mean_current` | mean of **\|current\|** over the cycle | **positive magnitude** (see §12) |
| `mean_power` | mean of `voltage × \|current\|` over the cycle | positive |
| `energy_wh` | `∫ (voltage × \|current\|) dt / 3600` over the cycle | positive, in Wh |

The model feature vector column order is fixed: `["mean_voltage", "mean_current", "mean_power", "energy_wh"]`. This order must be identical in training (`train_models.py`) and inference (`ml_inference.py`).

> **Backend note.** At inference the backend supplies `energy_delta = max(energy) − min(energy)` (the Wh consumed across the buffered cycle) as the fourth feature, positionally matching `energy_wh`. Both represent watt-hours delivered during the discharge, so they are consistent. See §9.5.

### 7.2 SOH — Equation 3.8

```
SOH(n) = ( Q_measured(n) / Q_rated ) × 100      with Q_rated = 2.0 Ah
```

Implemented in `utils.compute_soh(capacity_actual, capacity_rated=2.0)`.

### 7.3 RUL — Equation 3.9

RUL is first computed in **cycles remaining until EOL**, then converted to **days**:

```
RUL(n) = max(0, (n_EOL − n) × cycle_duration_days)     with cycle_duration_days = 0.5
```

- `n_EOL` is the first cycle index at which `SOH < 70%`. If a cell never crosses 70% within the data, `n_EOL` is taken as its last observed cycle.
- `cycle_duration_days = 0.5` encodes the modelling assumption of **~2 full cycles per day** in deployment. This is the single biggest assumption in the RUL figure and must be stated plainly (Section 17). It is a linear scale factor: change it and every RUL-in-days scales proportionally; the underlying cycles-to-EOL prediction is unaffected.

Implemented in `utils.compute_rul(n_eol, n_current, cycle_duration_days=0.5)`.

### 7.4 End-of-life threshold

`EOL_CAPACITY = 1.4 Ah` = 70% of 2.0 Ah rated. Equivalent to `SOH < 70%`.

### 7.5 Min-Max normalisation — Equation 3.10 & clipping 3.11

```
x_scaled = (x − x_min) / (x_max − x_min)         # per feature
x_clipped = min(1, max(0, x_scaled))             # live values only
```

The scaler's `x_min`/`x_max` are learned from the **training cells only** (never from validation/test), then reused unchanged everywhere. `clip_normalized` is available in `utils.py`; note that scikit-learn's `MinMaxScaler.transform` does **not** clip on its own, which is exactly why live features must be well-formed (Section 12).

### 7.6 Evaluation metrics — Equations 3.17–3.20

Reported on the held-out **B0018** test cell for each of the four models (RF-SOH, RF-RUL, XGB-SOH, XGB-RUL):

| Metric | Meaning |
|---|---|
| **MAE** | Mean Absolute Error (same units as target). |
| **RMSE** | Root Mean Squared Error (penalises large misses). |
| **MAPE** | Mean Absolute Percentage Error (`utils.mean_absolute_percentage_error`, zero-safe mask). |
| **R²** | Coefficient of determination (fraction of variance explained). |

---

## 8. ML pipeline specification (`ml/train_models.py`)

A single script performs the whole training pipeline and saves five artifacts. Full source in **Appendix C**; the required behaviour, step by step:

1. **Extract** — For each of the four cells, load the `.mat`, iterate cycles, keep `type == "discharge"`, and compute the four features (§7.1) plus the raw `capacity`. Uses `np.abs(current)` for the current-dependent features. Energy uses `np.trapezoid(voltage × |current|, time) / 3600`.
2. **Label** — Add `soh` (Eq 3.8) and `rul` (Eq 3.9). RUL labelling finds each cell's `n_EOL` as the first cycle with `soh < 70`, else the last cycle.
3. **Clean** — Drop rows with implausible SOH (keep `65 ≤ SOH ≤ 103`) to remove measurement artefacts.
4. **Split by cell** — Train = {B0005, B0006}, Validation = {B0007}, Test = {B0018} (§6.1).
5. **Scale** — Fit `MinMaxScaler` on **train features only**; transform train/val/test. Save `scaler.pkl`.
6. **Train RF** — `RandomForestRegressor` tuned with `GridSearchCV` (5-fold, scoring `r2`) for SOH and again for RUL. Grid in Appendix C / §13.
7. **Train XGB** — `XGBRegressor(objective="reg:squarederror")` tuned with `RandomizedSearchCV` (50 iters, 5-fold, scoring `r2`) for SOH and RUL.
8. **Evaluate** — Report MAE/RMSE/MAPE/R² for all four models on the **test** cell.
9. **Persist** — `joblib.dump` the five artifacts to `ml/models/`: `scaler.pkl`, `rf_soh.pkl`, `rf_rul.pkl`, `xgb_soh.pkl`, `xgb_rul.pkl`.

**Determinism.** All estimators and searches use `random_state=42` so results are reproducible.

**Artifacts contract.** The backend later loads exactly these five files by name. Do not rename them.

---

## 9. Backend specification (FastAPI)

Three files: `database.py` (persistence), `ml_inference.py` (models), `main.py` (API + cycle state machine). Full source in Appendices D, E, F.

### 9.1 Database schema (`database.py`)

SQLite at `sqlite:///./battery.db`, created relative to the backend's working directory. `create_engine` uses `connect_args={"check_same_thread": False}` because FastAPI may touch the session from different threads. Three tables:

**`readings`** — every raw reading, stored on arrival.

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | |
| `timestamp` | DateTime | defaults to `datetime.utcnow` |
| `voltage` | Float | V, not null |
| `current` | Float | A (signed as received), not null |
| `power` | Float | W, not null |
| `energy` | Float | Wh cumulative, not null |

**`cycles`** — one row per completed discharge cycle (the aggregated features).

| Column | Type |
|---|---|
| `id` | Integer PK |
| `cycle_number` | Integer |
| `cycle_start` | DateTime |
| `cycle_end` | DateTime |
| `mean_voltage` | Float |
| `mean_current` | Float |
| `mean_power` | Float |
| `energy_delta` | Float |

**`predictions`** — one row per completed cycle (the model outputs).

| Column | Type |
|---|---|
| `id` | Integer PK |
| `timestamp` | DateTime (default utcnow) |
| `cycle_number` | Integer |
| `soh_rf`, `soh_xgb`, `soh_final` | Float |
| `rul_rf`, `rul_xgb`, `rul_final` | Float |
| `alert_level` | String |

`create_tables()` runs `Base.metadata.create_all`. `get_db()` is the FastAPI dependency that yields a session and always closes it.

### 9.2 Inference module (`ml_inference.py`)

A singleton `inference = MLInference()` with:

- `load_models()` — loads the five `.pkl` files from `../ml/models` (resolved relative to this file). Sets `self.loaded = True` on success; on `FileNotFoundError` prints a helpful message and leaves `loaded = False` (the API still runs, but returns an `alert_level` of `"MODELS_NOT_LOADED"`).
- `predict(mean_voltage, mean_current, mean_power, energy_delta)` —
  1. Build a one-row `DataFrame` with columns `["mean_voltage","mean_current","mean_power","energy_wh"]` (order matters).
  2. `scaler.transform(...)`.
  3. `soh_rf = rf_soh.predict`, `soh_xgb = xgb_soh.predict`, `soh_final = (soh_rf + soh_xgb)/2`.
  4. Clip all SOH to `[0, 100]`.
  5. `rul_rf`, `rul_xgb` each clamped to `≥ 0`; `rul_final = (rul_rf + rul_xgb)/2`.
  6. Derive `alert_level` from `rul_final` (thresholds in §9.6).
  7. Return a dict rounded to sensible precision (SOH 2 dp, RUL 1 dp).

**Ensemble rationale.** Averaging RF and XGBoost reduces variance and hedges the two models' opposite failure modes; it is simple, defensible, and needs no extra training.

### 9.3 API surface (`main.py`)

FastAPI app with a **lifespan** context manager (not the deprecated `@app.on_event`). On startup it calls `create_tables()`, `inference.load_models()`, and **recovers `cycle_counter`** from the DB (`max(Prediction.cycle_number)`), so cycle numbering survives restarts. CORS is fully open (`allow_origins=["*"]`) so the Streamlit app (different port) can call it.

| Method & path | Purpose | Response |
|---|---|---|
| `POST /readings` | Ingest one reading; drives the cycle state machine. | `{"status":"ok","id":<int>}` |
| `GET /latest` | Most recent reading + most recent prediction. | `LatestOut` |
| `GET /history?limit=100` | Last `limit` readings, oldest-first, for charts. | `List[ReadingOut]` |
| `GET /predictions` | All predictions, oldest-first. | `List[PredictionOut]` |
| `GET /health` | Liveness + state introspection. | see below |

`GET /health` returns `{status, models_loaded, is_discharging, cycle_count, buffer_size}` — useful for the dashboard status dot and for debugging the state machine.

### 9.4 Request/response schemas (Pydantic v2)

**`ReadingInput`** (the contract the ESP32/simulator must satisfy):

| Field | Type | Constraint | Post-processing |
|---|---|---|---|
| `voltage` | float | `> 0` and `< 350` | rejected if `< 0.05` ("check PZEM-017 connections"); rounded to 3 dp |
| `current` | float | `> −350` and `< 350` | rounded to 3 dp (sign preserved as received) |
| `power` | float | `≥ 0` and `< 100000` | rounded to 2 dp |
| `energy` | float | `≥ 0` | rounded to 1 dp |
| `timestamp` | int | optional | epoch-ms from device; DB uses its own server time |

Use Pydantic v2 idioms: `Field(...)`, `@field_validator`, and `model_config = ConfigDict(from_attributes=True)` on the output models so they can be built directly from ORM rows.

**`ReadingOut`**: `id, timestamp, voltage, current, power, energy`.
**`PredictionOut`**: `id, timestamp, cycle_number, soh_rf, soh_xgb, soh_final, rul_rf, rul_xgb, rul_final, alert_level` (SOH/RUL optional to allow the not-loaded case).
**`LatestOut`**: `{reading: Optional[ReadingOut], prediction: Optional[PredictionOut]}`.

### 9.5 The discharge-cycle state machine (the heart of the backend)

Global state: `discharge_buffer` (list of reading dicts), `is_discharging` (bool), `cycle_start_time`, `cycle_counter`. Threshold constant `DISCHARGE_THRESHOLD = -0.5` (amps). On every `POST /readings`:

1. **Always persist** the raw reading to `readings`.
2. **If `current < −0.5`** (the cell is discharging, with a −0.5 A noise guard so tiny negative noise near rest doesn't trigger a cycle):
   - If we were *not* already discharging, this is a **rising edge**: set `is_discharging = True`, stamp `cycle_start_time = now`, clear the buffer, log "Discharge started".
   - Append this reading to `discharge_buffer`.
3. **Else** (`current ≥ −0.5`):
   - If we *were* discharging, this is a **falling edge** (the discharge just ended): set `is_discharging = False`, log the buffer size, and call `process_discharge_cycle(db)`.

`process_discharge_cycle(db)`:

1. If the buffer has **fewer than 3 readings**, discard it and return (too short to be a real cycle — rejects blips).
2. Aggregate the buffer into the four features:
   - `mean_voltage = mean(voltage)`
   - **`mean_current = mean(|current|)`** ← *the corrected line; absolute value so the sign matches training (Section 12).*
   - `mean_power = mean(power)` (already ≥ 0 from the device/schema)
   - `energy_delta = max(energy) − min(energy)`
3. Increment `cycle_counter`; write a `Cycle` row (start, end = now, the four features).
4. Call `inference.predict(...)`; write a `Prediction` row.
5. Commit. On any exception: `rollback`, **decrement `cycle_counter`** (so a failed cycle doesn't consume a number), log the error.
6. `finally:` clear the buffer.

**How a cycle "ends" in software.** The simulator (and, later, a resting battery) sends a few readings with `current = 0` after the discharge. Those cross the threshold from below, produce the falling edge, and trigger inference. This is exactly how a real battery returning to rest closes a cycle.

### 9.6 Alert levels (from `rul_final`, in days)

| Condition | `alert_level` | Meaning |
|---|---|---|
| `rul_final ≤ 7` | `URGENT` | Replace now. |
| `7 < rul_final ≤ 15` | `CRITICAL` | Schedule replacement soon. |
| `15 < rul_final ≤ 30` | `WARNING` | Ageing; plan ahead. |
| `rul_final > 30` | `NORMAL` | Healthy. |
| models not loaded | `MODELS_NOT_LOADED` | Train models first. |

---

## 10. Dashboard specification (Streamlit)

Single file `dashboard/dashboard.py`. Full source in **Appendix G**. It talks to the backend at `BACKEND_URL = "http://localhost:8000"` and refreshes every `REFRESH_INTERVAL = 5` seconds by calling `time.sleep(5)` then `st.rerun()` at the end of the render pass (a simple, reliable polling loop).

### 10.1 Page layout (top to bottom)

1. **Header** — 🔋 title "Battery Predictive Maintenance", a **status dot** (green `#2ECC71` when `/health` responds, red `#E74C3C` when the backend is offline), and a subtitle with the project name.
2. **Alert banner** — full-width, colour-coded by `alert_level` (§10.3). Before the first cycle completes it shows a neutral "⏳ Waiting for first discharge cycle to complete…" banner.
3. **Four live metric cards** — Voltage (V, 2 dp), Current (A, 2 dp), Power (W, 1 dp), Energy (Wh, 1 dp), from `/latest.reading`. Show `--` when there is no reading yet.
4. **Three panels in a row** — SOH gauge (§10.2), RUL countdown (big number of days + "Replace before <date>"), and a Model-Comparison table (RF / XGBoost / Ensemble × SOH% / RUL-days) from the prediction.
5. **Two time-series charts** — Voltage over Time (blue `#2563EB`) and Current over Time (amber `#D97706`), built from `/history` with Plotly Express.
6. **Raw Readings table** — the last 100 readings (`timestamp, voltage, current, power, energy`).
7. **Footer caption** — "Last updated: HH:MM:SS · Auto-refreshes every 5s".

All cards use the custom CSS injected at the top (light theme, rounded cards, subtle shadow). The design is intentionally light-mode.

### 10.2 SOH gauge (colours & thresholds)

A Plotly `go.Indicator(mode="gauge+number")` with a semicircular arc, the SOH number in the centre, and a coloured bar whose colour depends on the value:

| SOH range | Bar colour | Hex |
|---|---|---|
| `≥ 80` | green | `#2ECC71` |
| `60 – 80` | amber | `#F5B301` |
| `< 60` | red | `#E74C3C` |

Static background "steps" tint the arc for context: `[0,60] → #FDECEC`, `[60,80] → #FFF6DD`, `[80,100] → #E9F7EF`. Axis range is `[0,100]`. When there is no prediction the gauge shows `0`.

### 10.3 Alert banner styles

CSS classes and the message template for each level (RUL rounded to whole days; `replace_by = today + RUL days`):

| Level | CSS class | bg / text | Message |
|---|---|---|---|
| URGENT | `alert-urgent` | `#FDECEC` / `#B3261E` | 🚨 URGENT — Replace battery now. `{rul}` days remaining. |
| CRITICAL | `alert-critical` | `#FFF1E0` / `#B5540A` | ⚠️ CRITICAL — Schedule replacement soon. `{rul}` days remaining. |
| WARNING | `alert-warning` | `#FFF8E1` / `#8A6D00` | ⚡ WARNING — Battery ageing. Replace by `{replace_by}`. |
| NORMAL | `alert-normal` | `#E9F7EF` / `#1E7E34` | ✅ Battery healthy — estimated `{rul}` days remaining (replace by `{replace_by}`). |
| (pending) | `alert-pending` | `#EEF2F7` / `#4A5568` | ⏳ Waiting for first discharge cycle to complete… |

### 10.4 Resilience

Every backend call (`fetch_latest`, `fetch_history`, `fetch_health`) is wrapped in try/except with a 3-second timeout and returns a safe empty value on failure, so the dashboard degrades gracefully (shows `--`, flips the status dot to red) instead of crashing when the backend is briefly unavailable.

---

## 11. Simulator specification (`tools/simulate_discharge.py`)

The simulator is what lets you demonstrate the entire real-time pipeline **with no hardware**. It plays the exact role the ESP32 will later play, honouring the same `POST /readings` contract. Full source in **Appendix H**.

### 11.1 What it does

1. Loads a **real** discharge cycle from a NASA `.mat` file (same access pattern as training, §6.3), so the features are authentic.
2. Streams it reading-by-reading into `POST /readings`, sending:
   - `current` as a **negative** value (forced `−|i|`) so the backend's `current < −0.5` rule opens the buffer — this is the real discharge sign, and it is exactly why the backend must take `|current|` when forming the feature (Section 12).
   - `power` as `|voltage × current| ≥ 0` (matches the PZEM register and the schema).
   - `energy` as a **monotonically rising cumulative Wh**, so `energy_delta = max − min` is meaningful.
3. After the discharge, sends a few **idle readings** (`current = 0`) to produce the falling edge that closes the cycle and triggers inference.
4. Reads back `GET /latest` and prints the produced prediction next to the dataset's **true** SOH, so you can eyeball model-vs-ground-truth.

### 11.2 Command-line interface

```
python tools/simulate_discharge.py [--stage early|late] [--battery B0005|B0006|B0007|B0018]
                                    [--cycle N] [--delay S] [--samples N] [--url URL]
```

| Flag | Default | Meaning |
|---|---|---|
| `--stage` | `late` | `early` = first (healthy) cycle → high SOH, green; `late` = most-degraded cycle → low SOH, amber/red. |
| `--battery` | `B0018` | Which cell to replay. |
| `--cycle N` | — | Replay a specific discharge-cycle index (overrides `--stage`). |
| `--delay S` | `0.15` | Seconds between readings. Use `5` to replay at the true real-time cadence. |
| `--samples N` | `60` | How many readings to send across the cycle (down-sampled evenly). |
| `--url` | `http://localhost:8000` | Backend base URL. |

It performs a `GET /health` reachability check first and exits with a helpful message if the backend isn't running.

---

## 12. The current-sign correction (the one deliberate fix)

**This section documents the single correctness change this blueprint makes relative to the original code. Implement it as described.**

### 12.1 The problem

During discharge, current is **negative** (energy leaving the cell). Two places consume current, and they originally disagreed on sign:

- **Training** (`train_models.py`) built the `mean_current` and `mean_power` features from `np.abs(current)` — i.e. a **positive magnitude** (~2 A for the NASA cells). The `MinMaxScaler` therefore learned its min/max on **positive** current.
- **Backend inference** (`main.py`) originally averaged the **raw signed** current from the buffer, which is **negative** during discharge (~ −2 A). It then fed that negative value into the same scaler.

Because `MinMaxScaler.transform` does **not** clip, a negative `mean_current` scales to a value outside the `[0, 1]` range the models were trained on. The models were never shown negative current, so this is an out-of-distribution input — a genuine train/serve skew.

### 12.2 The fix

Standardise on **absolute-value current magnitude everywhere the feature is formed**. Concretely, in `process_discharge_cycle` compute:

```python
currents = [r["current"] for r in discharge_buffer]   # signed, negative during discharge
mean_current = sum(abs(c) for c in currents) / len(currents)   # ← positive magnitude, matches training
```

`mean_power` needs no change: `power` is already delivered as a non-negative register value (the schema enforces `power ≥ 0`, and the simulator sends `|V×I|`), so the backend's `mean_power` is already the positive quantity training used. `energy_delta` is a difference of a rising cumulative counter, so it is already positive.

### 12.3 Why the fix belongs in the backend, not the simulator

The simulator (and the real ESP32) **must** send negative current during discharge, because that negative sign is precisely what the backend's `current < −0.5` detector uses to recognise a discharge. So the raw signed value is correct on the wire and correct in the `readings` table; only the **feature derived for the model** must be made positive. Hence the `abs()` lives in `process_discharge_cycle`, at the moment the feature is formed — not on the wire.

### 12.4 Practical impact

For the NASA cells the discharge current is roughly constant (~2 A), so `mean_current` carries relatively little discriminative weight versus voltage/energy, and the original sign bug had a small numerical effect. But it is a real train/serve-skew defect, it is trivial to fix, and fixing it removes an obvious line of questioning at a viva. The clean blueprint fixes it.

---

## 13. Configuration reference (all the magic numbers in one place)

Every tunable constant, where it lives, and what it means. Keep these identical across files where they appear in more than one.

| Constant | Value | Location | Meaning |
|---|---|---|---|
| `RATED_CAPACITY` | `2.0` Ah | `train_models.py`, `utils.compute_soh`, simulator | New-cell capacity; SOH denominator. |
| `EOL_CAPACITY` | `1.4` Ah | `train_models.py` | 70% of rated; end-of-life. |
| `cycle_duration_days` | `0.5` | `utils.compute_rul` | ~2 cycles/day; converts RUL cycles→days. |
| `FEATURE_COLS` | `["mean_voltage","mean_current","mean_power","energy_wh"]` | `train_models.py`, `ml_inference.py` | Feature order — must match. |
| `TRAIN_BATTERIES` | `["B0005","B0006"]` | `train_models.py` | Training cells. |
| `VAL_BATTERY` | `"B0007"` | `train_models.py` | Validation cell. |
| `TEST_BATTERY` | `"B0018"` | `train_models.py` | Held-out test cell. |
| Outlier SOH window | `65 ≤ SOH ≤ 103` | `train_models.py` | Drops implausible capacity rows. |
| RF grid | see Appendix C | `train_models.py` | `GridSearchCV`, 5-fold, `r2`. |
| XGB dist | see Appendix C | `train_models.py` | `RandomizedSearchCV`, 50 iters, 5-fold, `r2`. |
| `random_state` | `42` | everywhere | Reproducibility. |
| `DATABASE_URL` | `sqlite:///./battery.db` | `database.py` | SQLite file. |
| `DISCHARGE_THRESHOLD` | `-0.5` A | `main.py` | Discharge-detect + noise guard. |
| Min buffer length | `3` readings | `main.py` | Reject too-short cycles. |
| Reading validation | V∈(0,350), I∈(−350,350), P∈[0,1e5), E≥0 | `main.py` | Pydantic bounds; V<0.05 rejected. |
| Alert thresholds | `7 / 15 / 30` days | `ml_inference.py` | URGENT/CRITICAL/WARNING/NORMAL. |
| SOH gauge bands | `80 / 60` | `dashboard.py` | green/amber/red. |
| Server bind | `0.0.0.0:8000` | `main.py` | Uvicorn host:port. |
| `BACKEND_URL` | `http://localhost:8000` | `dashboard.py`, simulator | Where clients call. |
| `REFRESH_INTERVAL` | `5` s | `dashboard.py` | Dashboard poll cadence. |

---

## 14. Phased build plan (follow in order)

This is the build order for Claude Code. Each phase lists its goal, the files to produce (from the appendices), and an **acceptance test** that must pass before continuing. Do not proceed past a failing acceptance test.

### Phase 0 — Scaffold & environment

**Goal:** a clean repo skeleton and a working virtual environment.

Do:
1. Create the directory tree from Section 5 (empty `ml/models/`, `backend/`, `dashboard/`, `tools/`, and `ml/data/`).
2. Write `requirements.txt`, `.gitignore`, and `README.md` (Appendices I, J).
3. Create and activate a venv; `pip install -r requirements.txt`.

**Acceptance test:** `python -c "import fastapi, uvicorn, sqlalchemy, pydantic, sklearn, xgboost, numpy, pandas, scipy, joblib, streamlit, plotly, requests; print('ok')"` prints `ok`.

### Phase 1 — Data in place

**Goal:** the four NASA cells are available to the pipeline.

Do: place `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` in `ml/data/`.

**Acceptance test:** this snippet prints four non-zero discharge-cycle counts:
```python
from scipy.io import loadmat
for b in ["B0005","B0006","B0007","B0018"]:
    m = loadmat(f"ml/data/{b}.mat")
    n = sum(1 for e in m[b][0,0]["cycle"][0] if e["type"][0]=="discharge")
    print(b, n)
```

### Phase 2 — ML helpers

**Goal:** `ml/utils.py` with the four helper functions (Appendix B).

**Acceptance test:** `compute_soh(1.4, 2.0) == 70.0`; `compute_rul(100, 40) == 30.0`; `clip_normalized(1.7) == 1.0`; `mean_absolute_percentage_error([100],[90])` ≈ `10.0`.

### Phase 3 — Train the models

**Goal:** produce the five artifacts in `ml/models/`.

Do: implement `ml/train_models.py` (Appendix C) exactly; run it from inside `ml/` (`cd ml && python train_models.py`).

**Acceptance test:** the run prints per-model MAE/RMSE/MAPE/R² on B0018 and the five files `scaler.pkl, rf_soh.pkl, rf_rul.pkl, xgb_soh.pkl, xgb_rul.pkl` now exist in `ml/models/`. Sanity: SOH R² on the test cell should be clearly positive (a working pipeline lands well above 0; treat a near-zero or negative R² as a bug to investigate, not a pass).

### Phase 4 — Persistence layer

**Goal:** `backend/database.py` with the three tables and session helpers (Appendix D).

**Acceptance test:** `cd backend && python -c "import database; database.create_tables(); print('tables ok')"` creates `backend/battery.db` and prints `tables ok`.

### Phase 5 — Inference module

**Goal:** `backend/ml_inference.py` loads the artifacts and predicts (Appendix E).

**Acceptance test:** from `backend/`:
```python
from ml_inference import inference
inference.load_models()          # prints "All ML models loaded successfully."
print(inference.predict(3.7, 2.0, 7.4, 4.5))   # positive current magnitude
```
returns a dict with numeric `soh_final`, `rul_final`, and a valid `alert_level` (not `MODELS_NOT_LOADED`).

### Phase 6 — Backend API & cycle state machine

**Goal:** `backend/main.py` (Appendix F), including the **corrected** `mean_current = mean(|current|)` (Section 12).

Do: implement all endpoints and the state machine of §9.5. Launch with `cd backend && python main.py`.

**Acceptance tests:**
1. `GET /health` → `{"status":"running","models_loaded":true,...}`.
2. Post a single charging reading (`current = 1.8`) → `readings` grows, **no** cycle/prediction created, `is_discharging` stays false. (Verifies the detector doesn't fire on charge.)
3. Post a short sequence of discharge readings (`current < −0.5`) followed by an idle reading (`current = 0`) → exactly one `Cycle` and one `Prediction` row appear, and `GET /latest` returns a prediction. (Verifies rising edge → buffer → falling edge → inference.)
4. Post only two discharge readings then idle → **no** prediction (buffer < 3 rejected).

### Phase 7 — Simulator & the live demo

**Goal:** `tools/simulate_discharge.py` (Appendix H) drives a real cycle end-to-end.

**Acceptance test:** with the backend running, `python tools/simulate_discharge.py --stage late` prints a `PREDICTION PRODUCED` block whose `SOH (Ensemble)` is a plausible percentage and whose printed **true SOH** from the dataset is in the same ballpark; a `Cycle`/`Prediction` pair is persisted.

### Phase 8 — Dashboard

**Goal:** `dashboard/dashboard.py` (Appendix G) visualises the live system.

Do: launch with `cd dashboard && streamlit run dashboard.py`.

**Acceptance test:** with backend + dashboard running, execute `python tools/simulate_discharge.py --stage late --delay 5` and observe: the status dot is green; the V/I/P/E cards and the two charts update while the discharge streams; the instant the cycle closes the SOH gauge fills with the right colour, the RUL day-count and "replace by" date appear, the model-comparison table populates, and the alert banner switches to the matching colour. Running `--stage early` afterwards drives SOH up and the gauge to green.

### Phase 9 — Full verification pass

Run the checks in Section 16. When they pass, the software build is complete.

---

## 15. Run & operations guide

Three processes, three terminals, venv activated in each. On Windows PowerShell, activate with `.\venv\Scripts\Activate.ps1` (if the script is blocked once, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`).

```
# Terminal 1 — backend (the brain)
cd backend
python main.py
# wait for: "All ML models loaded successfully." and "Waiting for ESP32 data..."

# Terminal 2 — dashboard (the screen)
cd dashboard
streamlit run dashboard.py
# a browser opens; it will say "Waiting for first discharge cycle..." until a cycle completes

# Terminal 3 — simulator (stands in for the ESP32)
python tools/simulate_discharge.py --stage late      # degraded battery → amber/red
python tools/simulate_discharge.py --stage early      # healthy battery → green
python tools/simulate_discharge.py --stage late --delay 5   # true 5-s real-time cadence
```

**Networking note.** Dashboard ↔ backend is pure loopback (`localhost:8000`); nothing touches the internet or a cloud. When the real ESP32 is added later, it joins the same local WiFi/router and POSTs to the PC's LAN IP on port 8000 — no code change on the backend.

**Editor note (VS Code).** If `main.py` shows "import could not be resolved" on `from database import …` / `from ml_inference import …`, that is Pylance not pointing at the venv, not a runtime error. Fix with `Ctrl+Shift+P → Python: Select Interpreter → .\venv`.

**Reset the system:** stop the processes and delete `backend/battery.db` to clear all readings/cycles/predictions; it is recreated empty on next start (and `cycle_counter` resets to 0).

---

## 16. Testing & verification plan

A green build satisfies all of the following. These are grouped so failures point straight at the responsible component.

**ML correctness.**
- The four `.mat` files each yield a non-zero discharge-cycle count (Phase 1 test).
- Training completes and reports a clearly positive SOH R² on the B0018 test cell; five artifacts written.
- The scaler is fit on training cells only (verify: `train_models.py` calls `scaler.fit` on `train_df[FEATURE_COLS]`, never on val/test).

**Feature-contract correctness (the fix).**
- `process_discharge_cycle` forms `mean_current` from `|current|`. Verify by posting discharge readings with `current = −2.0` and confirming the persisted `Cycle.mean_current` is `+2.0`, not `−2.0`.
- The inference DataFrame column order is exactly `["mean_voltage","mean_current","mean_power","energy_wh"]`.

**State-machine correctness.**
- Charge-only readings never create a cycle (Phase 6 test 2).
- A discharge burst + idle creates exactly one cycle/prediction (test 3).
- Buffers shorter than 3 readings are rejected (test 4).
- A failed cycle does not consume a `cycle_number` (force an exception path and confirm the counter is decremented).
- Restarting the backend resumes `cycle_counter` from the DB max (post one cycle, restart, confirm `/health.cycle_count` is preserved).

**API correctness.**
- `/latest`, `/history?limit=N`, `/predictions`, `/health` all return their documented shapes; `/history` is oldest-first.

**Dashboard correctness.**
- Status dot reflects backend up/down.
- Gauge colour matches SOH band (≥80 green, 60–80 amber, <60 red).
- Alert banner class matches `alert_level`.
- Cards/charts update at ~5 s; a completed cycle updates gauge/RUL/table within one refresh.

**End-to-end.**
- `--stage early` vs `--stage late` produce visibly different SOH/RUL/alert, each near the dataset's true SOH.

**Recommended:** capture screenshots of the green (early) and red/amber (late) dashboard states for the report's Chapter 4 results.

---

## 17. Design decisions & viva-defence notes

Be ready to justify each of these; the reasoning is written out so you can defend it confidently.

- **Why split by battery, not random rows?** Random splitting leaks a cell's own future cycles into training and inflates scores. Splitting by cell (train B0005/B0006, val B0007, test B0018) measures generalisation to an unseen battery — the real deployment condition.
- **Why an RF + XGBoost average?** Two strong, differently-biased tree ensembles; averaging cuts variance and avoids betting on one model. It needs no meta-learner and is easy to explain.
- **Why only four features?** They are the quantities a cheap DC meter (PZEM-017) actually exposes: voltage, current, power, energy. Keeping to sensed quantities is what makes the method low-cost and chemistry-general. Richer features (internal resistance, temperature, IC curves) would need more/《costlier sensing.
- **Why Min-Max scaling with clipping for the chemistry transfer?** It maps features into a normalised `[0,1]` space learned from training data; clipping keeps out-of-range live values bounded. It is the documented bridge from Li-ion training data to VRLA deployment — an approximation, not a claim of identical chemistry.
- **What is the RUL-in-days assumption?** RUL is predicted in cycles, then multiplied by `cycle_duration_days = 0.5` (~2 cycles/day). This linear factor should be tuned to the actual cycling frequency of the deployment; it scales the day figure but not the underlying cycles-to-EOL prediction.
- **Why predict once per cycle, not every reading?** SOH/RUL are cycle-level; health cannot change within one discharge. The 5-second stream drives *monitoring*; the completed cycle drives *prediction* (the "two clocks", §3.2).
- **Why `−0.5 A`, not `0`, as the discharge threshold?** A small negative dead-band rejects sensor noise around rest so idle jitter doesn't open spurious cycles.
- **Why reject buffers < 3 readings?** A couple of stray negative samples shouldn't count as a discharge cycle; three is a minimal guard against blips.
- **Known correction (own it proactively):** the train/serve current-sign skew, now fixed by taking `|current|` in the backend feature (Section 12). Small numerical impact for near-constant NASA discharge current, but corrected for rigour.
- **Why SQLite?** Zero-config, single-file, perfectly adequate for one monitored battery and a single dashboard; trivially swappable for Postgres via SQLAlchemy if scaled.

---

## 18. Known limitations & future work

- **Chemistry transfer is assumed, not yet validated on VRLA field data.** The models are trained on NASA Li-ion; quantifying VRLA accuracy needs labelled VRLA capacity data. State this as a limitation and a next step.
- **`cycle_duration_days` is a fixed assumption.** A deployment could log real timestamps per cycle and compute the cadence empirically instead of assuming 0.5.
- **Single-battery scope.** The schema tracks one battery; a `battery_id` column and per-battery buffers would generalise to a fleet.
- **Polling dashboard.** `time.sleep + st.rerun` is simple but re-fetches on a timer; a websocket/push model would be more efficient at scale.
- **Out of scope here (hardware track):** ESP32 firmware (Modbus-RTU read of the PZEM-017 over MAX485, WiFi POST to the backend, SPIFFS caching for WiFi dropouts) and the physical wiring. These are the natural next deliverable and slot in behind the existing `POST /readings` contract with no backend change.
- **No authentication on the API.** Fine for a LAN demo; add a token/reverse-proxy before any non-local exposure.

---

# Appendices — full corrected reference source

> These are the authoritative implementations. Where the main body and an appendix ever seem to differ, the appendix wins for code and the body wins for rationale. The **only** intentional change from the working original is the `mean_current = mean(|current|)` line in Appendix F (Section 12).

## Appendix A — index & conventions

| Appendix | File | Run from |
|---|---|---|
| B | `ml/utils.py` | imported |
| C | `ml/train_models.py` | `ml/` |
| D | `backend/database.py` | imported |
| E | `backend/ml_inference.py` | imported |
| F | `backend/main.py` | `backend/` |
| G | `dashboard/dashboard.py` | `dashboard/` |
| H | `tools/simulate_discharge.py` | project root |
| I | `requirements.txt`, `.gitignore` | — |
| J | `README.md` | — |

Conventions: Python 3.10+, 4-space indent, module docstrings kept, `random_state=42` throughout, flat imports inside `backend/` (Section 5).

## Appendix B — `ml/utils.py`

```python
"""
utils.py
Helper functions for SOH, RUL computation and evaluation metrics.
Equations reference Chapter 3 of the project report.
"""

import numpy as np


def compute_soh(capacity_actual, capacity_rated=2.0):
    """
    SOH = (Q_actual / Q_rated) x 100
    Equation 3.8
    """
    return (capacity_actual / capacity_rated) * 100


def compute_rul(n_eol, n_current, cycle_duration_days=0.5):
    """
    RUL(n) = n_EOL - n, converted to days
    Equation 3.9
    """
    return max(0, (n_eol - n_current) * cycle_duration_days)


def clip_normalized(x):
    """
    Clips a normalized value to the [0, 1] range
    Equation 3.11
    """
    return max(0.0, min(1.0, x))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE = (100/n) * sum(|y_true - y_pred| / |y_true|)
    Equation 3.19
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
```

## Appendix C — `ml/train_models.py`

```python
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
```

## Appendix D — `backend/database.py`

```python
"""
database.py
SQLite database setup using SQLAlchemy.
Defines the three tables: readings, cycles, predictions.
"""

from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

DATABASE_URL = "sqlite:///./battery.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Reading(Base):
    __tablename__ = "readings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    power = Column(Float, nullable=False)
    energy = Column(Float, nullable=False)


class Cycle(Base):
    __tablename__ = "cycles"

    id = Column(Integer, primary_key=True, index=True)
    cycle_number = Column(Integer)
    cycle_start = Column(DateTime)
    cycle_end = Column(DateTime)
    mean_voltage = Column(Float)
    mean_current = Column(Float)
    mean_power = Column(Float)
    energy_delta = Column(Float)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cycle_number = Column(Integer)
    soh_rf = Column(Float)
    soh_xgb = Column(Float)
    soh_final = Column(Float)
    rul_rf = Column(Float)
    rul_xgb = Column(Float)
    rul_final = Column(Float)
    alert_level = Column(String)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Appendix E — `backend/ml_inference.py`

```python
"""
ml_inference.py
Loads the trained scaler and models, and exposes a predict() method
used by the backend to turn live cycle features into SOH/RUL predictions.
"""

import os
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "models")


class MLInference:
    def __init__(self):
        self.loaded = False
        self.scaler = None
        self.rf_soh = None
        self.rf_rul = None
        self.xgb_soh = None
        self.xgb_rul = None

    def load_models(self):
        try:
            self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
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

        import pandas as pd
        features = pd.DataFrame(
            [[mean_voltage, mean_current, mean_power, energy_delta]],
            columns=["mean_voltage", "mean_current", "mean_power", "energy_wh"]
        )
        features_scaled = self.scaler.transform(features)

        soh_rf = float(self.rf_soh.predict(features_scaled)[0])
        soh_xgb = float(self.xgb_soh.predict(features_scaled)[0])
        soh_final = (soh_rf + soh_xgb) / 2.0

        soh_rf = max(0.0, min(100.0, soh_rf))
        soh_xgb = max(0.0, min(100.0, soh_xgb))
        soh_final = max(0.0, min(100.0, soh_final))

        rul_rf = max(0.0, float(self.rf_rul.predict(features_scaled)[0]))
        rul_xgb = max(0.0, float(self.xgb_rul.predict(features_scaled)[0]))
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
```

## Appendix F — `backend/main.py`  *(contains the Section 12 fix)*

The **only** change from the working original is one line in `process_discharge_cycle`, marked `# FIX (Section 12)`. Everything else is unchanged.

```python
"""
main.py
FastAPI backend for the IoT battery predictive maintenance system.
Receives sensor readings from ESP32, detects discharge cycles,
runs ML inference, and serves data to the Streamlit dashboard.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timezone
from typing import List, Optional
import uvicorn

from database import get_db, create_tables, Reading, Cycle, Prediction, SessionLocal
from ml_inference import inference


# ============================================================
# LIFESPAN -- replaces deprecated @app.on_event("startup")
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cycle_counter

    create_tables()
    inference.load_models()

    # Recover cycle_counter from the database instead of resetting to 0
    db = SessionLocal()
    try:
        max_cycle = db.query(func.max(Prediction.cycle_number)).scalar()
        cycle_counter = max_cycle if max_cycle is not None else 0
    finally:
        db.close()

    print(f"Backend started. Resuming from cycle_counter={cycle_counter}. "
          f"Waiting for ESP32 data...")

    yield  # app runs here

    print("Backend shutting down.")


app = FastAPI(
    title="Battery Predictive Maintenance API",
    description="IoT-Based Predictive Maintenance of VRLA and Li-ion Batteries",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

discharge_buffer = []
is_discharging = False
cycle_start_time = None
cycle_counter = 0

DISCHARGE_THRESHOLD = -0.5


# ============================================================
# PYDANTIC SCHEMAS -- Pydantic v2 syntax
# ============================================================
class ReadingInput(BaseModel):
    voltage: float = Field(..., gt=0, lt=350)
    current: float = Field(..., gt=-350, lt=350)
    power: float = Field(..., ge=0, lt=100000)
    energy: float = Field(..., ge=0)
    timestamp: Optional[int] = None

    @field_validator("voltage")
    @classmethod
    def voltage_check(cls, v):
        if v < 0.05:
            raise ValueError("Voltage too low -- check PZEM-017 connections")
        return round(v, 3)

    @field_validator("current")
    @classmethod
    def current_round(cls, v):
        return round(v, 3)

    @field_validator("power")
    @classmethod
    def power_round(cls, v):
        return round(v, 2)

    @field_validator("energy")
    @classmethod
    def energy_round(cls, v):
        return round(v, 1)


class ReadingOut(BaseModel):
    id: int
    timestamp: datetime
    voltage: float
    current: float
    power: float
    energy: float

    model_config = ConfigDict(from_attributes=True)


class PredictionOut(BaseModel):
    id: int
    timestamp: datetime
    cycle_number: int
    soh_rf: Optional[float]
    soh_xgb: Optional[float]
    soh_final: Optional[float]
    rul_rf: Optional[float]
    rul_xgb: Optional[float]
    rul_final: Optional[float]
    alert_level: str

    model_config = ConfigDict(from_attributes=True)


class LatestOut(BaseModel):
    reading: Optional[ReadingOut]
    prediction: Optional[PredictionOut]


# ============================================================
# DISCHARGE CYCLE PROCESSING
# ============================================================
def process_discharge_cycle(db: Session):
    global discharge_buffer, cycle_counter, cycle_start_time

    if len(discharge_buffer) < 3:
        discharge_buffer = []
        return

    try:
        voltages = [r["voltage"] for r in discharge_buffer]
        currents = [r["current"] for r in discharge_buffer]
        powers = [r["power"] for r in discharge_buffer]
        energies = [r["energy"] for r in discharge_buffer]

        mean_voltage = sum(voltages) / len(voltages)
        # FIX (Section 12): use |current| so the feature sign matches training,
        # which built mean_current from np.abs(current). Raw buffer currents are
        # negative during discharge; the models were trained on positive magnitude.
        mean_current = sum(abs(c) for c in currents) / len(currents)
        mean_power = sum(powers) / len(powers)
        energy_delta = max(energies) - min(energies)

        cycle_end_time = datetime.now(timezone.utc)
        cycle_counter += 1

        cycle = Cycle(
            cycle_number=cycle_counter,
            cycle_start=cycle_start_time,
            cycle_end=cycle_end_time,
            mean_voltage=round(mean_voltage, 4),
            mean_current=round(mean_current, 4),
            mean_power=round(mean_power, 4),
            energy_delta=round(energy_delta, 4),
        )
        db.add(cycle)

        result = inference.predict(mean_voltage, mean_current, mean_power, energy_delta)

        pred = Prediction(
            cycle_number=cycle_counter,
            soh_rf=result["soh_rf"],
            soh_xgb=result["soh_xgb"],
            soh_final=result["soh_final"],
            rul_rf=result["rul_rf"],
            rul_xgb=result["rul_xgb"],
            rul_final=result["rul_final"],
            alert_level=result["alert_level"],
        )
        db.add(pred)
        db.commit()

        print(f"Cycle {cycle_counter} complete -- SOH: {result['soh_final']}% | "
              f"RUL: {result['rul_final']} days | Alert: {result['alert_level']}")

    except Exception as e:
        db.rollback()
        cycle_counter -= 1  # undo the increment since this cycle failed
        print(f"Error processing discharge cycle: {e}")

    finally:
        discharge_buffer = []


# ============================================================
# ENDPOINTS
# ============================================================
@app.post("/readings", status_code=200)
def receive_reading(data: ReadingInput, db: Session = Depends(get_db)):
    global is_discharging, discharge_buffer, cycle_start_time

    try:
        reading = Reading(
            voltage=data.voltage,
            current=data.current,
            power=data.power,
            energy=data.energy,
        )
        db.add(reading)
        db.commit()
        db.refresh(reading)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save reading: {e}")

    reading_dict = {
        "voltage": data.voltage,
        "current": data.current,
        "power": data.power,
        "energy": data.energy,
    }

    if data.current < DISCHARGE_THRESHOLD:
        if not is_discharging:
            is_discharging = True
            cycle_start_time = datetime.now(timezone.utc)
            discharge_buffer = []
            print(f"Discharge started at {cycle_start_time}")
        discharge_buffer.append(reading_dict)
    else:
        if is_discharging:
            is_discharging = False
            print(f"Discharge ended. Buffer size: {len(discharge_buffer)}")
            process_discharge_cycle(db)

    return {"status": "ok", "id": reading.id}


@app.get("/latest", response_model=LatestOut)
def get_latest(db: Session = Depends(get_db)):
    latest_reading = db.query(Reading).order_by(Reading.id.desc()).first()
    latest_prediction = db.query(Prediction).order_by(Prediction.id.desc()).first()
    return LatestOut(reading=latest_reading, prediction=latest_prediction)


@app.get("/history", response_model=List[ReadingOut])
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    readings = db.query(Reading).order_by(Reading.id.desc()).limit(limit).all()
    return list(reversed(readings))


@app.get("/predictions", response_model=List[PredictionOut])
def get_predictions(db: Session = Depends(get_db)):
    return db.query(Prediction).order_by(Prediction.id.asc()).all()


@app.get("/health")
def health_check():
    return {
        "status": "running",
        "models_loaded": inference.loaded,
        "is_discharging": is_discharging,
        "cycle_count": cycle_counter,
        "buffer_size": len(discharge_buffer),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
```

## Appendix G — `dashboard/dashboard.py`

```python
"""
dashboard.py
Streamlit dashboard for real-time battery health monitoring.
Polls the FastAPI backend every 5 seconds and displays live data,
SOH gauge, RUL countdown, and alert banners.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

BACKEND_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5

st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }

    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .dashboard-header h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #0F2A47;
        margin: 0;
    }
    .dashboard-subtitle {
        color: #6B7684;
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E7EAEE;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15, 42, 71, 0.06);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #8A93A0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F2A47;
    }
    .metric-unit {
        font-size: 1rem;
        font-weight: 500;
        color: #9AA3AF;
        margin-left: 4px;
    }

    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E7EAEE;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px rgba(15, 42, 71, 0.06);
        height: 100%;
    }
    .panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F2A47;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .alert-banner {
        border-radius: 12px;
        padding: 16px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .alert-urgent   { background: #FDECEC; color: #B3261E; border: 1px solid #F5C2C0; }
    .alert-critical { background: #FFF1E0; color: #B5540A; border: 1px solid #F7D6A8; }
    .alert-warning  { background: #FFF8E1; color: #8A6D00; border: 1px solid #F2E2A0; }
    .alert-normal   { background: #E9F7EF; color: #1E7E34; border: 1px solid #B7E4C7; }
    .alert-pending  { background: #EEF2F7; color: #4A5568; border: 1px solid #DCE2EA; }

    .section-gap { margin-top: 28px; margin-bottom: 10px; }

    .status-dot {
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background: #2ECC71; }
    .status-offline { background: #E74C3C; }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E7EAEE;
    }
</style>
""", unsafe_allow_html=True)


def fetch_latest():
    try:
        r = requests.get(f"{BACKEND_URL}/latest", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_history():
    try:
        r = requests.get(f"{BACKEND_URL}/history?limit=100", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def metric_card(label, value, unit):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
        </div>
    """, unsafe_allow_html=True)


def soh_gauge(soh_value):
    if soh_value is None:
        soh_value = 0.0

    if soh_value >= 80:
        color = "#2ECC71"
    elif soh_value >= 60:
        color = "#F5B301"
    else:
        color = "#E74C3C"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh_value,
        number={"suffix": "%", "font": {"size": 40, "color": "#0F2A47"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#B0B8C1", "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60], "color": "#FDECEC"},
                {"range": [60, 80], "color": "#FFF6DD"},
                {"range": [80, 100], "color": "#E9F7EF"},
            ],
        }
    ))
    fig.update_layout(
        height=230,
        margin=dict(t=10, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig


def styled_line_chart(df, y_col, title, color):
    fig = px.line(df, x="timestamp", y=y_col)
    fig.update_traces(line=dict(color=color, width=2.5))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#0F2A47")),
        height=230,
        margin=dict(t=36, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=None, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#F0F2F5", title=None, tickfont=dict(size=10)),
    )
    return fig


def main():
    health = fetch_health()
    latest = fetch_latest()
    history = fetch_history()

    reading = latest.get("reading") if latest else None
    prediction = latest.get("prediction") if latest else None

    # ---- Header ----
    status_class = "status-online" if health else "status-offline"
    status_text = "Backend connected" if health else "Backend offline"

    st.markdown(f"""
        <div class="dashboard-header">
            <span style="font-size: 2.2rem;">🔋</span>
            <h1>Battery Predictive Maintenance</h1>
        </div>
        <div class="dashboard-subtitle">
            <span class="status-dot {status_class}"></span>{status_text}
            &nbsp;·&nbsp; IoT-Based Predictive Maintenance of VRLA and Li-ion Batteries
        </div>
    """, unsafe_allow_html=True)

    # ---- Alert banner ----
    if prediction:
        alert = prediction.get("alert_level", "NORMAL")
        rul = prediction.get("rul_final", 0) or 0
        replace_by = (datetime.now() + timedelta(days=rul)).strftime("%d %b %Y")

        banners = {
            "URGENT":   ("alert-urgent", f"🚨 URGENT — Replace battery now. {rul:.0f} days remaining."),
            "CRITICAL": ("alert-critical", f"⚠️ CRITICAL — Schedule replacement soon. {rul:.0f} days remaining."),
            "WARNING":  ("alert-warning", f"⚡ WARNING — Battery ageing. Replace by {replace_by}."),
            "NORMAL":   ("alert-normal", f"✅ Battery healthy — estimated {rul:.0f} days remaining (replace by {replace_by})."),
        }
        cls, msg = banners.get(alert, banners["NORMAL"])
        st.markdown(f'<div class="alert-banner {cls}">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="alert-banner alert-pending">⏳ Waiting for first discharge cycle to complete...</div>',
            unsafe_allow_html=True
        )

    # ---- Live metrics ----
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Voltage", f"{reading['voltage']:.2f}" if reading else "--", "V")
    with c2: metric_card("Current", f"{reading['current']:.2f}" if reading else "--", "A")
    with c3: metric_card("Power", f"{reading['power']:.1f}" if reading else "--", "W")
    with c4: metric_card("Energy", f"{reading['energy']:.1f}" if reading else "--", "Wh")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- SOH / RUL / Model comparison ----
    col_soh, col_rul, col_models = st.columns([1.1, 1, 1.3])

    with col_soh:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🔵 State of Health</div>', unsafe_allow_html=True)
        soh_val = prediction["soh_final"] if prediction else 0
        st.plotly_chart(soh_gauge(soh_val), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_rul:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">⏳ Remaining Useful Life</div>', unsafe_allow_html=True)
        if prediction:
            rul = prediction.get("rul_final", 0) or 0
            replace_by = (datetime.now() + timedelta(days=rul)).strftime("%d %b %Y")
            st.markdown(f"""
                <div style="text-align:center; padding-top: 18px;">
                    <div style="font-size: 2.6rem; font-weight: 800; color: #0F2A47;">{rul:.0f}</div>
                    <div style="color:#8A93A0; font-size:0.85rem; margin-top:-4px;">days remaining</div>
                    <div style="margin-top:16px; color:#4A5568; font-size:0.85rem;">Replace before <b>{replace_by}</b></div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align:center; padding-top: 30px; color:#9AA3AF;">
                    No prediction yet
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_models:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🤖 Model Comparison</div>', unsafe_allow_html=True)
        if prediction:
            df_models = pd.DataFrame({
                "Model": ["Random Forest", "XGBoost", "Ensemble"],
                "SOH (%)": [prediction.get("soh_rf"), prediction.get("soh_xgb"), prediction.get("soh_final")],
                "RUL (days)": [prediction.get("rul_rf"), prediction.get("rul_xgb"), prediction.get("rul_final")],
            })
            st.dataframe(df_models, hide_index=True, use_container_width=True)
        else:
            st.markdown('<div style="color:#9AA3AF; padding-top:20px;">No data yet</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # ---- Time series charts ----
    if history:
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.plotly_chart(styled_line_chart(df, "voltage", "Voltage over Time", "#2563EB"),
                             use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with ch2:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.plotly_chart(styled_line_chart(df, "current", "Current over Time", "#D97706"),
                             use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📋 Raw Readings</div>', unsafe_allow_html=True)
        st.dataframe(
            df[["timestamp", "voltage", "current", "power", "energy"]],
            use_container_width=True, hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No readings yet. Waiting for ESP32 data...")

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} · Auto-refreshes every {REFRESH_INTERVAL}s")

    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
```

## Appendix H — `tools/simulate_discharge.py`

```python
"""
simulate_discharge.py
---------------------------------------------------------------
Pretends to be the ESP32 + PZEM-017 hardware.

It opens a REAL discharge cycle from the NASA Battery Dataset and
replays it, reading-by-reading, into your running FastAPI backend
(POST /readings) -- exactly the way the ESP32 will once the hardware
is wired. This lets you watch a full cycle complete and a real
SOH / RUL prediction appear on your Streamlit dashboard, with no
hardware connected yet.

HOW IT MAPS TO YOUR SYSTEM
  - Sends current as a NEGATIVE value during discharge, so your
    backend's rule `current < -0.5` (DISCHARGE_THRESHOLD) opens the
    buffer, just like real discharge current from the shunt.
  - Sends power >= 0 and a monotonically rising cumulative energy,
    matching the PZEM-017 registers and your Pydantic schema.
  - At the end it sends a few "idle" readings (current = 0), which
    makes the backend close the buffer and run inference -- the
    moment your prediction is produced.

USAGE (run from the project root, with your venv active):
  python tools/simulate_discharge.py --stage late
  python tools/simulate_discharge.py --stage early --delay 5
  python tools/simulate_discharge.py --battery B0018 --cycle 100

  --stage early|late : pick a healthy (early) or degraded (late) cycle
  --battery          : B0005 | B0006 | B0007 | B0018  (default B0018)
  --cycle N          : use a specific discharge cycle index instead of --stage
  --delay S          : seconds between readings (default 0.15 = fast demo;
                       use 5 to replay at the true real-time cadence)
  --samples N        : how many readings to send across the cycle (default 60)
  --url              : backend URL (default http://localhost:8000)
---------------------------------------------------------------
"""

import argparse
import os
import sys
import time

import numpy as np
import requests
from scipy.io import loadmat

RATED_CAPACITY = 2.0  # Ahr, per Table 3.4 -- used only to print the TRUE SOH


def load_discharge_cycles(battery):
    """Return a list of discharge cycles from a NASA .mat file.

    Mirrors the exact structure used in ml/train_models.py so the
    features here match how the models were trained.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(here, "..", "ml", "data", f"{battery}.mat")
    if not os.path.exists(filepath):
        sys.exit(f"Could not find {filepath}. Check the battery name / data folder.")

    mat = loadmat(filepath)
    raw_cycles = mat[battery][0, 0]["cycle"][0]

    cycles = []
    for entry in raw_cycles:
        if entry["type"][0] != "discharge":
            continue
        data = entry["data"][0, 0]
        voltage = np.asarray(data["Voltage_measured"][0], dtype=float)
        current = np.asarray(data["Current_measured"][0], dtype=float)
        t = np.asarray(data["Time"][0], dtype=float)
        capacity = float(data["Capacity"][0][0])
        # keep only clean, usable cycles
        if len(voltage) > 5 and voltage.min() > 0:
            cycles.append(
                {"voltage": voltage, "current": current, "time": t, "capacity": capacity}
            )
    if not cycles:
        sys.exit(f"No usable discharge cycles found in {battery}.")
    return cycles


def pick_cycle(cycles, stage, cycle_idx):
    if cycle_idx is not None:
        idx = max(0, min(cycle_idx, len(cycles) - 1))
        return cycles[idx], idx
    # early = first healthy cycle; late = the most degraded (lowest capacity)
    if stage == "early":
        idx = 0
    else:
        idx = int(np.argmin([c["capacity"] for c in cycles]))
    return cycles[idx], idx


def downsample(arr, n):
    """Evenly pick n indices across the array length."""
    if len(arr) <= n:
        return np.arange(len(arr))
    return np.linspace(0, len(arr) - 1, n).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="B0018")
    ap.add_argument("--stage", choices=["early", "late"], default="late")
    ap.add_argument("--cycle", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.15)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/readings"

    print(f"Loading discharge cycles from {args.battery} ...")
    cycles = load_discharge_cycles(args.battery)
    cycle, idx = pick_cycle(cycles, args.stage, args.cycle)

    true_soh = cycle["capacity"] / RATED_CAPACITY * 100.0
    print(f"Using discharge cycle #{idx}  (of {len(cycles)} in {args.battery})")
    print(f"Measured capacity: {cycle['capacity']:.3f} Ahr  ->  TRUE SOH = {true_soh:.1f}%")
    print(f"Streaming to {endpoint}\n")

    # quick backend reachability check
    try:
        requests.get(args.url.rstrip("/") + "/health", timeout=3)
    except Exception:
        sys.exit(f"Backend not reachable at {args.url}. Start it first:\n"
                 f"    cd backend && python main.py")

    v = cycle["voltage"]
    i = cycle["current"]
    t = cycle["time"]
    picks = downsample(v, args.samples)

    # build a running cumulative energy (Wh) so energy_delta is meaningful
    cum_energy = 0.0
    prev_t = t[picks[0]]
    sent = 0

    for k in picks:
        voltage = float(v[k])
        # NASA discharge current is negative already; force negative to be safe
        current = -abs(float(i[k]))
        dt = max(0.0, float(t[k]) - prev_t)
        prev_t = float(t[k])
        power = abs(voltage * current)              # PZEM power register >= 0
        cum_energy += power * dt / 3600.0           # Wh, monotonically rising

        payload = {
            "voltage": round(voltage, 3),
            "current": round(current, 3),
            "power": round(power, 2),
            "energy": round(cum_energy, 3),
            "timestamp": int(time.time() * 1000),
        }
        try:
            requests.post(endpoint, json=payload, timeout=3)
        except Exception as e:
            print(f"  POST failed: {e}")
        sent += 1
        print(f"  reading {sent:3d}/{len(picks)}  V={voltage:5.2f}  I={current:6.2f}  "
              f"P={power:6.2f}  E={cum_energy:6.2f}   (discharging)")
        time.sleep(args.delay)

    # ---- close the cycle: idle readings (current = 0) trigger inference ----
    print("\nDischarge finished -> sending idle readings to close the cycle "
          "and trigger prediction ...")
    last_v = float(v[picks[-1]])
    for _ in range(3):
        payload = {
            "voltage": round(last_v, 3),
            "current": 0.0,
            "power": 0.0,
            "energy": round(cum_energy, 3),
            "timestamp": int(time.time() * 1000),
        }
        try:
            requests.post(endpoint, json=payload, timeout=3)
        except Exception as e:
            print(f"  POST failed: {e}")
        time.sleep(args.delay)

    # ---- read back the prediction the backend just produced ----
    time.sleep(0.5)
    try:
        latest = requests.get(args.url.rstrip("/") + "/latest", timeout=3).json()
        pred = latest.get("prediction")
        if pred:
            print("\n================  PREDICTION PRODUCED  ================")
            print(f"  Cycle #        : {pred.get('cycle_number')}")
            print(f"  SOH (RF)       : {pred.get('soh_rf')} %")
            print(f"  SOH (XGBoost)  : {pred.get('soh_xgb')} %")
            print(f"  SOH (Ensemble) : {pred.get('soh_final')} %   <- gauge value")
            print(f"  RUL (Ensemble) : {pred.get('rul_final')} days")
            print(f"  Alert level    : {pred.get('alert_level')}")
            print(f"  (TRUE SOH from dataset for this cycle: {true_soh:.1f} %)")
            print("======================================================")
            print("Now look at your Streamlit dashboard -- the gauge and RUL are live.")
        else:
            print("\nNo prediction found. Is the discharge buffer >= 3 readings? "
                  "Are the models loaded (check /health)?")
    except Exception as e:
        print(f"Could not read /latest: {e}")


if __name__ == "__main__":
    main()
```

## Appendix I — `requirements.txt` & `.gitignore`

**`requirements.txt`**
```
fastapi
uvicorn[standard]
sqlalchemy
pydantic>=2
scikit-learn
xgboost
numpy
pandas
scipy
joblib
streamlit
plotly
requests
```

**`.gitignore`**
```
# Python
__pycache__/
*.py[cod]
venv/
.venv/

# Project artifacts
backend/battery.db
ml/models/*.pkl

# Dataset (large; distribute separately)
ml/data/*.mat

# OS / editor
.DS_Store
.vscode/
.idea/
```

> Note: `ml/models/*.pkl` is git-ignored so trained artifacts aren't committed. If you prefer to ship the trained models with the repo (so the backend runs without retraining), remove that line. The `.mat` files are ignored because of their size — document where to download them in the README.

## Appendix J — `README.md`

````markdown
# IoT-Based Predictive Maintenance of VRLA and Li-ion Batteries using ML

Real-time battery State-of-Health (SOH) and Remaining-Useful-Life (RUL)
monitoring. Terminal readings (voltage, current, power, energy) stream into a
FastAPI backend, which detects each discharge cycle, runs a Random-Forest +
XGBoost ensemble, and serves predictions to a live Streamlit dashboard with
colour-coded alerts. Trained on the NASA PCoE Li-ion battery dataset.

See `PROJECT_SPEC.md` for the full build specification.

## Quick start

```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# 1) place B0005/B0006/B0007/B0018 .mat files in ml/data/
# 2) train the models (creates ml/models/*.pkl)
cd ml && python train_models.py && cd ..

# 3) run the three processes (three terminals)
cd backend && python main.py                 # http://localhost:8000
cd dashboard && streamlit run dashboard.py    # opens the dashboard
python tools/simulate_discharge.py --stage late   # demo a real cycle
```

## Layout
- `ml/` — training pipeline, dataset, saved models
- `backend/` — FastAPI API, SQLite, inference
- `dashboard/` — Streamlit UI
- `tools/` — NASA discharge simulator + demo guide
````

---

*End of specification. Build by following Section 14; the appendices are the source of truth for implementation.*

