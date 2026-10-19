"""
main.py
FastAPI backend for the IoT battery predictive maintenance system.
Receives sensor readings from ESP32, detects discharge cycles,
runs ML inference, and serves data to the Streamlit dashboard.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timezone
from typing import List, Optional
import os
import time
import uvicorn

# Secret shared with the ESP32. When set (always, when deployed online), a
# reading is accepted only if it carries this key in its X-API-Key header, so
# nobody else on the internet can post readings. Left unset for local use.
API_KEY = os.environ.get("API_KEY", "")

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
    description="IoT-Based Predictive Maintenance of Li-ion Batteries",
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
last_rx_wall = None     # laptop clock time of the last reading received
rest_peak_v = None      # highest resting (no-load) V/cell seen since the last discharge ended
discharge_rest_v = None # resting V/cell captured when the current discharge started
load_off_since = None   # laptop clock time the load current first dropped away mid-discharge

# Safety / robustness limits
LOW_CELL_ALARM_V = 2.8  # per-cell voltage under load that means "switch OFF now"
STALE_SECONDS = 180     # no readings this long during a discharge -> close it as INTERRUPTED
BMS_CUTOFF_WAIT_SECONDS = 30   # silence this long after reaching the cut-off region -> BMS cut-off
LOAD_OFF_CONFIRM_SECONDS = 30  # load current must stay off this long to end a discharge, so a
                               # brief contact dropout does not split one discharge into two

DISCHARGE_THRESHOLD = -0.5
# Rated capacity of ONE cell. In a series pack every cell carries the same
# current, so the pack's capacity equals one cell's.
RATED_CAPACITY_MAH = 2000.0
# A discharge only counts as complete (and gets an SOH/RUL prediction) if the
# per-cell voltage fell to this level. The models were trained on full
# discharges, so a partial one would be misread as a worn battery.
FULL_DISCHARGE_MAX_CELL_V = 3.2
# ...and it must also have STARTED from a full pack. "Full" is judged on the
# RESTING (no-load) voltage just before the load is switched on, because the
# voltage under load sags with the pack's wiring/holder resistance. A per-cell
# resting voltage in this range means the cells were charged first; above the
# upper limit the reading is implausible (a wiring or sensor fault).
FULL_START_MIN_CELL_V = 3.99
FULL_START_MAX_CELL_V = 4.5


def charge_mah(buf):
    """Trapezoidal integral of |current| over the device timestamps, in mAh."""
    return sum(
        (abs(buf[k]["current"]) + abs(buf[k - 1]["current"])) / 2 * max(0.0, buf[k]["t"] - buf[k - 1]["t"])
        for k in range(1, len(buf))
    ) / 3.6


# ============================================================
# PYDANTIC SCHEMAS -- updated to Pydantic v2 syntax
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
        # 3 dp: a cycle's energy_delta is only a few Wh, so 0.1 Wh
        # rounding would distort the energy_wh model feature
        return round(v, 3)


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


class CycleOut(BaseModel):
    cycle_number: int
    cycle_start: Optional[datetime]
    cycle_end: Optional[datetime]
    capacity_mah: Optional[float]
    min_cell_voltage: Optional[float]
    complete: Optional[int]
    measured_soh: Optional[float] = None   # capacity_mah / RATED_CAPACITY_MAH x 100

    model_config = ConfigDict(from_attributes=True)


class LatestOut(BaseModel):
    reading: Optional[ReadingOut]
    prediction: Optional[PredictionOut]
    cycle: Optional[CycleOut] = None


# ============================================================
# DISCHARGE CYCLE PROCESSING
# ============================================================
def process_discharge_cycle(db: Session, interrupted: bool = False, start_rest_v=None):
    """Close the buffered discharge. interrupted=True means readings stopped
    before the discharge ended (e.g. the PZEM lost power): record it, never predict.
    start_rest_v is the resting V/cell measured just before the load went on."""
    global discharge_buffer, cycle_counter, cycle_start_time

    if len(discharge_buffer) < 3:
        discharge_buffer = []
        return

    try:
        # Order by device time in case cached readings arrived late
        buf = sorted(discharge_buffer, key=lambda r: r["t"])
        voltages = [r["voltage"] for r in buf]
        # |current|: discharge current arrives negative, but the model was
        # trained on positive magnitude (Equation 3.5)
        currents = [abs(r["current"]) for r in buf]
        powers = [v * i for v, i in zip(voltages, currents)]
        times = [r["t"] for r in buf]

        # Equations 3.4-3.6, computed exactly as in ml/train_models.py
        mean_voltage = sum(voltages) / len(voltages)
        mean_current = sum(currents) / len(currents)
        mean_power = sum(powers) / len(powers)
        # Equation 3.7: trapezoidal integral of V*|I| over time, in Wh -- the
        # same method training uses. The PZEM-017 energy register only
        # resolves 1 Wh, too coarse for a single cell (~4-7 Wh per cycle).
        energy_delta = sum(
            (powers[k] + powers[k - 1]) / 2 * max(0.0, times[k] - times[k - 1])
            for k in range(1, len(buf))
        ) / 3600.0

        # Measured capacity (the "true SOH" reference) and completeness check
        capacity_mah = charge_mah(buf)
        min_cell_voltage = min(voltages)
        # Resting voltage before the load went on; if none was received (e.g. the
        # backend was restarted mid-run) fall back to the first ~30 s under load
        start_cell_voltage = start_rest_v if start_rest_v is not None else max(voltages[:6])
        started_full = FULL_START_MIN_CELL_V <= start_cell_voltage <= FULL_START_MAX_CELL_V
        complete = (min_cell_voltage <= FULL_DISCHARGE_MAX_CELL_V and started_full
                    and not interrupted)

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
            capacity_mah=round(capacity_mah, 1),
            min_cell_voltage=round(min_cell_voltage, 3),
            complete=int(complete),
        )
        db.add(cycle)

        if complete:
            result = inference.predict(mean_voltage, mean_current, mean_power, energy_delta)
        else:
            # Stopped before the cut-off: record it, but do not predict --
            # the models only understand complete discharges.
            if interrupted:
                level = "INTERRUPTED"
            elif not started_full:
                level = "NOT_STARTED_FULL"
            else:
                level = "PARTIAL_DISCHARGE"
            result = {"soh_rf": None, "soh_xgb": None, "soh_final": None,
                      "rul_rf": None, "rul_xgb": None, "rul_final": None,
                      "alert_level": level}

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

        if complete:
            print(f"Cycle {cycle_counter} complete -- SOH: {result['soh_final']}% | "
                  f"RUL: {result['rul_final']} days | Alert: {result['alert_level']} | "
                  f"delivered {capacity_mah:.0f} mAh (measured SOH "
                  f"{capacity_mah / RATED_CAPACITY_MAH * 100:.1f}%)")
        elif interrupted:
            print(f"Cycle {cycle_counter} INTERRUPTED -- readings stopped mid-discharge "
                  f"({capacity_mah:.0f} mAh recorded); no prediction")
        elif not started_full:
            print(f"Cycle {cycle_counter} NOT STARTED FULL -- began at {start_cell_voltage:.2f} V/cell "
                  f"(needs {FULL_START_MIN_CELL_V}-{FULL_START_MAX_CELL_V} V/cell at rest: charge the cells first), "
                  f"{capacity_mah:.0f} mAh delivered; no prediction")
        else:
            print(f"Cycle {cycle_counter} PARTIAL -- stopped at {min_cell_voltage:.2f} V/cell "
                  f"(full discharge needs <= {FULL_DISCHARGE_MAX_CELL_V} V/cell), "
                  f"{capacity_mah:.0f} mAh delivered; no prediction")

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
def receive_reading(data: ReadingInput, db: Session = Depends(get_db),
                    x_api_key: Optional[str] = Header(default=None)):
    global is_discharging, discharge_buffer, cycle_start_time, last_rx_wall, rest_peak_v, discharge_rest_v
    global load_off_since

    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or wrong API key")

    # A discharge whose readings stopped for too long is closed as INTERRUPTED,
    # so new readings never get glued onto a broken run.
    close_if_stale(db)
    last_rx_wall = time.time()

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
        # seconds; prefer the device clock so readings cached in SPIFFS
        # during a WiFi outage keep their true spacing
        "t": data.timestamp / 1000.0 if data.timestamp is not None else time.time(),
    }

    if data.current < DISCHARGE_THRESHOLD and data.voltage < LOW_CELL_ALARM_V:
        print(f"!!! SAFETY: {data.voltage:.2f} V/cell under load is below {LOW_CELL_ALARM_V} V "
              f"-- SWITCH THE LOAD OFF NOW (the BMS should have cut off)")

    if data.current < DISCHARGE_THRESHOLD:
        if not is_discharging:
            is_discharging = True
            cycle_start_time = datetime.now(timezone.utc)
            discharge_buffer = []
            discharge_rest_v = rest_peak_v
            rest_peak_v = None
            rest_txt = f"{discharge_rest_v:.2f} V/cell" if discharge_rest_v is not None else "not measured"
            print(f"Discharge started at {cycle_start_time} (resting voltage before load: {rest_txt})")
        elif load_off_since is not None:
            print(f"Load current back after {time.time() - load_off_since:.0f} s -- same discharge continues")
        load_off_since = None
        discharge_buffer.append(reading_dict)
    else:
        if is_discharging:
            # Only end the discharge once the load has stayed off for a while;
            # a single 0 A reading is usually a loose contact, not the end.
            if load_off_since is None:
                load_off_since = time.time()
                print(f"Load current dropped to {data.current:.2f} A -- waiting "
                      f"{LOAD_OFF_CONFIRM_SECONDS} s before ending the discharge")
            elif time.time() - load_off_since >= LOAD_OFF_CONFIRM_SECONDS:
                end_discharge(db)
        else:
            # idle (no load): remember the highest resting voltage for the next start check
            rest_peak_v = data.voltage if rest_peak_v is None else max(rest_peak_v, data.voltage)

    return {"status": "ok", "id": reading.id}


@app.get("/latest", response_model=LatestOut)
def get_latest(db: Session = Depends(get_db)):
    latest_reading = db.query(Reading).order_by(Reading.id.desc()).first()
    latest_prediction = db.query(Prediction).order_by(Prediction.id.desc()).first()
    latest_cycle = db.query(Cycle).order_by(Cycle.id.desc()).first()
    cycle_out = None
    if latest_cycle is not None:
        cycle_out = CycleOut.model_validate(latest_cycle)
        if latest_cycle.capacity_mah is not None:
            cycle_out.measured_soh = round(latest_cycle.capacity_mah / RATED_CAPACITY_MAH * 100, 1)
    return LatestOut(reading=latest_reading, prediction=latest_prediction, cycle=cycle_out)


@app.get("/history", response_model=List[ReadingOut])
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    readings = db.query(Reading).order_by(Reading.id.desc()).limit(limit).all()
    return list(reversed(readings))


@app.get("/predictions", response_model=List[PredictionOut])
def get_predictions(db: Session = Depends(get_db)):
    return db.query(Prediction).order_by(Prediction.id.asc()).all()


def end_discharge(db: Session):
    """The load has been off for LOAD_OFF_CONFIRM_SECONDS: close the discharge."""
    global is_discharging, load_off_since
    is_discharging = False
    load_off_since = None
    print(f"Discharge ended (load off for {LOAD_OFF_CONFIRM_SECONDS} s). Buffer size: {len(discharge_buffer)}")
    process_discharge_cycle(db, start_rest_v=discharge_rest_v)


def close_if_stale(db: Session):
    """Close a discharge whose readings have stopped.

    On the rig the PZEM-017's V- sits on BMS P-, so when the BMS cuts off at the
    end of a discharge the PZEM loses power too and the final 0 A reading never
    arrives. If the readings stop AFTER the pack has already reached the
    end-of-discharge region (<= FULL_DISCHARGE_MAX_CELL_V per cell), that is a
    BMS cut-off: close the cycle as complete and predict. If they stop earlier,
    the run really was interrupted: record it without a prediction.
    """
    global is_discharging, load_off_since
    if not (is_discharging and last_rx_wall is not None and discharge_buffer):
        return
    if load_off_since is not None and time.time() - load_off_since >= LOAD_OFF_CONFIRM_SECONDS:
        end_discharge(db)   # load switched off and it stayed off
        return
    silent = time.time() - last_rx_wall
    reached_end = min(r["voltage"] for r in discharge_buffer) <= FULL_DISCHARGE_MAX_CELL_V
    if reached_end and silent > BMS_CUTOFF_WAIT_SECONDS:
        is_discharging = False
        load_off_since = None
        print(f"No readings for {silent:.0f} s after the pack reached the cut-off region "
              f"-- treating it as a BMS cut-off: discharge complete")
        process_discharge_cycle(db, start_rest_v=discharge_rest_v)
    elif silent > STALE_SECONDS:
        is_discharging = False
        load_off_since = None
        print(f"No readings for {silent:.0f} s during a discharge -- closing it as INTERRUPTED")
        process_discharge_cycle(db, interrupted=True, start_rest_v=discharge_rest_v)


@app.get("/health")
def health_check():
    db = SessionLocal()
    try:
        close_if_stale(db)
    finally:
        db.close()
    return {
        # seconds since the last reading arrived (None before the first one)
        "seconds_since_reading": round(time.time() - last_rx_wall, 1) if last_rx_wall else None,
        "low_cell_alarm_v": LOW_CELL_ALARM_V,
        "status": "running",
        "models_loaded": inference.loaded,
        "is_discharging": is_discharging,
        "cycle_count": cycle_counter,
        "buffer_size": len(discharge_buffer),
        # live progress of the discharge in progress (0 when idle)
        "discharge_mah": round(charge_mah(sorted(discharge_buffer, key=lambda r: r["t"])), 1)
        if len(discharge_buffer) > 1 else 0.0,
        "discharge_minutes": round((max(r["t"] for r in discharge_buffer)
                                    - min(r["t"] for r in discharge_buffer)) / 60, 1)
        if len(discharge_buffer) > 1 else 0.0,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)