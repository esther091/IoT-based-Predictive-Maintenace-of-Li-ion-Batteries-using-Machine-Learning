"""
simulate_discharge.py
---------------------------------------------------------------
Pretends to be the ESP32 + PZEM-017 hardware.

It opens a REAL discharge cycle from the NASA Battery Dataset and
replays it, reading-by-reading, into your running FastAPI backend
(POST /readings) — exactly the way the ESP32 will once the hardware
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
    makes the backend close the buffer and run inference — the
    moment your prediction is produced.

USAGE (run from the project root, with your venv active)://
  python tools/simulate_discharge.py --stage late
  python tools/simulate_discharge.py --stage early --delay 5
  python tools/simulate_discharge.py --battery B0018 --cycle 100

  --stage early|late : pick a healthy (early) or degraded (late) cycle
  --battery          : B0005 | B0006 | B0007 | B0018  (default B0018)
  --cycle N          : use a specific discharge cycle index instead of --stage
  --delay S          : seconds between readings (default 0.15 = fast demo;
                       use 5 to replay at the true real-time cadence)
  --samples N        : how many readings to send across the cycle (default 60)
  --url              : backend URL (default http://127.0.0.1:8000)
---------------------------------------------------------------
"""

import argparse
import os
import sys
import time

import numpy as np
import requests
from scipy.io import loadmat

RATED_CAPACITY = 2.0  # Ahr, per Table 3.4 — used only to print the TRUE SOH


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
    # 127.0.0.1 avoids Windows' ~2 s IPv6 fallback delay on "localhost"
    ap.add_argument("--url", default="http://127.0.0.1:8000")
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

    # build a running cumulative energy (Wh), like the PZEM-017 register
    cum_energy = 0.0
    prev_t = t[picks[0]]
    sent = 0
    # Timestamps follow the dataset's own time axis (not wall-clock), so the
    # backend's V*I*dt energy integral sees the cycle's real duration even
    # when the replay is sped up with a short --delay.
    base_ms = int(time.time() * 1000) - int(float(t[picks[0]]) * 1000)

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
            "timestamp": base_ms + int(float(t[k]) * 1000),
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
    last_ms = base_ms + int(float(t[picks[-1]]) * 1000)
    for n in range(3):
        payload = {
            "voltage": round(last_v, 3),
            "current": 0.0,
            "power": 0.0,
            "energy": round(cum_energy, 3),
            "timestamp": last_ms + 5000 * (n + 1),
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
            print("Now look at your Streamlit dashboard — the gauge and RUL are live.")
        else:
            print("\nNo prediction found. Is the discharge buffer >= 3 readings? "
                  "Are the models loaded (check /health)?")
    except Exception as e:
        print(f"Could not read /latest: {e}")


if __name__ == "__main__":
    main()
