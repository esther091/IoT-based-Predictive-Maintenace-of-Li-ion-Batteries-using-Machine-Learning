# Setup and Run Guide — Single 18650 Li-ion Cell Test Rig

Follow the parts in order. Part A is done once. Parts C and D are what you do
every time you run a test (and on defence day).

---

## Part A — One-time setup on the laptop

### A1. Python environment (skip if `venv` already works)
Open PowerShell in the project folder and run:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
The project's `venv` was created when the folder was on the Desktop. Some of
its shortcuts (`activate.bat`, `streamlit.exe`, `pip.exe`) still point there, so
always use `python -m streamlit` and `python -m pip` rather than the bare
`streamlit` / `pip` commands. `start_system.bat` already does this.

### A2. Let the ESP32 reach the laptop (Windows Firewall)
Open **PowerShell as Administrator** and run once:
```powershell
netsh advfirewall firewall add rule name="Battery backend 8000" dir=in action=allow protocol=TCP localport=8000
```
Without this the ESP32 connects to WiFi but every reading fails to send.

### A3. Arduino IDE
1. Install the **ESP32 board package** (Tools > Board > Boards Manager > "esp32 by Espressif").
2. Install libraries (Tools > Manage Libraries): **ModbusMaster** (Doc Walker) and **ArduinoJson** (Benoit Blanchon).
3. Open `firmware/esp32_firmware_v2/esp32_firmware_v2.ino`.

### A4. Edit four lines in the firmware
```cpp
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://192.168.1.100:8000/readings";   // your laptop's IP
#define SHUNT_RANGE_CODE 0x0002   // 0x0001 = 50 A, 0x0000 = 100 A, 0x0002 = 200 A, 0x0003 = 300 A
```
- Find the laptop IP: run `ipconfig` and copy the **IPv4 Address** (or run
  `start_system.bat`, which prints it).
- The ESP32 only works on **2.4 GHz WiFi**. A phone hotspot is fine (turn on
  "maximise compatibility" / 2.4 GHz). Laptop and ESP32 must be on the **same** network.
- If you change networks (for example on defence day), the IP changes: update
  `SERVER_URL` and upload again.

### A5. Upload
Tools > Board > **ESP32 Dev Module**, Tools > Port > your ESP32's COM port, then **Upload**.

---

## Part B — Wiring (everything unplugged while you wire)

### Parts
One 18650 cell (about 2 Ah) in a holder, TP4056 charger module, constant-current
electronic load, PZEM-017 with its shunt, 5 V USB adapter, MAX485 module,
ESP32, 1 kΩ and 2 kΩ resistors, jumper wires, multimeter.

### B1. Current path (thick wires)
```
Cell (+) ─────────────────────────────► Electronic load (+)
Cell (−) ──► SHUNT big bolt 1
             SHUNT big bolt 2 ─────────► Electronic load (−)
```

### B2. PZEM-017 measuring side
- **Voltage wires** of the PZEM-017 → across the cell (+ and −).
- **Shunt sense wires** (thin) → the two **small screws** on the shunt.
- Follow the terminal labels printed on your PZEM-017.
- Plug the **5 V USB adapter** into the PZEM-017 **micro-USB** port
  (a single cell is below the 7 V it needs to power itself).

### B3. PZEM-017 → MAX485
| PZEM-017 (RS485 side) | Connects to |
|---|---|
| A | MAX485 A |
| B | MAX485 B |
| 5V | ESP32 5V (VIN) — **required**, the RS485 side needs its own 5 V |
| GND | ESP32 GND |

### B4. MAX485 → ESP32
| MAX485 | ESP32 | Note |
|---|---|---|
| VCC | 5V (VIN) | |
| GND | GND | common ground |
| RO | GPIO16 | through a divider: RO → 1 kΩ → GPIO16, and GPIO16 → 2 kΩ → GND |
| DI | GPIO17 | |
| DE and RE (joined) | GPIO4 | |

### B5. Discharge signal
A jumper wire from **GPIO27 to GND** tells the ESP32 "the battery is discharging".
Put it **on** during a discharge and **off** while charging.
**Never connect GPIO27 to the battery.**

### B6. Check before power
- Multimeter across the cell: between 2.7 V and 4.2 V.
- No bare wires touching. All grounds joined (ESP32, MAX485, PZEM GND).

---

## Part C — Start the software

1. Double-click **`start_system.bat`** in the project folder.
   Two windows open:
   - **Backend** — wait for `All ML models loaded successfully.`
   - **Dashboard** — your browser opens the dashboard.
2. Plug the ESP32 into the laptop by USB and open the Arduino **Serial Monitor**
   at **115200** baud. You should see:
   ```
   [WiFi] Connected, IP 192.168.x.x
   [PZEM] current range already set (code 2)
   [PZEM] V=4.15 V  I=0.00 A  P=0.0 W  E=0 Wh
   ```
   and the dashboard's voltage card starts updating every 5 seconds.

---

## Part D — Run one test cycle

1. **Charge**: disconnect the load, jumper **off**, connect the TP4056 to the
   cell, charge until its LED shows full (4.2 V). Disconnect the charger.
2. **Set the load**: 2.0 A constant current, cut-off 2.7 V.
3. Jumper **on** (GPIO27 → GND), then **start the load**.
   The backend window prints `Discharge started`.
4. Wait. The load stops by itself at 2.7 V (about an hour for a good cell,
   less for a worn one).
5. When the load stops, the current drops to zero and the backend closes the
   cycle: it prints `Cycle 1 complete -- SOH ... RUL ... Alert ...` and the
   dashboard updates the gauge, RUL and alert. Then take the jumper **off**.
6. **Record** for Table 4.9 of the report: the mAh shown by the electronic load
   (true SOH = mAh ÷ rated mAh × 100), the predicted SOH/RUL, and multimeter
   checks of voltage and current. Take a photo of the rig for Figure 4.1.
7. Repeat with the **worn cell** from an old laptop battery.

---

## Part E — Demo without hardware (backup plan)

With `start_system.bat` running, open a third terminal in the project folder:
```powershell
.\venv\Scripts\Activate.ps1
python tools\simulate_discharge.py --stage early   # healthy cell -> NORMAL
python tools\simulate_discharge.py --stage late    # worn cell    -> URGENT
```
To clear old results, close both windows, delete `backend\battery.db`, and start again.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[PZEM] read error` / `cannot read current range` | Swap A and B; check the PZEM-017 has 5 V on micro-USB; check GND is shared |
| WiFi never connects | 2.4 GHz network only; check name and password |
| `[HTTP] Failed` in Serial Monitor | Wrong `SERVER_URL` IP, backend not running, or firewall rule (A2) missing |
| Current reads about half the true value | `SHUNT_RANGE_CODE` does not match your shunt |
| Readings arrive but no prediction | Jumper was not on during the discharge (current must arrive as negative), or the load has not stopped yet |
| Dashboard says models not loaded | `ml/models/` is missing the `.pkl` files: run `python ml/train_models.py` |
| Everything worked yesterday but not today | Laptop IP changed: re-check `ipconfig`, update `SERVER_URL`, re-upload |
