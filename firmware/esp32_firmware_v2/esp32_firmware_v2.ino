/*
  esp32_firmware_v2.ino
  IoT-Based Predictive Maintenance of Li-ion Batteries Using Machine Learning
  ================================================================================
  Reads the PZEM-017 DC power meter over RS485 / Modbus RTU and sends one reading
  every 5 seconds to the FastAPI backend (POST /readings). Readings are cached in
  SPIFFS while WiFi is down and flushed when the connection returns.

  CHANGES FROM esp32_firmware.ino (v1)
  ------------------------------------
  1. Register map corrected to the PZEM-017 datasheet. v1 used the PZEM-004T
     (AC) layout, which treats current as a 32-bit value across 0x0001-0x0002.
     PZEM-017 layout:
        0x0000        voltage  (16-bit, 0.01 V/LSB)
        0x0001        current  (16-bit, 0.01 A/LSB)
        0x0002-0x0003 power    (32-bit low/high, 0.1 W/LSB)
        0x0004-0x0005 energy   (32-bit low/high, 1 Wh/LSB)
  2. Energy is sent in Wh (v1 divided by 1000 and sent kWh). The backend and the
     trained models use Wh.
  3. Current sign. The PZEM-017 reports current MAGNITUDE only. The backend
     detects a discharge when current < -0.5 A, so the ESP32 must supply the
     sign. A load-state input (LOAD_SENSE_PIN) is read: when the discharge load
     switch/relay is closed the pin is pulled LOW and the current is sent as a
     negative value. Wire one spare contact of the load switch between
     GPIO27 and GND.
  4. JSON key "timestamp" (the backend schema field) instead of "time_ms".
     The backend uses it to integrate V*I over time, so cached readings keep
     their true spacing.
  5. MAX_CACHE_LINES is now enforced so SPIFFS cannot fill up.
  6. On start-up the PZEM-017 current range is set to match the shunt
     (holding register 0x0003). The module ships set to 100 A; with a 200 A
     shunt left on the 100 A setting, every current reading is wrong by 2x.
     Set SHUNT_RANGE_CODE below to match YOUR shunt.
  7. No load switch? For a simple test, put a jumper wire from GPIO27 to GND
     while the battery is discharging and remove it while charging.
  8. A packet the backend rejects (HTTP 4xx, e.g. 422 validation error) is
     dropped instead of cached. Previously it was cached and retried forever,
     which blocked every cached reading queued behind it.
  9. Readings stay in time order after a WiFi outage: cached readings are
     delivered before any new reading, and a flush interrupted by another drop
     keeps only the undelivered readings (nothing is lost or sent twice).

  SINGLE 18650 Li-ion CELL TEST RIG
  ---------------------------------
  - The PZEM-017 measuring side needs >= 7 V from the battery to power itself;
    a single cell (2.5-4.2 V) is below that, so power the PZEM-017 through its
    micro-USB port (5 V).
  - A 2 A discharge is only 1% of a 200 A shunt. If you have a 50 A shunt, use
    it (SHUNT_RANGE_CODE 0x0001) for better current accuracy.
  - Stop the discharge at 2.7 V to protect the cell.

  HARDWARE CONNECTIONS
  --------------------
  PZEM-017 A -> MAX485 A          MAX485 RO    -> ESP32 GPIO16 (RX2) via divider
  PZEM-017 B -> MAX485 B          MAX485 DI    -> ESP32 GPIO17 (TX2)
  PZEM-017 5V/GND (RS485 side) -> ESP32 5V/GND   MAX485 DE+RE -> ESP32 GPIO4
  MAX485 VCC -> 5V, GND -> GND    Load switch aux contact -> GPIO27 and GND

  LIBRARIES: ModbusMaster (Doc Walker), ArduinoJson (Benoit Blanchon), WiFi,
             HTTPClient, SPIFFS (ESP32 core)
*/

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ModbusMaster.h>
#include <SPIFFS.h>

// ---------------- configuration ----------------
// WIFI_SSID, WIFI_PASSWORD and SERVER_URL live in secrets.h (copy secrets.example.h)
#include "secrets.h"

#define RS485_EN_PIN     4
#define RXD2             16
#define TXD2             17
#define LOAD_SENSE_PIN   27      // LOW = discharge load connected

#define READ_INTERVAL_MS  5000
#define FLUSH_INTERVAL_MS 30000
#define WIFI_TIMEOUT_MS   10000
#define CACHE_FILENAME    "/cache.jsonl"
#define MAX_CACHE_LINES   500

// PZEM-017 current range: 0x0000 = 100 A, 0x0001 = 50 A, 0x0002 = 200 A, 0x0003 = 300 A
#define SHUNT_RANGE_CODE 0x0002  // <-- 200 A shunt; change if your shunt is different

// Number of 18650 cells in series in the pack under test. The PZEM-017 needs
// >= 7 V to power itself, so a 3S pack (8.1-12.6 V) is used. Voltage, power
// and energy are divided by this before sending, so the backend sees one
// average cell -- the same scale as the NASA single-cell training data.
// (Current and capacity are the same for every cell in a series pack.)
#define CELLS_IN_SERIES  3

// Test rig: the pack is charged OFF-board (each cell on its own TP4056), so
// any current through the shunt can only be a discharge. true = always send
// current as negative (discharging) and ignore GPIO27 -- no jumper needed.
// Set false where charging and discharging share the same terminals; GPIO27
// (load-switch auxiliary contact to GND) then tells the ESP32 the direction.
#define DISCHARGE_ONLY_RIG  true

#define PZEM_ADDR        0x01
#define REG_START        0x0000
#define REG_COUNT        6       // 0x0000 .. 0x0005

// sendToServer() results
#define SEND_OK       0          // delivered (HTTP 200)
#define SEND_RETRY    1          // no WiFi / network error / server error -> cache and retry
#define SEND_REJECTED 2          // HTTP 4xx: backend refused the packet -> retrying won't help

ModbusMaster pzem;

struct BatteryReading {
  float voltage;   // V
  float current;   // A, negative while discharging
  float power;     // W
  float energy;    // Wh
  bool  valid;
};

// Function prototypes (declared explicitly so the Arduino IDE does not need to generate them)
void preTransmission();
void postTransmission();
void connectWiFi();
void configureShuntRange();
BatteryReading readPZEM017();
String buildJSON(const BatteryReading& r);
int sendToServer(const String& payload);
int countCacheLines();
void cacheReading(const String& payload);
void flushCache();
void setup();
void loop();

unsigned long lastReadTime  = 0;
unsigned long lastFlushTime = 0;

void preTransmission()  { digitalWrite(RS485_EN_PIN, HIGH); }
void postTransmission() {
  digitalWrite(RS485_EN_PIN, LOW);
  delayMicroseconds(100);
  while (Serial2.available()) Serial2.read();   // drop the glitch byte made when the bus switches to receive
}

void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected, IP %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] Not connected - readings will be cached");
  }
}

// Make sure the PZEM-017 is set to the same current range as the shunt fitted
void configureShuntRange() {
  uint8_t r = pzem.readHoldingRegisters(0x0003, 1);
  if (r != pzem.ku8MBSuccess) {
    Serial.printf("[PZEM] cannot read current range (error 0x%02X) - check A/B wiring and 5 V supply\n", r);
    return;
  }
  uint16_t current = pzem.getResponseBuffer(0);
  if (current == SHUNT_RANGE_CODE) {
    Serial.printf("[PZEM] current range already set (code %u)\n", current);
    return;
  }
  r = pzem.writeSingleRegister(0x0003, SHUNT_RANGE_CODE);
  Serial.printf("[PZEM] current range %u -> %u : %s\n", current, SHUNT_RANGE_CODE,
                r == pzem.ku8MBSuccess ? "OK" : "FAILED");
}

void setup() {
  Serial.begin(115200);
  pinMode(RS485_EN_PIN, OUTPUT);
  digitalWrite(RS485_EN_PIN, LOW);
  pinMode(LOAD_SENSE_PIN, INPUT_PULLUP);

  Serial2.begin(9600, SERIAL_8N2, RXD2, TXD2);   // PZEM-017: 9600 baud, 8N2
  pzem.begin(PZEM_ADDR, Serial2);
  pzem.preTransmission(preTransmission);
  pzem.postTransmission(postTransmission);

  configureShuntRange();

  if (!SPIFFS.begin(true)) Serial.println("[SPIFFS] mount failed - caching disabled");
  connectWiFi();
}

BatteryReading readPZEM017() {
  BatteryReading r = {0, 0, 0, 0, false};
  uint8_t result = 0;
  for (int attempt = 0; attempt < 3; attempt++) {   // retry: one bad reply should not lose the reading
    result = pzem.readInputRegisters(REG_START, REG_COUNT);
    if (result == pzem.ku8MBSuccess) break;
    delay(50);
  }
  if (result != pzem.ku8MBSuccess) {
    Serial.printf("[PZEM] read error 0x%02X\n", result);
    return r;
  }
  r.voltage = pzem.getResponseBuffer(0) / 100.0f;
  float magnitude = pzem.getResponseBuffer(1) / 100.0f;
  uint32_t pRaw = ((uint32_t)pzem.getResponseBuffer(3) << 16) | pzem.getResponseBuffer(2);
  uint32_t eRaw = ((uint32_t)pzem.getResponseBuffer(5) << 16) | pzem.getResponseBuffer(4);
  r.power  = pRaw / 10.0f;
  r.energy = (float)eRaw;                                   // already in Wh

  bool discharging = DISCHARGE_ONLY_RIG || (digitalRead(LOAD_SENSE_PIN) == LOW);
  r.current = discharging ? -magnitude : magnitude;

  r.valid = (r.voltage > 0.05f && r.voltage < 300.0f && magnitude < 300.0f);

  // per-cell normalisation (see CELLS_IN_SERIES)
  r.voltage /= CELLS_IN_SERIES;
  r.power   /= CELLS_IN_SERIES;
  r.energy  /= CELLS_IN_SERIES;
  return r;
}

String buildJSON(const BatteryReading& r) {
  StaticJsonDocument<192> doc;
  doc["voltage"]   = r.voltage;
  doc["current"]   = r.current;
  doc["power"]     = r.power;
  doc["energy"]    = r.energy;
  doc["timestamp"] = millis();
  String out;
  serializeJson(doc, out);
  return out;
}

int sendToServer(const String& payload) {
  if (WiFi.status() != WL_CONNECTED) return SEND_RETRY;
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);
  int code = http.POST(payload);
  http.end();
  if (code == 200) return SEND_OK;
  if (code >= 400 && code < 500) {
    Serial.printf("[HTTP] backend rejected packet (%d) - dropped: %s\n", code, payload.c_str());
    return SEND_REJECTED;
  }
  return SEND_RETRY;   // negative code = network error, 5xx = server error
}

int countCacheLines() {
  File f = SPIFFS.open(CACHE_FILENAME, FILE_READ);
  if (!f) return 0;
  int n = 0;
  while (f.available()) { f.readStringUntil('\n'); n++; }
  f.close();
  return n;
}

void cacheReading(const String& payload) {
  if (countCacheLines() >= MAX_CACHE_LINES) {
    Serial.println("[Cache] full - reading dropped");
    return;
  }
  File f = SPIFFS.open(CACHE_FILENAME, FILE_APPEND);
  if (!f) return;
  f.println(payload);
  f.close();
}

void flushCache() {
  if (WiFi.status() != WL_CONNECTED || !SPIFFS.exists(CACHE_FILENAME)) return;
  File f = SPIFFS.open(CACHE_FILENAME, FILE_READ);
  if (!f) return;
  File rest;              // readings still undelivered if the network drops mid-flush
  bool failed = false;
  int sent = 0;
  while (f.available()) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;
    if (!failed) {
      // a rejected (4xx) line is dropped, not retried, so it cannot block the queue
      if (sendToServer(line) != SEND_RETRY) { sent++; delay(100); continue; }
      failed = true;
    }
    // keep this line and everything after it, so nothing is lost or sent twice
    if (!rest) rest = SPIFFS.open("/cache.tmp", FILE_WRITE);
    if (rest) rest.println(line);
  }
  f.close();
  if (rest) rest.close();
  SPIFFS.remove(CACHE_FILENAME);
  if (failed) SPIFFS.rename("/cache.tmp", CACHE_FILENAME);
  if (sent > 0) Serial.printf("[Cache] delivered %d cached readings\n", sent);
}

void loop() {
  unsigned long now = millis();
  if (WiFi.status() != WL_CONNECTED) connectWiFi();

  if (now - lastFlushTime >= FLUSH_INTERVAL_MS) {
    flushCache();
    lastFlushTime = now;
  }

  if (now - lastReadTime >= READ_INTERVAL_MS) {
    lastReadTime = now;
    BatteryReading r = readPZEM017();
    if (!r.valid) return;
    Serial.printf("[PZEM] pack=%.2f V  (per cell %.2f V)  I=%.2f A  P=%.1f W/cell  E=%.1f Wh/cell\n",
                  r.voltage * CELLS_IN_SERIES, r.voltage, r.current, r.power, r.energy);
    String payload = buildJSON(r);
    // Keep readings in time order: deliver any cached (older) readings first,
    // and if some are still waiting, queue this one behind them. Sending a new
    // reading ahead of the cache could close a discharge cycle early.
    if (SPIFFS.exists(CACHE_FILENAME)) flushCache();
    if (SPIFFS.exists(CACHE_FILENAME)) cacheReading(payload);
    else if (sendToServer(payload) == SEND_RETRY) cacheReading(payload);
  }
  delay(100);
}
