/*
  pzem_diagnostic.ino
  -------------------
  Test sketch for the ESP32 <-> MAX485 <-> PZEM-017 link. Upload it instead of
  the main firmware, open the Serial Monitor at 115200 baud, and read the
  verdict it prints. Re-upload esp32_firmware_v2 when the link is fixed.

  Uses the same pins as the main firmware:
    GPIO16 = RX2 (from MAX485 RO, via divider)
    GPIO17 = TX2 (to MAX485 DI)
    GPIO4  = MAX485 DE + RE

  Every 10 s it runs 20 reads at the normal address (0x01) and 20 at the
  PZEM "general" address (0xF8, which any single PZEM answers), then prints
  the success rate and the error mix.
*/

#include <Arduino.h>
#include <ModbusMaster.h>

#define RS485_EN_PIN 4
#define RXD2         16
#define TXD2         17
#define TRIES        20

ModbusMaster pzem;

void preTransmission()  { digitalWrite(RS485_EN_PIN, HIGH); }
void postTransmission() { digitalWrite(RS485_EN_PIN, LOW);  }

const char* errName(uint8_t e) {
  switch (e) {
    case 0x00: return "OK";
    case 0xE0: return "0xE0 garbled reply (wrong address byte)";
    case 0xE1: return "0xE1 garbled reply (wrong function)";
    case 0xE2: return "0xE2 no reply (timeout)";
    case 0xE3: return "0xE3 garbled reply (bad checksum)";
    default:   return "other Modbus error";
  }
}

void testAddress(uint8_t addr) {
  pzem.begin(addr, Serial2);
  pzem.preTransmission(preTransmission);
  pzem.postTransmission(postTransmission);

  int ok = 0, timeouts = 0, garbled = 0;
  float lastV = -1, lastI = -1;
  for (int i = 0; i < TRIES; i++) {
    uint8_t r = pzem.readInputRegisters(0x0000, 2);
    if (r == pzem.ku8MBSuccess) {
      ok++;
      lastV = pzem.getResponseBuffer(0) / 100.0f;
      lastI = pzem.getResponseBuffer(1) / 100.0f;
    } else if (r == 0xE2) {
      timeouts++;
    } else {
      garbled++;
    }
    delay(150);
  }
  Serial.printf("  address 0x%02X: %2d/%d OK, %2d no reply, %2d garbled",
                addr, ok, TRIES, timeouts, garbled);
  if (ok) Serial.printf("   (last: V=%.2f V, I=%.2f A)", lastV, lastI);
  Serial.println();

  if (addr == 0xF8 && ok) {
    // read the device's real address (holding register 0x0002)
    if (pzem.readHoldingRegisters(0x0002, 1) == pzem.ku8MBSuccess)
      Serial.printf("  -> this PZEM's configured address is 0x%02X\n", pzem.getResponseBuffer(0));
  }

  // verdict
  if (ok == TRIES) {
    Serial.println("  VERDICT: link is solid at this address.");
  } else if (ok == 0 && garbled == 0) {
    Serial.println("  VERDICT: silence. The meter is not answering at all. Check: PZEM RS485 5V + GND "
                   "to ESP32, DE AND RE both on GPIO4, TX2->DI, A->A and B->B (try swapping A/B once), "
                   "micro-USB 5 V on the PZEM.");
  } else if (garbled > 0) {
    Serial.println("  VERDICT: replies arrive but are corrupted. Check: common GND between ESP32, "
                   "MAX485 and PZEM; the RO divider orientation (1k from RO, 2k to GND); loose A/B wires.");
  } else {
    Serial.println("  VERDICT: unreliable (some replies). Check: common GND, RO divider orientation, "
                   "RE pin connected, loose jumpers, PZEM micro-USB power.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(RS485_EN_PIN, OUTPUT);
  digitalWrite(RS485_EN_PIN, LOW);
  Serial2.begin(9600, SERIAL_8N2, RXD2, TXD2);
  Serial.println("\n=== PZEM-017 link diagnostic ===");
}

void loop() {
  Serial.println("\n--- test run ---");
  // Is the RO line idling HIGH? (MAX485 RO idles at logic 1 when the bus is quiet)
  int idle = digitalRead(RXD2);
  Serial.printf("  RX2 (GPIO16) idle level: %s\n",
                idle ? "HIGH (good)" : "LOW - RO not connected, divider wrong way round, or MAX485 unpowered");
  testAddress(0x01);
  testAddress(0xF8);
  delay(10000);
}
