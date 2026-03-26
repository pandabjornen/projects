#include <Arduino.h>
#include <ArduinoBLE.h>

// Samma UUID som i din riktiga kod
BLEService timerService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLEByteCharacteristic commandChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite);

void setup() {
    Serial.begin(9600);
    while (!Serial);  

    Serial.println("Startar BLE...");

    if (!BLE.begin()) {
        Serial.println("FAIL: BLE kunde inte starta");
        while (1);
    }

    BLE.setLocalName("SprintTimer");
    BLE.setAdvertisedService(timerService);
    timerService.addCharacteristic(commandChar);
    BLE.addService(timerService);
    commandChar.writeValue(0);
    BLE.advertise();

    Serial.println("OK: BLE igång, advertiser sänder");
    Serial.println("Öppna nRF Connect och leta efter 'SprintTimer'");
}

void loop() {
    BLE.poll();

    // Skriv ut när någon ansluter
    BLEDevice central = BLE.central();
    if (central) {
        Serial.print("Ansluten: ");
        Serial.println(central.address());

        while (central.connected()) {
            BLE.poll();

            // Om commandChar får ett nytt värde
            if (commandChar.written()) {
                uint8_t val = commandChar.value();
                Serial.print("Mottog värde: ");
                Serial.println(val);

                if (val == 1) {
                    Serial.println(">> Kommando '1' mottaget – skulle starta timer");
                    commandChar.writeValue(0); // Nollställ
                }
            }
        }

        Serial.println("Frånkopplad");
    }
}