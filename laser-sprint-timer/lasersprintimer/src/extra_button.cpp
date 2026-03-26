#include <Arduino.h>
#include "TFLuna.h"
#include "DisplayManager.h"

// --- Hårdvaruinställningar ---
const int kButtonPin = 2;           // Knapp kopplad till Pin 2 och GND
const uint16_t kTriggerDistHigh = 200;  // Trigga vid mätningar under denna dist
const uint16_t kTriggerDistLow = 20;   // Trigga vid mätningar över denna dist
const unsigned long kCountdownMs = 3000; // 3 sekunder nedräkning

// --- Objekt ---
TFLunaParser tfParser;
TFLunaData latestData;
DisplayManager oled;

// --- Systemets tillstånd (State Machine) ---
enum class SystemState { IDLE, COUNTDOWN, MEASURING, DONE };
SystemState currentState = SystemState::IDLE;

// --- Tidtagning och logik ---
unsigned long stateStart = 0;
unsigned long raceStart = 0;
unsigned long lastBtn = 0;
const unsigned long kDebounceDelay = 50;

void setup() {
    // Starta kommunikation med TF-Luna
    Serial1.begin(115200);
    
    // Konfigurera knappen med inbyggt pullup-motstånd
    pinMode(kButtonPin, INPUT_PULLUP);
    
    // Starta OLED-skärmen
    if (!oled.begin()) {
        // Om skärmen inte hittas stannar vi här (felsökning)
        while(1); 
    }
}

void loop() {
    // 1. Läs sensordata kontinuerligt (viktigt för 250Hz prestanda)
    while (Serial1.available()) {
        if (tfParser.processByte(Serial1.read())) {
            latestData = tfParser.getData();
        }
    }

    // 2. Hantera knapptryck (Debouncing)
    bool btnPressed = false;
    if (digitalRead(kButtonPin) == LOW && (millis() - lastBtn) > kDebounceDelay) {
        btnPressed = true;
        lastBtn = millis();
    }

    // 3. Logik baserat på var i loppet vi befinner oss
    switch (currentState) {
        
        case SystemState::IDLE:
            // Visa live-monitor så man kan se att lasern siktar rätt
            oled.showIdle(latestData);
            if (btnPressed) {
                stateStart = millis();
                currentState = SystemState::COUNTDOWN;
            }
            break;

        case SystemState::COUNTDOWN: {
            // Beräkna hur många sekunder som är kvar (3, 2, 1)
            unsigned long elapsed = millis() - stateStart;
            int remaining = 3 - (elapsed / 1000);
            
            oled.showCountdown(remaining);
            
            if (elapsed >= kCountdownMs) {
                raceStart = millis(); // Nu startar klockan!
                currentState = SystemState::MEASURING;
            }
            break;
        }

        case SystemState::MEASURING:
            // Visa att vi mäter
            oled.showRacing(latestData.distance, latestData.amplitude);
            
            // Logik: Inom kTriggerDist, inte i blindzon (kTriggerDistLow) och bra signal (Amp >= 100)
            if (latestData.distance < kTriggerDistHigh && 
                latestData.distance >= kTriggerDistLow && 
                latestData.amplitude >= 100 && 
                latestData.amplitude != 65535) {
                
                float finalTime = (millis() - raceStart) / 1000.0;
                oled.showResult(finalTime);
                currentState = SystemState::DONE;
            }
            break;

        case SystemState::DONE:
            // Vänta på nytt knapptryck för att nollställa och köra igen
            if (btnPressed) {
                currentState = SystemState::IDLE;
            }
            break;
    }
}