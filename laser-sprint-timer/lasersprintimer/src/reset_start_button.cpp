/** 
 * @file reset_start_button.cpp
 * @brief Använd reset knapp på Arduino för att starta mätning 
*/

#include <Arduino.h>
#include "TFLuna.h"
#include "DisplayManager.h"

const uint16_t kTriggerDist = 50; // cm 
const unsigned long kCountdownMs = 3000;

TFLunaParser tfParser;
TFLunaData latestData;
DisplayManager oled;

// Starta direkt i COUNTDOWN istället för IDLE
enum class SystemState { COUNTDOWN, MEASURING, DONE };
SystemState currentState = SystemState::COUNTDOWN;

unsigned long stateStart = 0;
unsigned long raceStart = 0;

void setup() {
    Serial1.begin(115200);
    oled.begin();
    
    // Sätt starttiden för nedräkningen direkt vid uppstart
    stateStart = millis();
}

void loop() {
    // Fortsätt läsa sensorn
    while (Serial1.available()) {
        if (tfParser.processByte(Serial1.read())) {
            latestData = tfParser.getData();
        }
    }

    switch (currentState) {
        case SystemState::COUNTDOWN: {
            unsigned long elapsed = millis() - stateStart;
            int remaining = 3 - (elapsed / 1000);
            
            oled.showCountdown(remaining);
            
            if (elapsed >= kCountdownMs) {
                raceStart = millis();
                currentState = SystemState::MEASURING;
            }
            break;
        }

        case SystemState::MEASURING:
            oled.showRacing(latestData.distance, latestData.amplitude);
            
            if (latestData.distance < kTriggerDist && latestData.distance >= 20 && latestData.amplitude >= 100) {
                float finalTime = (millis() - raceStart) / 1000.0;
                oled.showResult(finalTime);
                currentState = SystemState::DONE;
            }
            break;

        case SystemState::DONE:
            // Här gör vi ingenting. 
            // För att starta om trycker användaren på RESET-knappen på Arduinon.
            break;
    }
}