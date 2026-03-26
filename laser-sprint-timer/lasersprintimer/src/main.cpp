#include <Arduino.h>
#include <ArduinoBLE.h> 
#include "TFLuna.h"
#include "DisplayManager.h"

const bool DEBUG = false; 
const long LIDAR_BAUD_RATE = 115200; 

const int kButtonPin = 2;           
const uint16_t kTriggerDistHigh = 40;  
const uint16_t kTriggerDistLow = 20;   
const unsigned long kCountdownMs = 5000; 

TFLunaParser tfParser;
TFLunaData latestData;
DisplayManager oled;

BLEService timerService("19B10000-E8F2-537E-4F6C-D104768A1214"); // Som URL
BLEByteCharacteristic commandChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite); // som ett formulärfält på en hemsida

enum class SystemState { IDLE, COUNTDOWN, MEASURING, DONE };
SystemState currentState = SystemState::IDLE;

unsigned long stateStart = 0;
unsigned long raceStart = 0;
unsigned long lastBtn = 0;
const unsigned long kDebounceDelay = 50;

void setup() {

    if (DEBUG){ 
        Serial.begin(9600);
        while (!Serial);  
    }
    Serial1.begin(LIDAR_BAUD_RATE);
    
    pinMode(kButtonPin, INPUT_PULLUP);
    
    if (!oled.begin()) {
        Serial.println("OLED fail");
        while(1);
    }

    if (DEBUG) Serial.println("Startar BLE...");

    if (!BLE.begin()) {
        Serial.println("BLE fail");
        oled.showText("BLE fail");
        while(1);
    }  

    //Sätt namn synligt för andra enheter
    if (DEBUG) Serial.println("FOUND BLE --> setting name and service...."); 
    BLE.setLocalName("SprintTimer");
    BLE.setAdvertisedService(timerService);
    
    // liten minnes cell med 1 byte som vi kan skriva till senare
    timerService.addCharacteristic(commandChar);
    BLE.addService(timerService);
    commandChar.writeValue(0); // skriver 0 som default värde
    
    if (DEBUG) Serial.println("Starting advertise..."); 
    // Börja sända ut signalen (Advertise)
    BLE.advertise();
}

void loop() {
    // MAIN Processorn fågar aktivt efter att något har hänt istället för att bli meddelad när ngt har hänt
    BLE.poll();

    while (Serial1.available()) {
        if (tfParser.processByte(Serial1.read())) {
            latestData = tfParser.getData();
        }
    }

    //Fysisk knapp || Bluetooth
    bool btnPressed = false;
    
    // Fysisk knapp
    if (digitalRead(kButtonPin) == LOW && (millis() - lastBtn) > kDebounceDelay) {
        btnPressed = true;
        lastBtn = millis();
    }

    // Bluetooth-kommando
    if (commandChar.written()) {
        if (commandChar.value() == 1) { // Om får värde "1"
            btnPressed = true;
            commandChar.writeValue(0);  // Nollställ direkt
        }
    }

    switch (currentState) {
        
        case SystemState::IDLE:
            oled.showIdle(latestData);
            if (btnPressed) {
                stateStart = millis();
                currentState = SystemState::COUNTDOWN;
            }
            break;

        case SystemState::COUNTDOWN: {
            unsigned long elapsed = millis() - stateStart;
            int remaining = (kCountdownMs / 1000) - (elapsed / 1000);
            
            oled.showCountdown(remaining);
            
            if (elapsed >= kCountdownMs) {
                raceStart = millis(); 
                currentState = SystemState::MEASURING;
            }
            break;
        }

        case SystemState::MEASURING:
            oled.showRacing(latestData.distance, latestData.amplitude);
            
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
            if (btnPressed) {
                currentState = SystemState::IDLE;
            }
            break;
    }
}