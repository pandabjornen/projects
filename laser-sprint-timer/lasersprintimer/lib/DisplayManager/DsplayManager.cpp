#include "DisplayManager.h"

DisplayManager::DisplayManager() 
    : display_(kWidth, kHeight, kMosi, kClk, kDc, kReset, kCs) {}

bool DisplayManager::begin() {
    if (!display_.begin(SSD1306_SWITCHCAPVCC)) return false;
    display_.setTextColor(SSD1306_WHITE);
    return true;
}

void DisplayManager::showIdle(const TFLunaData& data) {
    display_.clearDisplay();
    
    // Rubrik
    display_.setTextSize(1);
    display_.setCursor(0, 0);
    display_.println("Vantar pa start...");

    // Live-data monitor
    display_.setCursor(0, 20);
    display_.print("Dist: "); 
    display_.print(data.distance); 
    display_.println(" cm"); // Enligt manualen är standarden cm 

    display_.print("Amp:  "); 
    display_.println(data.amplitude); // Signalstyrka 

    // Temperaturberäkning: Temp/8 - 256 
    float tempC = (data.temperature / 8.0) - 256.0;
    display_.print("Temp: "); 
    display_.print(tempC, 1); 
    display_.println(" C");

    // Statusmeddelande längst ner
    display_.setCursor(0, 55);
    if (data.amplitude < 100) {
        display_.println("Status: SVAG SIGNAL"); // Amp < 100 är opålitligt 
    } else {
        display_.println("Status: READY");
    }

    display_.display();
}

void DisplayManager::showCountdown(int seconds) {
    display_.clearDisplay();
    display_.setTextSize(4);
    display_.setCursor(50, 15);
    display_.println(seconds);
    display_.display();
}

void DisplayManager::showRacing(uint16_t dist, uint16_t amp) {
    display_.clearDisplay();
    display_.setTextSize(2);
    display_.setCursor(0, 0);
    display_.println("SPRING!");
    display_.setTextSize(1);
    display_.print("Dist: "); display_.print(dist); display_.println(" cm");
    display_.display();
}

void DisplayManager::showResult(float timeSec) {
    display_.clearDisplay();
    display_.setTextSize(1);
    display_.setCursor(0, 0);
    display_.println("MALGANG!");
    display_.setTextSize(2);
    display_.print(timeSec, 3); // 3 decimaler noggranhet men massa saker som egränsar faktiskt nggrhet
    display_.println(" s");
    display_.display();
}

void DisplayManager::showText(const char* text) {
    display_.clearDisplay();
    display_.setTextSize(1);
    display_.setCursor(0, 0);
    display_.println(text);
    display_.display();
}