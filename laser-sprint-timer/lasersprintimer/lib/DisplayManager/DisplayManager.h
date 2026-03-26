#pragma once
#include <Arduino.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "../TFLuna/TFLuna.h"

class DisplayManager {
public:
    // Definition av SPI-pinnar för Waveshare OLED
    static constexpr int kMosi = 11;
    static constexpr int kClk = 13;
    static constexpr int kDc = 9;
    static constexpr int kCs = 10;
    static constexpr int kReset = 8;
    static constexpr int kWidth = 128;
    static constexpr int kHeight = 64;

    DisplayManager();
    bool begin();
    void showIdle(const TFLunaData& data);
    void showCountdown(int seconds);
    void showRacing(uint16_t dist, uint16_t amp);
    void showResult(float timeSec);
    void showText(const char* text);

private:
    Adafruit_SSD1306 display_;
};