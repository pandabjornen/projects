#pragma once

#include <Arduino.h>

class Timer {
private:
    unsigned long interval;        // Timer interval in ms
    unsigned long lastTime;        // Last time timer was triggered
    void (*callback)();            // Function pointer for callback

public:
    // Constructor
    Timer(unsigned long intervalMs, void (*cb)());

    // Call this every loop
    void update();

    // Optional: reset the timer
    void reset();

    // Optional: change interval on the fly
    void setInterval(unsigned long intervalMs);
};