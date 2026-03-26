#include "Timer.h"


Timer::Timer(unsigned long intervalMs, void (*cb)())
    : interval(intervalMs), callback(cb), lastTime(0) {}

void Timer::update() {
    unsigned long currentTime = millis();
    if ((currentTime - lastTime) >= interval) {
        lastTime = currentTime;
        if (callback) callback();
    }
}

void Timer::reset() {
    lastTime = millis();
}

void Timer::setInterval(unsigned long intervalMs) {
    interval = intervalMs;
}