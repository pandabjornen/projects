#include "ServoCommandProcessor.h"

ServoCommandProcessor::ServoCommandProcessor() : _hasCommand(false) {}

void ServoCommandProcessor::update() {
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();

        int spaceIndex = line.indexOf(' ');
        if (spaceIndex > 0) {
            ServoCommand newCommand;
            newCommand.servo_id = line.substring(0, spaceIndex).toInt();
            newCommand.pwm = line.substring(spaceIndex + 1).toInt();

            _nextCommand = newCommand;
            _hasCommand = true;
        }
    }
}

bool ServoCommandProcessor::commandReady() const {
    return _hasCommand;
}

ServoCommand ServoCommandProcessor::getCommand() {
    _hasCommand = false; // reset flag
    return _nextCommand; // return a copy; caller cannot modify internal state
}