#pragma once
#include <Arduino.h>

struct ServoCommand {
    int servo_id;
    int pwm;
};


class ServoCommandProcessor {
public:
    ServoCommandProcessor();

    void update();                 // Read serial and prepare command
    bool commandReady() const;     // Check if a command is ready
    ServoCommand getCommand();     // Returns a new, immutable command

private:
    ServoCommand _nextCommand;     // temporary storage for incoming command
    bool _hasCommand;
};