#include "Arm.h"

Arm::Arm(const int* pins, const int* minPos, const int* maxPos, const int* initPos, int servoCount)
    : _servoCount(servoCount)
{
    // Allocate memory
    _servos = new Servo[_servoCount];
    _currPos = new int[_servoCount];
    _targetPos = new int[_servoCount];
    _minPos = new int[_servoCount];
    _maxPos = new int[_servoCount];

    // Initialize each servo
    for (int i = 0; i < _servoCount; ++i) {
        _servos[i].attach(pins[i]);
        delay(20);
        _currPos[i] = initPos[i];
        _targetPos[i] = initPos[i];
        _minPos[i] = minPos[i];
        _maxPos[i] = maxPos[i];

        // Write initial position immediately
        _servos[i].writeMicroseconds(initPos[i]);
    }
}

void Arm::takeCommand(const ServoCommand& cmd) {
    int id = cmd.servo_id;
    if (id < 0 || id >= _servoCount) return; // bounds check

    // Clamp PWM to min/max
    int pwm = cmd.pwm;
    if (pwm < _minPos[id]) pwm = _minPos[id];
    if (pwm > _maxPos[id]) pwm = _maxPos[id];

    _targetPos[id] = pwm;
}

void Arm::step(int stepLoops) {
    const int delta = 2; // how much to move per sub-step (PWM units)

    for (int s = 0; s < stepLoops; ++s) {
        for (int i = 0; i < _servoCount; ++i) {
            int curr = _currPos[i];
            int target = _targetPos[i];

            if (curr == target) continue;

            int diff = target - curr;
            int step = constrain(diff, -delta, delta); // move closer but not past

            int newPos = curr + step;

            // Clamp to safe range
            if (newPos < _minPos[i]) newPos = _minPos[i];
            if (newPos > _maxPos[i]) newPos = _maxPos[i];

            _servos[i].writeMicroseconds(newPos);
            _currPos[i] = newPos;
        }

        delay(20); // short pause between incremental moves
    }
}


bool Arm::targetsReached() const {
    for (int i = 0; i < _servoCount; ++i) {
        if (_currPos[i] != _targetPos[i]) {
            return false;  // found a servo that is not at target
        }
    }
    return true;  // all servos are at their target
}