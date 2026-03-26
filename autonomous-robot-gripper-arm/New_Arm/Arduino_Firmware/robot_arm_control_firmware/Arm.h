#pragma once
#include <Arduino.h>
#include <Servo.h>
#include "ServoCommandProcessor.h" // contains ServoCommand struct

class Arm {
public:
    Arm(const int* pins, const int* minPos, const int* maxPos, const int* initPos, int servoCount);

    void takeCommand(const ServoCommand& cmd);  // Update target PWM for a servo
    void step(int stepLoops);                    // Move servos toward target PWM
    bool targetsReached() const;

private:
    int _servoCount;
    Servo* _servos;          // dynamic array of Servo objects
    int* _currPos;           // current PWM of each servo
    int* _targetPos;         // target PWM of each servo
    int* _minPos;            // min allowed PWM
    int* _maxPos;            // max allowed PWM
};