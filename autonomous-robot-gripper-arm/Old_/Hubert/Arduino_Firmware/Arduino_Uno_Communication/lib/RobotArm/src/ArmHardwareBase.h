#pragma once

#include <Arduino.h>
#include <Servo.h>
#include <vector>

enum class ServoIndex {
    BODY = 0,  //automatically sets rest of ints to 1,2,3, ...
    HEAD_PAN,
    HEAD_TILT,
    SHOULDER,
    ELBOW,
    GRIPPER
};

struct AnglePWMCalibOneServo{
    std::vector<double> angles; 
    std::vector<double> PWMs; 
}; 

struct AnglePWMCalibData{
    AnglePWMCalibOneServo base_servo; 
    AnglePWMCalibOneServo shoulder_servo; 
    AnglePWMCalibOneServo elbow_servo; 
}; 

struct ArmConfig {
    
    const int SERVO_COUNT;
    const int* const PINS; // two const since constant pointer (adress) and constant value at that adress
    const int* const MIN_POS;
    const int* const MAX_POS;
    const int* const INIT_POS;
    
    
    const int SERIAL_CONNECTION_NUMBER;
    const int SETUP_DELAY;
    const unsigned int STATUS_PRINT_DELAY;
    const int ACTUATE_STEP_DELAY;
    const int ACTUATE_SETUP_DELAY;

    const unsigned int POSITION_BROADCAST_INTERVAL;  // how often send

    AnglePWMCalibData CALIBRATION_DATA; 
};

struct LinRegResults {
    double slope; 
    double intercept; 
}; 

class ArmHardwareBase
{
public:
    
    ArmHardwareBase(const ArmConfig& config); // constructur requirements
    
    // Virtual => need to be implemented; "=0" => not implemented. 
    virtual void begin() = 0;
    virtual void update() = 0;


    // geyter methods (like when using 'get' keyword in C# methods) to give to main.cpp setuo function. 
    int getSerialConnectionNumber() const { return arm_config.SERIAL_CONNECTION_NUMBER; }
    int getSetupDelay() const { return arm_config.SETUP_DELAY; }
    
protected: 
    
    virtual void actuate(int servo_index, int new_pos) = 0; // '=0' => 'pure virtual'. all robot arms need to be able to actuate


    virtual LinRegResults lin_reg(std::vector<double> y_vec, std::vector<double> x_vec);

    const ArmConfig& arm_config; 
    Servo* servos = nullptr; // Initialize to nullptr, allocate when child (Hubert) init, inside setup function in main.cpp
    int* current_positions = nullptr; 


};