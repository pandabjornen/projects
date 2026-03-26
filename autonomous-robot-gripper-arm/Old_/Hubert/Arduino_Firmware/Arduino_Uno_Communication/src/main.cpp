#include <Arduino.h>
#include "Hubert.h"


//BODY = 0, HEAD_PAN = 1, HEAD_TILT = 2, SHOULDER = 3,
// ELBOW = 4, GRIPPER = 5


// Hardware
const int HUBERT_SERVO_COUNT = 6;
const int HUBERT_PINS[HUBERT_SERVO_COUNT]     = {3, 5, 6, 9, 10, 11};
const int HUBERT_MIN_POS[HUBERT_SERVO_COUNT]  = {560, 1200, 1700, 750, 550, 550};
const int HUBERT_MAX_POS[HUBERT_SERVO_COUNT]  = {2330, 2340, 2400, 2300, 2400, 2150};
const int HUBERT_INIT_POS[HUBERT_SERVO_COUNT] = {1550, 1500, 2100, 2150, 1500, 1000};

// Calibration

const AnglePWMCalibData HUBERT_CALIBRATION_DATA = {
    .base_servo = {
        .angles = {0.0, 90.0, 45.0, -45.0},     
        .PWMs = {1550.0, 630.0, 1140.0, 2100}
    },
    .shoulder_servo = {
        .angles = {90.0, 0.0},      
        .PWMs = {1365.0, 2265.0} 
    },
    .elbow_servo = {
        .angles = {0.0, 90.0},     
        .PWMs = {1590.0, 650.0} 
    }
};


// Communication
const int SERIAL_CONNECTION_NUMBER = 9600;
const int POSITION_BROADCAST_INTERVAL = 5000; //(ms)

// Delays movement
const int INITIAL_SETUP_DELAY = 3000;
const int STATUS_PRINT_INTERVAL = 5000;
const int ACTUATE_MOVE_STEP_DELAY = 30; // (ms) 
const int ACTUATE_MOVE_SETUP_DELAY = 20; // (ms)

const ArmConfig HUBERT_CONFIG = {
    .SERVO_COUNT = HUBERT_SERVO_COUNT, // '.' before struct variable to not need to set variables in correct order
    .PINS = HUBERT_PINS,
    .MIN_POS = HUBERT_MIN_POS,
    .MAX_POS = HUBERT_MAX_POS,
    .INIT_POS = HUBERT_INIT_POS,
    .SERIAL_CONNECTION_NUMBER = SERIAL_CONNECTION_NUMBER,
    .SETUP_DELAY = INITIAL_SETUP_DELAY,
    .STATUS_PRINT_DELAY = STATUS_PRINT_INTERVAL,
    .ACTUATE_STEP_DELAY = ACTUATE_MOVE_STEP_DELAY,
    .ACTUATE_SETUP_DELAY = ACTUATE_MOVE_SETUP_DELAY, 
    .POSITION_BROADCAST_INTERVAL = POSITION_BROADCAST_INTERVAL,
    .CALIBRATION_DATA = HUBERT_CALIBRATION_DATA
};


Hubert hubert(HUBERT_CONFIG);

void setup() // need to allocate memory to heap ('new' or 'malloc') 
            // first here in setup because otherwise Arduino heap memory manager
            // not setup before trying to allocate memory to it.  
{ 
    Serial.begin(hubert.getSerialConnectionNumber());
    delay(hubert.getSetupDelay());
    Serial.println("Init start:");

    // or just use the hardcoded values above, if dont want to complicate it like this: 
    // Serial.begin(SERIAL_CONNECTION_NUMBER);
    // delay(INITIAL_SETUP_DELAY);
    hubert.begin();
}

void loop() { 
    hubert.update();
}


// void setup(){}
// void loop(){}