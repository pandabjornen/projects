#include <Arduino.h>
#include "ServoCommandProcessor.h"
#include "Arm.h"
#include "Timer.h"

// const int HUBERT_SERVO_COUNT = 6;
// const int HUBERT_PINS[HUBERT_SERVO_COUNT]     = {3, 5, 6, 9, 10, 11};
// const int HUBERT_MIN_POS[HUBERT_SERVO_COUNT]  = {560, 1200, 1700, 750, 550, 550};
// const int HUBERT_MAX_POS[HUBERT_SERVO_COUNT]  = {2330, 2340, 2400, 2300, 2400, 2150};
// const int HUBERT_INIT_POS[HUBERT_SERVO_COUNT] = {1550, 1500, 2100, 2150, 1500, 1000};

// const int SERVO_COUNT = 5;
// const int SERVO_PINS[SERVO_COUNT]     = {11, 10, 9, 6, 5};
// const int SERVO_MIN_POS[SERVO_COUNT]  = {560, 1100, 1800, 1000, 500};
// const int SERVO_MAX_POS[SERVO_COUNT]  = {2330,1800, 2500, 1800, 2500};
// const int SERVO_INIT_POS[SERVO_COUNT] = {1550,1100, 1800, 1800, 1500};

// After Gabriel: 
// const int SERVO_COUNT = 5;
// const int SERVO_PINS[SERVO_COUNT]     = {11, 10, 9, 6, 5};
// const int SERVO_MIN_POS[SERVO_COUNT]  = {560, 1100, 1600, 1000, 500};
// const int SERVO_MAX_POS[SERVO_COUNT]  = {2330,1800, 2500, 1800, 2500};
// const int SERVO_INIT_POS[SERVO_COUNT] = {1550,1100, 1800, 1800, 1500};


// FIX Shoulder: 
const int SERVO_COUNT = 5;
const int SERVO_PINS[SERVO_COUNT]     = {11, 10, 9, 6, 5};
const int SERVO_MIN_POS[SERVO_COUNT]  = {560, 
                                        1025, // = 44°
                                        1550, // 45°
                                        1160, // 1200 ms = -25° 
                                        1300}; // may want to change but set these now to not accidently destroy something  
const int SERVO_MAX_POS[SERVO_COUNT]  = {2500,
                                        1190,   // 15°
                                        2250, //-16° 
                                        2200, // -90° - some marginal
                                        1800};  // may want to change but set these now to not accidently destroy something 
const int SERVO_INIT_POS[SERVO_COUNT] = {1550, // 0°
                                        1100, //30°
                                        1700, // 30°
                                        1690, // -90
                                        1500};

const int PRESSURE_SENSOR_COUNT = 3;
const int PRESSURE_SENSOR_PINS[PRESSURE_SENSOR_COUNT] = {A1, A3, A5};

// Communication
const int SERIAL_CONNECTION_NUMBER = 9600;

// Delays movement
const int INITIAL_SETUP_DELAY = 3000; //ms
const int SEND_TARGET_REACHED_INTERVAL = 200; //ms
const int SEND_PRESSURE_SENSOR_READ_INTERVAL = 50; //ms

// Objects
ServoCommandProcessor* servoCommandProcessor;
Arm* arm;
Timer* send_target_reached_timer;
Timer* send_pressure_sensor_read_timer;



void setup()
{
    Serial.begin(SERIAL_CONNECTION_NUMBER);
    delay(INITIAL_SETUP_DELAY);
    Serial.println("INFO: Started");

    servoCommandProcessor = new ServoCommandProcessor();
    arm = new Arm(SERVO_PINS, SERVO_MIN_POS, SERVO_MAX_POS, SERVO_INIT_POS, SERVO_COUNT);
    send_target_reached_timer = new Timer(SEND_TARGET_REACHED_INTERVAL, send_target_reached);
    send_pressure_sensor_read_timer = new Timer(SEND_PRESSURE_SENSOR_READ_INTERVAL, send_pressure_sensor_read);

}


void loop() { 
    servoCommandProcessor->update();

    if (servoCommandProcessor->commandReady()) {
        ServoCommand command = servoCommandProcessor->getCommand();
        arm->takeCommand(command);
        Serial.println("DATA:armAtTarget FALSE");
    }

    arm->step(10);

    send_target_reached_timer->update();
    send_pressure_sensor_read_timer->update();

}


void send_target_reached(){
    if (arm->targetsReached()) {
        Serial.println("DATA:armAtTarget TRUE");
    }
    else{
        Serial.println("DATA:armAtTarget FALSE");
    }
}

void send_pressure_sensor_read(){
    String output = "DATA:pressure";

    for (int i = 0; i < PRESSURE_SENSOR_COUNT; i++) {
        int value = analogRead(PRESSURE_SENSOR_PINS[i]);
        output += " sensor";
        output += (i);
        output += ":";
        output += value;
        
    }

    Serial.println(output);
} 






