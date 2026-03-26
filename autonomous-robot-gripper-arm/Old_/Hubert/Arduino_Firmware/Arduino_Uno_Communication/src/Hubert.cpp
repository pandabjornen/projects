#include "Hubert.h"

Hubert::Hubert(const ArmConfig& config) 
    : ArmHardwareBase(config) 
{}

void Hubert::begin() {

    Serial.println("just quickly calculating som lin reg for angle calib..."); 

    CalcLinRegs(); 


    servos = new Servo[arm_config.SERVO_COUNT]; // "new" newer better version of malloc
                                                // to allocate memory on the heap, knows automatically which datatype stored. 
                                                // Also runs the constructor for each Servo object created
    current_positions = new int[arm_config.SERVO_COUNT];
    Serial.println("Positioning servos to defined minimums...");
    for (int i = 0; i < arm_config.SERVO_COUNT; i++) {
        servos[i].attach(arm_config.PINS[i]);
        servos[i].writeMicroseconds(arm_config.MIN_POS[i]); 
        delay(500); 
        current_positions[i] = arm_config.MIN_POS[i]; 
    }

    
    delay(1000);
    
    for (int i = 0; i < arm_config.SERVO_COUNT; i++) {
        actuate(i, arm_config.INIT_POS[i]);
    }
    Serial.println("Hubert robot initialized.");
}

void Hubert::CalcLinRegs() {
    
    base_servo_x_equals_angles = lin_reg(
        arm_config.CALIBRATION_DATA.base_servo.PWMs, 
        arm_config.CALIBRATION_DATA.base_servo.angles
    );
    
    shoulder_servo_x_equals_angles = lin_reg(
        arm_config.CALIBRATION_DATA.shoulder_servo.PWMs, 
        arm_config.CALIBRATION_DATA.shoulder_servo.angles
    );

    elbow_servo_x_equals_angles = lin_reg(
        arm_config.CALIBRATION_DATA.elbow_servo.PWMs, 
        arm_config.CALIBRATION_DATA.elbow_servo.angles
    );
    
    
    Serial.print("Base Servo: slope="); Serial.print(base_servo_x_equals_angles.slope);
    Serial.print(", intercept="); Serial.println(base_servo_x_equals_angles.intercept);

    Serial.print("Shoulder Servo: slope="); Serial.print(shoulder_servo_x_equals_angles.slope);
    Serial.print(", intercept="); Serial.println(shoulder_servo_x_equals_angles.intercept);

    Serial.print("Elbow Servo: slope="); Serial.print(elbow_servo_x_equals_angles.slope);
    Serial.print(", intercept=");  Serial.println(elbow_servo_x_equals_angles.intercept);
}