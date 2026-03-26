#include "ArmSerialControl.h"
#include "ArmMovementController.h" 

ArmSerialControl::ArmSerialControl()
{
    last_status_print = 0;
    last_position_broadcast = 0; 
}

void ArmSerialControl::update() {
    processSerialCommand();
    printStatus();
    broadcastPositions(); 
}



void ArmSerialControl::broadcastPositions() {
    if (millis() - last_position_broadcast > arm_config.POSITION_BROADCAST_INTERVAL) {
        last_position_broadcast = millis();
        Serial.print("POS,"); 
        for (int i = 0; i < arm_config.SERVO_COUNT; i++) {
            
            int angle = PWM_to_angle(current_positions[i], i);
            Serial.print(angle);
            if (i < arm_config.SERVO_COUNT - 1) {
                Serial.print(",");
            }
        }
        Serial.println(); 
    }
}
void ArmSerialControl::processSerialCommand() {
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        
        int spaceIndex = line.indexOf(' ');
        if (spaceIndex > 0) {
            int servo_id = line.substring(0, spaceIndex).toInt();
            double angle_deg = line.substring(spaceIndex + 1).toDouble();
            
            int pwm = Angle_to_PWM(angle_deg, servo_id); 

            if (servo_id >= 0 && servo_id < arm_config.SERVO_COUNT) {
                int constrained_pwm = constrain(pwm, arm_config.MIN_POS[servo_id], arm_config.MAX_POS[servo_id]);
                actuate(servo_id, constrained_pwm);
                Serial.print("Moved servo ");
                Serial.print(servo_id);
                Serial.print(" to ");
                Serial.println(constrained_pwm); 
            }
        }
    }
}

void ArmSerialControl::printStatus() {
    
    if (millis() - last_status_print > arm_config.STATUS_PRINT_DELAY) {
        last_status_print = millis();
        Serial.println("Current positions: ");
        for (int i = 0; i < arm_config.SERVO_COUNT; i++) {
            Serial.print(current_positions[i]);
            if (i < arm_config.SERVO_COUNT - 1) Serial.print(", ");
        }
        Serial.println();
    }
}


int ArmSerialControl::Angle_to_PWM(double angle_deg, int servo_id) {
    double pwm_double = 1500;

    if (servo_id == (int)ServoIndex::BODY) {
        pwm_double = base_servo_x_equals_angles.slope * angle_deg + base_servo_x_equals_angles.intercept;
    }
    else if (servo_id == (int)ServoIndex::SHOULDER) {
        pwm_double = shoulder_servo_x_equals_angles.slope * angle_deg + shoulder_servo_x_equals_angles.intercept;
    }
    else if (servo_id == (int)ServoIndex::ELBOW) {
        pwm_double = elbow_servo_x_equals_angles.slope * angle_deg + elbow_servo_x_equals_angles.intercept;
    }
    else {
        Serial.print("WARNING: Not calibrated, using naive mapping for SERVO: ");
        Serial.println(servo_id);
        pwm_double = map(angle_deg, -90, 90, 544, 2400);
    }

    return (int)pwm_double;
}


int ArmSerialControl::PWM_to_angle(int PWM, int servo_id) {
    double angle_double = 0.0; 

    LinRegResults calib;
    bool calibrated = true;

    if (servo_id == (int)ServoIndex::BODY) {
        calib = base_servo_x_equals_angles;
    }
    else if (servo_id == (int)ServoIndex::SHOULDER) {
        calib = shoulder_servo_x_equals_angles;
    }
    else if (servo_id == (int)ServoIndex::ELBOW) {
        calib = elbow_servo_x_equals_angles;
    }
    else {
        calibrated = false;
    }

    if (calibrated) 
    {
        angle_double = (PWM - calib.intercept) / calib.slope;
    }
    else {
        // Fallback 
        angle_double = map(PWM, 544, 2400, -90, 90);
    }

    return (int)angle_double; 
}