#pragma once

#include "ArmHardwareBase.h"

class ArmSerialControl : virtual public ArmHardwareBase {
public:
    ArmSerialControl();
    void update() override;

protected: 
    LinRegResults base_servo_x_equals_angles; 
    LinRegResults shoulder_servo_x_equals_angles; 
    LinRegResults elbow_servo_x_equals_angles; 

private:
    void processSerialCommand();
    void printStatus();
    void broadcastPositions(); 

    int Angle_to_PWM(double angle_deg, int servo_id); 
    int PWM_to_angle(int PWM, int servo_id); 

    unsigned long last_status_print;
    unsigned long last_position_broadcast; 
};