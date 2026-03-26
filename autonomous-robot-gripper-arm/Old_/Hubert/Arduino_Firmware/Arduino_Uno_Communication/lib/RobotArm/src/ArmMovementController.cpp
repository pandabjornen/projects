#include "ArmMovementController.h"

ArmMovementController::ArmMovementController(){}


void ArmMovementController::actuate(int servo_index, int new_pos) 
{
    int now = current_positions[servo_index];
    if (now == new_pos) return;

    int diff = (new_pos > now) ? 1 : -1;
    int steps = abs(new_pos - now);
    int delta = 6;
    
    delay(arm_config.ACTUATE_SETUP_DELAY); 
    for (int i = 0; i < steps; i += delta) {
        now += delta * diff;
        servos[servo_index].writeMicroseconds(now);
        delay(arm_config.ACTUATE_STEP_DELAY); 
    }
    current_positions[servo_index] = now;
    delay(arm_config.ACTUATE_SETUP_DELAY); 
}