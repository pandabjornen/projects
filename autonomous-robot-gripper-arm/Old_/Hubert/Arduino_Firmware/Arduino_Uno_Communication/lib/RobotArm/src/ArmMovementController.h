#pragma once

#include "ArmHardwareBase.h"

class ArmMovementController : virtual public ArmHardwareBase //virtual because is a sibling class 
{
public:
    ArmMovementController();

protected:
    void actuate(int servo_index, int new_pos) override; // intedning to implement the actuate function. 
};