#pragma once

#include <ArmMovementController.h>
#include <ArmSerialControl.h>

class Hubert : public ArmMovementController, public ArmSerialControl {
public:
    Hubert(const ArmConfig& config);
    
    void begin() override; 
    void CalcLinRegs(); 
};