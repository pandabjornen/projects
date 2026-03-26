#pragma once
#include <Arduino.h>

struct TFLunaData {
    uint16_t distance;
    uint16_t amplitude;
    uint16_t temperature;
};

class TFLunaParser {
public:
    TFLunaParser();
    
    bool processByte(uint8_t b);
    TFLunaData getData() const;

private:
    enum class State { WAIT_HEADER1, WAIT_HEADER2, READ_DATA, WAIT_CHECKSUM };
    State state_;
    uint8_t buffer_[8]; // Sparar byten 0-7 för att beräkna checksumma
    uint8_t dataIndex_;
    uint8_t checksum_;
    TFLunaData lastData_;
};