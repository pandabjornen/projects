#include "TFLuna.h"

TFLunaParser::TFLunaParser() {
    state_ = State::WAIT_HEADER1;
    dataIndex_ = 0;
    checksum_ = 0;
}

bool TFLunaParser::processByte(uint8_t b) {
    switch (state_) {
        case State::WAIT_HEADER1:
            if (b == 0x59) {
                buffer_[0] = b;
                checksum_ = b;
                state_ = State::WAIT_HEADER2;
            }
            break;

        case State::WAIT_HEADER2:
            if (b == 0x59) {
                buffer_[1] = b;
                checksum_ += b;
                dataIndex_ = 2; // Vi har redan läst index 0 och 1
                state_ = State::READ_DATA;
            } else {
                state_ = State::WAIT_HEADER1; // Felaktig header, börja om
            }
            break;

        case State::READ_DATA:
            buffer_[dataIndex_] = b;
            checksum_ += b;
            dataIndex_++;
            
            if (dataIndex_ == 8) { // Vi har läst byte 0 till 7 
                state_ = State::WAIT_CHECKSUM;
            }
            break;

        case State::WAIT_CHECKSUM:
            state_ = State::WAIT_HEADER1; // Oavsett resultat börjar vi om nästa gång
            
            // Kolla om vår uträknade checksumma matchar den mottagna byten
            if (checksum_ == b) { 
                // Pussla ihop datan med bitshift och OR
                lastData_.distance = (buffer_[3] << 8) | buffer_[2]; 
                lastData_.amplitude = (buffer_[5] << 8) | buffer_[4];
                lastData_.temperature = (buffer_[7] << 8) | buffer_[6]; 
                return true; // Giltigt paket!
            }
            break;
    }
    return false; // Inte ett färdigt/giltigt paket än
}

TFLunaData TFLunaParser::getData() const {
    return lastData_;
}