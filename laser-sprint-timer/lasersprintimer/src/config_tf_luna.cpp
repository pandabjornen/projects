#include <Arduino.h>

const long SERIAL_BAUD = 115200;

uint8_t cmd_freq_250Hz[] ={0x5A, 0x06, 0x03, 0xFA, 0x00, 0x00}; 
uint8_t cmd_save[] ={0x5A, 0x04, 0x11, 0x00}; 

const int DELAY_BETWEEN_CMDS_MS = 100; 

void setup() {
    Serial1.begin(SERIAL_BAUD);

    while (!Serial1) {
    ; 
    }
    Serial1.write(cmd_freq_250Hz, sizeof(cmd_freq_250Hz));
    delay(DELAY_BETWEEN_CMDS_MS);
    Serial1.write(cmd_save, sizeof(cmd_save)); 
}

void loop() { 

}  