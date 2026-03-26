#include <unity.h>
#include <Arduino.h>

const long SERIAL_BAUD_RATE = 115200;

void test_lidar_svarar() {
    Serial1.begin(SERIAL_BAUD_RATE);
    delay(500);  // Ge Lidar tid att starta
    
    TEST_ASSERT_MESSAGE(Serial1.available() > 0, "Lidar svarar inte!");
}

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    while (!Serial);

    UNITY_BEGIN();
    RUN_TEST(test_lidar_svarar);
    UNITY_END();
}

void loop() {}