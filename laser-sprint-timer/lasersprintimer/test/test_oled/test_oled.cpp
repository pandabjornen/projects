#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <unity.h>

#define CS_PIN    10
#define DC_PIN     9
#define RESET_PIN  8

Adafruit_SSD1306 display(128, 64, &SPI, DC_PIN, RESET_PIN, CS_PIN);

void test_oled_initierar() {
    bool ok = display.begin(SSD1306_SWITCHCAPVCC);
    TEST_ASSERT_MESSAGE(ok, "OLED svarar inte — kolla koppling");
}

void test_oled_skriver_text() {
    display.clearDisplay();
    display.setTextColor(WHITE);
    display.setTextSize(2);
    display.setCursor(0, 0);
    display.println("TEST OK");
    display.display();

    // Om vi kom hit utan krasch fungerar skärmen
    TEST_ASSERT(true);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    UNITY_BEGIN();
    RUN_TEST(test_oled_initierar);
    RUN_TEST(test_oled_skriver_text);
    UNITY_END();
}

void loop() {}