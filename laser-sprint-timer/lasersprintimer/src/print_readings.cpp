#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "TFLuna.h" // Se till att filerna ligger i lib/TFLuna/

// OLED SPI Pins
#define OLED_MOSI   11
#define OLED_CLK    13
#define OLED_DC      9
#define OLED_CS     10
#define OLED_RESET   8

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

// Initiera skärmen för SPI
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, OLED_MOSI, OLED_CLK, OLED_DC, OLED_RESET, OLED_CS);

TFLunaParser tfParser;
TFLunaData latestData;

void setup() {
    Serial.begin(115200);
    Serial1.begin(115200); // UART-kommunikation med TF-Luna 

    // Starta OLED
    if(!display.begin(SSD1306_SWITCHCAPVCC)) {
        Serial.println(F("SSD1306 allocation failed"));
        for(;;);
    }

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0,0);
    display.println("TF-Luna Test");
    display.display();
}

void loop() { 
    while (Serial1.available()) {
        uint8_t b = Serial1.read();
        
        // Om parsern hittar ett giltigt paket 
        if (tfParser.processByte(b)) {
            latestData = tfParser.getData();

            // Rensa bufferten och skriv ny data
            display.clearDisplay();
            display.setCursor(0,0);
            
            // Distans i cm 
            display.setTextSize(2);
            display.print("Dist: ");
            display.print(latestData.distance);
            display.println("cm");

            // Signalstyrka (Amp) 
            display.setTextSize(1);
            display.print("Amp: ");
            display.println(latestData.amplitude);

            // Temperaturkonvertering enligt manual: Temp/8 - 256 
            float tempC = (latestData.temperature / 8.0) - 256.0;
            display.print("Temp: ");
            display.print(tempC, 1);
            display.println(" C");

            // Varning om svag signal 
            if (latestData.amplitude < 100) {
                display.setCursor(0, 50);
                display.print("LOW SIGNAL!");
            }

            display.display(); // Skicka bufferten till skärmen
        }
    }
}