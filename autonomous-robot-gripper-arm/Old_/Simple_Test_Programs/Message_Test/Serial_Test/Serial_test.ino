char triggerChar = 'A';  // Character to trigger the LED
const int ledPin = 13;   // Onboard LED

void setup() {
  Serial.begin(57600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char received = Serial.read();
    if (received == triggerChar) {
      digitalWrite(ledPin, HIGH);  // Turn on LED
      Serial.println("LED is ON!"); // Send message back
      delay(5000);                 // Keep LED on for a second
      digitalWrite(ledPin, LOW);   // Turn off LED
    }
  }
}