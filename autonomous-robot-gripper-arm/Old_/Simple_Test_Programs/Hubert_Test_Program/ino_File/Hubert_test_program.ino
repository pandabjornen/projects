/*
  Hubert robot test program.

  This sketch control Hubert's servos.
  A write.microseconds value in approx. [500,2350], mid pos.: 1350
  NOTE: Go back to NeutralPos. before terminating.

  created 20 Jul 2020
  by Krister Wolff
  modified 11 Sep 2025
  by Vivien Lacorre

  This example code is in the public domain.

  http://www.arduino.cc/en/
*/

#include <Arduino.h>
#include <Servo.h>

//Servos
Servo body;
Servo headPan;
Servo headTilt;
Servo shoulder;
Servo elbow;
Servo gripper;

// Enum for servo indices
enum ServoIndex {
  BODY = 0,
  HEAD_PAN = 1,
  HEAD_TILT = 2,
  SHOULDER = 3,
  ELBOW = 4,
  GRIPPER = 5,
  SERVO_COUNT = 6
};

//Init position of all servos
const int servo_pins[SERVO_COUNT] = {3, 5, 6, 9, 10, 11};

int curr_pos[SERVO_COUNT];

const int pos_min[SERVO_COUNT] = {560, 1200, 950, 750, 550, 550};
const int pos_max[SERVO_COUNT] = {2330, 2340, 2400, 2200, 2400, 2150};

const int pos_init[SERVO_COUNT] = {1700, 1500, 2000, 1650, 1650, 1600};

void actuate_servo(Servo &servo, int servo_index, const int new_pos) {
  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  // current servo value
  now = curr_pos[servo_index];
  CurrPwm = now;
  NewPwm = new_pos;

  if (CurrPwm == NewPwm) return; // No movement needed

  // determine direction (+1 or -1)
  diff = (NewPwm - CurrPwm) / abs(NewPwm - CurrPwm);
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta * diff;
    servo.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[servo_index] = now;
  delay(10);
}

void setup() {
  Serial.begin(57600); // Starts the serial communication

	//Attach each joint servo
	//and write each init position
  body.attach(servo_pins[0]);
  body.writeMicroseconds(pos_init[0]);

  headPan.attach(servo_pins[1]);
  headPan.writeMicroseconds(pos_init[1]);

  headTilt.attach(servo_pins[2]);
  headTilt.writeMicroseconds(pos_init[2]);

  shoulder.attach(servo_pins[3]);
	shoulder.writeMicroseconds(pos_init[3]);

	elbow.attach(servo_pins[4]);
	elbow.writeMicroseconds(pos_init[4]);

	gripper.attach(servo_pins[5]);
  gripper.writeMicroseconds(pos_init[5]);

  Serial.println("Hubert test program started.");

  // We keep track of the current poses in the curr_pos array
  byte i;
  for (i=0; i<(sizeof(pos_init)/sizeof(int)); i++){
    curr_pos[i] = pos_init[i];
  }

	delay(2000);
}

void loop() {
  // Move all body parts to min positions
  Serial.println("Moving body to min position...");
  actuate_servo(body, BODY, pos_min[BODY]);
  Serial.println("Moving head pan to min position...");
  actuate_servo(headPan, HEAD_PAN, pos_min[HEAD_PAN]);
  Serial.println("Moving head tilt to min position...");
  actuate_servo(headTilt, HEAD_TILT, pos_min[HEAD_TILT]);
  Serial.println("Moving shoulder to min position...");
  actuate_servo(shoulder, SHOULDER, pos_min[SHOULDER]);
  Serial.println("Moving elbow to min position...");
  actuate_servo(elbow, ELBOW, pos_min[ELBOW]);
  Serial.println("Moving gripper to min position...");
  actuate_servo(gripper, GRIPPER, pos_min[GRIPPER]);

  // Move all body parts to max positions
  Serial.println("Moving body to max position...");
  actuate_servo(body, BODY, pos_max[BODY]);
  Serial.println("Moving head pan to max position...");
  actuate_servo(headPan, HEAD_PAN, pos_max[HEAD_PAN]);
  Serial.println("Moving head tilt to max position...");
  actuate_servo(headTilt, HEAD_TILT, pos_max[HEAD_TILT]);
  Serial.println("Moving shoulder to max position...");
  actuate_servo(shoulder, SHOULDER, pos_max[SHOULDER]);
  Serial.println("Moving elbow to max position...");
  actuate_servo(elbow, ELBOW, pos_max[ELBOW]);
  Serial.println("Moving gripper to max position...");
  actuate_servo(gripper, GRIPPER, pos_max[GRIPPER]);
}
