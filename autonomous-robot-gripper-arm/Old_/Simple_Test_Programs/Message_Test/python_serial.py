import serial
import time

# Replace 'COM3' with your Arduino's port on Windows
# On Linux/Mac it might be something like '/dev/ttyACM0' or '/dev/ttyUSB0'
arduino_port = "COM5"
baud_rate = 57600
trigger_char = 'A'

# Open serial connection
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

# Send the trigger character
ser.write(trigger_char.encode())

# Read response from Arduino
while True:
    if ser.in_waiting > 0:
        message = ser.readline().decode().strip()
        if message:
            print("Arduino says:", message)
            break

ser.close()