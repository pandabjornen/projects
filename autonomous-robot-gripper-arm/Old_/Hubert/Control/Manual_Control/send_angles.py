import serial
import time

# Port för usb till Arduino
#ser = serial.Serial('COM3', 57600, timeout=1)  # Windows
#ser = serial.Serial('/dev/tty.usbmodem14101', 57600, timeout=1)  # Mac
#ser = serial.Serial('/dev/ttyACM0', 57600, timeout=1) # Linux/Mac
#ser = serial.Serial('/dev/cu.usbmodem2101', 9600, timeout=1)
ser = serial.Serial('COM5', 57600, timeout=1)



time.sleep(2)  # Let Arduino reset

def move_servo(servo_id, angle_deg, ser):
    ser.reset_input_buffer()
    command = f"{servo_id} {angle_deg}\n"
    ser.write(command.encode('utf-8'))

while True: 
    servo_id = int(input("Servo ID:"))
    # pwm = int(input("Pulse Width [µs]:"))
    angle_deg = float(input("Angle (deg):"))
    move_servo(servo_id, angle_deg, ser)

    time.sleep(1)  # Let it move and then get current position, but doesnt work below anyway currently
    
    
    
    # while ser.in_waiting > 0:
    #     line = ser.readline().decode('utf-8').strip()
    #     print(line)

    
    # while ser.in_waiting > 0:
    #     line = ser.readline().decode('utf-8').strip()
    #     print(line)



