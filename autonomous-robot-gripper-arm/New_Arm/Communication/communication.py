
"""
Message Types from Arduino:

1. Data Messages (machine-readable)
   - Prefix: "DATA:"
   - Format: DATA:<key> <value1> <value2> ...
   - Example:
       DATA:armAtTarget 1

2. Info / Log Messages (human-readable)
   - Prefix: "INFO:"
   - Example:
       INFO: Initialization complete
"""


import numpy as np
import serial
import threading
import time
import serial.tools.list_ports

def find_arduino_uno():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if (port.vid, port.pid) in [
            (0x2341, 0x0043),  # Official Uno
            (0x2341, 0x0001),  # Older official Uno
            (0x1A86, 0x7523),  # CH340 clone
        ]:
            print(f"Arduino Uno found on {port.device} ({port.description})")
            return port.device
    print("No Arduino Uno detected.")
    return None


def parse_bool(robot, key, values):
    """Parse messages like:
    DATA:armAtTarget FALSE"
    """

    if not values:
        return
    val = values[0].lower()
    if val in ('true', '1'):
        result = True
    elif val in ('false', '0'):
        result = False
    else:
        print(f"Unknown boolean value for {key}: {values[0]}")
        return

    with robot._data_lock:
        robot.data[key] = result


def parse_numeric_dict(robot, key, values):
    """Parse messages like:
    DATA: pressure sensor1:120 sensor2:122 sensor3:121
    
    Stores results in robot.data[key] as a nested dict:
        robot.data["pressure"] = {"sensor1": 120.0, "sensor2": 122.0, ...}
    """
    if not values:
        return

    with robot._data_lock:
        # Ensure the key exists and is a dict
        if key not in robot.data or not isinstance(robot.data[key], dict):
            robot.data[key] = {}

        for pair in values:
            if ':' not in pair:
                print(f"Invalid pair in {key}: {pair}")
                continue

            subkey, val_str = pair.split(':', 1)
            try:
                val = float(val_str)
            except ValueError:
                print(f"Invalid numeric value for {key}.{subkey}: {val_str}")
                continue

            robot.data[key][subkey] = val


class RobotInterface:
    def __init__(self, port='COM5', baudrate=9600, timeout=0.1):
        # Locks
        self._ser_lock = threading.Lock()
        self._data_lock = threading.Lock()

        # Shared data
        self.data = {}

        # Serial setup
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
        except:
            print("Warning ------Did not connect to the robot---------")
        time.sleep(2)  # allow Arduino reset

        # Parser registry
        self._parsers = {}
        self.setup_standard_parsers()
    

        # Listener thread
        self._stop_thread = False
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()


    def setup_standard_parsers(self):
        self.register_parser("armAtTarget", parse_bool)
        self.register_parser("pressure", parse_numeric_dict)

    # ---------- Public API ----------
    def register_parser(self, key: str, func):
        """Register a parser function for a given data key."""
        self._parsers[key] = func

    def move_servo(self, servo_id, pwm):
        """Send servo command to Arduino."""
        with self._ser_lock:
            self.ser.reset_input_buffer()
            self.ser.write(f"{servo_id} {pwm}\n".encode('utf-8'))

        #reset arm state
        with self._data_lock:
            self.data['armAtTarget'] = False

    def get_data(self, key):
        with self._data_lock:
            return self.data.get(key)
    
    def get_available_data_keys(self):
        with self._data_lock:
            return list(self.data.keys())

    def stop(self):
        """Stop background thread and close serial port."""
        self._stop_thread = True
        self._thread.join(timeout=2)
        with self._ser_lock:
            if self.ser.is_open:
                self.ser.close()

    # ---------- Thread internals ----------
    def _listen(self):
        """Continuously read serial input in background."""
        while not self._stop_thread:
            time.sleep(0.05)
            with self._ser_lock:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                except Exception:
                    continue

            if not line:
                continue

            if line.startswith("DATA:"):
                self._handle_data(line[5:].strip())
            elif line.startswith("INFO:"):
                print("Arduino Info:", line[5:].strip())
            # else ignore

    def _handle_data(self, payload: str):
        """Parse 'DATA:' messages and dispatch to registered handler."""
        parts = payload.split()
        if not parts:
            return

        key, *values = parts
        parser = self._parsers.get(key)
        if parser is not None:
            parser(self, key, values)
        else:
            print(f"Unknown key from Arduino: {key}")




# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from angles_to_pwm import convert_to_pwm
    # robot = RobotInterface(port='COM6', baudrate=9600)
    port = find_arduino_uno()
    
    robot = RobotInterface(port=port, baudrate=9600)
    try:    
        time.sleep(5)
        angles_or_pwm = input("angles or pwm? (a/p)")
        while True:

            print(robot.get_data("pressure"))
            
            if angles_or_pwm.lower() == 'a':
                servo_id = int(input("Servo ID: "))
                angle_deg = float(input("Angle (degree): "))
                angle_rad = np.deg2rad(angle_deg)
                pwm = convert_to_pwm(servo_id, angle_rad)
                print("PWM (ms):", pwm)
                pwm = int(pwm)
                robot.move_servo(servo_id, pwm)
                input("\ncontinue?\n")
                print("\nCurrent armAtTarget:", robot.get_data('armAtTarget'), "\n")
            elif angles_or_pwm.lower() == 'p':    
                servo_id = int(input("Servo ID: "))
                pwm = int(input("PWM (ms): "))
                robot.move_servo(servo_id, pwm)
                input("\ncontinue?\n")
                print("\nCurrent armAtTarget:", robot.get_data('armAtTarget'), "\n")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        robot.stop()