import sys, os
import serial
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FK')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'IK')))
from FK_Hubert import fk_hubert 
from IK_Hubert import ik_hubert 


#NOTE REMEMBER L1! 
#WORKSPACE LIMITs: 
X_MIN = -0.32726780493007823
X_MAX =  0.34308204391465863
Y_MIN = -0.21174003670616093
Y_MAX = 0.34308204391465863
Z_MIN = 0.06800000000000003
Z_MAX = 0.6532678049300782



def send_position(x, y, z, port, baud_rate): 
    
    #ser = serial.Serial(port, baud_rate, timeout=1)
    if not (X_MIN <= x <= X_MAX):
        raise ValueError(f"x={x} is out of bounds ({X_MIN}, {X_MAX})")
    if not (Y_MIN <= y <= Y_MAX):
        raise ValueError(f"y={y} is out of bounds ({Y_MIN}, {Y_MAX})")
    if not (Z_MIN <= z <= Z_MAX):
        raise ValueError(f"z={z} is out of bounds ({Z_MIN}, {Z_MAX})")

    pos_tolerance_ik = 1e-5
    solutions_theta2_and_theta3_degrees, solutions_theta1_degrees = ik_hubert(x, y, z)
    solutions = find_combinations_of_correct_angles(solutions_theta2_and_theta3_degrees, solutions_theta1_degrees, x, y, z, pos_tolerance_ik)
    
    
    print_stuff(solutions, x, y, z)

    choice = int(input("Which option do you want? (1/2)\n").strip())

    if choice == 1 : 
        theta1 = solutions[0][0]
        theta2 = solutions[0][1]
        theta3 = solutions[0][2]
    elif choice == 2: 
        theta1 = solutions[1][0]
        theta2 = solutions[1][1]
        theta3 = solutions[1][2]
    else:
        raise ValueError("choose option 1 or 2") 

    move_servo(0, theta1, ser)
    move_servo(3, theta2, ser)
    move_servo(4, theta3, ser)



def find_combinations_of_correct_angles(solutions_theta2_and_theta3_degrees, solutions_theta1_degrees, x, y, z, pos_tolerance_ik): 
    solutions =[]
    for theta1_pred in solutions_theta1_degrees: 
        for theta2_pred, theta3_pred in solutions_theta2_and_theta3_degrees: 
            x_after_ik, y_after_ik, z_after_ik = fk_hubert(theta1_pred, theta2_pred, theta3_pred)

            if (abs(x_after_ik - x) <= pos_tolerance_ik and
                abs(y_after_ik - y) <= pos_tolerance_ik and
                abs(z_after_ik - z) <= pos_tolerance_ik):

                solutions.append((theta1_pred, theta2_pred, theta3_pred))
    return solutions


def move_servo(servo_id, angle_deg, ser):
    ser.reset_input_buffer()
    command = f"{servo_id} {angle_deg}\n"
    ser.write(command.encode('utf-8'))

def print_stuff(solutions, x, y, z): 
    if not solutions: 
        raise ValueError("SHIT no solution found!")
    
        
    print(f"Found {len(solutions)} combinations of angles for x = {x}, y = {y}, z = {z} \n")

    print("Program assumes finds 2 solutions...\n")

    if len(solutions) < 2: 
        raise ValueError("Well apperently it wasnt...")

    print(f"\t 1: Theta 1 = {solutions[0][0]}, Theta 2 = {solutions[0][1]}, Theta 3 = {solutions[0][2]}")
    print(f"\t 2: Theta 1 = {solutions[1][0]}, Theta 2 = {solutions[1][1]}, Theta 3 = {solutions[1][2]}\n")



port = 'COM5'
baud_rate =  9600
# port = input("serial port: ")
# baud_rate = int(input("serial baud_rate "))

workspace_points = np.load("workspace_points.npy")
n_samples = 10  
indices = np.random.choice(len(workspace_points), n_samples, replace=False)
sample_points = workspace_points[indices]

for x, y, z in sample_points:
    print(x, y, z)

_ = input("continue?")

x=0.2
y=0.2
z=0.2

send_position(x, y, z, port, baud_rate)