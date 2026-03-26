import time

def move_servos(robot, ik, convert_to_pwm, position, delay_between_moves, TESTING_NO_ARDUINO):

    x, y, z = position[0], position[1], position[2]
    angles = ik.get_angles(x, y, z) 
    
    if not TESTING_NO_ARDUINO: 
        for i in range(len(angles)): 
            pwm = convert_to_pwm(i, angles[i])
            robot.move_servo(i, pwm)
            time.sleep(delay_between_moves)
    
