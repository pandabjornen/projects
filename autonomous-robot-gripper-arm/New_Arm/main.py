from dataclasses import dataclass
import numpy as np, time
from robot_system import RobotSystem
from Perception.Object_Detection.classify_yolo import classify
from Verbose_Functions.testing_utils import visualize_frame
#NOTE: end_point = torch.tensor([0.16, -0.01, -0.02609], device=device)

@dataclass
class Params:
    prompt: list
    init_angles: np.ndarray
    camera_device_index: int
    verbose: bool
    testing_no_arduino: bool
    nr_pictures_when_searching: int
    z_top_object: float
    delay_between_moves: float
    delta_z_object: float
    hand_to_obj_z_grip: float
    step_size_to_object_z: float
    pwm_gripping_delta: int
    delay_gripping: float
    drop_off_point: list
    box_mode_align: str
    box_mode_search_move: str
    init_angles_sleep: float
    z_ascend_above_object: float
    nr_pictures_align : int
    pwm_releasing_shorter_time_than_grip : float
    pwm_release_delta: float
params = Params(

    ####### NOT USED BUT TO LAZY TO REMOVE ###########################################
    nr_pictures_align = 60, 
    testing_no_arduino=False,    #used to test function with no arduino

    ######################################################################################E 
    ####### 1. Init Robot System ###########################################
    init_angles_sleep = 3, # (s)
    camera_device_index=0, # 0 or 1 for computer or web camera 
    verbose=True,   # Show frames used

    #AUTO_DETECT = False: 
    #delay_between_moves=0.5, # (s) delay between sending commands to different servos to arduino for one new pose. 
    #AUTO_DETECT = True: 
    delay_between_moves=0.5,

    init_angles=np.array([np.deg2rad(0.0), np.deg2rad(45.0),np.deg2rad(25.0),np.deg2rad(-100.0)]),    
    
    z_top_object=0.075, # (m) tennis ball on top of tape roll
    delta_z_object=0.075, # (m) meters above object

    ####### 2. Search Objects ############################################
    #AUTO_DETECT = False: 
    #prompt = ["yellow-green tennis ball on white roll of tape"], 
    #AUTO_DETECT = True: 
    prompt = ["yellow-green ball"],  
    
    box_mode_search_move = "center",  #bottom or center 
    box_mode_align = "center",  #bottom or center 

    nr_pictures_when_searching=10, # When using verbose and when serching for object before arm moved towards object
    
    ####### 5. Step Down Until Gripping Height #################################

    #AUTO_DETECT = False: 
    step_size_to_object_z=0.02, #(m) steps per iteration in SECTION 5
    hand_to_obj_z_grip=0.02, # (m) meters to object before starting to grip
    #AUTO_DETECT = True: 
    #step_size_to_object_z=0.01, #(m) steps per iteration in SECTION 5
    #hand_to_obj_z_grip=0.02, # (m) meters to object before starting to grip

    ####### 6. Gripping #################################
    pwm_gripping_delta=150, # (ms) this will be subtracted to 1500 ms when gripping and added when releasing
    pwm_release_delta = 100, 
    delay_gripping=12,
    pwm_releasing_shorter_time_than_grip = 7.0,
    # (s) time PWM of gripper servo ≠ 1500 (neutral)

    ####### 7. Step up #################################
    z_ascend_above_object = 0.15, # step up so moves at a higher horizontal plane, same step as step down

    ####### 8. Move To Drop-Off Point ############################################
    drop_off_point=[0.1, 0.175, 0.15],  # [x, y, z] (m)

)   

AUTO_DETECT = True
AUTO_PAN_SEARCH = True


HARDCODED_PAN = -50
HARDCODE_SEND_ANGLE_1 = -20


# old offsets:  # object_pos[0] -= 0.03
    # object_pos[1] += 0.01
RADIUS_OFFSET = -0.025 # += this value

Y_PIXEL_FROM_EDGE = 100
X_PIXEL_FROM_EDGE = 300
####### 1. Init Robot System ###########################################
print(f"\nInitializing Robot System... (Z_TOP_OBJECT = {params.z_top_object} m)\n")
robot_system = RobotSystem(params)


####### 2. Search Objects ############################################

if AUTO_DETECT: 

    if AUTO_PAN_SEARCH:

        ########################2.1 Pan search (rotate base servo)#################################### 
        detections = []
        angle_0_deg = 0.0
        obj_detect_model = robot_system.model

        # if not at 0° move to 0°
        if angle_0_deg > 10: # small offset since baseplate sucks
            angles = params.init_angles
            angles[0] = 0.0
            robot_system._send_angles(angles, params.delay_between_moves)
        else: 
            angle_0_deg = np.rad2deg(params.init_angles[0])
            angles = params.init_angles
        
        print("\nBASE SERVO ANGLE: ", angle_0_deg)
        while not detections or angle_0_deg > -45.0: 
            frame = robot_system.camera.get_frame()
            for i in range(params.nr_pictures_when_searching): 
                detections, _ = classify(frame, obj_detect_model)
                # if params.verbose:
                #     _ = visualize_frame(frame, detections, i + 1, params.nr_pictures_when_searching, params.prompt, params.z_top_object)
                #     time.sleep(2.0)
                time.sleep(2.0)
                if detections: 
                    print("found object")
                    break
            if detections:

                if params.box_mode_search_move == "center": 
                    x_px, y_px = robot_system._get_detection_box_center(detections[0])
                elif params.box_mode_search_move == "bottom": 
                    x_px, y_px = robot_system._get_detection_box_bottom_middle(detections[0])
                else: 
                    raise ValueError("param box mode needs to be 'center' or 'bottom'. ")
                
                print(f"\n Detected object at x_px: {x_px}, y_px: {y_px} ")

                if (Y_PIXEL_FROM_EDGE < y_px < 720 - Y_PIXEL_FROM_EDGE) and \
                    (X_PIXEL_FROM_EDGE < x_px < 1280 - X_PIXEL_FROM_EDGE):
                    print("Object within center region.")
                    break
                else:
                    print("Object outside center region.")
                    angle_0_deg -= 10
                    angles[0] = np.deg2rad(angle_0_deg)
                    robot_system._send_angles(angles, params.delay_between_moves)
                    while not robot_system.interface.get_data("armAtTarget"):
                        time.sleep(0.25) 
                    

            else:
                angle_0_deg -= 10
                angles[0] = np.deg2rad(angle_0_deg)
                robot_system._send_angles(angles, params.delay_between_moves)
                while not robot_system.interface.get_data("armAtTarget"):
                    time.sleep(0.25)
                print("\n Could not detect object")
                print("\nBASE SERVO ANGLE: ", angle_0_deg)
        
    else: 
        while not robot_system.interface.get_data("armAtTarget"):
            time.sleep(0.25)

        angles = params.init_angles
        angles[0] = np.deg2rad(HARDCODED_PAN)
        robot_system._send_angles(angles, params.delay_between_moves)

        while not robot_system.interface.get_data("armAtTarget"):
            time.sleep(0.25)
    ##########################################################################################
    

    #################### 2.2 RAY METHOD SERACH ########################################
    object_pos = robot_system.search_object()
    print(f"\nObject found at: {object_pos}\n")

    #Manual offsets: 
    # object_pos[0] -= 0.03
    # object_pos[1] += 0.01

    x = object_pos[0]
    y = object_pos[1]

    r = np.sqrt(x**2 + y**2)
    r += RADIUS_OFFSET


    theta = np.arctan2(y, x)
    # theta = angles[0]
    print(f"\n angle from stored angles: {angles[0]}")
    print(f"\n angle from x, y : {theta}")

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    object_pos[0] = x
    object_pos[1] = y



    ################################################################################
else: 

    ####### Hard coded positions:  ############################################

    ### Point 1  (written on table)#####
    obj_x = 0.14  # 0.15
    obj_y = -0.14 #  -0.12

    # obj_x = 0.15  # 0.15
    # obj_y = -0.12 #  -0.12
    #obj_z = 0.05 ## NOTE: remmeber addition of params.z_top_object

    obj_z = params.z_top_object
    #####################################################

    ### Point 2 (written on table)#####
    # obj_x = 0.11 
    # obj_y = -0.05
    # obj_z = params.z_top_object
    #####################################################


    object_pos = np.array([obj_x, obj_y, obj_z])

    # [OPTIOINAL] Pan up to not hit tennis ball: 

    angles = params.init_angles
    angles[2] += np.deg2rad(HARDCODE_SEND_ANGLE_1)
    robot_system._send_angles(angles, params.delay_between_moves)
while not robot_system.interface.get_data("armAtTarget"):
    time.sleep(0.1)


####### 3. Move Above Object ############################################
hand_pos = object_pos.copy()
hand_pos[2] += params.delta_z_object
robot_system.move_servos(hand_pos, params.delay_between_moves)

print(f"\nMoved above object at {hand_pos}\n")

while not robot_system.interface.get_data("armAtTarget"):
    time.sleep(0.1)

####### 5. Step Down Until Gripping Height #################################
while abs(hand_pos[2] - object_pos[2]) > params.hand_to_obj_z_grip:
    hand_pos[2] -= params.step_size_to_object_z
    robot_system.move_servos(hand_pos, params.delay_between_moves)
    if params.verbose:
        print(f"[STEP-DOWN] Hand now at {hand_pos}")

# if AUTO_DETECT:     
#     input("continue")
###### 6. Grip Object #######################################################
robot_system.grip_or_release(grip=True)


#TODO: Implement feedback from force sensors to replace hardcoded variant. 
#We can keep the line robot_system.grip_or_release(grip=True) and then just add
# feedback from sensors while lifting up the object in 7. and moving it in 8. 


####### 7. Step up ############################################
while abs(hand_pos[2] - object_pos[2]) < params.z_ascend_above_object:
    hand_pos[2] += params.step_size_to_object_z
    robot_system.move_servos(hand_pos, params.delay_between_moves)
    if params.verbose:
        print(f"[STEP-UP] Hand now at {hand_pos}")


####### 8. Move To Drop-Off Point ############################################
print(f"\nMoving to drop-off point {params.drop_off_point}\n")
robot_system.move_servos(params.drop_off_point, params.delay_between_moves)

while not robot_system.interface.get_data("armAtTarget"):
    time.sleep(0.1)
####### 9. Release Object #######################################################
robot_system.grip_or_release(grip=False)


####### 9.5 Step up ############################################
while abs(hand_pos[2] - object_pos[2]) < params.z_ascend_above_object:
    hand_pos[2] += params.step_size_to_object_z
    robot_system.move_servos(hand_pos, params.delay_between_moves)
    if params.verbose:
        print(f"[STEP-UP] Hand now at {hand_pos}")

print("\nObject placed. Task complete.\n")
####### 10. Return to Init #######################################################
robot_system.init_position()
