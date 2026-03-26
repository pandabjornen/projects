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
    
    # Detection
    # prompt=["Tennis ball"], 
     
    prompt = ["White roll of tape"], 
     
    box_mode_search_move = "center",  #bottom or center 
    box_mode_align = "center",  #bottom or center 

    init_angles_sleep = 3, # (s)
    delay_between_moves=1.5, # (s) delay between sending commands to different servos to arduino for one new pose. 

    camera_device_index=0, # 0 or 1 for computer or web camera 

    verbose=True,   # Show frames used
    testing_no_arduino=False,    #used to test function with no arduino

    nr_pictures_when_searching=10, # When using verbose and when serching for object before arm moved towards object
    nr_pictures_align = 60, 

    init_angles=np.array([np.deg2rad(0.0), np.deg2rad(15.0),np.deg2rad(10.0),np.deg2rad(-90.0)]),    
    # z_top_object=0.1, # (m) NOTE: MEASURE the top of the objec
    # z_top_object=0.075, # (m) tennis ball on top of tape roll
                        # NOTE: prob wrong because measured from table not servo. 
    delta_z_object=0.0, # (m) meters above object
    z_top_object= 0.02, 

    
    
    ####### 5. Step Down Until Gripping Height #################################
    step_size_to_object_z=0.05, #(m) steps per iteration in SECTION 5
    hand_to_obj_z_grip=0.02, # (m) meters to object before starting to grip

    ####### 6. Gripping #################################
    pwm_gripping_delta=150, # (ms) this will be subtracted to 1500 ms when gripping and added when releasing
    pwm_release_delta = 100, 
    delay_gripping=12,
    pwm_releasing_shorter_time_than_grip = 7.0,
     # (s) time PWM of gripper servo ≠ 1500 (neutral)

    ####### 7. Step up #################################
    z_ascend_above_object = 0.15, 

    ####### 8. Move To Drop-Off Point ############################################
    #drop_off_point=[0.15, -0.12, 0.05], # [x, y, z] (m)
    drop_off_point=[0.15, -0.12, 0.20], 
)   

####### 1. Init Robot System ###########################################
print(f"\nInitializing Robot System... (Z_TOP_OBJECT = {params.z_top_object} m)\n")
robot_system = RobotSystem(params)





input("continue")

#TODO: Include Force Sensor feed back to stop early if any reading

####### 6. Grip Object #######################################################
robot_system.grip_or_release(grip=True)

#TODO: Implement feedback from force sensors to replace hardcoded variant. 

input("continue?")

####### 8. Release Object #######################################################
robot_system.grip_or_release(grip=False)

