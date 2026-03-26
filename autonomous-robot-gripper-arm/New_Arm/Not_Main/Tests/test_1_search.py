from dataclasses import dataclass
import numpy as np, time
from robot_system import RobotSystem


@dataclass
class Params:
    prompt: list
    init_angles: np.ndarray
    camera_device_index: int
    verbose: bool
    testing_no_arduino: bool
    nr_pictures_when_searching: int
    z_top_object: float
    tol: float
    step: float
    delay_between_moves: float
    max_iterations: int
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
    
params = Params(
    
    # Detection
    # prompt=["Tennis ball"], 
     
    # prompt = ["Roll of tape"], 
    prompt = ["tennis ball"], 
     
    box_mode_search_move = "bottom",  #bottom or center 
    box_mode_align = "center",  #bottom or center 

    init_angles_sleep = 3, # (s)
    delay_between_moves=1, # (s) delay between sending commands to different servos to arduino for one new pose. 

    camera_device_index=0, # 0 or 1 for computer or web camera 

    verbose=True,   # Show frames used
    testing_no_arduino=False,    #used to test function with no arduino

    nr_pictures_when_searching=10, # When using verbose and when serching for object before arm moved towards object
    nr_pictures_align = 60, 

    init_angles=np.array([np.deg2rad(0.0), np.deg2rad(40.0),np.deg2rad(30.0),np.deg2rad(-90.0)]),    
    # z_top_object=0.1, # (m) NOTE: MEASURE the top of the objec
    z_top_object=0.075, # (m) tennis ball on top of tape roll
                        # NOTE: prob wrong because measured from table not servo. 
    delta_z_object=0.08, # (m) meters above object

    ####### 4. Align Hand To Object ############################################
    tol=0.01, # (m) dx or dy between hand_pos and obj_pos to keep aligning
    step=0.001, # (m) step size in to minimize dx or dy. 
    max_iterations=1_000, # for align function
    
    ####### 5. Step Down Until Gripping Height #################################
    step_size_to_object_z=0.01, #(m) steps per iteration in SECTION 5
    hand_to_obj_z_grip=0.02, # (m) meters to object before starting to grip

    ####### 6. Gripping #################################
    pwm_gripping_delta=100, # (ms) this will be subtracted to 1500 ms when gripping and added when releasing
    delay_gripping=8, # (s) time PWM of gripper servo ≠ 1500 (neutral)

    ####### 7. Step up #################################
    z_ascend_above_object = 0.15, 

    ####### 8. Move To Drop-Off Point ############################################
    #drop_off_point=[0.15, -0.12, 0.05], # [x, y, z] (m)
    drop_off_point=[0.15, -0.12, 0.20], 
)   


####### 1. Init Robot System ###########################################
print(f"\nInitializing Robot System... (Z_TOP_OBJECT = {params.z_top_object} m)\n")
robot_system = RobotSystem(params)


####### 2. Search Objects ############################################
object_pos = robot_system.search_object()
print(f"\nObject found at: {object_pos}\n")

