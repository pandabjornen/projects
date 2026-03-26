from types import SimpleNamespace
from robot_system_const_r import RobotSystemConstR
import numpy as np
params = SimpleNamespace(
    prompt=["Roll of tape"],
    init_angles=np.array([np.deg2rad(0.0), np.deg2rad(40.0),np.deg2rad(30.0),np.deg2rad(-60.0)]), 
    camera_device_index=0,
    verbose=True,
    testing_no_arduino=False,
    nr_pictures_when_searching=10,
    z_top_object=0.1,
    object_radius_m=0.3,
    tol=1e-3,
    step=0.01,
    delay_between_moves=0.1,
    max_iterations=10,
    init_angles_sleep=5,
    pwm_gripping_delta=200,
    delay_gripping=0.5
)

robot = RobotSystemConstR(params)
pos = robot.search_object()
print("Found object at:", pos)
