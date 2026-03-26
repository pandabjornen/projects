import numpy as np
import matplotlib.pyplot as plt

def get_all_servo_coefficients():

    base_servo_dict = { #NOTE: WRONG -90°?  # Ranges: [-torch.pi/2, torch.pi/4] = [-90°, 45°]
        "angles (deg)": [-45.0, 0, 45.0, 90],
        "PWM (ms)": [2000, 1550, 1040.0, 600]
    }

    # shoulder_dict = {
    #     "angles (deg)": [-25, 0, 25],  
    #     "PWM (ms)": [1500, 1250, 1100]
    # }
    shoulder_dict = { # Ranges: [torch.pi/12, torch.pi/4] = [15°, 45°]
        "angles (deg)": [ 0, 15, 20, 25,30, 40, 44],   # got 36 degrees once for 1050 ms. kinda depends from which way you go to a position
        "PWM (ms)": [1275, 1190, 1160, 1120,1100, 1050, 1025]
    }



    # elbow_dict = {
    #     "angles (deg)": [-90, 0, 25],
    #     "PWM (ms)": [1810, 920, 700]
    # }

    elbow_dict = { # RANGE:  [-torch.pi/12, torch.pi/4 = -15° , 45°]
         "angles (deg)": [-16, 0, 30, 45], # 45 ≈ 44-47
         "PWM (ms)": [2250, 2050, 1700, 1550]
     }


    camera_servo_dict = { # RANGE: [-torch.pi/2, -torch.pi/8] = [- 90° , - 22.5°]
        "angles (deg)": [-90, -45, 0],  # pwm for 0° seems to correct. Also for -90°. Dont remember if we did the calibration with the full hand on or not anyway seems to be ok. 
                                        # big error for -25° : (°, pwm) = (-25°, 1200)
        "PWM (ms)": [1690, 1375,1025]
    }

    
    servo_configs = [
        (0, base_servo_dict),
        (1, shoulder_dict),
        (2, elbow_dict),
        (3, camera_servo_dict)
    ]

    coeffs_dict = {}
    degree = 1  

    for servo_id, data in servo_configs:
        angles_key = "angles (deg)" 
        angles = np.array(data[angles_key])
        pwms = np.array(data["PWM (ms)"])
        
        coeffs = np.polyfit(angles, pwms, degree)
        coeffs_dict[servo_id] = coeffs.tolist()  # Lista: [slope, intercept]

    return coeffs_dict


def convert_to_pwm(Servo_ID, angle_radians) -> float:
    angle_deg = np.degrees(angle_radians)
    
    coeffs_dict = get_all_servo_coefficients()
    
    if Servo_ID not in coeffs_dict:
        raise ValueError(f"Unknown Servo_ID: {Servo_ID}")
    
    slope, intercept = coeffs_dict[Servo_ID]
    pwm = slope * angle_deg + intercept
    
    return pwm

if __name__ == "__main__":
    coeffs = get_all_servo_coefficients()
    for sid, (slope, intercept) in coeffs.items():
        x = np.linspace(-90, 90, 100)
        y = slope * x + intercept
        plt.plot(x, y, label=f"Servo {sid}")
    plt.xlabel("Angle (deg)")
    plt.ylabel("PWM (ms)")
    plt.legend()
    plt.show()