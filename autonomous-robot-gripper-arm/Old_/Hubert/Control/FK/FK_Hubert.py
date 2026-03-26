import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fk_hubert(theta_1, theta_2, theta_3): 
    

    L1 = 0.055
    L2 = 0.315
    L3 = 0.045
    L4 = 0.108
    L5 = 0.005
    L6 = 0.034
    L7 = 0.015
    L8 = 0.088
    L9 = 0.204
    theta_1 = np.deg2rad(theta_1)
    theta_2 = np.deg2rad(theta_2)
    theta_3 = np.deg2rad(theta_3)

    s1 = np.sin(theta_1)
    s2 = np.sin(theta_2)
    s3 = np.sin(theta_3)
    c1 = np.cos(theta_1)
    c2 = np.cos(theta_2)
    c3 = np.cos(theta_3)

    x = L4*s1 - L5*s1 + L6*c1 + L9*c1*c2*s3 + L9*c1*c3*s2 + c1*(L7*c2 + L8*s2)
    y = -L4*c1 + L5*c1 + L6*s1 + L9*c2*s1*s3 + L9*c3*s1*s2 + s1*(L7*c2 + L8*s2)
    z = L2 + L3 + L7*s2 - L8*c2 - L9*c2*c3 + L9*s2*s3    

    return x, y, z
