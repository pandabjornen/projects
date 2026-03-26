import numpy as np
import sys
import os


L1 = 0.055
L2 = 0.315
L3 = 0.045
L4 = 0.108
L5 = 0.005
L6 = 0.034
L7 = 0.015
L8 = 0.088
L9 = 0.204

Ls = [0,0,L3, L3, L4, L5, L6, L7, L8, L9]


def ik_hubert(x0, y0, z0) -> tuple[list[tuple[float, float]], list[float]]:
    
    z1 = -L5
    r0 = np.sqrt(x0**2 + y0**2)
    x1 = np.sqrt(r0**2 - (L4+z1)**2) -L6 # NOTE: assume x1 > 0 : mathematically ± np.sqrt(r0**2 - (L4+z1)**2) -L6,
    y1 = z0 - L2 - L3
    
   
    # print("computing th2...")
    theta3s_rad = compute_theta3(x1, y1)

    # print("computing th3...")
    solutions_theta2_and_theta3_degrees = compute_theta2(theta3s_rad, x1, Ls)    
    # print("computing th1...")
    solutions_theta1_degrees = compute_theta1(x0, y0, x1, z1)

    
    return solutions_theta2_and_theta3_degrees, solutions_theta1_degrees


def compute_theta3(x1, y1): 
    u3 = Ls[8]
    v3 = Ls[7]
    w3 = (x1**2 + y1**2 - Ls[7]**2  - Ls[8]**2  - Ls[9]**2)/(2*Ls[9])
    theta3s_rad = linear_combination_sin_cos(u3, v3, w3)
    return theta3s_rad

def compute_theta2(theta3s_rad, x1,Ls): 
    solutions_degrees = []
    for theta3 in theta3s_rad: 
        u2 = Ls[7] + Ls[9]*np.sin(theta3)
        v2 = Ls[8] + Ls[9]*np.cos(theta3)
        w2 = x1
        theta2s_rad = linear_combination_sin_cos(u2, v2, w2)

        for theta2 in theta2s_rad: 
            
            t2_deg = np.degrees(theta2) % 360
            t3_deg = np.degrees(theta3) % 360
            solutions_degrees.append((t2_deg, t3_deg))
    return solutions_degrees

def compute_theta1(x0, y0, x1, z1):

    u1 = x1 +L6
    v1 = L4 + z1
    w1 = x0
    theta1s_rad = linear_combination_sin_cos(u1, v1, w1)
    solutions_theta1_degrees = []
    for theta1_rad in theta1s_rad:
        t1_deg = np.degrees(theta1_rad) % 360
        solutions_theta1_degrees.append(t1_deg)
    return solutions_theta1_degrees


def linear_combination_sin_cos(a, b, c):
    """
    Solves the equation a*cos(theta) + b*sin(theta) = c for theta.
    """
    R = np.sqrt(a**2 + b**2)
    
    
    if abs(c / R) > 1:
        raise ValueError("arccos arg has abs >1")
        # print('errror')
        # return []
         
    alpha = np.arctan2(b, a)
    beta = np.arccos(c / R)

    return [alpha + beta, alpha - beta]




