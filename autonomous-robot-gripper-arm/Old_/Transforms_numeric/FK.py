import numpy as np

def transform_vector(vector : np.ndarray, angle : np.ndarray, L2=0.315, L3=0.045, L4=0.108, L5=0.005, L6=0.034, L7=0.015, L8=0.088, L9=0.204):
    x3 = vector[0]
    y3 = vector[1]
    z3 = vector[2]


    theta1 = angle[0] 
    theta2 = angle[1]
    theta3 = angle[2]
    
    return np.array([L4*np.sin(theta1) - L5*np.sin(theta1) + L6*np.cos(theta1) + L7*np.cos(theta1)*np.cos(theta2) + L8*np.sin(theta2)*np.cos(theta1) + L9*np.sin(theta2 + theta3)*np.cos(theta1) + x3*np.cos(theta1)*np.cos(theta2 + theta3) - y3*np.sin(theta2 + theta3)*np.cos(theta1) + z3*np.sin(theta1), -L4*np.cos(theta1) + L5*np.cos(theta1) + L6*np.sin(theta1) + L7*np.sin(theta1)*np.cos(theta2) + L8*np.sin(theta1)*np.sin(theta2) + L9*np.sin(theta1)*np.sin(theta2 + theta3) + x3*np.sin(theta1)*np.cos(theta2 + theta3) - y3*np.sin(theta1)*np.sin(theta2 + theta3) - z3*np.cos(theta1), L2 + L3 + L7*np.sin(theta2) - L8*np.cos(theta2) - L9*np.cos(theta2 + theta3) + x3*np.sin(theta2 + theta3) + y3*np.cos(theta2 + theta3)])
