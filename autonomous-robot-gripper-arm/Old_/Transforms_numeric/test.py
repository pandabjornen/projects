import numpy as np


def transform_vector(v, theta1, theta2, theta3, L2=0.315, L3=0.045, L4=0.108, L5=0.005, L6=0.034, L7=0.015, L8=0.088, L9=0.204):
    T1 = np.array([
        [np.cos(theta1), -np.sin(theta1), 0, 0],
        [np.sin(theta1), np.cos(theta1), 0, 0],
        [0, 0, 1,  0],
        [0, 0, 0, 1]
    ])

    T12 = np.array([
        [1, 0, 0, L6],
        [0, 1, 0, -L4],
        [0, 0, 1,  L2 + L3],
        [0, 0, 0, 1]
    ])

    T2 = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])


    T3 = np.array([
        [np.cos(theta2), -np.sin(theta2), 0,0],
        [np.sin(theta2), np.cos(theta2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T32 = np.array([
        [1, 0, 0, L7],
        [0, 1, 0, -L8],
        [0, 0, 1, -L5],
        [0, 0, 0, 1]
    ])
 


    T4 = np.array([
        [np.cos(theta3), -np.sin(theta3), 0, 0],
        [np.sin(theta3), np.cos(theta3), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


    T42 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -L9],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


    # Combineing all transformations
    T = T1 @ T12 @ T2 @ T3 @ T32 @ T4 @ T42

    # make sure the input vector has homogeneous coordinates
    if len(v) == 3:
        v_h = np.array([v[0], v[1], v[2], 1])
    else:
        v_h = np.array(v)

    # Multiply
    result = T @ v_h

    res2 = np.array([[np.cos(theta1)*np.cos(theta2 + theta3), -np.sin(theta2 + theta3)*np.cos(theta1), np.sin(theta1), L4*np.sin(theta1) - L5*np.sin(theta1) + L6*np.cos(theta1) + L7*np.cos(theta1)*np.cos(theta2) + L8*np.sin(theta2)*np.cos(theta1) + L9*np.sin(theta2 + theta3)*np.cos(theta1)], [np.sin(theta1)*np.cos(theta2 + theta3), -np.sin(theta1)*np.sin(theta2 + theta3), -np.cos(theta1), -L4*np.cos(theta1) + L5*np.cos(theta1) + L6*np.sin(theta1) + L7*np.sin(theta1)*np.cos(theta2) + L8*np.sin(theta1)*np.sin(theta2) + L9*np.sin(theta1)*np.sin(theta2 + theta3)], [np.sin(theta2 + theta3), np.cos(theta2 + theta3), 0, L2 + L3 + L7*np.sin(theta2) - L8*np.cos(theta2) - L9*np.cos(theta2 + theta3)], [0, 0, 0, 1]]) @ v_h

    print(result)
    print(res2)
    x3 = v_h[0]
    y3 = v_h[1]
    z3 = v_h[2]
    print(np.array([L4*np.sin(theta1) - L5*np.sin(theta1) + L6*np.cos(theta1) + L7*np.cos(theta1)*np.cos(theta2) + L8*np.sin(theta2)*np.cos(theta1) + L9*np.sin(theta2 + theta3)*np.cos(theta1) + x3*np.cos(theta1)*np.cos(theta2 + theta3) - y3*np.sin(theta2 + theta3)*np.cos(theta1) + z3*np.sin(theta1), -L4*np.cos(theta1) + L5*np.cos(theta1) + L6*np.sin(theta1) + L7*np.sin(theta1)*np.cos(theta2) + L8*np.sin(theta1)*np.sin(theta2) + L9*np.sin(theta1)*np.sin(theta2 + theta3) + x3*np.sin(theta1)*np.cos(theta2 + theta3) - y3*np.sin(theta1)*np.sin(theta2 + theta3) - z3*np.cos(theta1), L2 + L3 + L7*np.sin(theta2) - L8*np.cos(theta2) - L9*np.cos(theta2 + theta3) + x3*np.sin(theta2 + theta3) + y3*np.cos(theta2 + theta3), 1]))


    




transform_vector(np.array([1,0,2]), 1,-4,2)