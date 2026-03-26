
import numpy as np
import torch



def arm_end_to_base_transform_matrix(a, b, c, d) -> np.ndarray:
    """
    angles in radians
    """
    L1x, L1y, L1z = 0.01031, -0.0321, 0.07324
    L2x, L2y, L2z = 0.0427, 0.19096, -0.05834
    L3x, L3y, L3z = 0.22335, 0.01031, -0.05834



    ca, sa = np.cos(a), np.sin(a)
    cbcd = np.cos(b - c + d)
    sbcd = np.sin(b - c + d)
    cb = np.cos(b)
    sb = np.sin(b)
    cbc = np.cos(b - c)
    sbc = np.sin(b - c)

    T = np.array([
        [
            ca * cbcd,
            -sbcd * ca,
            sa,
            L1x*ca - L1y*sa
            + L2x*ca*cb - L2y*sb*ca + L2z*sa
            + L3x*ca*cbc + L3y*ca*sbc - L3z*sa
        ],
        [
            sa * cbcd,
            -sa * sbcd,
            -ca,
            L1x*sa + L1y*ca
            + L2x*sa*cb - L2y*sa*sb - L2z*ca
            + L3x*sa*cbc + L3y*sa*sbc + L3z*ca
        ],
        [
            sbcd,
            cbcd,
            0,
            L1z + L2x*sb + L2y*cb + L3x*sbc - L3y*cbc
        ],
        [0, 0, 0, 1]
    ])
    return T


def camera_to_arm_end_transform_matrix():
    
    Lx, Ly, Lz = 0.04365, 0.1275, -0.02609

    T = np.array([
        [0, 0, 1, Lx],
        [0, -1, 0, Ly],
        [1, 0, 0, Lz],
        [0, 0, 0, 1]
    ])

    return T



def arm_transform_points_torch(points : torch.Tensor, angles :  torch.Tensor) -> torch.Tensor:

    # Ensure points is 2D
    if points.ndim == 1:
        points = points.unsqueeze(0)

    a = angles[:,0]
    b = angles[:,1]
    c = angles[:,2]
    d = angles[:,3]

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    L1x, L1y, L1z = 0.01031, -0.0321, 0.07324
    L2x, L2y, L2z = 0.0427, 0.19096, -0.05834
    L3x, L3y, L3z = 0.22335, 0.01031, -0.05834


    cos_a = torch.cos(a)
    sin_a = torch.sin(a)
    cos_b = torch.cos(b)
    sin_b = torch.sin(b)
    cos_b_c = torch.cos(b - c)
    sin_b_c = torch.sin(b - c)
    cos_b_c_d = torch.cos(b - c + d)
    sin_b_c_d = torch.sin(b - c + d)

    new_x = L1x * cos_a - L1y * sin_a + L2x * cos_a * cos_b - L2y * sin_b * cos_a + L2z * sin_a + L3x * cos_a * cos_b_c + L3y * sin_b_c * cos_a - L3z * sin_a + x * cos_a * cos_b_c_d - y * sin_b_c_d * cos_a + z * sin_a

    new_y = L1x * sin_a + L1y * cos_a + L2x * sin_a * cos_b - L2y * sin_a * sin_b - L2z * cos_a + L3x * sin_a * cos_b_c + L3y * sin_a * sin_b_c + L3z * cos_a + x * sin_a * cos_b_c_d - y * sin_a * sin_b_c_d - z * cos_a

    new_z = L1z + L2x * sin_b + L2y * cos_b + L3x * sin_b_c - L3y * cos_b_c + x * sin_b_c_d + y * cos_b_c_d

    return torch.stack([new_x, new_y, new_z], axis=1)



def arm_transform_points_numpy(points : np.ndarray, angles : np.ndarray):
    #Example:
    #points = np.array([[0,0,0],[1,0,1]])
    #angles = np.array([[0,0,0,0], [1.8, 0, 1.5, 2]])

    a = angles[:,0]
    b = angles[:,1]
    c = angles[:,2]
    d = angles[:,3]

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    L1x, L1y, L1z = 0.01031, -0.0321, 0.07324
    L2x, L2y, L2z = 0.0427, 0.19096, -0.05834
    L3x, L3y, L3z = 0.22335, 0.01031, -0.05834



    cos_a = np.cos(a)
    sin_a = np.sin(a)
    cos_b = np.cos(b)
    sin_b = np.sin(b)
    cos_b_c = np.cos(b - c)
    sin_b_c = np.sin(b - c)
    cos_b_c_d = np.cos(b - c + d)
    sin_b_c_d = np.sin(b - c + d)

    new_x = L1x * cos_a - L1y * sin_a + L2x * cos_a * cos_b - L2y * sin_b * cos_a + L2z * sin_a + L3x * cos_a * cos_b_c + L3y * sin_b_c * cos_a - L3z * sin_a + x * cos_a * cos_b_c_d - y * sin_b_c_d * cos_a + z * sin_a

    new_y = L1x * sin_a + L1y * cos_a + L2x * sin_a * cos_b - L2y * sin_a * sin_b - L2z * cos_a + L3x * sin_a * cos_b_c + L3y * sin_a * sin_b_c + L3z * cos_a + x * sin_a * cos_b_c_d - y * sin_a * sin_b_c_d - z * cos_a

    new_z = L1z + L2x * sin_b + L2y * cos_b + L3x * sin_b_c - L3y * cos_b_c + x * sin_b_c_d + y * cos_b_c_d
    
    return np.stack([new_x, new_y, new_z], axis=1)



def arm_calc():

    points = torch.tensor([1,0,0])
    angles = torch.tensor([[0,0,0,-np.pi/2], [0,0, 0,0]])

    answer = arm_transform_points_torch(points, angles)

    T = arm_end_to_base_transform_matrix(0,0,0,-np.pi/2)
    
    print(answer)
    print(T@np.array([1,0,0,1]))


    import sympy as sp

    # Define symbols
    theta1, theta2, theta3, theta4 = sp.symbols('a b c d', real=True)
    x, y, z = sp.symbols('x y z', real=True)

    #Link translations
    L1x, L1y, L1z = sp.symbols('L1x L1y L1z', real=True)
    L2x, L2y, L2z = sp.symbols('L2x L2y L2z', real=True)
    L3x, L3y, L3z = sp.symbols('L3x L3y L3z', real=True)

    # Transformation matrices
    R1_2 = sp.Matrix([
        [sp.cos(theta1), -sp.sin(theta1), 0, 0],
        [sp.sin(theta1),  sp.cos(theta1), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T1_2 = sp.Matrix([
        [1, 0, 0, L1x],
        [0,  1, 0, L1y],
        [0, 0, 1, L1z],
        [0, 0, 0, 1]
    ])

    RC2 = sp.Matrix([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    R2_3 = sp.Matrix([
        [sp.cos(theta2), -sp.sin(theta2), 0, 0],
        [sp.sin(theta2),  sp.cos(theta2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T2_3 = sp.Matrix([
        [1, 0, 0, L2x],
        [0,  1, 0, L2y],
        [0, 0, 1, L2z],
        [0, 0, 0, 1]
    ])


    RC3 = sp.Matrix([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    R3_4 = sp.Matrix([
        [sp.cos(theta3), -sp.sin(theta3), 0, 0],
        [sp.sin(theta3),  sp.cos(theta3), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])



    T3_4 = sp.Matrix([
        [1, 0, 0, L3x],
        [0,  1, 0, L3y],
        [0, 0, 1, L3z],
        [0, 0, 0, 1]
    ])


    RC4 = sp.Matrix([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    R4_end = sp.Matrix([
        [sp.cos(theta4), -sp.sin(theta4), 0, 0],
        [sp.sin(theta4),  sp.cos(theta4), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Input vector
    v = sp.Matrix([x, y, z, 1])

    # Symbolically precompute the transformation
    T_total = sp.simplify(R1_2 * T1_2 * RC2 * R2_3 * T2_3 * RC3 * R3_4 * T3_4 * RC4 * R4_end)

    #print(T_total)

    v_out = sp.simplify(T_total * v)

    # Optionally extract the 3D position only
    v_out_xyz = v_out[:3]

    # Print symbolic expressions
    sp.pprint(T_total)


def camera_calc():
    import sympy as sp
    Lx, Ly, Lz = sp.symbols('Lx Ly Lz', real=True)

    T = sp.Matrix([
        [1, 0, 0, Lx],
        [0,  1, 0, Ly],
        [0, 0, 1, Lz],
        [0, 0, 0, 1]
    ])

    ZC = sp.Matrix([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    YC = sp.Matrix([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    T_total = sp.simplify(T * YC * ZC)

    print(T_total[0][0])


if __name__ == "__main__":
    arm_calc()
    C = camera_to_arm_end_transform_matrix()
    A = arm_end_to_base_transform_matrix(0,0,0,0)

    point = np.array([0,0,0,1])
    
    print(A @ point)
    
