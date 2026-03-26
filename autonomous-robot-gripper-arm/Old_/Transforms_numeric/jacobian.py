import sympy as sp

# symbols
theta1, theta2, theta3 = sp.symbols("theta1 theta2 theta3")
L2, L3, L4, L5, L6, L7, L8, L9 = sp.symbols("L2 L3 L4 L5 L6 L7 L8 L9")
x3, y3, z3 = sp.symbols("x3 y3 z3")

# matrices
T1 = sp.Matrix([
    [sp.cos(theta1), -sp.sin(theta1), 0, 0],
    [sp.sin(theta1),  sp.cos(theta1), 0, 0],
    [0,               0,              1, 0],
    [0,               0,              0, 1]
])

T12 = sp.Matrix([
    [1, 0, 0, L6],
    [0, 1, 0, -L4],
    [0, 0, 1, L2 + L3],
    [0, 0, 0, 1]
])

T2 = sp.Matrix([
    [1, 0,  0, 0],
    [0, 0, -1, 0],
    [0, 1,  0, 0],
    [0, 0,  0, 1]
])

T3 = sp.Matrix([
    [sp.cos(theta2), -sp.sin(theta2), 0, 0],
    [sp.sin(theta2),  sp.cos(theta2), 0, 0],
    [0,               0,              1, 0],
    [0,               0,              0, 1]
])

T32 = sp.Matrix([
    [1, 0, 0, L7],
    [0, 1, 0, -L8],
    [0, 0, 1, -L5],
    [0, 0, 0, 1]
])

T4 = sp.Matrix([
    [sp.cos(theta3), -sp.sin(theta3), 0, 0],
    [sp.sin(theta3),  sp.cos(theta3), 0, 0],
    [0,               0,              1, 0],
    [0,               0,              0, 1]
])

T42 = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, -L9],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Full transformation
T = T1 * T12 * T2 * T3 * T32 * T4 * T42
T = sp.simplify(T)


# Apply to point
p3 = sp.Matrix([x3, y3, z3, 1])
p_final = T * p3

print(p_final)

J = p_final[:3, :].jacobian([theta1, theta2, theta3])
print("--")
print(J)

#np.array([[cos(theta1)*cos(theta2 + theta3), -sin(theta2 + theta3)*cos(theta1), sin(theta1), L4*sin(theta1) - L5*sin(theta1) + L6*cos(theta1) + L7*cos(theta1)*cos(theta2) + L8*sin(theta2)*cos(theta1) + L9*sin(theta2 + theta3)*cos(theta1)], [sin(theta1)*cos(theta2 + theta3), -sin(theta1)*sin(theta2 + theta3), -cos(theta1), -L4*cos(theta1) + L5*cos(theta1) + L6*sin(theta1) + L7*sin(theta1)*cos(theta2) + L8*sin(theta1)*sin(theta2) + L9*sin(theta1)*sin(theta2 + theta3)], [sin(theta2 + theta3), cos(theta2 + theta3), 0, L2 + L3 + L7*sin(theta2) - L8*cos(theta2) - L9*cos(theta2 + theta3)], [0, 0, 0, 1]])