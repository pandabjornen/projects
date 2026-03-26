import numpy as np
import math
from FK import transform_vector


class Register:

    L2=0.315
    L3=0.045
    L4=0.108
    L5=0.005
    L6=0.034
    L7=0.015
    L8=0.088
    L9=0.204

    constants = np.array([-1, 1, 2, np.e, 3, np.pi, 5, 7, 11, 13, L2+L3, L4, L5, L6, L7, L8, L9], dtype=float)
    input_size = 3
    output_size = 3
    working_memory_size = 4

    variable_range = (0, input_size + output_size + working_memory_size - 1)
    size = input_size + output_size + working_memory_size + constants.size

    def __init__(self, input_array: np.ndarray):
        self.register = np.zeros(Register.input_size + Register.output_size + Register.working_memory_size, dtype=float)
        self.register = np.concatenate([self.register, Register.constants])

        self.register[:3] = input_array

    
    def __getitem__(self, key):
        return self.register[key]
    
    def __setitem__(self, key, value):
        max_index = Register.variable_range[1]
        if key > max_index:
            raise IndexError(f"Index {key} is out of bounds for the variable register)")
        self.register[key] = value


    def get_output(self) -> np.ndarray:
        return self.register[3:6]





# Basic arithmetic functions
def addition(o1, o2):
    return o1 + o2

def subtraction(o1, o2):
    return o1 - o2

def division(o1, o2):
    if o2 == 0:
        raise ValueError("divide by zero")
    return o1 / o2

def multiplication(o1, o2):
    return o1 * o2

# Trigonometric functions (input in radians)
def sine(x, _):
    return math.sin(x)

def cosine(x, _ ):
    return math.cos(x)

def tangent(x, _):
    return math.tan(x)

# Logarithmic functions
def natural_log(x, _):
    return math.log(x)  # ln(x)

def power(o1, o2):
    return math.pow(o1, o2)

def sqrt(x, _):
    if (x < 0):
        raise ValueError(f"sqrt domain error: received {x}, must be >= 0")
    return math.sqrt(x)

def arctan(x, _):
    return math.atan(x)

def arcsin(x, _):
    if ((x < -1) | (x > 1)):
        raise ValueError(f"arcsin domain error: received {x}, must be in [-1, 1]")
    return math.asin(x)

def arccos(x, _):
    if ((x < -1) | (x > 1)):
        raise ValueError(f"arccos domain error: received {x}, must be in [-1, 1]")
    return math.acos(x)



operators = [
    addition,
    subtraction,
    division,
    multiplication,
    sine,
    cosine,
    tangent,
    natural_log,
    power,
    arctan,
    arcsin,
    arccos,
    sqrt
]




def run_lgp(register : Register, operators: list, genes: np.ndarray):

    genes = genes.astype(int) 

    if genes.ndim != 2 or genes.shape[1] != 4:
        raise ValueError(f"genes must have shape (n,4), got {genes.shape}")

    for op_idx, dst_idx, src_a_idx, src_b_idx in genes:
        register[dst_idx] = operators[op_idx](
            register[src_a_idx], register[src_b_idx]
        )

    return register


def delta_theta(theta1, theta2, theta3, target, current_pos, alpha, L2=0.315, L3=0.045, L4=0.108, L5=0.005, L6=0.034, L7=0.015, L8=0.088, L9=0.204):


    J = np.array([
        [
            L4*np.cos(theta1) - L5*np.cos(theta1) - L6*np.sin(theta1)
            - L7*np.sin(theta1)*np.cos(theta2) - L8*np.sin(theta1)*np.sin(theta2)
            - L9*np.sin(theta1)*np.sin(theta2 + theta3),

            -L7*np.sin(theta2)*np.cos(theta1)
            + L8*np.cos(theta1)*np.cos(theta2)
            + L9*np.cos(theta1)*np.cos(theta2 + theta3),

            L9*np.cos(theta1)*np.cos(theta2 + theta3)
        ],

        [
            L4*np.sin(theta1) - L5*np.sin(theta1) + L6*np.cos(theta1)
            + L7*np.cos(theta1)*np.cos(theta2) + L8*np.sin(theta2)*np.cos(theta1)
            + L9*np.sin(theta2 + theta3)*np.cos(theta1),

            -L7*np.sin(theta1)*np.sin(theta2)
            + L8*np.sin(theta1)*np.cos(theta2)
            + L9*np.sin(theta1)*np.cos(theta2 + theta3),

            L9*np.sin(theta1)*np.cos(theta2 + theta3)
        ],

        [
            0,
            L7*np.cos(theta2) + L8*np.sin(theta2) + L9*np.sin(theta2 + theta3),
            L9*np.sin(theta2 + theta3)
        ]
    ])

    error = target - current_pos

    delta = alpha * J.T @ error
    return delta

def distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def fk(a):
    return transform_vector(np.array([0,0,0]), np.array(a))

def ik(point : np.ndarray, chromosome, tolaracne = 0.01):
    register = Register(point)
    run_lgp(register, operators, chromosome)
    
    initial_angle_guess = register.get_output()
    initial_point = fk(initial_angle_guess)

    dist = distance(point, initial_point)

    if (dist < tolaracne):
        return initial_angle_guess, dist

    current_angle = initial_angle_guess
    current_point = initial_point
    for x in range(50000):
        delta = delta_theta(current_angle[0], current_angle[1], current_angle[2], point, current_point, 0.1)
        current_angle += delta
        current_point = fk(current_angle)
        dist = distance(point, current_point)
        #if (dist < tolaracne):
        #    return current_angle, dist
    
    return current_angle, dist
    raise RuntimeError(f"Solution not found after {10000} iterations")



chromosome = np.load("best_individual.npy", allow_pickle=True)

angles, dist = ik(np.array([0.2, 0.2, 0.2]), chromosome, 0.02)
print(np.rad2deg(angles))
print(dist)

exit()

















servo_1_range = [-np.pi/2, np.pi/2]
servo_2_range = [0, np.pi]
servo_3_range = [0, np.pi/2]


# Resolution
steps = 15 

# Angle grids
theta1 = np.linspace(servo_1_range[0], servo_1_range[1], steps)
theta2 = np.linspace(servo_2_range[0], servo_2_range[1], steps)
theta3 = np.linspace(servo_3_range[0], servo_3_range[1], steps)


T1, T2, T3 = np.meshgrid(theta1, theta2, theta3, indexing="ij")

angles = np.vstack([T1.ravel(), T2.ravel(), T3.ravel()]).T
points = np.array([transform_vector(np.array([0,0,0]), a) for a in angles])
print(points.shape[0])

error_count = 1

d = np.zeros(steps*steps*steps, dtype=float)

chromosome = np.load("best_individual.npy", allow_pickle=True)

for i,point in enumerate(points):
    print(i)
    
    _, d[i] = ik(point, chromosome)



import matplotlib.pyplot as plt

plt.hist(d, bins=5, edgecolor='black', weights=np.ones_like(d)/len(d)*100)
plt.ylabel('Percentage')
plt.xlabel('Value')
plt.title('Histogram')
plt.show()