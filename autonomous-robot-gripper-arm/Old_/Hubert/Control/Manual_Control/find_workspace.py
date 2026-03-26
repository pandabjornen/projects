import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FK')))

import numpy as np
import matplotlib.pyplot as plt
from FK_Hubert import fk_hubert

theta1_range = np.arange(0, 181, 1)
theta2_range = np.arange(0, 181, 1)
theta3_range = np.arange(0, 91, 1)

workspace_points = []

for t1 in theta1_range:
    print(f"Theta 1: {t1}/180")
    for t2 in theta2_range:
        for t3 in theta3_range:
            x, y, z = fk_hubert(t1, t2, t3)
            workspace_points.append([x, y, z])

workspace_points = np.array(workspace_points)
np.save("workspace_points.npy", workspace_points)

x_min, y_min, z_min = workspace_points.min(axis=0)
x_max, y_max, z_max = workspace_points.max(axis=0)
limitsbool = input("overwrite limits ? (y/n)").strip().lower() == 'y'
if limitsbool: 
    with open("workspace_limits.txt", "w") as f:
        f.write(f"x_min = {x_min}\n")
        f.write(f"x_max = y{x_max}\n")
        f.write(f"y_min = {y_min}\n")
        f.write(f"y_max = {y_max}\n")
        f.write(f"z_min = {z_min}\n")
        f.write(f"z_max = {z_max}\n")

plotbool = input("plot, maybe laggy... ? (y/n)").strip().lower() == 'y'
if plotbool: 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(workspace_points[:,0], workspace_points[:,1], workspace_points[:,2], s=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


