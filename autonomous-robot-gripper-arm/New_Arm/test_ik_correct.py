
from Control.Kinematics.Ffnn_IK.ik import Ik_hand
from Control.Kinematics.fk import arm_transform_points_numpy, arm_transform_points_torch
from pathlib import Path
import numpy as np

ffnnIK_hand_model_path = Path("Control")/"Kinematics"/"Ffnn_IK"/"best_model_hand.pt"
ik_hand = Ik_hand(ffnnIK_hand_model_path)

a,b,c,d = ik_hand.get_angles(0.1239, -0.1672, 0.0)

print(arm_transform_points_numpy(np.array([[0.16, -0.01, -0.02609]]), np.array([[a,b,c,d]])))



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

batch_size = 5000000
device = "cpu"

a = torch.empty(batch_size, device=device).uniform_(-torch.pi/2, torch.pi/4)
b = torch.empty(batch_size, device=device).uniform_(torch.pi/12, torch.pi/4)
c = torch.empty(batch_size, device=device).uniform_(-torch.pi/12, torch.pi/4)
d = torch.empty(batch_size, device=device).uniform_(-torch.pi/2, -torch.pi/8)
angles = torch.stack([a, b, c, d], dim=1)

print(angles.shape)

# Compute z offset (how far the arm moves along z-axis)
z = (arm_transform_points_torch(torch.tensor([1, 0, 0], device=device), angles)
        - arm_transform_points_torch(torch.tensor([0, 0, 0], device=device), angles))[:, 2]

# Filter out angles where z is not close to 0
mask = torch.abs(z + 1) < 0.01


end_point = torch.tensor([0.16, -0.01, -0.02609])
# Transform only valid ones
transformed = arm_transform_points_torch(end_point, angles[mask])

points = transformed.detach().cpu().numpy()
points = points[points[:,2] <= 0]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')



# --- Circle parameters ---
r = 0.1770                     # radius
center = np.array([0, 0, 0])  # circle center (x0, y0, z0)
normal = np.array([0, 0, 1])  # normal vector of the plane

# --- Generate circle points in plane defined by 'normal' ---
theta = np.linspace(0, 2 * np.pi, 200)
# Find two orthonormal vectors perpendicular to 'normal'
normal = normal / np.linalg.norm(normal)
# Pick an arbitrary vector not parallel to 'normal'
not_parallel = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
v1 = np.cross(normal, not_parallel)
v1 /= np.linalg.norm(v1)
v2 = np.cross(normal, v1)

# Parametric equation of circle in 3D
circle_points = np.array([
    center + r * (np.cos(t) * v1 + np.sin(t) * v2)
    for t in theta
])

# --- Plot ---
ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 'r', lw=2)

ax.scatter(points[:,0], points[:,1], points[:,2], s=5, alpha=0.6)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim((-0.3, 0.3))
ax.set_ylim((-0.3, 0.3))
ax.set_title("End-effector positions (valid configurations)")
ax.view_init(elev=30, azim=45)
plt.show()