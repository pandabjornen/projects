from Control.Kinematics.Ffnn_IK.ik import Ik_hand
from Control.Kinematics.fk import arm_transform_points_numpy, arm_transform_points_torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

ffnnIK_hand_model_path = Path("Control")/"Kinematics"/"Ffnn_IK"/"best_model_hand.pt"
ik_hand = Ik_hand(ffnnIK_hand_model_path)

batch_size = 500000
device = "cpu"

a = torch.empty(batch_size, device=device).uniform_(-torch.pi/2, torch.pi/4)
b = torch.empty(batch_size, device=device).uniform_(torch.pi/12, torch.pi/4)
c = torch.empty(batch_size, device=device).uniform_(-torch.pi/12, torch.pi/4)
d = torch.empty(batch_size, device=device).uniform_(-torch.pi/2, -torch.pi/8)
angles = torch.stack([a, b, c, d], dim=1)

z = (arm_transform_points_torch(torch.tensor([1, 0, 0], device=device), angles)
     - arm_transform_points_torch(torch.tensor([0, 0, 0], device=device), angles))[:, 2]
mask = torch.abs(z + 1) < 0.01

end_point = torch.tensor([0.16, -0.01, -0.02609])
transformed = arm_transform_points_torch(end_point, angles[mask])
points = transformed.detach().cpu().numpy()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
idx = np.random.choice(len(points), 10000, replace=False)  # sample fewer for plotting
ax.scatter(points[idx,0], points[idx,1], points[idx,2], s=5, alpha=0.6)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim((-0.3, 0.3))
ax.set_ylim((-0.3, 0.3))
ax.set_title("End-effector positions (valid configurations)")
ax.view_init(elev=30, azim=45)
plt.show()

# higher resolution slice
batch_slice = 3000000
a = torch.empty(batch_slice, device=device).uniform_(-torch.pi/2, torch.pi/4)
b = torch.empty(batch_slice, device=device).uniform_(torch.pi/12, torch.pi/4)
c = torch.empty(batch_slice, device=device).uniform_(-torch.pi/12, torch.pi/4)
d = torch.empty(batch_slice, device=device).uniform_(-torch.pi/2, -torch.pi/8)
angles2 = torch.stack([a, b, c, d], dim=1)

z2 = (arm_transform_points_torch(torch.tensor([1, 0, 0], device=device), angles2)
      - arm_transform_points_torch(torch.tensor([0, 0, 0], device=device), angles2))[:, 2]
mask2 = torch.abs(z2 + 1) < 0.01
transformed2 = arm_transform_points_torch(end_point, angles2[mask2])
points2 = transformed2.detach().cpu().numpy()

z_target = 0.075
tol = 0.002
mask_z = (points2[:,2] > z_target - tol) & (points2[:,2] < z_target + tol)
plane_points = points2[mask_z]

radii = np.sqrt(plane_points[:,0]**2 + plane_points[:,1]**2)
r_in, r_out = radii.min(), radii.max()
print(f"Inner radius = {r_in:.4f} m, Outer radius = {r_out:.4f} m")

angles_plane = np.arctan2(plane_points[:,1], plane_points[:,0])
angle_range = np.degrees(angles_plane.max() - angles_plane.min())
print(f"Angle interval in horizontal plane ≈ {angle_range:.1f}°")

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(plane_points[:,0], plane_points[:,1], s=1, alpha=0.4)
circle_in = plt.Circle((0,0), r_in, color='r', fill=False, linestyle='--', label=f"r_in={r_in:.3f}")
circle_out = plt.Circle((0,0), r_out, color='g', fill=False, linestyle='--', label=f"r_out={r_out:.3f}")
ax.add_artist(circle_in)
ax.add_artist(circle_out)
ax.set_aspect('equal')
ax.set_title(f"Workspace slice at z={z_target} m\nAngle span ≈ {angle_range:.1f}°")
ax.legend()
plt.show()

z_bins = np.linspace(points[:,2].min(), points[:,2].max(), 30)
r_in_list, r_out_list = [], []
for i in range(len(z_bins)-1):
    mask = (points[:,2] > z_bins[i]) & (points[:,2] <= z_bins[i+1])
    if np.any(mask):
        r = np.sqrt(points[mask,0]**2 + points[mask,1]**2)
        r_in_list.append(r.min())
        r_out_list.append(r.max())
    else:
        r_in_list.append(np.nan)
        r_out_list.append(np.nan)

z_mid = 0.5*(z_bins[:-1] + z_bins[1:])
plt.plot(z_mid, r_in_list, 'r--', label="Inner radius")
plt.plot(z_mid, r_out_list, 'g-', label="Outer radius")
plt.xlabel("z (m)")
plt.ylabel("Radius (m)")
plt.legend()
plt.title("Inner/Outer radius vs z")
plt.show()
