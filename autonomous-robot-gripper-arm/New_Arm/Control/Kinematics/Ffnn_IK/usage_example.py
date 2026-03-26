import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fk import arm_transform_points_torch 
from models import IKNet          
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load model ---
model = IKNet().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()


a_vals = torch.linspace(-torch.pi, torch.pi, 10)
b_vals = torch.linspace(-torch.pi/3, torch.pi/3, 5)
c_vals = torch.linspace(-torch.pi/2, torch.pi/3, 5)
d_vals = torch.linspace(-torch.pi/2, 0, 5)

# Create 4D grid (indexing='ij' gives structured coordinates)
A, B, C, D = torch.meshgrid(a_vals, b_vals, c_vals, d_vals, indexing='ij')

# Stack into (N, 4)
angles = torch.stack([A, B, C, D], dim=-1).reshape(-1, 4)


end_point = torch.tensor([0, 0, 0])
original_points = arm_transform_points_torch(end_point, angles)

print(original_points)

# --- Predict angles from points ---
with torch.no_grad():
    predicted_angles = model(original_points.to(device))
    mins = predicted_angles.min(dim=0).values
    maxs = predicted_angles.max(dim=0).values

    # Print neatly
    for i, (mn, mx) in enumerate(zip(mins, maxs)):
        print(f"Column {i}: min = {mn.item():.4f}, max = {mx.item():.4f}")
    predicted_points = arm_transform_points_torch(end_point.to(device), predicted_angles)
    predicted_points = predicted_points.cpu()


# --- Plot in 3D ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(original_points[:,0], original_points[:,1], original_points[:,2],
           c='blue', label='Original', s=20)
ax.scatter(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2],
           c='red', label='Predicted', s=20)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('IK Prediction vs Original Points')
ax.legend()
ax.grid(True)
plt.show()