import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fk import arm_transform_points_torch
from models import IKNet




def ik_training_step(model, optimizer, target_points, end_point):
    optimizer.zero_grad()

    pred_angles = model(target_points)


    z = (arm_transform_points_torch(torch.tensor([1,0,0], device=end_point.device), pred_angles) - arm_transform_points_torch(torch.tensor([0,0,0], device=end_point.device), pred_angles))[:,2]

    pred_points = arm_transform_points_torch(end_point, pred_angles)

    loss = torch.mean(torch.norm(pred_points - target_points, dim=1) ** 2 + ((z+1)**2) * 0.005)


    loss.backward()
    optimizer.step()

    return loss.item()


def ik_validation_step(model, val_loader, device, end_point): 
    model.eval()
    losses = []
    msre = []
    zl = []
    with torch.no_grad():
        for (batch,) in val_loader:
            #batch = batch.to(device)
            pred_angles = model(batch)
            z = (arm_transform_points_torch(torch.tensor([1,0,0], device=end_point.device), pred_angles) - arm_transform_points_torch(torch.tensor([0,0,0], device=end_point.device), pred_angles))[:,2]
            pred_points = arm_transform_points_torch(end_point, pred_angles)
            loss = torch.mean(torch.norm(pred_points - batch, dim=1) ** 2 + ((z+1)**2) * 0.005)
            losses.append(loss.item())

            msre.append(torch.sqrt(torch.mean(torch.norm(pred_points - batch, dim=1) ** 2)).item())
            zl.append(torch.sqrt(torch.mean(((z+1)**2))).item())

    model.train()
    
    return np.array(losses).mean(), np.array(msre).mean(), np.array(zl).mean()



def generate_dataset_torch(end_point, N=200_000, device="cpu", z_tolerance=0.01, batch_size=500_000):
    """Generate N valid transformed points where z ≈ 0."""

    valid_points = []
    total_collected = 0

    while total_collected < N:
        # Sample a batch of random angles
        a = torch.empty(batch_size, device=device).uniform_(-torch.pi/2, torch.pi/4)
        b = torch.empty(batch_size, device=device).uniform_(torch.pi/12, torch.pi/4)
        c = torch.empty(batch_size, device=device).uniform_(-torch.pi/12, torch.pi/4)
        d = torch.empty(batch_size, device=device).uniform_(-torch.pi/2, -torch.pi/8)
        angles = torch.stack([a, b, c, d], dim=1)

        # Compute z offset (how far the arm moves along z-axis)
        z = (arm_transform_points_torch(torch.tensor([1, 0, 0], device=device), angles) - arm_transform_points_torch(torch.tensor([0, 0, 0], device=device), angles))[:, 2]

        # Filter out angles where z is not close to -1
        mask = torch.abs(z + 1) < z_tolerance
        if mask.sum() == 0:
            continue  # no valid samples in this batch

        # Transform only valid ones
        transformed = arm_transform_points_torch(end_point, angles[mask])

        valid_points.append(transformed)
        total_collected += transformed.shape[0]

    # Concatenate and truncate to exactly N
    valid_points = torch.cat(valid_points, dim=0)[:N]

    return valid_points




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    end_point = torch.tensor([0.16, -0.01, -0.02609], device=device)

    # Generate dataset
    full_data = generate_dataset_torch(end_point,5_000_000, device=device)
    print("made data set")
    dataset = TensorDataset(full_data)

    # Train/val split (80/20)
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096)

    # Model and optimizer
    model = IKNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1e-3)

    best_val_loss = float('inf')

    for epoch in range(1000):
        train_losses = []
        for (batch,) in train_loader:
            #batch = batch.to(device)
            loss = ik_training_step(model, optimizer, batch, end_point)
            train_losses.append(loss)

        val_loss, msre, zl = ik_validation_step(model, val_loader, device, end_point)
        print(f"Epoch {epoch+1:02d} | Train loss: {torch.tensor(train_losses).mean():.8f} | Val loss: {val_loss:.8f} | Val msre: {msre:.8f} | Val zl: {zl:.8f} ")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  --> New best model saved with val loss: {val_loss:.6f}")