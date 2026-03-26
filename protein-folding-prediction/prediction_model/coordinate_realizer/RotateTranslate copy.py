import torch
def KabashAlign(pred: torch.Tensor, true: torch.Tensor):
    # Move tensors to CPU for operations not supported on MPS
    device = pred.device
    pred_cpu = pred.cpu()
    true_cpu = true.cpu()
    
    # centroid
    p_cent = pred_cpu.mean(dim=0)
    t_cent = true_cpu.mean(dim=0)
    # centralisera
    P = pred_cpu - p_cent
    T = true_cpu - t_cent
    # SVD för kovarians
    C = P.T @ T
    U, S, Vt = torch.linalg.svd(C)
    # rotation
    R = Vt.T @ U.T
    if torch.det(R) < 0:
        # rättar handighet
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # applicera och translat
    pred_aligned = (pred_cpu - p_cent) @ R + t_cent
    # RMSE
  
    pred_aligned = pred_aligned
    diff = (pred_aligned - true_cpu).pow(2)
    mse   = diff.sum() / (true_cpu.size(0)**2 -pred_aligned.size(0))
    rmse  = torch.sqrt(mse)

    # Move result back to original device
    pred_aligned = pred_aligned.to(device)
    return pred_aligned, rmse