import torch

def OptimizeCoordsFromDM(distance_matrix, learning_rate, max_iterations,
                         epochs, print_every, patience, optimizer_type):
    device = distance_matrix.device
    d_target = distance_matrix.cpu()
    L = d_target.size(0)

    # --- 1) MDS initiering ---
    D2 = d_target ** 2
    J = torch.eye(L) - torch.ones(L, L) / L
    B = -0.5 * (J @ D2 @ J)
    eigvals, eigvecs = torch.linalg.eigh(B)
    top3 = eigvals.argsort(descending=True)[:3]
    L_sqrt = torch.sqrt(eigvals[top3].clamp(min=0))
    V3 = eigvecs[:, top3]
    X = (V3 * L_sqrt)  # (L,3)

   
    

    
    X = X.clone().detach().requires_grad_(True)

    
    W = (d_target > 0).float()

    
    if optimizer_type == "adam":
        opt = torch.optim.Adam([X], lr=learning_rate)
    else:
        opt = torch.optim.LBFGS([X], lr=learning_rate,
                                max_iter=max_iterations,
                                line_search_fn="strong_wolfe")

    best_rmse = float('inf')
    no_improve = 0

    for i in range(epochs):
        def compute_loss():
            dist = torch.cdist(X, X)
            return torch.sum(W * (dist - d_target) ** 2)

        if optimizer_type == "adam":
            opt.zero_grad()
            loss = compute_loss()
            loss.backward()
            opt.step()
        else:
            def closure():
                opt.zero_grad()
                loss = compute_loss()
                loss.backward()
                return loss
            opt.step(closure)

        with torch.no_grad():
            dist = torch.cdist(X, X)
            rmse = torch.sqrt(torch.sum((dist - d_target) ** 2) / (L**2 - L))

        if i % print_every == 0 or i == epochs - 1:
            print(f"Epoch {i}: RMSE = {rmse:.6f}")

        if rmse < best_rmse - 1e-8:
            best_rmse = rmse
            best_X = X.clone().detach()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {i}.")
            break

    return best_X.to(device), best_rmse.to(device)
