import torch

def OptimizeCoordsFromDM(distance_matrix, learning_rate, max_iterations, epochs, print_every, patience, optimizer_type):

    device = distance_matrix.device
    
    
    d_target = distance_matrix.cpu()
    print("d_target size, ", d_target.size(0))
    

    L = d_target.size(0)
    

    X = torch.randn(L, 3, device="cpu", dtype=torch.float, requires_grad=True)
    
    if optimizer_type == "adam":
        opt = torch.optim.Adam([X], lr=learning_rate)
    elif optimizer_type == "lbfgs":
        opt = torch.optim.LBFGS([X], lr=learning_rate, max_iter=max_iterations, line_search_fn="strong_wolfe")
    
    
    best_rmse = float('inf')
    epochs_since_improve = 0
    
    
    for i in range(epochs):
    
        if optimizer_type == "adam":
            opt.zero_grad()
            current_distances = torch.cdist(X, X)
    
            squared_errors = torch.sum((current_distances - d_target)**2) / (L**2 - L)
            loss = 0.5 * squared_errors
            loss.backward()
            opt.step()
            
    
            with torch.no_grad():
                current_distances_for_rmse = torch.cdist(X, X)
                squared_error_for_rmse = torch.sum((current_distances_for_rmse - d_target)**2) / (L**2 - L)
                rmse = torch.sqrt(squared_error_for_rmse)
        
        
        else:
            def closure():
                opt.zero_grad()
                current_distances = torch.cdist(X, X)
                squared_errors = torch.sum((current_distances - d_target)**2) / (L**2 - L)
                loss = 0.5 * squared_errors
                loss.backward()
                return loss
            
            opt.step(closure)
            
            
            with torch.no_grad():
                current_distances_for_rmse = torch.cdist(X, X)
                squared_error_for_rmse = torch.sum((current_distances_for_rmse - d_target)**2) / (L**2 - L)
                rmse = torch.sqrt(squared_error_for_rmse)
        
        
        if i % print_every == 0 or i == epochs - 1:
            print(f"Epoch {i}: RMSE = {rmse.item():.8f}")
        
        
        if rmse.item() < best_rmse - 1e-8:
            best_rmse = rmse.item()
            best_X = X.clone().detach()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            
        if epochs_since_improve >= patience:
            print(f"Early stopping at epoch {i} due to no improvement for {patience} epochs.")
            break
    
    
    return best_X.to(device)