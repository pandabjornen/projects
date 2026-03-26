import torch

def OptimizeCoordsFromDM(DistanceMatrices, learningRate, maxIterations, epochs, nrEpochPrint,patience,  optimizer_type):
    def _OptimizeCoord(d):
        L = d.size(0)
        X = torch.randn(L, 3, requires_grad=True, device=d.device)

        if optimizer_type == "adam":
            opt = torch.optim.Adam([X], lr=learningRate)
        elif optimizer_type == "lbfgs":
            opt = torch.optim.LBFGS([X], lr=learningRate, max_iter=maxIterations)
        else:
            raise ValueError("Invalid optimizer_type. Use 'adam' or 'lbfgs'.")

        best_rmse = float('inf')
        epochs_since_improve = 0
        

        for i in range(epochs):
            if optimizer_type == "adam":
                for _ in range(maxIterations):
                    opt.zero_grad()
                    dist = torch.cdist(X, X)
                    loss = 0.5 * ((dist - d) ** 2).mean()
                    loss.backward()
                    opt.step()
            else:
                def closure():
                    opt.zero_grad()
                    dist = torch.cdist(X, X)
                    loss = 0.5 * ((dist - d) ** 2).mean()
                    loss.backward()
                    return loss
                opt.step(closure)

            if i % nrEpochPrint == 0:
                with torch.no_grad():
                    dist = torch.cdist(X, X)
                    mask_data = (d > 0).float()   
                    dist = dist * mask_data
                    diff2 =((dist - d) ** 2)
                    mse = diff2.sum()/mask_data.sum()
                    rmse = torch.sqrt(mse)
                    print(f"Epoch {i}: RMSE = {rmse:.8f}")
                    if rmse < best_rmse - 1e-6:
                        best_rmse = rmse
                        epochs_since_improve = 0
                    if epochs_since_improve >= patience:
                        print(f"Early stopping at epoch {i}")
                        break

            epochs_since_improve += 1         
        return X

    if DistanceMatrices.dim() == 2:
        return _OptimizeCoord(DistanceMatrices)
    elif DistanceMatrices.dim() == 3:
        return torch.stack([_OptimizeCoord(DistanceMatrices[i]) for i in range(DistanceMatrices.size(0))], dim=0)
    else:
        raise ValueError(f"Wrong dim of D got {DistanceMatrices.dim()}D.")
