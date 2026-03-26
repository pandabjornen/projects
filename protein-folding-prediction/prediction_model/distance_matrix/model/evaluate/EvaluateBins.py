import torch
import matplotlib.pyplot as plt
import numpy as np

def EvaluateBINS(seq_tensors_test, msa_tensors_test, targetTensorsTest, original_distance_matrices_test,
                 model, device, bin_edges: torch.Tensor):
    """
    Evaluates the model's performance on bin matrices (classification task), including
    input MSA features. Computes accuracy, RMSE, and detects protein lengths. Skips invalid cases.
    """
    model.to(device)
    accuracies_list = []
    predicted_distance_matrices_list = []
    rmse_list = []
    lengths_list = []

    # User choice for plotting
    show_plots = input("Show plots? (y/n): ").strip().lower() == 'y'

    # Prepare bin midpoints
    bin_edges = bin_edges.to(device)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    model.eval()
    with torch.no_grad():
        for i in range(seq_tensors_test.shape[0]):
            # Sequence and MSA inputs
            seq_tensor = seq_tensors_test[i].to(device)
            msa_tensor = msa_tensors_test[i].to(device)
            logits = model(seq_tensor.unsqueeze(0), msa_tensor.unsqueeze(0)).squeeze(0)
            pred_bins = torch.argmax(logits, dim=-1)
            pred_bins.fill_diagonal_(0)

            true_bins = targetTensorsTest[i].to(device)
            true_dist = original_distance_matrices_test[i].to(device)

            # Determine protein length
            col0 = true_dist[:, 0]
            zeros = torch.where(col0 == 0)[0]
            if len(zeros) > 1 and zeros[0].item() == 0:
                length = zeros[1].item()
            elif len(zeros) > 0:
                length = zeros[0].item()
            else:
                length = true_dist.size(0)
            lengths_list.append(length)

            # Map bins to distances
            pred_dist = bin_midpoints[pred_bins]
            pred_dist.fill_diagonal_(0.0)
            masked_true = true_dist.clone(); masked_true.fill_diagonal_(0.0)
            if length < pred_dist.size(0):
                mask = torch.ones_like(masked_true, dtype=torch.bool)
                mask[:length, :length] = False
                pred_dist = pred_dist.masked_fill(mask, 0.0)
                masked_true = masked_true.masked_fill(mask, 0.0)

            # Create validity mask
            mask_diag = torch.eye(true_bins.size(0), device=device, dtype=torch.bool)
            mask_pad = torch.zeros_like(mask_diag)
            if length < mask_pad.size(0):
                mask_pad[length:, :] = True; mask_pad[:, length:] = True
            valid = ~mask_diag & ~mask_pad

            # Skip if any NaNs in valid region
            if torch.isnan(pred_dist[valid]).any() or torch.isnan(masked_true[valid]).any():
                print(f"Protein {i+1}: contains NaNs, skipping.")
                continue

            # Compute metrics
            acc = ((pred_bins == true_bins) & valid).sum().float() / valid.sum().float()
            rmse = torch.sqrt(((pred_dist - masked_true)[valid]**2).mean())
            # Skip if acc or rmse is NaN
            if torch.isnan(acc) or torch.isnan(rmse):
                print(f"Protein {i+1}: invalid metrics (NaN), skipping.")
                continue

            accuracies_list.append(acc)
            rmse_list.append(rmse)
            predicted_distance_matrices_list.append(pred_dist.cpu().numpy())
            print(f"Protein {i+1}/{seq_tensors_test.size(0)}: bin acc={acc:.4f}, RMSE={rmse:.4f} Å")

            # Plotting
            if show_plots:
                tb = true_bins.cpu().numpy(); pb = pred_bins.cpu().numpy()
                td = masked_true.cpu().numpy(); pd = pred_dist.cpu().numpy()
                fig, axes = plt.subplots(1, 4, figsize=(24, 6))
                fig.suptitle(f'Protein {i+1}: acc={acc:.4f}, RMSE={rmse:.4f}', fontsize=16)

                axes[0].imshow(tb,  origin='lower'); axes[0].set_title('True Bins')
                axes[1].imshow(pb,  origin='lower'); axes[1].set_title('Predicted Bins')

                im_true = axes[2].imshow(td, cmap='plasma', origin='lower', vmin=0, vmax=bin_edges[-1].item())
                axes[2].set_title('True Distances (Å)')
                im_pred = axes[3].imshow(pd, cmap='plasma', origin='lower', vmin=0, vmax=bin_edges[-1].item())
                axes[3].set_title('Predicted Distances (Å)')

                for ax in axes:
                    ax.set_xlabel('Residue Index'); ax.set_ylabel('Residue Index')

                # Single colorbar for distances
                fig.colorbar(im_pred, ax=axes[2:4], location='right', shrink=0.8, label='Distance (Å)')
                plt.show()

        # Summary
        # Filter out any NaNs (should be none after skipping)
        if accuracies_list:
            valid_acc = torch.stack(accuracies_list)
            valid_rmse = torch.stack(rmse_list)
            avg_acc = valid_acc.mean()
            avg_rmse = valid_rmse.mean()
            print(f"\nAverage bin accuracy: {avg_acc:.4f}")
            print(f"Average RMSE: {avg_rmse:.4f} Å")
        else:
            print("\nNo valid proteins evaluated.")

    return predicted_distance_matrices_list, accuracies_list, rmse_list, lengths_list
