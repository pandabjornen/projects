import torch

def distances_to_bins(distance_matrix: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of distance matrices to a tensor of bin indices.

    Args:
        distance_matrix (torch.Tensor): A tensor of shape (N, L, L) or (L, L)
                                        containing distances (float).
        bin_edges (torch.Tensor): A 1D tensor of strictly increasing bin edge values.
                                  e.g., torch.tensor([0.0, 4.0, 5.0, ..., 50.0])

    Returns:
        torch.Tensor: A tensor of the same shape as distance_matrix, but with
                      integer bin indices.
    """
    # Ensure bin_edges is on the same device as distance_matrix
    bin_edges = bin_edges.to(distance_matrix.device)

    # Use searchsorted to find which bin each distance falls into
    # It returns the index into bin_edges where the distance would be inserted
    # to maintain sorted order. This index corresponds to the bin.
    # For example, if bin_edges = [0, 5, 10]
    # distance = 3 -> index 1 (falls into [0, 5))
    # distance = 7 -> index 2 (falls into [5, 10))
    # distance = 10 -> index 2 (falls into [5, 10), due to right=False default)
    # The returned indices will be 0 to len(bin_edges) - 1.
    # We want 0-indexed bins, so if distance_matrix[i,j] < bin_edges[0], it will get bin 0.
    # If distance_matrix[i,j] >= bin_edges[-1], it will get bin len(bin_edges) - 1.
    # This is correct if bin_edges defines `num_bins` intervals.
    bin_indices = torch.searchsorted(bin_edges, distance_matrix, right=False)

    # Handle edge case: distances larger than or equal to the last bin edge
    # searchsorted with right=False will put values >= last_edge into len(bin_edges) - 1.
    # So, if you have len(bin_edges) edges defining len(bin_edges)-1 bins,
    # the maximum valid index for a bin is len(bin_edges)-2.
    # Example: edges = [0, 5, 10]. bins are [0,5), [5,10). Max index is 1.
    # searchsorted will return 0, 1, or 2. Max index should be 1.
    # So we clamp to the maximum valid bin index.
    num_bins = len(bin_edges) - 1
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    return bin_indices.long() # Return as long integers for CrossEntropyLoss