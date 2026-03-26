from Bio.SVDSuperimposer import SVDSuperimposer
import torch

def KabschAlign(predicted_coords: torch.Tensor, target_coords: torch.Tensor):

    pred = predicted_coords.cpu().numpy()
    targ = target_coords.cpu().numpy()

    sup = SVDSuperimposer()
    sup.set(targ, pred)  
    sup.run()
    rmsd = sup.get_rms()

    rot, tran = sup.get_rotran()

    aligned = (pred @ rot) + tran
    aligned = torch.tensor(aligned, device=predicted_coords.device, dtype=predicted_coords.dtype)

    return aligned, rmsd
