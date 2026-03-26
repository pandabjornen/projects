import torch
from torch.utils.data import TensorDataset, DataLoader

def SplitData(seq_tensors, msa_tensors, target_tensors, batchSize, val_split):
    total = seq_tensors.shape[0]
    val_size = int(total * val_split)
    idx = torch.randperm(total)
    train_idx, val_idx = idx[val_size:], idx[:val_size]

    
    seq_tr = seq_tensors[train_idx]
    msa_tr = msa_tensors[train_idx]
    tgt_tr = target_tensors[train_idx]
    seq_val = seq_tensors[val_idx]
    msa_val = msa_tensors[val_idx]
    tgt_val = target_tensors[val_idx]

    
    train_ds = TensorDataset(seq_tr, msa_tr, tgt_tr)
    val_ds   = TensorDataset(seq_val, msa_val, tgt_val)

    train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batchSize, shuffle=False)
    return train_dl, val_dl
