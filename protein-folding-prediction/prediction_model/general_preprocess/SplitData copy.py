import torch
from torch.utils.data import TensorDataset, DataLoader
def SplitData(inputTensors, targetTensors, batchSize, val_split): 

    print("\n \t START SPLIT DATA")

    total_size = inputTensors.shape[0]
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    indices = torch.randperm(total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    

    inputTensorsTrain = inputTensors[train_indices]
    targetTensorsTrain = targetTensors[train_indices]
    inputTensorsVal = inputTensors[val_indices]
    targetTensorsVal = targetTensors[val_indices]


    train_dataset = TensorDataset(inputTensorsTrain, targetTensorsTrain)
    val_dataset = TensorDataset(inputTensorsVal, targetTensorsVal)

    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=False)
    

    return train_dataloader, val_dataloader