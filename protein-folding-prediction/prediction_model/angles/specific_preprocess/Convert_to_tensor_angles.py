import torch
from angles.specific_preprocess.CalculateAngles import CalculateNeighbourAngles
import numpy as np
def ConvertToTensorNNAngles(CaCoords_arr, inputVectors, maxLengthProtein, device): 

    targetTensors_list = []

    for CaCoordsList in CaCoords_arr: 

        neighbouringangles = CalculateNeighbourAngles(CaCoordsList, maxLengthProtein)
        targetTensor = torch.tensor(neighbouringangles, dtype=torch.float, device=device)
        targetTensors_list.append(targetTensor)

    targetTensors = torch.stack(targetTensors_list)
    inputTensors = torch.tensor(np.array(inputVectors), dtype=torch.long, device=device)

    return inputTensors, targetTensors
