import torch
from distance_matrix.specific_preprocess.CalculateDistanceMatrix import CalculateDistanceMatrix
import numpy as np
def ConvertToTensor(CaCoords_arr, inputVectors, maxLengthProtein, device): 

    targetTensors_list = []

    for CaCoordsList in CaCoords_arr: 

        distanceMatrix = CalculateDistanceMatrix(CaCoordsList, maxLengthProtein)
        targetTensor = torch.tensor(distanceMatrix, dtype=torch.float, device=device)
        targetTensors_list.append(targetTensor)

    targetTensors = torch.stack(targetTensors_list)
    inputTensors = torch.tensor(np.array(inputVectors), dtype=torch.long, device=device)

    return inputTensors, targetTensors
