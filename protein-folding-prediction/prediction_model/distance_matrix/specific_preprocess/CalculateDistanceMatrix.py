import numpy as np
def CalculateDistanceMatrix(CaCoordsOfOneProtein, sizeOfDistanceMatrix):
    proteinLength = len(CaCoordsOfOneProtein)
    distanceMatrix = np.zeros((sizeOfDistanceMatrix, sizeOfDistanceMatrix))
    for row in range(proteinLength): 
        x0, y0, z0 = CaCoordsOfOneProtein[row]
        for column in range(proteinLength): 
            x1, y1, z1 = CaCoordsOfOneProtein[column]

            distance = np.sqrt((x1-x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
            distanceMatrix[row, column] = distance
    np.fill_diagonal(distanceMatrix, 0)
    return distanceMatrix