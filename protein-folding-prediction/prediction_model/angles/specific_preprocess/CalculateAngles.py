import numpy as np

def CalculateNeighbourAngles(CaCoordsOfOneProtein, sizeOfAngleArr): 
    proteinLength = len(CaCoordsOfOneProtein)
    # print(f"Lenght of protein = {CaCoordsOfOneProtein}" )



    angleArr = np.zeros((sizeOfAngleArr))

    idx = 0
    for idx in range(proteinLength):  
        x0, y0, z0 = CaCoordsOfOneProtein[(idx - 1) % proteinLength]  
        x1, y1, z1 = CaCoordsOfOneProtein[idx]
        x2, y2, z2 = CaCoordsOfOneProtein[(idx + 1) % proteinLength]  


        vector1 = np.array([x0 - x1, y0 - y1, z0 - z1])
        vector2 = np.array([x2 - x1, y2 - y1, z2 - z1])
        
        magnitude1 = np.sqrt(np.sum(vector1**2))
        magnitude2 = np.sqrt(np.sum(vector2**2))
        
        dotProduct = np.dot(vector1, vector2)
        
        # cos(θ) = (v1·v2)/(|v1|·|v2|)
        cos = max(min(dotProduct / (magnitude1 * magnitude2), 1.0), -1.0)
        angle = np.degrees(np.arccos(cos))


        angleArr[idx] = angle
        idx += 1

    return angleArr