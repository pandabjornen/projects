from general_preprocess.GetLetterToIndexDict import GetLetterToIndexDict
import numpy as np
def GetInputVectors(AminoAcidSequence_list, maxLengthProtein): 
    inputVectors = []
    for aminoAcidSequence in AminoAcidSequence_list: 
        inputVector = np.zeros(maxLengthProtein)
        for j, residueIndex in enumerate(aminoAcidSequence): 
            inputVector[j] = residueIndex
        inputVectors.append(inputVector)
    return inputVectors