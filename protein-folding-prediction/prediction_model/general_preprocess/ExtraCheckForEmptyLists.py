import numpy as np

def ExtraCheckForEmptyLists(CaCoords_arr, AminoAcids_arr, minLenghtExtraCheck): 
    CaCoords_list = []
    AA_list = []
    for i in range(len(CaCoords_arr)): 
        Cacoords = CaCoords_arr[i]
        if len(Cacoords) < minLenghtExtraCheck: 
            continue
        else: 
            CaCoords_list.append(Cacoords)
            AA_list.append(AminoAcids_arr[i])


    AA_arr = np.array(AA_list, dtype = object)
    CaCoords_arr = np.array(CaCoords_list, dtype=object)

    return AA_arr, CaCoords_arr