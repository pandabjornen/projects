
def GetLetterToIndexDict():
    threeLetterToOneLetterDict = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", 
            "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", 
            "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", 
            "TYR": "Y", "VAL": "V", 
            "SEC": "U", "PYL": "O", 
            "XAA": "X", "ASX": "B", "GLX": "Z", "XLE": "J"  
        }
    OneLetterAminoAcidToIndex = {}
    for i, value in enumerate(threeLetterToOneLetterDict.values()): 
        OneLetterAminoAcidToIndex[value] = i
    return OneLetterAminoAcidToIndex