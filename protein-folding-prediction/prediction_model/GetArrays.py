from general_preprocess.FindIDs import SearchForIDsForSingleChains
from general_preprocess.DownloadCIFsFromIds import download_mmcif_files
from general_preprocess.ExtractNpArrayFromCIF import extract_chain_from_cif
import os


ACCEPTABLE_RATIO_OF_UKNW_AAS = 0.1

MIN_LENGTH_PROTEIN = 20 #NOTE: Set atleast to 2, otherwise can have som useless ones.
MAX_LENGTH_PROTEIN = 70


nrOfProteinsToTryAndDownload = 4000
test_split = 0.1
max_resolution = 3.5 #Å

dataDir = "./data/"+str(MIN_LENGTH_PROTEIN)+"_"+str(MAX_LENGTH_PROTEIN)+"/cif_files"
np_arr_dir = "./data/"+str(MIN_LENGTH_PROTEIN)+"_"+str(MAX_LENGTH_PROTEIN)+"/np_arrays"


def DownloadFilesAndGetNpArray(min_length, max_length, max_resolution, nrOfChains ,max_ratio_unknown, mmCif_dir, np_arr_dir):

    downloadedIds = [f.split('.')[0] for f in os.listdir(mmCif_dir) if f.endswith('.cif')]

    searchForIds= input("Search for ids (y/n)?")
    if searchForIds.lower().strip() ==  "y": 
        ids = SearchForIDsForSingleChains(min_length, max_length, max_resolution, nrOfChains, downloadedIds)
    
    downloadIds= input("Download ids (y/n)?")
    if downloadIds.lower().strip() ==  "y": 
        file_paths = download_mmcif_files(ids, mmCif_dir)
    else: 
        file_paths = [dataDir + "/" + f for  f in os.listdir(mmCif_dir)]
    eactractChain= input("Extract chain (y/n)?")
    if eactractChain.lower().strip() ==  "y": 
        valid_paths = extract_chain_from_cif(file_paths, np_arr_dir, min_length, max_length, max_ratio_unknown, test_split )

    print("downloaded ", len(valid_paths))


def main(): 
    DownloadFilesAndGetNpArray(MIN_LENGTH_PROTEIN,
    MAX_LENGTH_PROTEIN,
    max_resolution,
    nrOfProteinsToTryAndDownload ,ACCEPTABLE_RATIO_OF_UKNW_AAS, 
    dataDir, np_arr_dir)

if __name__ == "__main__":
    main()
