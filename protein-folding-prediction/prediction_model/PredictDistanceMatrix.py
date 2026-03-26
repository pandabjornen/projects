import numpy as np
import matplotlib.pyplot as plt 
import torch

from general_preprocess.GetInputsAndTargets import GetInputVectors
from distance_matrix.specific_preprocess.ConvertToTensor import ConvertToTensor
from distance_matrix.model.architecture.encoderOnlyTransformer.EncoderOnlyTransformer import EncoderOnlyTransformer
from distance_matrix.model.train.trainTE import trainTE
from distance_matrix.model.train.trainBins import trainBINS
from general_preprocess.SplitData import SplitData
from distance_matrix.specific_preprocess.BinMatrix import distances_to_bins
from general_preprocess.ExtraCheckForEmptyLists import ExtraCheckForEmptyLists
from Constants import DM_Constants
from distance_matrix.model.architecture.encoderOnlyTransformer.BinsEncoderOnlyTransformer import BinsEncoderOnlyTransformer
from MSA import build_msa_tensor
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS,PATH_LAST_EPOCH_MODEL_WEIGHTS, \
PATH_TO_CORRECT_DISTANCE_MATRICES, PATH_TO_PREDICTED_DISTANCE_MATRICES,PATH_TO_TEST_OUTPUTS_DM, PATH_TO_TEST_INPUTS_DM = DM_Constants()


d_z = EMBEDDING_DIMENSION//16
accumBatches = 2 # 2 for 20_70
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
def main(): 
    torch.mps.set_per_process_memory_fraction(0.95)  # < 1.0!
    #--------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------
    wantToTrain = input("Do you want to train (y/n)?")
    if wantToTrain.strip().lower() == "y": 
        #-------------------------------------------------------------------------------------------------------------------
        CaCoords_arr = np.load(PATH_TO_CA_COORDS, allow_pickle=True)
        AminoAcids_arr = np.load(PATH_TO_AA, allow_pickle=True)
        AA_arr, CaCoords_arr = ExtraCheckForEmptyLists(CaCoords_arr, AminoAcids_arr, MIN_LENGTH_EXTRA_CHECK)

        
        msa_tensor = build_msa_tensor(AA_arr, MAX_LENGTH_PROTEIN)
    
        
        
        inputVectors = GetInputVectors(AA_arr[:MAX_DATA_LENGTH], MAX_LENGTH_PROTEIN)
        inputTensors, targetTensors  = ConvertToTensor(CaCoords_arr[:MAX_DATA_LENGTH], inputVectors, MAX_LENGTH_PROTEIN, DEVICE)
        
        suitable_bin_edges = torch.tensor([0.0, 3.5 ,4.0 , 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.5, 13.0, 15.0, 17.0, 21.0, 25.0])

        targetTensors = distances_to_bins(targetTensors, suitable_bin_edges)
        msa_tensor = msa_tensor[:MAX_DATA_LENGTH] 
        trainDataLoader, valDataLoader = SplitData(inputTensors, msa_tensor,targetTensors, BATCH_SIZE, VAL_SPLIT)
        #-------------------------------------------------------------------------------------------------------------------
        model  = BinsEncoderOnlyTransformer(MAX_LENGTH_PROTEIN, EMBEDDING_DIMENSION, \
                                         PRINT_EVERY_EPOCH, NR_OF_ATTENTION_HEADS, \
                                             DROPOUT_RATE, LEARNING_RATE, NR_T_ENCODE_LAYERS, d_z, len(suitable_bin_edges)-1)
        # model  = EncoderOnlyTransformer(MAX_LENGTH_PROTEIN, EMBEDDING_DIMENSION, \
        #                                 PRINT_EVERY_EPOCH, NR_OF_ATTENTION_HEADS, \
        #                                     DROPOUT_RATE, LEARNING_RATE, NR_T_ENCODE_LAYERS, d_z)
        model = trainBINS(model, trainDataLoader, valDataLoader,FOLDER_MODEL_WEIGHTS, NR_OF_EPOCHS, accumBatches)
        # model = trainTE(model, trainDataLoader, valDataLoader,FOLDER_MODEL_WEIGHTS, NR_OF_EPOCHS, accumBatches)
    #--------------------------------------------------------------------------------------------------------------------------------------------
    # #--------------------------------------------------------------------------------------------------------------------------------------------   
    wantToPlot = input("Do you want to plot training statistics (y/n)?")
    if wantToPlot.strip().lower() == "y":
        plt.plot(model.train_rmse_log, label="Train (epoch) rmse")
        plt.plot(model.val_rmse_log, label="Val (epoch) rmse")
        plt.legend(); plt.show()
    #--------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

