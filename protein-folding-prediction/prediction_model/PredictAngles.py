import numpy as np
import matplotlib.pyplot as plt 

from general_preprocess.GetInputsAndTargets import GetInputVectors
from angles.specific_preprocess.Convert_to_tensor_angles import ConvertToTensorNNAngles
from angles.model.architecture.EncoderOnlyTransformer import EncoderOnlyTransformerNNAngles
from distance_matrix.model.train.trainTE import trainTE
from general_preprocess.SplitData import SplitData
import torch

from general_preprocess.ExtraCheckForEmptyLists import ExtraCheckForEmptyLists
from Constants import NN_Angles_Constants

MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS,PATH_LAST_EPOCH_MODEL_WEIGHTS, \
PATH_TO_CORRECT_ANGLES, PATH_TO_PREDICTED_ANGLES,PATH_TO_TEST_OUTPUTS_DM, PATH_TO_TEST_INPUTS_DM = NN_Angles_Constants()
def main():  

    torch.mps.set_per_process_memory_fraction(0.95)  # < 1.0!

    wantToTrain = input("Do you want to train (y/n)?")
    wantToTrain = "y"
    if wantToTrain.strip().lower() == "y": 
        CaCoords_arr = np.load(PATH_TO_CA_COORDS, allow_pickle=True)
        AminoAcids_arr = np.load(PATH_TO_AA, allow_pickle=True)

        AA_arr, CaCoords_arr = ExtraCheckForEmptyLists(CaCoords_arr, AminoAcids_arr, MIN_LENGTH_EXTRA_CHECK)
        print("Nr of proteins used for training: ", len(CaCoords_arr[:MAX_DATA_LENGTH]))

        inputVectors = GetInputVectors(AA_arr[:MAX_DATA_LENGTH], MAX_LENGTH_PROTEIN)

        inputTensors, targetTensors  = ConvertToTensorNNAngles(CaCoords_arr[:MAX_DATA_LENGTH], inputVectors, MAX_LENGTH_PROTEIN, DEVICE)

        trainDataLoader, valDataLoader  = SplitData(inputTensors, targetTensors, BATCH_SIZE, VAL_SPLIT)
        
        model  = EncoderOnlyTransformerNNAngles(MAX_LENGTH_PROTEIN, EMBEDDING_DIMENSION, \
                                        PRINT_EVERY_EPOCH, NR_OF_ATTENTION_HEADS, \
                                            DROPOUT_RATE, LEARNING_RATE, NR_T_ENCODE_LAYERS)
        model = trainTE(model, trainDataLoader, valDataLoader,FOLDER_MODEL_WEIGHTS, NR_OF_EPOCHS)
        


    wantToPlot = input("Do you want to plot training statistics (y/n)?")
    if wantToPlot.strip().lower() == "y":
        plt.plot(model.train_rmse_log, label="Train (epoch) rmse  ")
        plt.plot(model.val_rmse_log, label="Val (epoch) rmse ")
        # plt.yscale("log")
        plt.legend(); plt.show()

if __name__ == "__main__":
    main()

