import torch
import numpy as np

from Constants import DM_Constants
from Constants import NN_Angles_Constants
from distance_matrix.model.architecture.encoderOnlyTransformer.EncoderOnlyTransformer import EncoderOnlyTransformer
from angles.model.architecture.EncoderOnlyTransformer import EncoderOnlyTransformerNNAngles
from distance_matrix.model.evaluate.EvaluateTE import EvaluateDM
from angles.model.evaluate.Evaluate import EvaluateNNAngles
from distance_matrix.model.evaluate.EvaluateBins import EvaluateBINS


from general_preprocess.GetInputsAndTargets import GetInputVectors
from distance_matrix.specific_preprocess.ConvertToTensor import ConvertToTensor
from angles.specific_preprocess.Convert_to_tensor_angles import ConvertToTensorNNAngles

from distance_matrix.specific_preprocess.BinMatrix import distances_to_bins
from distance_matrix.model.architecture.encoderOnlyTransformer.BinsEncoderOnlyTransformer import BinsEncoderOnlyTransformer
from MSA import build_msa_tensor

wantToEvaluate = input("Do you want to evaluate for DISTANCE MATRICES (y/n)?")

if wantToEvaluate.strip().lower() == "y": 

    MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
    NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
    MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS,PATH_LAST_EPOCH_MODEL_WEIGHTS, \
    PATH_TO_CORRECT_DISTANCE_MATRICES, PATH_TO_PREDICTED_DISTANCE_MATRICES,PATH_TO_TEST_OUTPUTS, PATH_TO_TEST_INPUTS = DM_Constants()


    inputTestArr = np.load(PATH_TO_TEST_INPUTS, allow_pickle=True)
    CaCoords_arr = np.load(PATH_TO_TEST_OUTPUTS, allow_pickle=True)

    inputVectors = GetInputVectors(inputTestArr, MAX_LENGTH_PROTEIN)

    inputTensorsTest, targetTensorsTest  = ConvertToTensor(CaCoords_arr, inputVectors, MAX_LENGTH_PROTEIN, DEVICE)
    
    suitable_bin_edges = torch.tensor([0.0, 3.5 ,4.0 , 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.5, 13.0, 15.0, 17.0, 21.0, 25.0])
    msa_tensor = build_msa_tensor(inputTestArr, MAX_LENGTH_PROTEIN)
    targetTensorsTest = distances_to_bins(targetTensorsTest, suitable_bin_edges)
    torch.save(targetTensorsTest, PATH_TO_CORRECT_DISTANCE_MATRICES)
    ckpt = torch.load(PATH_MODEL_WEIGHTS, map_location=DEVICE)
    sd   = ckpt["state_dict"]       
    d_z = EMBEDDING_DIMENSION // 8
    model = EncoderOnlyTransformer(MAX_LENGTH_PROTEIN, EMBEDDING_DIMENSION, \
                                PRINT_EVERY_EPOCH, NR_OF_ATTENTION_HEADS, \
                                    DROPOUT_RATE, LEARNING_RATE, NR_T_ENCODE_LAYERS, d_z).to(DEVICE)

    model.load_state_dict(sd)
    
    predictions_list , predictions_np_list, rmse_list = EvaluateDM(inputTensorsTest,msa_tensor ,targetTensorsTest, model, device = DEVICE)
    
    torch.save(torch.tensor(np.array(predictions_np_list), device=DEVICE), PATH_TO_PREDICTED_DISTANCE_MATRICES)


wantToEvaluate = input("Do you want to evaluate for NN-Angles (y/n)?")

if wantToEvaluate.strip().lower() == "y": 
    MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
    NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
    MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS, PATH_LAST_EPOCH_MODEL_WEIGHTS, \
    PATH_TO_CORRECT_ANGLES, PATH_TO_PREDICTED_ANGLES,PATH_TO_TEST_OUTPUTS, PATH_TO_TEST_INPUTS = NN_Angles_Constants()

    inputTestArr = np.load(PATH_TO_TEST_INPUTS, allow_pickle=True)
    CaCoords_arr = np.load(PATH_TO_TEST_OUTPUTS, allow_pickle=True)

    inputVectors = GetInputVectors(inputTestArr, MAX_LENGTH_PROTEIN)

    inputTensorsTest, targetTensorsTest  = ConvertToTensorNNAngles(CaCoords_arr, inputVectors, MAX_LENGTH_PROTEIN, DEVICE)

    torch.save(targetTensorsTest, PATH_TO_CORRECT_ANGLES)

    ckpt = torch.load(PATH_MODEL_WEIGHTS, map_location=DEVICE)
    sd   = ckpt["state_dict"] 
    d_z = EMBEDDING_DIMENSION // 16 
    model = EncoderOnlyTransformerNNAngles(MAX_LENGTH_PROTEIN, EMBEDDING_DIMENSION, \
                                PRINT_EVERY_EPOCH, NR_OF_ATTENTION_HEADS, \
                                    DROPOUT_RATE, LEARNING_RATE, NR_T_ENCODE_LAYERS, d_z).to(DEVICE)

    model.load_state_dict(sd)  
    predictions_list , predictions_np_list, rmse_list = EvaluateNNAngles(inputTensorsTest, targetTensorsTest, model, device = DEVICE)
    
    torch.save(torch.tensor(np.array(predictions_np_list), device=DEVICE), PATH_TO_PREDICTED_ANGLES)


wantToEvaluate = input("Do you want to evaluate for DISTOGRAMS (y/n)?")

if wantToEvaluate.strip().lower() == "y": 

    MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
    NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
    MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS,PATH_LAST_EPOCH_MODEL_WEIGHTS, \
    PATH_TO_CORRECT_DISTANCE_MATRICES, PATH_TO_PREDICTED_DISTANCE_MATRICES,PATH_TO_TEST_OUTPUTS, PATH_TO_TEST_INPUTS = DM_Constants()


    inputTestArr = np.load(PATH_TO_TEST_INPUTS, allow_pickle=True)
    CaCoords_arr = np.load(PATH_TO_TEST_OUTPUTS, allow_pickle=True)

    inputVectors = GetInputVectors(inputTestArr, MAX_LENGTH_PROTEIN)

    inputTensorsTest, targetTensorsTest_DM = ConvertToTensor(CaCoords_arr, inputVectors, MAX_LENGTH_PROTEIN, DEVICE)
    suitable_bin_edges = torch.tensor([0.0, 3.5 ,4.0 , 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.5, 13.0, 15.0, 17.0, 21.0, 25.0])
    msa_tensor = build_msa_tensor(inputTestArr, MAX_LENGTH_PROTEIN)
    targetTensorsTest = distances_to_bins(targetTensorsTest_DM, suitable_bin_edges)
    torch.save(targetTensorsTest, PATH_TO_CORRECT_DISTANCE_MATRICES)
    ckpt = torch.load(PATH_MODEL_WEIGHTS, map_location=DEVICE)
    sd   = ckpt["state_dict"]       

    d_z = EMBEDDING_DIMENSION // 16
    model  = BinsEncoderOnlyTransformer(MAX_LENGTH_PROTEIN, EMBEDDING_DIMENSION, \
                                         PRINT_EVERY_EPOCH, NR_OF_ATTENTION_HEADS, \
                                             DROPOUT_RATE, LEARNING_RATE, NR_T_ENCODE_LAYERS, d_z, len(suitable_bin_edges)-1).to(DEVICE)

    model.load_state_dict(sd)
    
    predicted_distance_matrices_list, accuracies_list, rmse_list, lengths_list = EvaluateBINS(inputTensorsTest, msa_tensor,targetTensorsTest,targetTensorsTest_DM, model, device = DEVICE, bin_edges=suitable_bin_edges)
    
    torch.save(torch.tensor(np.array(predicted_distance_matrices_list), device=DEVICE), PATH_TO_PREDICTED_DISTANCE_MATRICES)



