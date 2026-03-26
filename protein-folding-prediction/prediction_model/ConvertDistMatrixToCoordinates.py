import torch
import matplotlib.pyplot as plt
import numpy as np
from coordinate_realizer.GradDescentRealizer import OptimizeCoordsFromDM
from coordinate_realizer.RotateTranslate import KabschAlign
from show.PlotOneProteinAAsColors import PlotOneProtein
from show.PlotTwoProteins import PlotTwoProteins
from Constants import DM_Constants
from general_preprocess.GetInputsAndTargets import GetInputVectors
from distance_matrix.specific_preprocess.CalculateDistanceMatrix import CalculateDistanceMatrix
#--------------------------------------------------------------------------------------------------------------------------------------------
MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS,PATH_LAST_EPOCH_MODEL_WEIGHTS, \
PATH_TO_CORRECT_DISTANCE_MATRICES, PATH_TO_PREDICTED_DISTANCE_MATRICES,PATH_TO_TEST_OUTPUTS, PATH_TO_TEST_INPUTS = DM_Constants()

pathWhereToSavePredictedCoordinates = "coordinate_realizer/predicted_coordinates/pc"+RUN_COUNT+".npy"
#--------------------------------------------------------------------------------------------------------------------------------------------
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 100
EPOCHS = 1000
PRINT_EVERY_EPOCH = 10
NR_OF_MATRICES_TO_TEST_INIT = 1
patienceNrOfEpochs = 40
optimizer_type = "adam"

LEARNING_RATE = 0.1
MAX_ITERATIONS = 100
EPOCHS = 5
PRINT_EVERY_EPOCH = 1
NR_OF_MATRICES_TO_TEST_INIT = 100
patienceNrOfEpochs = 5
optimizer_type = "lbfgs"
#--------------------------------------------------------------------------------------------------------------------------------------------
def main():

    D = torch.load(PATH_TO_PREDICTED_DISTANCE_MATRICES)
    
    NR_OF_MATRICES_TO_TEST = min(len(D), NR_OF_MATRICES_TO_TEST_INIT)
    
    # predicted_distance_matrices_list= []

    # for i in range(D.size(0)): 
    #     for j in range(1, D.size(2)): 
    #         if D[i, 0, j] == 0: 
    #             noPaddingD = D[i, :j, :j]
    #             predicted_distance_matrices_list.append(noPaddingD)
    #             break

    # D: (N, N, N) tensor, zero-padded on the right/bottom
    # find first zero in D[i,0,1:]
    mask = D[:, 0] == 0             # shape (N, N)
    mask[:, 0] = False              # ignore the [i,0,0] diagonal
    lengths = mask.int().argmax(1)  # first True per row → cut index

    # now slice each matrix exactly once
    predicted_distance_matrices_list = [
        D[i, :L, :L]
        for i, L in enumerate(lengths.tolist())
    ]

    print("first loop done")
    
    correctCoordinates_np_arr =np.load(PATH_TO_TEST_OUTPUTS, allow_pickle=True)
    testInputAA = np.load(PATH_TO_TEST_INPUTS, allow_pickle=True)
    
    correctCoordinates_list = []

    testInputAA_list = []

    correctDistanceMatrices_list= []
    for j, CaCoords in enumerate(correctCoordinates_np_arr[:NR_OF_MATRICES_TO_TEST]):
        correctDistanceMatrix = CalculateDistanceMatrix(CaCoords, len(CaCoords))
        correctDistanceMatrices_list.append(torch.tensor(correctDistanceMatrix, device=DEVICE, dtype=torch.float))

        coords = torch.tensor(CaCoords, dtype=torch.float, device=DEVICE)
        correctCoordinates_list.append(coords)

        aa_tensor = torch.tensor(testInputAA[j][:len(CaCoords)], dtype=torch.long, device=DEVICE)
        testInputAA_list.append(aa_tensor)

    print("second loop done")

    print("second loop done")

    # bara behåll proteiner med ≥ MIN_ATOMS atomer **och** motsvarande DM har ≥ MIN_ATOMS rader
    MIN_ATOMS = 2
    valid_idx = [
        i for i in range(len(correctCoordinates_list))
        if correctCoordinates_list[i].size(0) >= MIN_ATOMS
        and predicted_distance_matrices_list[i].size(0) >= MIN_ATOMS
    ]

    # filtrera ALLA listor med samma index
    predicted_distance_matrices_list = [predicted_distance_matrices_list[i] for i in valid_idx]
    correctDistanceMatrices_list      = [correctDistanceMatrices_list[i]      for i in valid_idx]
    correctCoordinates_list           = [correctCoordinates_list[i]           for i in valid_idx]
    testInputAA_list                  = [testInputAA_list[i]                  for i in valid_idx]

    # nu kan du göra cappen mot NR_OF_MATRICES_TO_TEST
    testInputAA_list                = testInputAA_list[:NR_OF_MATRICES_TO_TEST]
    correctCoordinates_list         = correctCoordinates_list[:NR_OF_MATRICES_TO_TEST]
    predicted_distance_matrices_list = predicted_distance_matrices_list[:NR_OF_MATRICES_TO_TEST]
    correctDistanceMatrices_list    = correctDistanceMatrices_list[:NR_OF_MATRICES_TO_TEST]

    NR_OF_MATRICES_TO_TEST = min(NR_OF_MATRICES_TO_TEST, len(predicted_distance_matrices_list))
    
    wantToPlot = input("want to plot distance matrices (y/n)?")
    if wantToPlot.lower().strip() == "y":
        for i in range(D.shape[0]):
           
            fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

            im1 = axes[0].imshow(predicted_distance_matrices_list[i].cpu().detach().numpy())
            axes[0].set_title("PREDICTED")
            fig.colorbar(im1, ax=axes[0])
            
            correct_distance_matrix = CalculateDistanceMatrix(correctCoordinates_np_arr[i], MAX_LENGTH_PROTEIN)
            im2 = axes[1].imshow(correct_distance_matrix)
            axes[1].set_title("CORRECT")
            fig.colorbar(im2, ax=axes[1])

            plt.tight_layout() 
            plt.show()
            
    
    optmCorrDM = input("opt correct DM (y/n)?")
    optmPredDM = input("opt pred DM (y/n)?")

    if optmCorrDM.lower().strip() != "y" and optmPredDM.lower().strip() != "y": 
        return 

    if optmCorrDM.lower().strip() == "y": 
        predictedCoordinatesFromCorrectDistanceMatrices_list = []
        rmse_list = []
        for i in range(NR_OF_MATRICES_TO_TEST):
            predictedCoordinatesFromCorrectDistanceMatrix, rmse = OptimizeCoordsFromDM(correctDistanceMatrices_list[i], LEARNING_RATE, MAX_ITERATIONS, EPOCHS, PRINT_EVERY_EPOCH, patienceNrOfEpochs, optimizer_type)
            predictedCoordinatesFromCorrectDistanceMatrices_list.append(predictedCoordinatesFromCorrectDistanceMatrix)
            rmse_list.append(rmse.cpu().detach())
        print(f"FOR {NR_OF_MATRICES_TO_TEST} PROTEINS. CRRECT RMSE AVERAGE AFTER OPT: {np.mean(rmse_list):.8f}")
    if optmPredDM.lower().strip() == "y":  
        predictedCoordinates_list = [] 
        rmse_list_pred =[]
        for i in range(NR_OF_MATRICES_TO_TEST):
            predictedCoordinates, rsmepred = OptimizeCoordsFromDM(predicted_distance_matrices_list[i], LEARNING_RATE,MAX_ITERATIONS , EPOCHS, PRINT_EVERY_EPOCH, patienceNrOfEpochs, optimizer_type)
            predictedCoordinates_list.append(predictedCoordinates) 
            rmse_list_pred.append(rsmepred.cpu().detach())
        print(f"FOR {NR_OF_MATRICES_TO_TEST} PROTEINS. CRRECT RMSE AVERAGE AFTER OPT: {np.mean(rmse_list_pred):.8f}")
    wantToPlot = input("want to plot proteins before alignment (y/n)?")
    if wantToPlot.lower().strip() == "y":
        if optmPredDM.lower().strip() == "y": 
            for n, predTorch in enumerate(predictedCoordinates_list): 
                prednpArr = predTorch.cpu().detach().numpy()
                PlotTwoProteins(prednpArr, correctCoordinates_np_arr[n], title = "predicted DM")
        if optmCorrDM.lower().strip() == "y": 
            for n, corrpredtorch in enumerate(predictedCoordinatesFromCorrectDistanceMatrices_list): 
                corrprednpArr = corrpredtorch.cpu().detach().numpy()
                PlotTwoProteins(corrprednpArr, correctCoordinates_np_arr[n], title = "Correct DM")
        

    predictionsAlignedList = []
    rmse_before_aligned_list = []
    rmse_after_aligned_list = []

    correct_predictionsAlignedList = []
    correct_rmse_before_aligned_list = []
    correct_rmse_after_aligned_list = []

    for i in range(NR_OF_MATRICES_TO_TEST):
                
        corr = correctCoordinates_list[i] 

        if optmPredDM.lower().strip() == "y":

            pred = predictedCoordinates_list[i] 
            L = min(pred.size(0), corr.size(0))
            
            pred = pred[:L]
            corr = corr[:L]
            diff_before = (pred - corr).pow(2)
            mse_before = diff_before.sum() / (pred.size(0) * 3)
            rmse_b = torch.sqrt(mse_before)
        
            
            rmse_before_aligned_list.append(rmse_b.cpu().detach())
            print(f"Before rotations and translations: Coordinate RMSE: {rmse_b:.4f}")

            pred_aligned, rmseA = KabschAlign(pred.cpu(), corr.cpu())
            
            print(f"After rotations and translations: Coordinate RMSE: {rmseA:.4f}")
            rmse_after_aligned_list.append(rmseA)
            predAlignedNP = pred_aligned.cpu().detach().numpy()
            predictionsAlignedList.append(predAlignedNP)

        if optmCorrDM.lower().strip() == "y":
            
            pred_correct = predictedCoordinatesFromCorrectDistanceMatrices_list[i] 
            
            L = min(pred_correct.size(0), corr.size(0))
            pred_correct = pred_correct[:L]
            corr = corr[:L]
            diff_correct_before = (pred_correct - corr).pow(2)
            mse_correct_before = diff_correct_before.sum() / (pred_correct.size(0)*3)
            rmse_correct_b = torch.sqrt(mse_correct_before)
            

            correct_rmse_before_aligned_list.append(rmse_correct_b.cpu().detach())
            print(f"CORRECT DM: Before rotations and translations: Coordinate RMSE: {rmse_correct_b:.4f}")

            
            pred_aligned_correct, rmseA_correct = KabschAlign(pred_correct.cpu(), corr.cpu())

            print(f"CORRECT DM: After rotations and translations: Coordinate RMSE: {rmseA_correct:.4f}")
            correct_rmse_after_aligned_list.append(rmseA_correct)
            predAlignedNP_correct = pred_aligned_correct.cpu().detach().numpy()
            correct_predictionsAlignedList.append(predAlignedNP_correct)

    if rmse_before_aligned_list:
        print(f"FOR {NR_OF_MATRICES_TO_TEST} PROTEINS. Before rotations and translations: AVERAGE Coordinate RMSE: {np.mean(rmse_before_aligned_list):.4f}")
    if rmse_after_aligned_list:
        print(f"FOR {NR_OF_MATRICES_TO_TEST} PROTEINS. After rotations and translations: AVERAGE Coordinate RMSE: {np.mean(rmse_after_aligned_list):.4f}")
    if correct_rmse_before_aligned_list:
        print(f"FOR {NR_OF_MATRICES_TO_TEST} PROTEINS. CORRECT DM: Before rotations and translations: AVERAGE Coordinate RMSE: {np.mean(correct_rmse_before_aligned_list):.4f}")
    if correct_rmse_after_aligned_list:
        print(f"FOR {NR_OF_MATRICES_TO_TEST} PROTEINS. CORRECT DM: After rotations and translations: AVERAGE Coordinate RMSE: {np.mean(correct_rmse_after_aligned_list):.4f}")

    wantToPlot = input("want to plot proteins after alignment (y/n)?")
    if wantToPlot.lower().strip() == "y":
        if optmPredDM.lower().strip() == "y": 
            for n, prednpArr in enumerate(predictionsAlignedList): 
                PlotTwoProteins(prednpArr, correctCoordinates_np_arr[n], title = "predicted DM")
        if optmCorrDM.lower().strip() == "y": 
            for n, corr_prednpArr in enumerate(correct_predictionsAlignedList): 
                PlotTwoProteins(corr_prednpArr, correctCoordinates_np_arr[n], title = "Correct DM")


    if optmPredDM.lower().strip() == "y": 
        np.save(pathWhereToSavePredictedCoordinates, np.array(predictionsAlignedList, dtype=object), allow_pickle=True)

if __name__ == "__main__":
    main()