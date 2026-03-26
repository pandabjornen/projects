import os
def DM_Constants(): 
    MIN_LENGTH_PROTEIN = 20
    MAX_LENGTH_PROTEIN = 70
    #NOTE: CHANGE HERE BEFORE DOWNLOAD DATA!
    EMBEDDING_DIMENSION = 256
    BATCH_SIZE = 128
    #Training Hyperparams
    VAL_SPLIT = 0.20
    NR_OF_EPOCHS = 1000
    PRINT_EVERY_EPOCH = NR_OF_EPOCHS // 100
    PRINT_EVERY_EPOCH = 10

    #NEW CONSTANTS: 
    NR_OF_ATTENTION_HEADS  = 8
    LEARNING_RATE = 0.0001
    DROPOUT_RATE = 0.1
    NR_T_ENCODE_LAYERS = 2
    DEVICE = "mps"
    MIN_LENGTH_EXTRA_CHECK = 4
    MAX_DATA_LENGTH = 2000

    
    PATH_TO_CA_COORDS = "./data/"+str(MIN_LENGTH_PROTEIN)+"_"+str(MAX_LENGTH_PROTEIN)+"/np_arrays/train_val_coords.npy"
    PATH_TO_AA ="./data/"+str(MIN_LENGTH_PROTEIN)+"_"+str(MAX_LENGTH_PROTEIN)+"/np_arrays/train_val_seqs.npy"

    PATH_TO_TEST_OUTPUTS = "./data/"+str(MIN_LENGTH_PROTEIN)+"_"+str(MAX_LENGTH_PROTEIN)+"/np_arrays/test_coords.npy"
    PATH_TO_TEST_INPUTS = "./data/"+str(MIN_LENGTH_PROTEIN)+"_"+str(MAX_LENGTH_PROTEIN)+"/np_arrays/test_seqs.npy"

    RUN_COUNT = "07_20_70"

    
    MODEL_FILENAME = "epoch=318-val_rmse=3.1847.ckpt" #NOTE: FILL IN!
    FOLDER_MODEL_WEIGHTS = "distance_matrix/trained_models/model_"+RUN_COUNT+"_weights/"
    os.makedirs(FOLDER_MODEL_WEIGHTS, exist_ok=True)
    PATH_MODEL_WEIGHTS = FOLDER_MODEL_WEIGHTS+MODEL_FILENAME #NOTE Change path!
    PATH_LAST_EPOCH_MODEL_WEIGHTS = "distance_matrix/trained_models/model_0_weights/model_weights_"+RUN_COUNT+".pth"



    PATH_TO_CORRECT_DISTANCE_MATRICES = "distance_matrix/output_data/target_distance_matrices/cdm"+RUN_COUNT+".pt"
    PATH_TO_PREDICTED_DISTANCE_MATRICES = "distance_matrix/output_data/predicted_distance_matrices/pdm"+RUN_COUNT+".pt"


    DROPOUT_RATE = 0.1
    return MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
        NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
        MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS,PATH_LAST_EPOCH_MODEL_WEIGHTS, \
        PATH_TO_CORRECT_DISTANCE_MATRICES, PATH_TO_PREDICTED_DISTANCE_MATRICES,PATH_TO_TEST_OUTPUTS, PATH_TO_TEST_INPUTS

#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

def NN_Angles_Constants(): 
    MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
    NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
    MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS,PATH_LAST_EPOCH_MODEL_WEIGHTS, \
    PATH_TO_CORRECT_DISTANCE_MATRICES, PATH_TO_PREDICTED_DISTANCE_MATRICES,PATH_TO_TEST_OUTPUTS, PATH_TO_TEST_INPUTS = DM_Constants()

    PATH_TO_CORRECT_ANGLES = "angles/output_data/target_angles/ca"+RUN_COUNT+".pt"
    PATH_TO_PREDICTED_ANGLES = "angles/output_data/predicted_angles/pa"+RUN_COUNT+".pt" 

    MODEL_FILENAME = "epoch=2999-val_rmse=3.3255.ckpt" #NOTE: FILL IN!
    FOLDER_MODEL_WEIGHTS = "angles/trained_models/model_0_weights/"
    PATH_MODEL_WEIGHTS = FOLDER_MODEL_WEIGHTS+MODEL_FILENAME #NOTE Change path!

    LEARNING_RATE = 3e-5
    NR_T_ENCODE_LAYERS = 1
    BATCH_SIZE = 128


    return MIN_LENGTH_PROTEIN, MAX_LENGTH_PROTEIN,EMBEDDING_DIMENSION, BATCH_SIZE, VAL_SPLIT, NR_OF_EPOCHS, PRINT_EVERY_EPOCH,\
        NR_OF_ATTENTION_HEADS, LEARNING_RATE, DROPOUT_RATE, NR_T_ENCODE_LAYERS, DEVICE, MIN_LENGTH_EXTRA_CHECK, \
        MAX_DATA_LENGTH, PATH_TO_CA_COORDS, PATH_TO_AA, RUN_COUNT, MODEL_FILENAME, PATH_MODEL_WEIGHTS, FOLDER_MODEL_WEIGHTS, PATH_LAST_EPOCH_MODEL_WEIGHTS, \
        PATH_TO_CORRECT_ANGLES, PATH_TO_PREDICTED_ANGLES,PATH_TO_TEST_OUTPUTS, PATH_TO_TEST_INPUTS


        