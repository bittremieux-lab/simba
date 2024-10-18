class Config:
    # default configuration
    # Spectra and spectrum pairs to include with the following settings.
    def __init__(self):
        

        # MULTITASKING
        self.COLUMN_EDIT_DISTANCE=2
        self.COLUMN_MCES20 = 3
        
        #PREPROCESSING
        self.PREPROCESSING_BATCH_SIZE=1000
        self.PREPROCESSING_NUM_WORKERS=60
        self.PREPROCESSING_OVERWRITE=False #overwrite the output file during generation
        self.COMPUTE_SPECIFIC_PAIRS=True
        self.FORMAT_FILE_SPECIFIC_PAIRS='INPUT_SPECIFIC_PAIRS_indexes_tani_incremental' # the prefix of the file containing the indexes to be computed
        self.USE_EDIT_DISTANCE=False ## If using edit distance for generating data, not for training!!! 
        self.SUBSAMPLE_PREPROCESSING=False
        self.RANDOM_MCES_SAMPLING = False
        self.CHARGES = 0, 1
        self.MIN_N_PEAKS = 6
        self.FRAGMENT_MZ_TOLERANCE = 0.1
        self.MIN_MASS_DIFF = 0  # Da
        self.MAX_MASS_DIFF = 200  # Da
        self.THRESHOLD_MCES=20

        # training
        self.USE_MCES20_LOG_LOSS=False ### apply log function to increase the weight of the differences in the low range
        self.USE_EDIT_DISTANCE_REGRESSION=False
        self.USE_MULTITASK=True
        self.EDIT_DISTANCE_N_CLASSES=6
        self.EDIT_DISTANCE_USE_GUMBEL=False
        self.USE_TANIMOTO =False # using Tanimoto or MCES20 for training
        self.MCES20_MAX_VALUE=40 # value used as midpoint for normalization. 19 it is chosen to make NORMALIZED_MCES to be in the range below 0.49 and make it consider a low similarity pair
        self.USE_LOSS_WEIGHTS_SECOND_SIMILARITY= True # use weights for training the second similarity of multitasking
        self.N_LAYERS = 5  # transformer parameters
        self.D_MODEL = 256  # transformer parameters
        self.EMBEDDING_DIM=512
        self.use_cosine_distance = True
        self.LR = 1e-4
        self.epochs = 1000
        self.BATCH_SIZE = 128
        self.enable_progress_bar = True
        self.threshold_class = 0.7  # threshold classification binary
        self.load_maldi_embedder = False
        self.INFERENCE_USE_LAST_MODEL=False
        self.maldi_embedder_path = (
            "/scratch/antwerpen/209/vsc20939/data/maldi_embedder/best_model.ckpt"
        )
        self.load_pretrained = False  # a whole SIMBA model
        self.dataset_path=  "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240319_unique_smiles_100_million_v2_no_identity.pkl"
        self.use_uniform_data_TRAINING = False
        self.bins_uniformise_TRAINING = 10
        self.use_uniform_data_INFERENCE = True
        self.bins_uniformise_INFERENCE = 10
        self.validate_after_ratio = 0.0010  # it indicates the interval between validations. O.1 means 10 validations in 1 epoch
        self.extra_info = "_multitasking_mces20raw_gumbelhard_20241004"
        self.derived_variables()
        self.PREPROCESSING_DIR=f"/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925/"
        self.PREPROCESSING_PICKLE_FILE= f"edit_distance_neurips_nist_exhaustive.pkl"

    def derived_variables(self):
        self.MODEL_CODE = f"{self.D_MODEL}_units_{self.N_LAYERS}_layers_{self.epochs}_epochs_{self.LR}_lr_{self.BATCH_SIZE}_bs{self.extra_info}"
        self.CHECKPOINT_DIR = f"/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_checkpoints_{self.MODEL_CODE}/"
        
        self.BEST_MODEL_NAME=f"best_model.ckpt"
        self.PRETRAINED_MODEL_NAME=f"pretrained_model.ckpt"
        self.pretrained_path = self.CHECKPOINT_DIR + self.PRETRAINED_MODEL_NAME
        self.best_model_path = self.CHECKPOINT_DIR + self.BEST_MODEL_NAME
        
