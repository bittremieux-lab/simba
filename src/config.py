class Config:
    # default configuration
    # Spectra and spectrum pairs to include with the following settings.
    def __init__(self):

        #PREPROCESSING
        self.PREPROCESSING_NUM_WORKERS=60
        self.USE_EDIT_DISTANCE=True
        self.SUBSAMPLE_PREPROCESSING=False
        self.RANDOM_MCES_SAMPLING = True
        self.CHARGES = 0, 1
        self.MIN_N_PEAKS = 6
        self.FRAGMENT_MZ_TOLERANCE = 0.1
        self.MIN_MASS_DIFF = 0  # Da
        self.MAX_MASS_DIFF = 200  # Da

        # training
        self.N_LAYERS = 5  # transformer parameters
        self.D_MODEL = 128  # transformer parameters
        self.EMBEDDING_DIM=256
        self.use_cosine_distance = True
        self.LR = 1e-4
        # self.LR = 1e-3
        self.epochs = 1000
        self.BATCH_SIZE = 128
        self.enable_progress_bar = True
        self.threshold_class = 0.7  # threshold classification binary

        self.load_maldi_embedder = False
        self.maldi_embedder_path = (
            "/scratch/antwerpen/209/vsc20939/data/maldi_embedder/best_model.ckpt"
        )
        self.load_pretrained = False  # a whole SIMBA model

        #self.dataset_path = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240319_unique_smiles_1_million_v2_no_sim1.pkl"
        self.dataset_path=  "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240319_unique_smiles_100_million_v2_no_identity.pkl"
        self.use_uniform_data_TRAINING = False
        self.bins_uniformise_TRAINING = 10

        self.use_uniform_data_INFERENCE = True
        self.bins_uniformise_INFERENCE = 10
        self.validate_after_ratio = 0.0010  # it indicates the interval between validations. O.1 means 10 validations in 1 epoch
        self.extra_info = "_edit_distance_loaded_full"
        self.derived_variables()
        self.PREPROCESSING_DIR=f"/scratch/antwerpen/209/vsc20939/data/preprocessing_edit_distance_loaded_full/"
        self.PREPROCESSING_PICKLE_FILE= f"edit_distance_neurips_nist_exhaustive.pkl"

    def derived_variables(self):
        self.MODEL_CODE = f"{self.D_MODEL}_units_{self.N_LAYERS}_layers_{self.epochs}_epochs_{self.LR}_lr_{self.BATCH_SIZE}_bs{self.extra_info}"
        self.CHECKPOINT_DIR = f"/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_checkpoints_{self.MODEL_CODE}/"
        self.pretrained_path = self.CHECKPOINT_DIR + f"best_model.ckpt"
        self.best_model_path = self.CHECKPOINT_DIR + f"best_model.ckpt"
        
