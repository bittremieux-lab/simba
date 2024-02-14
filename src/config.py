class Config:
    # default configuration
    # Spectra and spectrum pairs to include with the following settings.
    def __init__(self):
        self.CHARGES = 0, 1
        self.MIN_N_PEAKS = 6
        self.FRAGMENT_MZ_TOLERANCE = 0.1
        self.MIN_MASS_DIFF = 1  # Da
        self.MAX_MASS_DIFF = 200  # Da

        # training
        self.N_LAYERS = 10  # transformer parameters
        self.D_MODEL = 128  # transformer parameters
        self.LR = 1e-4
        self.epochs = 50
        self.BATCH_SIZE = 1024
        self.enable_progress_bar = True
        self.threshold_class = 0.7  # threshold classification binary

        self.load_pretrained = False

        self.dataset_path = "/scratch/antwerpen/209/vsc20939/data/merged_gnps_nist_20240207_gnps_nist_janssen_15_millions.pkl"

        self.use_uniform_data_TRAINING = False
        self.bins_uniformise_TRAINING = 10

        self.use_uniform_data_INFERENCE = True
        self.bins_uniformise_INFERENCE = 10
        self.validate_after_ratio = 0.0010  # it indicates the interval between validations. O.1 means 10 validations in 1 epoch
        self.derived_variables()

    def derived_variables(self):
        self.MODEL_CODE = f"{self.D_MODEL}_units_{self.N_LAYERS}_layers_{self.epochs}_epochs_{self.LR}_lr_{self.BATCH_SIZE}_bs"
        self.CHECKPOINT_DIR = f"/scratch/antwerpen/209/vsc20939/data/model_checkpoints/model_checkpoints_{self.MODEL_CODE}/"
        self.pretrained_path = self.CHECKPOINT_DIR + f"best_model.ckpt"
        self.best_model_path = self.CHECKPOINT_DIR + f"best_model.ckpt"
