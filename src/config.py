



class Config:
    # default configuration
    # Spectra and spectrum pairs to include with the following settings.
    def __init__(self):
        self.CHARGES = 0, 1
        self.MIN_N_PEAKS = 6
        self.FRAGMENT_MZ_TOLERANCE = 0.1
        self.MIN_MASS_DIFF = 0    # Da
        self.MAX_MASS_DIFF = 200    # Da
        
        # training
        self.N_LAYERS=5 #transformer parameters
        self.D_MODEL=32 #transformer parameters
        self.LR=1e-6
        self.epochs=200
        self.BATCH_SIZE=128
        self.enable_progress_bar=True
        self.threshold_class=0.6 #threshold classification binary
        
        self.load_pretrained=False

        self.derived_variables()
        
    def derived_variables(self):
        self.MODEL_CODE= f'{self.D_MODEL}_units_{self.N_LAYERS}_layers_{self.epochs}_epochs_{self.LR}_lr_{self.BATCH_SIZE}_bs'
        self.pretrained_path = f'/scratch/antwerpen/209/vsc20939/metabolomics/model_checkpoints_{self.MODEL_CODE}/best_model_pretrained.ckpt'
        self.CHECKPOINT_DIR=f'./model_checkpoints_{self.MODEL_CODE}/'
