



class Config:
    # Spectra and spectrum pairs to include with the following settings.
    CHARGES = 0, 1
    MIN_N_PEAKS = 6
    FRAGMENT_MZ_TOLERANCE = 0.1
    MIN_MASS_DIFF = 0    # Da
    MAX_MASS_DIFF = 200    # Da
    
    # training
    n_layers=2 #transformer parameters
    d_model=16 #transformer parameters
    epochs=200
    enable_progress_bar=False
    threshold_class=0.6 #threshold classification binary
    MODEL_CODE= f'{d_model}_units_{n_layers}_layers_{epochs}_epochs'
    load_pretrained=True
    pretrained_path = f'/scratch/antwerpen/209/vsc20939/metabolomics/model_checkpoints_{d_model}_{n_layers}/best_model_pretrained.ckpt'
