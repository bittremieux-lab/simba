



class Config:
    # Spectra and spectrum pairs to include with the following settings.
    CHARGES = 0, 1
    MIN_N_PEAKS = 6
    FRAGMENT_MZ_TOLERANCE = 0.1
    MIN_MASS_DIFF = 1    # Da
    MAX_MASS_DIFF = 200    # Da
    
    # training
    n_layers=1 #transformer parameters
    d_model=8 #transformer parameters
    epochs=3
    enable_progress_bar=True
    threshold_class=0.6 #threshold classification binary
    MODEL_CODE= f'{d_model}_units_{n_layers}_layers_{epochs}_epochs'

