
from spectrum_utils.spectrum import MsmsSpectrum
import numpy as np
from typing import Iterable, Union
import numpy as np

class SpectrumExt(MsmsSpectrum):
    ''''
    extended spectrum class that incorporates the binned vector
    '''
    #def __init__(self, **kwargs):
    #        super().__init__(**kwargs)  # Call the constructor of the base class
    #
    #        # extra variables
    #        self.spectrum_vector = np.array([])
    #        self.smiles = '' 

    def __init__(self,
        identifier: str,
        precursor_mz: float,
        precursor_charge: int,
        mz: Union[np.ndarray, Iterable],
        intensity: Union[np.ndarray, Iterable],
        retention_time: float = np.nan,):

        super().__init__(
        identifier,
        precursor_mz,
        precursor_charge,
        mz,
        intensity,
        retention_time) 

        # extra variables
        self.params= None
        self.intensity_array = None
        self.mz_array = None
        self.spectrum_vector=None
        self.smiles =None
        self.max_peak=None
        
    def set_params(self, params):
         self.params = params 

    def set_mz_array(self, mz_array):
         self.mz_array=mz_array 

    def set_intesity_array(self, intensity_array):
         self.intensity_array= intensity_array
    def set_spectrum_vector(self, spectrum_vector):
          self.spectrum_vector =spectrum_vector

    def set_smiles(self, smiles):
          self.smiles = smiles

    def set_max_peak(self, max_peak):
         '''
         set the maximum amplitude in the spectrum
         '''
         self.max_peak = max_peak