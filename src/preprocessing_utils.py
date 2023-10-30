
import numpy as np
from itertools import groupby

class PreprocessingUtils:

    @staticmethod
    def is_centroid(intensity_array):
        return np.all(intensity_array > 0)
    
    @staticmethod
    def order_by_charge(spectrums): 
        spectrums_new = spectrums.copy()

        # Sort the list based on the property 'x' (optional, but required for groupby)
        spectrums_new.sort(key=lambda a: a.precursor_charge)

        # Group the elements based on the 'x' property
        spectrums_by_charge = {}
        for key, group in groupby(spectrums_new, key=lambda a: a.precursor_charge):
            spectrums_by_charge[key] = list(group)
        return spectrums_by_charge

    @staticmethod
    def order_spectrums_by_mz(spectrums):
        '''
        in order to take into account mass differences
        '''

        spectrums_by_charge =  PreprocessingUtils.order_by_charge(spectrums) # return a dictionary
        
        total_spectrums=[]
        for charge in spectrums_by_charge:

            # order by mz
            mzs = np.array([s.precursor_mz for s in spectrums_by_charge[charge]])
            ordered_indexes= np.argsort(mzs)
            temp_spectrums = [spectrums_by_charge[charge][r] for r in ordered_indexes]
            total_spectrums = total_spectrums + temp_spectrums

        return total_spectrums
        
