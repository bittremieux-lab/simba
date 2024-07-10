import numpy as np


class OrdinalClassification:

    @staticmethod
    def from_float_to_class(array, N_classes):
        '''
        convert a float between 0 and 1 to an integer value between 0 and N_max
        '''
        return OrdinalClassification.custom_random(array*(N_classes-1)).astype(int)
    
    @staticmethod
    def custom_random(array):
        '''
        round the percentage values to integer, letting 1.5 -> 2
        '''
        return np.floor(array+0.51)