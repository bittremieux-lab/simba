


class OrdinalClassification:

    @staticmethod
    def from_float_to_class(array, N_classes):
        '''
        convert a float between 0 and 1 to an integer value between 0 and N_max
        '''
        return (array*(N_classes-1)).astype(int)