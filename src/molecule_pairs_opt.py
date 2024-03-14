from src.molecular_pairs_set import MolecularPairsSet
import numpy as np
from src.molecule_pair import MoleculePair

class MoleculePairsOpt(MolecularPairsSet):
    """
    optimized version of molecule pairs set with the possiblitiy of working over unique smiles
    """

    def __init__(
        self, spectrums_original, spectrums_unique, df_smiles, indexes_tani_unique
    ):
        """
        it receives a set of spectrums, and a tuple with indexes i,j, tani tuple
        """
        self.spectrums_original = spectrums_original
        self.spectrums = spectrums_unique
        self.df_smiles = df_smiles  # table containing the indexes to map unique to repetitions of the same smiles
        # treat the first 2 columns as int and the 3 column as float
        self.indexes_tani = MolecularPairsSet.adjust_data_format(
            np.array(indexes_tani_unique)
        )

    
    def __add__(self, other):
        # only to be used when the spectrums are the same

        if self.are_spectrums_the_same(self.spectrums_original, other.spectrums_original):
            new_indexes_tani = np.concatenate(
                (self.indexes_tani, other.indexes_tani), axis=0
            )
            return MoleculePairsOpt(
                spectrums_unique=self.spectrums,
                spectrums_original=self.spectrums_original,
                indexes_tani_unique=new_indexes_tani,
                df_smiles=self.df_smiles,
            )
        else:
            print("ERROR: Attempting to add 2 set of spectrums with different content")
            return 0
        
    #def get_molecular_pair(self, index):
    #    raise Exception('Not implemented functionality')