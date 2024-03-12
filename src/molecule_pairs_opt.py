from src.molecular_pairs_set import MolecularPairsSet
import numpy as np


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
