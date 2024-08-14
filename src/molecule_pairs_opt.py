from src.molecular_pairs_set import MolecularPairsSet
import numpy as np
from src.molecule_pair import MoleculePair


class MoleculePairsOpt(MolecularPairsSet):
    """
    optimized version of molecule pairs set with the possiblitiy of working over unique smiles
    """

    def __init__(
        self, spectrums_original, spectrums_unique, df_smiles, indexes_tani_unique, tanimotos=None
    ):
        """
        it receives a set of spectrums, and a tuple with indexes i,j, tani tuple
        """
        self.spectrums_original = spectrums_original
        self.spectrums = spectrums_unique
        self.df_smiles = df_smiles  # table containing the indexes to map unique to repetitions of the same smiles
        # treat the first 2 columns as int and the 3 column as float
        #self.indexes_tani = MolecularPairsSet.adjust_data_format(
        #    np.array(indexes_tani_unique)
        #)
        self.indexes_tani = indexes_tani_unique
        self.tanimotos=tanimotos

    def __add__(self, other):
        # only to be used when the spectrums are the same

        if self.are_spectrums_the_same(
            self.spectrums_original, other.spectrums_original
        ):
            new_indexes_tani = np.concatenate(
                (self.indexes_tani, other.indexes_tani), axis=0
            )
            if (self.tanimotos is not None) and (other.tanimotos is not None):
                tanimotos= np.concatenate( (self.tanimotos, other.tanimotos), axis=0)
            else:
                tanimotos=None
            return MoleculePairsOpt(
                spectrums_unique=self.spectrums,
                spectrums_original=self.spectrums_original,
                indexes_tani_unique=new_indexes_tani,
                df_smiles=self.df_smiles,
                tanimotos=tanimotos,
            )
        else:
            print("ERROR: Attempting to add 2 set of spectrums with different content")
            return 0

    def get_molecular_pair(self, index):
        """
        get a molecular pair.
        For the first molecule of the pair, retrieve the first element, for the second element retrieve the last index
        this is to avoid to retrieve the same spectrum when the indexes are the same : sim=1
        """
        # i,j,tani = self.indexes_tani[index]
        i = int(self.indexes_tani[index, 0])
        j = int(self.indexes_tani[index, 1])
        tani = self.indexes_tani[index, 2]

        molecule_pair = MoleculePair(
            vector_0=None,
            vector_1=None,
            smiles_0=self.spectrums[i].smiles,
            smiles_1=self.spectrums[j].smiles,
            similarity=tani,
            global_feats_0=MolecularPairsSet.get_global_variables(self.spectrums[i]),
            global_feats_1=MolecularPairsSet.get_global_variables(self.spectrums[j]),
            index_in_spectrum_0=self.get_original_index_from_unique_index(
                i, 0
            ),  # index in the spectrum list used as input
            index_in_spectrum_1=self.get_original_index_from_unique_index(j, 1),
            spectrum_object_0=self.get_original_spectrum_from_unique_index(i, 0),
            spectrum_object_1=self.get_original_spectrum_from_unique_index(j, 1),
            params_0=self.get_original_spectrum_from_unique_index(i,0).params,
            params_1=self.get_original_spectrum_from_unique_index(j,1).params,
        )

        return molecule_pair

    def get_original_spectrum_from_unique_index(self, unique_index, pair):

        return self.spectrums_original[
            self.get_original_index_from_unique_index(unique_index, pair)
        ]

    def get_original_index_from_unique_index(self, index, pair):
        """
        obtain the mapped spectrum from index computed in the unique compound space
        if pair=0, return the first index, else return the last index
        """
        if pair == 0:
            return self.df_smiles.loc[index, "indexes"][0]
        else:
            return self.df_smiles.loc[index, "indexes"][-1]

    def get_spectrums_from_indexes(self, pair_index):
        # pair index refers if it is 0 or 1 in the pair
        indexes = [index for index in self.indexes_tani[:, pair_index]]
        original_indexes = [
            self.get_original_index_from_unique_index(index, pair_index)
            for index in indexes
        ]
        return [self.spectrums_original[index] for index in original_indexes]

    def get_sampled_spectrums(self):
        """
        retrieve the sampled spectrums for the first and second molecule of the pairs
        """
        spectrums_index_0 = self.get_spectrums_from_indexes(0)
        spectrums_index_1 = self.get_spectrums_from_indexes(1)
        return spectrums_index_0, spectrums_index_1

