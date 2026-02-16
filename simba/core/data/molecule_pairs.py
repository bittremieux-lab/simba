import numpy as np
import pandas as pd

from simba.core.data.spectrum import SpectrumExt


class MoleculePair:

    def __init__(
        self,
        vector_0=None,
        vector_1=None,
        smiles_0=None,
        smiles_1=None,
        similarity=None,
        global_feats_0=None,
        global_feats_1=None,
        index_in_spectrum_0=None,
        index_in_spectrum_1=None,
        spectrum_object_0=None,
        spectrum_object_1=None,
        params_0=None,
        params_1=None,
        fingerprint_0=None,
        fingerprint_1=None,
    ):

        self.spectrum_object_0 = spectrum_object_0
        self.spectrum_object_1 = spectrum_object_1
        self.vector_0 = vector_0
        self.vector_1 = vector_1
        self.global_feats_0 = global_feats_0
        self.global_feats_1 = global_feats_1
        self.smiles_0 = smiles_0
        self.smiles_1 = smiles_1
        self.similarity = similarity
        self.index_in_spectrum_0 = index_in_spectrum_0
        self.index_in_spectrum_1 = index_in_spectrum_1
        self.params_0 = params_0
        self.params_1 = params_1
        self.fingerprint_0 = fingerprint_0
        self.fingerprint_1 = fingerprint_1
        self.deterministic_similarity = {}

    def set_det_similarity_score(self, score, similarity_score):

        self.deterministic_similarity[similarity_score] = score

    def __str__(self):
        return f"Molecular pair with similarity: {self.similarity} for smiles_0: {self.smiles_0} and smiles_1: {self.smiles_1}"


class MolecularPairsSet:
    """
    class that encapsulates the indexes and the spectra from where they are retrieved
    """

    def __init__(self, spectra, pair_distances):
        """
        it receives a list of spectra, and a 2D array with the indexes of the spectra
        and the distances between them
        """
        self.spectra = spectra
        self.pair_distances = pair_distances

    @staticmethod
    def adjust_data_format(indexes_tani):
        # Extracting the first two columns and changing their data type to int
        # int_columns = indexes_tani[:, 0:2].astype(np.int32)

        # Extracting the last column and changing its data type to float
        # float_column = indexes_tani[:, 2].astype(np.float16)

        # Combining the modified columns to create a new array
        # new_indexes_tani = np.column_stack((int_columns, float_column))
        return indexes_tani

    def __len__(self):
        return len(self.pair_distances)

    def spectra_equal(self, spectra_0, spectra_1):
        spectra_hash_0 = [s.spectrum_hash for s in spectra_0]
        spectra_hash_1 = [s.spectrum_hash for s in spectra_1]
        return all(
            [s0 == s1 for s0, s1 in zip(spectra_hash_0, spectra_hash_1, strict=False)]
        )

    def __add__(self, other):
        # only to be used when the spectra are the same

        if self.spectra_equal(self.spectra, other.spectra):
            new_spectra = self.spectra
            new_pair_distances = np.concatenate(
                (self.pair_distances, other.pair_distances), axis=0
            )
            return MolecularPairsSet(
                spectra=new_spectra, pair_distances=new_pair_distances
            )
        else:
            print(
                "ERROR: Attempting to add 2 set of spectra with different content"
            )
            return 0

    def __getitem__(self, index):
        return self.get_molecular_pair(index)

    @staticmethod
    def get_global_variables(spectrum):
        """
        get global variables from a spectrum such as precursor mass
        """
        list_global_variables = [
            spectrum.precursor_mz,
            spectrum.precursor_charge,
        ]
        return np.array(list_global_variables)

    def get_molecular_pair(self, index):
        # i,j,tani = self.indexes_tani[index]
        i = int(self.pair_distances[index, 0])
        j = int(self.pair_distances[index, 1])
        tani = self.pair_distances[index, 2]

        molecule_pair = MoleculePair(
            vector_0=None,
            vector_1=None,
            smiles_0=self.spectra[i].smiles,
            smiles_1=self.spectra[j].smiles,
            similarity=tani,
            global_feats_0=MolecularPairsSet.get_global_variables(
                self.spectra[i]
            ),
            global_feats_1=MolecularPairsSet.get_global_variables(
                self.spectra[j]
            ),
            index_in_spectrum_0=i,  # index in the spectrum list used as input
            index_in_spectrum_1=j,
            spectrum_object_0=self.spectra[i],
            spectrum_object_1=self.spectra[j],
            params_0=self.spectra[i].params,
            params_1=self.spectra[j].params,
        )

        return molecule_pair

    def get_molecular_pairs(self, indexes):
        # create dataset
        molecule_pairs = []

        if indexes is None:
            iterator = self.pair_distances
        else:
            iterator = self.pair_distances[indexes]

        for i, j, tani in iterator:
            molecule_pair = MoleculePair(
                vector_0=None,
                vector_1=None,
                smiles_0=self.spectra[i].smiles,
                smiles_1=self.spectra[j].smiles,
                similarity=tani,
                global_feats_0=MolecularPairsSet.get_global_variables(
                    self.spectra[i]
                ),
                global_feats_1=MolecularPairsSet.get_global_variables(
                    self.spectra[j]
                ),
                index_in_spectrum_0=i,  # index in the spectrum list used as input
                index_in_spectrum_1=j,
                spectrum_object_0=self.spectra[i],
                spectrum_object_1=self.spectra[j],
                params_0=self.spectra[i].params,
                params_1=self.spectra[j].params,
            )
            molecule_pairs.append(molecule_pair)

        return molecule_pairs

    def remove_duplicates(self):
        self.pair_distances = np.unique(self.pair_distances, axis=0)
        return self

    def get_janssen_pairs(self):
        """
        filter our pairs that are not from janssen
        """
        indexes_tani = []
        for i, m in enumerate([mol for mol in self]):
            if (m.spectrum_object_0.library == "janssen") and (
                m.spectrum_object_1.library == "janssen"
            ):
                # molecule_pairs.append(m)
                indexes_tani.append(self.pair_distances[i])

        molecule_pairs = MolecularPairsSet(
            spectra=self.spectra, pair_distances=np.array(indexes_tani)
        )
        return molecule_pairs

    def get_gnps_pairs(self):
        """
        filter only pairs that have exclusively gnps data
        """
        indexes_tani = []
        for i, m in enumerate([mol for mol in self]):
            if (
                "spectrumid" in m.params_0.keys()
                and "spectrumid" in m.params_1.keys()
            ):
                if m.params_0["spectrumid"].startswith(
                    "CCMSLIB"
                ) and m.params_1["spectrumid"].startswith("CCMSLIB"):
                    # molecule_pairs.append(m)
                    indexes_tani.append(self.pair_distances[i])

        molecule_pairs = MolecularPairsSet(
            spectra=self.spectra, pair_distances=np.array(indexes_tani)
        )
        return molecule_pairs

    def get_no_gnps_pairs(self):
        """
        filter any of the gnps data out
        """
        indexes_tani = []
        for i, m in enumerate([mol for mol in self]):
            if "spectrumid" in m.params_0.keys():
                if m.params_0["spectrumid"].startswith("CCMSLIB"):
                    pass
                else:
                    indexes_tani.append(self.pair_distances[i])
            elif "spectrumid" in m.params_1.keys():
                if m.params_1["spectrumid"].startswith("CCMSLIB"):
                    pass
                else:
                    indexes_tani.append(self.pair_distances[i])
            else:
                indexes_tani.append(self.pair_distances[i])

        molecule_pairs = MolecularPairsSet(
            spectra=self.spectra, pair_distances=np.array(indexes_tani)
        )
        return molecule_pairs

    # remove janssen pairs from training and validation
    def remove_library_pairs(self, library):
        spectrums = self.spectra
        indexes_tani = self.pair_distances
        new_indexes_tani = [
            row
            for row in indexes_tani
            if (
                (spectrums[int(row[0])].library != library)
                and (spectrums[int(row[1])].library != library)
            )
        ]
        return MolecularPairsSet(
            spectra=spectrums, pair_distances=new_indexes_tani
        )

    def filter_by_similarity(self, min_sim, max_sim):
        new_indexes_tani = self.pair_distances[
            (self.pair_distances[:, 2] >= min_sim)
            & (self.pair_distances[:, 2] <= max_sim)
        ]
        new_mols = MolecularPairsSet(
            spectra=self.spectra, pair_distances=new_indexes_tani
        )
        return new_mols


class MoleculePairsOpt(MolecularPairsSet):
    """
    optimized version of molecule pairs set with the possiblitiy of working over unique smiles
    """

    def __init__(
        self,
        original_spectra: list[SpectrumExt],
        unique_spectra: list[SpectrumExt],
        df_smiles: pd.DataFrame,
        pair_distances: np.ndarray,
        extra_distances: np.ndarray | None = None,
    ):
        """
        Initialize the optimized molecule pairs.

        Parameters
        ----------
        original_spectra: List[SpectrumExt]
            list of all the spectra, including repetitions of the same compound
        unique_spectra: List[SpectrumExt]
            list of unique spectra, one per compound
        df_smiles: pd.DataFrame
            dataframe containing the mapping from unique smiles to original spectra
        pair_distances: np.ndarray
            array of shape (num_pairs, 3) with the indexes of the pairs and their distance
            - first column: index of the first spectrum in the pair
            - second column: index of the second spectrum in the pair
            - third column: distance (e.g., substructure edit distance) between the two compounds
        extra_distances: Optional[np.ndarray]
            array of shape (num_pairs, 1) with an extra distance metric (e.g., MCES)
        """
        self.original_spectra = original_spectra
        self.spectra = unique_spectra
        self.df_smiles = df_smiles  # table containing the indexes to map unique to repetitions of the same smiles
        # treat the first 2 columns as int and the 3 column as float
        # self.indexes_tani = MolecularPairsSet.adjust_data_format(
        #    np.array(indexes_tani_unique)
        # )
        self.pair_distances = pair_distances
        self.extra_distances = extra_distances

    def __add__(self, other):
        # only to be used when the spectrums are the same

        if self.spectra_equal(self.original_spectra, other.original_spectra):
            new_indexes_tani = np.concatenate(
                (self.pair_distances, other.pair_distances), axis=0
            )
            if (self.extra_distances is not None) and (
                other.extra_distances is not None
            ):
                extra_distances = np.concatenate(
                    (self.extra_distances, other.extra_distances), axis=0
                )
            else:
                extra_distances = None
            return MoleculePairsOpt(
                unique_spectra=self.spectra,
                original_spectra=self.original_spectra,
                pair_distances=new_indexes_tani,
                df_smiles=self.df_smiles,
                extra_distances=extra_distances,
            )
        else:
            print("ERROR: Attempting to add 2 set of spectrums with different content")
            return 0

    def get_molecular_pair(self, index: int) -> MoleculePair:
        """
        get a molecular pair.
        For the first molecule of the pair, retrieve the first element, for the second element retrieve the last index
        this is to avoid to retrieve the same spectrum when the indexes are the same : sim=1
        """
        # i,j,tani = self.indexes_tani[index]
        i = int(self.pair_distances[index, 0])
        j = int(self.pair_distances[index, 1])
        dist = self.pair_distances[index, 2]

        molecule_pair = MoleculePair(
            vector_0=None,
            vector_1=None,
            smiles_0=self.spectra[i].smiles,
            smiles_1=self.spectra[j].smiles,
            similarity=dist,
            global_feats_0=MolecularPairsSet.get_global_variables(self.spectra[i]),
            global_feats_1=MolecularPairsSet.get_global_variables(self.spectra[j]),
            index_in_spectrum_0=self.get_original_index_from_unique_index(
                i, 0
            ),  # index in the spectrum list used as input
            index_in_spectrum_1=self.get_original_index_from_unique_index(j, 1),
            spectrum_object_0=self.get_original_spectrum_from_unique_index(i, 0),
            spectrum_object_1=self.get_original_spectrum_from_unique_index(j, 1),
            params_0=self.get_original_spectrum_from_unique_index(i, 0).params,
            params_1=self.get_original_spectrum_from_unique_index(j, 1).params,
        )

        return molecule_pair

    def get_original_spectrum_from_unique_index(self, unique_index, pair):
        return self.original_spectra[
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
        indexes = list(self.pair_distances[:, pair_index])
        original_indexes = [
            self.get_original_index_from_unique_index(index, pair_index)
            for index in indexes
        ]
        return [self.original_spectra[index] for index in original_indexes]

    def get_sampled_spectrums(self):
        """
        retrieve the sampled spectrums for the first and second molecule of the pairs
        """
        spectrums_index_0 = self.get_spectrums_from_indexes(0)
        spectrums_index_1 = self.get_spectrums_from_indexes(1)
        return spectrums_index_0, spectrums_index_1
