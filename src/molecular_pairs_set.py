from src.molecule_pair import MoleculePair
import numpy as np


class MolecularPairsSet:
    """
    class that encapsulates the indexes and the spectrums from where they are retrieved
    """

    def __init__(self, spectrums, indexes_tani):
        """
        it receives a set of spectrums, and a tuple with indexes i,j, tani tuple
        """
        self.spectrums = spectrums

        # treat the first 2 columns as int and the 3 column as float
        self.indexes_tani = MolecularPairsSet.adjust_data_format(np.array(indexes_tani))

        
        

    @staticmethod
    def adjust_data_format(indexes_tani):
        # Extracting the first two columns and changing their data type to int
        int_columns = indexes_tani[:, 0:2].astype(np.int32)

        # Extracting the last column and changing its data type to float
        float_column = indexes_tani[:, 2].astype(np.float32)

        # Combining the modified columns to create a new array
        new_indexes_tani = np.column_stack((int_columns, float_column))    
        return new_indexes_tani
    
    def __len__(self):
        return len(self.indexes_tani)

    def are_spectrums_the_same(self, spectrums0, spectrums1):
        spectrum_hash_0 = [s.spectrum_hash for s in spectrums0]
        spectrum_hash_1 = [s.spectrum_hash for s in spectrums1]
        return all([ s0==s1 for s0,s1 in zip(spectrum_hash_0, spectrum_hash_1)])

    def __add__(self, other):
        # only to be used when the spectrums are the same

        if self.are_spectrums_the_same(self.spectrums, other.spectrums):
            new_spectrums = self.spectrums
            new_indexes_tani = np.concatenate(
                (self.indexes_tani, other.indexes_tani), axis=0
            )
            return MolecularPairsSet(spectrums=new_spectrums, indexes_tani=new_indexes_tani)
        else:
            print('ERROR: Attempting to add 2 set of spectrums with different content')
            return 0

    def __getitem__(self, index):
        return self.get_molecular_pair(index)

    @staticmethod
    def get_global_variables(spectrum):
        """
        get global variables from a spectrum such as precursor mass
        """
        list_global_variables = [spectrum.precursor_mz, spectrum.precursor_charge]
        return np.array(list_global_variables)

    def get_molecular_pair(self, index):
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
            index_in_spectrum_0=i,  # index in the spectrum list used as input
            index_in_spectrum_1=j,
            spectrum_object_0=self.spectrums[i],
            spectrum_object_1=self.spectrums[j],
            params_0=self.spectrums[i].params,
            params_1=self.spectrums[j].params,
        )

        return molecule_pair

    def get_molecular_pairs(self, indexes):
        # create dataset
        molecule_pairs = []

        if indexes is None:
            iterator = self.indexes_tani
        else:
            iterator = self.indexes_tani[indexes]

        for i, j, tani in iterator:
            molecule_pair = MoleculePair(
                vector_0=None,
                vector_1=None,
                smiles_0=self.spectrums[i].smiles,
                smiles_1=self.spectrums[j].smiles,
                similarity=tani,
                global_feats_0=MolecularPairsSet.get_global_variables(
                    self.spectrums[i]
                ),
                global_feats_1=MolecularPairsSet.get_global_variables(
                    self.spectrums[j]
                ),
                index_in_spectrum_0=i,  # index in the spectrum list used as input
                index_in_spectrum_1=j,
                spectrum_object_0=self.spectrums[i],
                spectrum_object_1=self.spectrums[j],
                params_0=self.spectrums[i].params,
                params_1=self.spectrums[j].params,
            )
            molecule_pairs.append(molecule_pair)

        return molecule_pairs

    def remove_duplicates(self):
        self.indexes_tani = np.unique(self.indexes_tani, axis=0)
        return self

    def get_gnps_pairs(self):
        """
        filter only pairs that have exclusively gnps data
        """
        indexes_tani = []
        for i, m in enumerate([mol for mol in self]):
            if "spectrumid" in m.params_0.keys() and "spectrumid" in m.params_1.keys():
                if m.params_0["spectrumid"].startswith("CCMSLIB") and m.params_1[
                    "spectrumid"
                ].startswith("CCMSLIB"):
                    # molecule_pairs.append(m)
                    indexes_tani.append(self.indexes_tani[i])

        molecule_pairs = MolecularPairsSet(
            spectrums=self.spectrums, indexes_tani=np.array(indexes_tani)
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
                    indexes_tani.append(self.indexes_tani[i])
            elif "spectrumid" in m.params_1.keys():
                if m.params_1["spectrumid"].startswith("CCMSLIB"):
                    pass
                else:
                    indexes_tani.append(self.indexes_tani[i])
            else:
                indexes_tani.append(self.indexes_tani[i])

        molecule_pairs = MolecularPairsSet(
            spectrums=self.spectrums, indexes_tani=np.array(indexes_tani)
        )
        return molecule_pairs

    # remove janssen pairs from training and validation
    def remove_library_pairs(self, library):
        spectrums=self.spectrums
        indexes_tani= self.indexes_tani
        new_indexes_tani = [row for row in indexes_tani if ((spectrums[int(row[0])].library!=library)and (spectrums[int(row[1])].library!=library))]
        return MolecularPairsSet(spectrums=spectrums, indexes_tani=new_indexes_tani)

    def filter_by_similarity(self, min_sim, max_sim):
        new_indexes_tani = self.indexes_tani[
            (self.indexes_tani[:, 2] >= min_sim) & (self.indexes_tani[:, 2] <= max_sim)
        ]
        new_mols = MolecularPairsSet(
            spectrums=self.spectrums, indexes_tani=new_indexes_tani
        )
        return new_mols
