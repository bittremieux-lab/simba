
from src.molecule_pair import MoleculePair
import numpy as np

class MolecularPairsSet:
    '''
    class that encapsulates the indexes and the spectrums from where they are retrieved
    '''
    def __init__(self, spectrums, indexes_tani):
        '''
        it receives a set of spectrums, and a tuple with indexes i,j, tani tuple
        '''
        self.spectrums= spectrums
        self.indexes_tani = indexes_tani

    def __len__(self):
        return len(self.indexes_tani)
    
    def __add__(self, other):
        # only to be used when the spectrums are the same
        new_spectrums = self.spectrums 
        new_indexes_tani = np.concatenate((self.indexes_tani, other.indexes_tani), axis=0)
        return MolecularPairsSet(spectrums=new_spectrums,
                                 indexes_tani=new_indexes_tani
                                 )
    
    def __getitem__(self, index):
        return self.get_molecular_pair(index)
    
    @staticmethod
    def get_global_variables(spectrum):
        '''
        get global variables from a spectrum such as precursor mass
        '''
        list_global_variables = [spectrum.precursor_mz, spectrum.precursor_charge]
        return np.array(list_global_variables)
    
    def get_molecular_pair(self, index):
        #i,j,tani = self.indexes_tani[index]
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
                        index_in_spectrum_0=i, #index in the spectrum list used as input
                        index_in_spectrum_1=j,
                        spectrum_object_0= self.spectrums[i],
                         spectrum_object_1 = self.spectrums[j],
                         params_0= self.spectrums[i].params,
                         params_1= self.spectrums[j].params,)

        
        return molecule_pair
    def get_molecular_pairs(self, indexes):
        # create dataset
        molecule_pairs=[]

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
                        global_feats_0=MolecularPairsSet.get_global_variables(self.spectrums[i]),
                        global_feats_1=MolecularPairsSet.get_global_variables(self.spectrums[j]),
                        index_in_spectrum_0=i, #index in the spectrum list used as input
                        index_in_spectrum_1=j,
                        spectrum_object_0= self.spectrums[i],
                         spectrum_object_1 = self.spectrums[j],
                         params_0= self.spectrums[i].params,
                         params_1= self.spectrums[j].params,)
            molecule_pairs.append(molecule_pair)
        
        return molecule_pairs 
    
    