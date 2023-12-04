
from src.transformers.CustomDataset import CustomDataset
import numpy as np

class LoadData:
    
    @staticmethod
    def from_molecule_pairs_to_dataset(molecule_pairs):
        '''
        load molecule pairs data and convert it for being used in Pytorch 
        '''
        ## Convert data into a dataset
        
        list_mz_0 = [m.spectrum_object_0.mz_array for m in molecule_pairs]
        list_intensity_0 = [m.spectrum_object_0.intensity_array for m in molecule_pairs]
        list_mz_1 = [m.spectrum_object_1.mz_array for m in molecule_pairs]
        list_intensity_1 = [m.spectrum_object_1.intensity_array for m in molecule_pairs] 
        list_similarity = [m.similarity for m in molecule_pairs]
        
        mz_0 =np.zeros((len(molecule_pairs),500))
        intensity_0 = np.zeros((len(molecule_pairs),500))
        mz_1=np.zeros((len(molecule_pairs),500))
        intensity_1 = np.zeros((len(molecule_pairs),500))
        similarity= np.zeros((len(molecule_pairs),1))
        
        # fill arrays
        for i,l in enumerate(molecule_pairs):
            #check for maximum length
            length_0 = len(list_mz_0[i]) if len(list_mz_0[i])<=500 else 500
            length_1 = len(list_mz_1[i]) if len(list_mz_1[i])<=500 else 500
            
            # assign the values to the array
            mz_0[i, 0:length_0] = np.array(list_mz_0[i][0:length_0])
            intensity_0[i, 0:length_0] = np.array(list_intensity_0[i][0:length_0])
            mz_1[i, 0:length_1] = np.array(list_mz_1[i][0:length_1])
            intensity_1[i, 0:length_1] = np.array(list_intensity_1[i][0:length_1])
            similarity[i] = list_similarity[i]
            
        dictionary_data = {"mz_0": mz_0,
                   "intensity_0":intensity_0,
                  "mz_1": mz_1,
                   "intensity_1": intensity_1,
                   "similarity":similarity,
                  }
        
        return CustomDataset(dictionary_data)
        