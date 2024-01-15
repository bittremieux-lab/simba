
from src.transformers.CustomDataset import CustomDataset
import numpy as np

class LoadData:
    
        
    @staticmethod
    def from_molecule_pairs_to_dataset(molecule_pairs, max_num_peaks=100):
        '''
        load molecule pairs data and convert it for being used in Pytorch 
        '''
        ## Convert data into a dataset
        
        list_mz_0 = [m.spectrum_object_0.mz_array for m in molecule_pairs]
        list_intensity_0 = [m.spectrum_object_0.intensity_array for m in molecule_pairs]
        list_mz_1 = [m.spectrum_object_1.mz_array for m in molecule_pairs]
        list_intensity_1 = [m.spectrum_object_1.intensity_array for m in molecule_pairs] 
        list_similarity = [m.similarity for m in molecule_pairs]
        
        if hasattr(molecule_pairs[0], 'fingerprint_0'):
          if molecule_pairs[0].fingerprint_0 is not None:
            list_fingerprints = [np.concatenate((m.fingerprint_0, m.fingerprint_1)) for m in molecule_pairs]
          else:
              list_fingerprints = [0 for m in molecule_pairs]
        else:
         list_fingerprints = [0 for m in molecule_pairs]

        mz_0 =np.zeros((len(molecule_pairs),max_num_peaks))
        intensity_0 = np.zeros((len(molecule_pairs),max_num_peaks))
        mz_1=np.zeros((len(molecule_pairs),max_num_peaks))
        intensity_1 = np.zeros((len(molecule_pairs),max_num_peaks))
        similarity= np.zeros((len(molecule_pairs),1))
        precursor_mass_0 = np.zeros((len(molecule_pairs),1))
        precursor_charge_0 = np.zeros((len(molecule_pairs),1))
        precursor_mass_1 = np.zeros((len(molecule_pairs),1))
        precursor_charge_1 = np.zeros((len(molecule_pairs),1))
        fingerprints = np.zeros((len(molecule_pairs), 128))

        # fill arrays
        for i,l in enumerate(molecule_pairs):
            #check for maximum length
            length_0 = len(list_mz_0[i]) if len(list_mz_0[i])<=max_num_peaks else max_num_peaks
            length_1 = len(list_mz_1[i]) if len(list_mz_1[i])<=max_num_peaks else max_num_peaks
            
            # assign the values to the array
            mz_0[i, 0:length_0] = np.array(list_mz_0[i][0:length_0])
            intensity_0[i, 0:length_0] = np.array(list_intensity_0[i][0:length_0])
            mz_1[i, 0:length_1] = np.array(list_mz_1[i][0:length_1])
            intensity_1[i, 0:length_1] = np.array(list_intensity_1[i][0:length_1])
            
            precursor_mass_0[i] = l.global_feats_0[0]
            precursor_charge_0[i] = l.global_feats_0[1]
            precursor_mass_1[i] = l.global_feats_1[0]
            precursor_charge_1[i] = l.global_feats_1[1]
            similarity[i] = list_similarity[i]
            fingerprints[i]=list_fingerprints[i]
            
        dictionary_data = {"mz_0": mz_0,
                   "intensity_0":intensity_0,
                  "mz_1": mz_1,
                   "intensity_1": intensity_1,
                   "similarity":similarity,
                           "precursor_mass_0": precursor_mass_0,
                           "precursor_mass_1":precursor_mass_1,
                           "precursor_charge_0": precursor_charge_0,
                           "precursor_charge_1": precursor_charge_1,
                           "fingerprint": fingerprints,
                  }
        
        return CustomDataset(dictionary_data)
        
