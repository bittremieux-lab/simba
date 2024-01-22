
from src.transformers.CustomDataset import CustomDataset
import numpy as np

class LoadData:
    
        
    @staticmethod
    def from_molecule_pairs_to_dataset(molecule_pairs, max_num_peaks=100):
        '''
        load molecule pairs data and convert it for being used in Pytorch 
        '''
        ## Convert data into a dataset

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
            length_0 = len(l.spectrum_object_0.mz_array) if len(l.spectrum_object_0.mz_array)<=max_num_peaks else max_num_peaks
            length_1 = len(l.spectrum_object_1.mz_array) if len(l.spectrum_object_1.mz_array)<=max_num_peaks else max_num_peaks
            
            # assign the values to the array
            mz_0[i, 0:length_0] = np.array(l.spectrum_object_0.mz_array[0:length_0])
            intensity_0[i, 0:length_0] = np.array(l.spectrum_object_0.intensity_array[0:length_0])
            mz_1[i, 0:length_1] = np.array(l.spectrum_object_1.mz_array[0:length_1])
            intensity_1[i, 0:length_1] = np.array(l.spectrum_object_1.intensity_array[0:length_1])
            
            precursor_mass_0[i] = l.global_feats_0[0]
            precursor_charge_0[i] = l.global_feats_0[1]
            precursor_mass_1[i] = l.global_feats_1[0]
            precursor_charge_1[i] = l.global_feats_1[1]
            similarity[i] = l.similarity
            fingerprints[i]=list_fingerprints[i]
            

        # Calculate the root of the sum of squares of the components unit vectors
        magnitude_0 = np.sqrt(np.sum(intensity_0**2, axis=1, keepdims=True))
        magnitude_1 = np.sqrt(np.sum(intensity_1**2, axis=1, keepdims=True))

        # Normalize the intensity array
        intensity_0 = intensity_0 / magnitude_0
        intensity_1 = intensity_1 / magnitude_1

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
        
