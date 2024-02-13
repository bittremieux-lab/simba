
from src.transformers.CustomDataset import CustomDataset
import numpy as np
from src.preprocessor import Preprocessor
from tqdm import tqdm
from src.molecular_pairs_set import MolecularPairsSet
import copy
class LoadData:
    
        
    @staticmethod
    def from_molecule_pairs_to_dataset(molecule_pairs_input, 
                                       max_num_peaks=100, 
                                       shuffle_order_pairs=False, #shuffle the spectrum 0 and 1 for data augmentation
                                       ):
        '''
        preprocess the spectra and convert it for being used in Pytorch 
        '''
        # copy spectrums to avoid overwriting
        molecule_pairs = MolecularPairsSet(spectrums=[copy.copy(s) for s in molecule_pairs_input.spectrums],
                                            indexes_tani = molecule_pairs_input.indexes_tani.copy())
        
        if shuffle_order_pairs:
            #exchange the order of the spectrums 0 and 1 for a molecule pair
            new_indexes_tani = np.array([[row[1], row[0], row[2]] for row in molecule_pairs.indexes_tani])
            molecule_pairs = MolecularPairsSet(spectrums=molecule_pairs.spectrums,
                                               indexes_tani= new_indexes_tani)
       
        ## Preprocess the data
        pp = Preprocessor()
        print('Preprocessing all the data ...')
        molecule_pairs.spectrums = pp.preprocess_all_spectrums(molecule_pairs.spectrums)
        print('Finished preprocessing ')
        
        ## Convert data into a dataset
        #if hasattr(molecule_pairs[0], 'fingerprint_0'):
        #  if molecule_pairs[0].fingerprint_0 is not None:
        #    list_fingerprints = [np.concatenate((m.fingerprint_0, m.fingerprint_1)) for m in molecule_pairs]
        #  else:
        #      list_fingerprints = [0 for i in range(0,len(molecule_pairs))]
        #else:
        # list_fingerprints = [0 for i in range(0,len(molecule_pairs))]

        mz_0 =np.zeros((len(molecule_pairs),max_num_peaks), dtype=np.float32)
        intensity_0 = np.zeros((len(molecule_pairs),max_num_peaks), dtype=np.float32)
        mz_1=np.zeros((len(molecule_pairs),max_num_peaks), dtype=np.float32)
        intensity_1 = np.zeros((len(molecule_pairs),max_num_peaks), dtype=np.float32)
        similarity= np.zeros((len(molecule_pairs),1), dtype=np.float32)
        precursor_mass_0 = np.zeros((len(molecule_pairs),1), dtype=np.float32)
        precursor_charge_0 = np.zeros((len(molecule_pairs),1), dtype=np.int32)
        precursor_mass_1 = np.zeros((len(molecule_pairs),1), dtype=np.float32)
        precursor_charge_1 = np.zeros((len(molecule_pairs),1), dtype=np.int32)
        #fingerprints = np.zeros((len(molecule_pairs), 128), dtype=np.float32)



        print('Starting the loading of the data ...')
        # fill arrays
        for i,l in (enumerate(molecule_pairs)):
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
            #fingerprints[i]=list_fingerprints[i]

        print('Normalizing intensities')
        # Normalize the intensity array
        intensity_0 = intensity_0 / np.sqrt(np.sum(intensity_0**2, axis=1, keepdims=True))
        intensity_1 = intensity_1 /  np.sqrt(np.sum(intensity_1**2, axis=1, keepdims=True))

        print('Creating dictionaries')
        dictionary_data = {"mz_0": mz_0,
                   "intensity_0":intensity_0,
                  "mz_1": mz_1,
                   "intensity_1": intensity_1,
                   "similarity":similarity,
                           "precursor_mass_0": precursor_mass_0,
                           "precursor_mass_1":precursor_mass_1,
                           "precursor_charge_0": precursor_charge_0,
                           "precursor_charge_1": precursor_charge_1,
                           #"fingerprint": fingerprints,
                  }
        
        return CustomDataset(dictionary_data)
        
