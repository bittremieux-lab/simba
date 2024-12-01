import random
import copy
import numpy as np

class Augmentation:

    @staticmethod
    def augment(data_sample, training=False):
        new_sample = copy.deepcopy(data_sample)
        #new_sample = Augmentation.inversion(new_sample)
        #new_sample = Augmentation.add_noise_to_precursor_mass(new_sample)  

        # peak augmentation
        new_sample = Augmentation.normalize_max(new_sample)
        new_sample = Augmentation.peak_augmentation_removal_noise(data_sample)
        new_sample = Augmentation.peak_augmentation_max_peaks(data_sample,)

        #precursor mass
        new_sample = Augmentation.add_false_precursor_masses_negatives(new_sample)
        new_sample = Augmentation.add_false_precursor_masses_positives(new_sample)

        # normalize
        #new_sample = Augmentation.normalize_intensities(new_sample)
        return new_sample

    @staticmethod
    def normalize_max(data_sample):
        for sufix in ['_0','_1']:
            #put a threshold
            intensity_column= 'intensity' + sufix
            mz_column = 'mz' + sufix 

            # normalization
            intensity = np.array(data_sample[intensity_column])
            mz = data_sample[mz_column]
            
            intensity = intensity / np.max(intensity,  keepdims=True)

            # apply 
            data_sample[intensity_column] = intensity
            data_sample[mz_column] = mz

        return data_sample


    @staticmethod
    def peak_augmentation_max_peaks(data_sample, p_augmentation=0.5, max_peaks=100):
         # first normalize to maximum

        if random.random()<p_augmentation:
            max_augmented_peaks= int(max(20, random.random()*max_peaks))

            for sufix in ['_0','_1']:
                #put a threshold
                intensity_column= 'intensity' + sufix
                mz_column = 'mz' + sufix 
                intensity = data_sample[intensity_column]
                mz = data_sample[mz_column]

                # order intensities by amplitude
                intensity_ordered_indexes = np.argsort(intensity)[::-1]  # flip the order to have the max at the beginning
                indexes_to_be_erased = intensity_ordered_indexes[max_augmented_peaks:-1]
                
                intensity[indexes_to_be_erased]=0
                mz[indexes_to_be_erased]=0

                # apply 
                data_sample[intensity_column] = intensity
                data_sample[mz_column] = mz
            return data_sample
        else:
            return data_sample


    @staticmethod
    def peak_augmentation_removal_noise(data_sample, max_percentage=0.01, p_augmentation=0.5):

        if random.random()<p_augmentation:
            # first normalize to maximum
            for sufix in ['_0','_1']:
                #put a threshold
                intensity_column= 'intensity' + sufix
                mz_column = 'mz' + sufix 
                intensity = data_sample[intensity_column]
                mz = data_sample[mz_column]

                # remove noise peaks
                max_amplitude= random.random()* max_percentage
                intensity[intensity < max_amplitude] = 0
                mz[intensity < max_amplitude] = 0

                # apply 
                data_sample[intensity_column] = intensity
                data_sample[mz_column] = mz
            # put a threshold based on peak amplitude

            return data_sample
        else:
            return data_sample

    @staticmethod 
    def normalize_intensities(data_sample, intensity_labels= ['intensity_0','intensity_1']):
        # sqrt root
        for intensity_column in  intensity_labels:
            intensity = data_sample[intensity_column]
            intensity = np.sqrt(intensity)
            intensity = intensity / np.sqrt(np.sum(intensity**2,  keepdims=True))
            data_sample[intensity_column]= intensity
        return data_sample
         
    @staticmethod
    def inversion(data_sample):
        # inversion

        new_sample = {}
        new_sample["mz_0"] = data_sample["mz_1"]
        new_sample["mz_1"] = data_sample["mz_0"]

        new_sample["intensity_0"] = data_sample["intensity_1"]
        new_sample["intensity_1"] = data_sample["intensity_0"]

        new_sample["precursor_mass_0"] = data_sample["precursor_mass_1"]
        new_sample["precursor_mass_1"] = data_sample["precursor_mass_0"]

        new_sample["precursor_charge_0"] = data_sample["precursor_charge_1"]
        new_sample["precursor_charge_1"] = data_sample["precursor_charge_0"]

        new_sample["similarity"] = data_sample["similarity"]
        return new_sample

  

    @staticmethod
    def add_false_precursor_masses_positives(sample, max_noise=0.01, p_augmentation=0.2):
        '''
        create a pair where the precursor masses are very different
        '''

        if random.random()< p_augmentation:
            added_noise_factor_0 = random.uniform(-max_noise, max_noise)
            added_noise_factor_1 = random.uniform(-max_noise, max_noise)

            pmz= sample["precursor_mass_0"].copy()
            sample["precursor_mass_0"] = pmz + added_noise_factor_0 * (pmz)
            sample["precursor_mass_1"] = pmz + added_noise_factor_1 * (pmz)
            return sample
        else:
            return sample
        
    
    @staticmethod
    def add_false_precursor_masses_negatives(sample, max_noise=1.0, p_augmentation=0.10):
        '''
        create a pair where the precursor masses are almost the same
        '''

        if random.random()< p_augmentation:
            added_noise_factor_0 = random.uniform(-max_noise, max_noise)
            added_noise_factor_1 = random.uniform(-max_noise, max_noise)

            sample["precursor_mass_0"] = sample["precursor_mass_0"] +  added_noise_factor_0 * (sample["precursor_mass_0"])
            sample["precursor_mass_1"] = sample["precursor_mass_1"]  + added_noise_factor_1 * (sample["precursor_mass_1"])
            return sample
        else:
            return sample