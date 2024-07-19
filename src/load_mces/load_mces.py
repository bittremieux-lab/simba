
import os
import numpy as np
from src.mces.mces_computation import MCES
class LoadMCES:

    def find_file(directory_path, prefix):
        """
        Searches for a .pkl file in the given directory and returns the path of the first one found.
        
        Args:
        directory_path (str): The path of the directory to search in.
        
        Returns:
        str: The path of the first .pkl file found, or None if no such file exists.
        """
        pickle_files=[]
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.startswith(prefix):
                    pickle_files.append(os.path.join(root, file))
        return pickle_files 


    def merge_numpy_arrays(directory_path, prefix):
        '''
        load np arrays containing data as well as apply normalization
        '''
        # find all np arrays
        files = LoadMCES.find_file(directory_path, prefix)
        
        # load np files
        list_arrays=[]
        for f in files:
            list_arrays.append(np.load(f))

        #merge
        merged_array= np.concatenate(list_arrays, axis=0)

        
        # normalize
        merged_array[:,2]= MCES.normalize_mces(merged_array[:,2])

        # remove excess low pairs
        merged_array = LoadMCES.remove_excess_low_pairs(merged_array)

        return merged_array

    def remove_excess_low_pairs(indexes_tani, remove_percentage=0.95):
        '''
        remove the 90% of the low pairs to reduce the data loaded
        '''
        # get the sample size for the low range pairs
        sample_size = indexes_tani.shape[0] - int(remove_percentage*indexes_tani.shape[0])

        # filter by high or low similarity
        indexes_tani_high = indexes_tani[indexes_tani[:,2]>0]
        indexes_tani_low = indexes_tani[indexes_tani[:,2]==0]

        # get some indexes to sample
        random_samples = np.random.randint(0,indexes_tani_low.shape[0], sample_size)
        
        # index
        indexes_tani_low = indexes_tani_low[random_samples]
        return np.concatenate((indexes_tani_low, indexes_tani_high), axis=0)