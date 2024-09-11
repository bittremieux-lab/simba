
import os
import numpy as np

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

    def load_raw_data(directory_path, prefix, partitions=10000000):
        '''
        load data for inspection purposes
        '''
        # find all np arrays
        files = LoadMCES.find_file(directory_path, prefix)
        
        # load np files
        print('Loading the partitioned files of the pairs')
        list_arrays=[]

        for i in list(range(0, min(len(files), partitions))):
            f= files[i]
            print(f'Processing batch {i}')
            np_array= np.load(f)
            print(f'Size: {np_array.shape[0]}')
            list_arrays.append(np_array)

        #merge
        print('Merging')
        merged_array= np.concatenate(list_arrays, axis=0)
        return merged_array
    
    def merge_numpy_arrays_mces(directory_path, prefix):
        '''
        load np arrays containing data as well as apply normalization for training
        '''
        # find all np arrays
        files = LoadMCES.find_file(directory_path, prefix)
        
        # load np files
        print('Loading the partitioned files of the pairs')
        list_arrays=[]
        for i,f in enumerate(files):
            print(f'Processing batch {i}')
            np_array= np.load(f)

            # select only the first 3 rows: index0, index1 and similarity
            np_array = np_array[:,0:3]

            print(f'Size without removal: {np_array.shape[0]}')
            np_array=LoadMCES.remove_excess_low_pairs(np_array, remove_percentage=remove_percentage)
            print(f'Size with removal: {np_array.shape[0]}')
            list_arrays.append(np_array)

        #merge
        print('Merging')
        merged_array= np.concatenate(list_arrays, axis=0)
        
        # normalize
        print('Normalizing')
        merged_array[:,2]= LoadMCES.normalize_mces(merged_array[:,2])

        print('Remove redundant pairs')
        merged_array = np.unique(merged_array, axis=0)
        # remove excess low pairs
        #merged_array = LoadMCES.remove_excess_low_pairs(merged_array)
        return merged_array

    def add_high_similarity_pairs_edit_distance(merged_array):
        max_index_spectrum = int(np.max(merged_array[:,0]))
        indexes_tani_high= np.zeros((max_index_spectrum,merged_array.shape[1]))
        indexes_tani_high[:,0]= np.arange(0,max_index_spectrum)
        indexes_tani_high[:,1]= np.arange(0,max_index_spectrum)
        indexes_tani_high[:,2]= 0
        # if there is the extra column corresponding to tanimoto
        if merged_array.shape[1]==4:
            indexes_tani_high[:,3]= 1

        merged_array= np.concatenate([merged_array,indexes_tani_high])
        return merged_array
    def merge_numpy_arrays_edit_distance(directory_path, prefix, remove_percentage=0.90):
        '''
        load np arrays containing data as well as apply normalization
        '''
        # find all np arrays
        files = LoadMCES.find_file(directory_path, prefix)
        
        # load np files
        print('Loading the partitioned files of the pairs')
        list_arrays=[]
        for i,f in enumerate(files):
            print(f'Processing batch {i}')
            np_array= np.load(f)

            # select only the first 3 rows: index0, index1 and similarity
            np_array = np_array[:,0:3]

            print(f'Size without removal: {np_array.shape[0]}')
            np_array=LoadMCES.remove_excess_low_pairs(np_array, remove_percentage=remove_percentage)
            print(f'Size with removal: {np_array.shape[0]}')
            list_arrays.append(np_array)

        #merge
        print('Merging')
        merged_array= np.concatenate(list_arrays, axis=0)
        
        # add the high similarity pairs
        merged_array= LoadMCES.add_high_similarity_pairs_edit_distance(merged_array)
        # normalize

        print('Normalizing')
        merged_array[:,2]= LoadMCES.normalize_mces(merged_array[:,2])

        # remove excess low pairs
        #merged_array = LoadMCES.remove_excess_low_pairs(merged_array)

        return merged_array

    def merge_numpy_arrays_multitask(directory_path, prefix, remove_percentage=0.90):
        '''
        load np arrays containing data as well as apply normalization
        '''
        # find all np arrays
        files = LoadMCES.find_file(directory_path, prefix)
        
        # load np files
        print('Loading the partitioned files of the pairs')
        list_arrays=[]
        for i,f in enumerate(files):
            print(f'Processing batch {i}')
            np_array= np.load(f)

            print(f'Size without removal: {np_array.shape[0]}')

            np_array=LoadMCES.remove_excess_low_pairs(np_array, remove_percentage=remove_percentage)
            print(f'Size with removal: {np_array.shape[0]}')
            list_arrays.append(np_array)

        #merge
        print('Merging')
        merged_array= np.concatenate(list_arrays, axis=0)
        
        # add the high similarity pairs
        merged_array= LoadMCES.add_high_similarity_pairs_edit_distance(merged_array)
        # normalize

        print('Normalizing')
        merged_array[:,2]= LoadMCES.normalize_mces(merged_array[:,2],)

        # remove excess low pairs
        #merged_array = LoadMCES.remove_excess_low_pairs(merged_array)

        return merged_array

    def merge_numpy_arrays(directory_path, prefix, use_edit_distance, use_multitask=False):
        '''
        load np arrays containing data as well as apply normalization
        '''
        if use_multitask:
            return LoadMCES.merge_numpy_arrays_multitask(directory_path, prefix,)
        else:
            if use_edit_distance:
                return LoadMCES.merge_numpy_arrays_edit_distance(directory_path, prefix,)
            else:
                return LoadMCES.merge_numpy_arrays_mces(directory_path, prefix,)


    def remove_excess_low_pairs(indexes_tani, remove_percentage=0.99, max_mces=5):
        '''
        remove the 90% of the low pairs to reduce the data loaded
        '''
        # get the sample size for the low range pairs
        sample_size = indexes_tani.shape[0] - int(remove_percentage*indexes_tani.shape[0])

        # filter by high or low similarity, assuming MCES distance
        indexes_tani_high = indexes_tani[indexes_tani[:,2]<max_mces]
        indexes_tani_low = indexes_tani[indexes_tani[:,2]>=max_mces]

        random_samples = np.random.randint(0,indexes_tani_low.shape[0], sample_size)
        
        # index
        indexes_tani_low = indexes_tani_low[random_samples]
        return np.concatenate((indexes_tani_low, indexes_tani_high), axis=0)

    @staticmethod
    def normalize_mces(mces, max_mces=5):
        # asuming series
        # normalize mces. the higher the mces the lower the similarity
        #mces_normalized = mces.apply(lambda x:x if x<=max_mces else max_mces)
        #return mces_normalized.apply(lambda x:(1-(x/max_mces)))

        ## asuming numpy
        print(f'Example of input mces: {mces}')
        mces_normalized = mces.copy()
        mces_normalized[mces_normalized >= max_mces] = max_mces 
        mces_normalized = 1 - (mces_normalized/max_mces)
        print(f'Example of normalized mces: {mces_normalized}')
        return mces_normalized

    def load_mces_20_data(directory_path, prefix, number_folders):
        '''
        loads the mces with threshold 20 across different folders
        '''
        list_arrays= []
        for index in range(0, number_folders):
            array= LoadMCES.load_raw_data(directory_path=directory_path + str(index),
                                                    prefix=prefix) 
            list_arrays.append(array)
        return np.concatenate(list_arrays, axis=0)