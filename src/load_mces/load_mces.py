
import os
import numpy as np
from src.config import Config
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
        print(directory_path)
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
        if len(list_arrays)>0:
            return np.concatenate(list_arrays, axis=0)
        else:
            return np.array([])

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

            #print(f'Size without removal: {np_array.shape[0]}')
            np_array=LoadMCES.remove_excess_low_pairs(np_array, 
                                                remove_percentage=remove_percentage,
                                                target_column=Config.COLUMN_EDIT_DISTANCE)
            #print(f'Size with removal: {np_array.shape[0]}')
            list_arrays.append(np_array)

        #merge
        print('Merging')
        merged_array= np.concatenate(list_arrays, axis=0)
        
        # normalize
        print('Normalizing')
        merged_array[:,Config.COLUMN_EDIT_DISTANCE]= LoadMCES.normalize_mces(merged_array[:,Config.COLUMN_EDIT_DISTANCE])

        print('Remove redundant pairs')
        merged_array = np.unique(merged_array, axis=0)
        # remove excess low pairs
        #merged_array = LoadMCES.remove_excess_low_pairs(merged_array)

        print(f'Size of data loaded: {merged_array.shape[0]}')
        return merged_array

    def add_high_similarity_pairs_edit_distance(merged_array):
        max_index_spectrum = int(np.max(merged_array[:,0]))
        indexes_tani_high= np.zeros((max_index_spectrum,merged_array.shape[1]))
        indexes_tani_high[:,0]= np.arange(0,max_index_spectrum)
        indexes_tani_high[:,1]= np.arange(0,max_index_spectrum)
        indexes_tani_high[:,2]= 1
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
        
        

        print('Normalizing')
        merged_array[:,2]= LoadMCES.normalize_mces(merged_array[:,2])

        # add the high similarity pairs
        merged_array= LoadMCES.add_high_similarity_pairs_edit_distance(merged_array)
        # normalize
        # remove excess low pairs
        #merged_array = LoadMCES.remove_excess_low_pairs(merged_array)

        return merged_array

    def merge_numpy_arrays_multitask(directory_path, prefix, remove_percentage=0.00, add_high_similarity_pairs=False):
        '''
        load np arrays containing data as well as apply normalization
        '''

        # call the configuration
        config=Config()
        # find all np arrays
        files = LoadMCES.find_file(directory_path, prefix)
        
        # load np files
        print('Loading the partitioned files of the pairs')
        list_arrays=[]
        for i,f in enumerate(files):
            print(f'Processing batch {i}')
            np_array= np.load(f)

            #print(f'Size without removal: {np_array.shape[0]}')

            np_array=LoadMCES.remove_excess_low_pairs(np_array, remove_percentage=remove_percentage,
                                                                target_column=config.COLUMN_EDIT_DISTANCE)
            #print(f'Size with removal: {np_array.shape[0]}')
            list_arrays.append(np_array)

        #merge
        print('Merging')
        merged_array= np.concatenate([l for l in list_arrays if len(l)>0], axis=0)
        
        

        print('Normalizing')
        merged_array[:,config.COLUMN_EDIT_DISTANCE]= LoadMCES.normalize_ed(merged_array[:,config.COLUMN_EDIT_DISTANCE],)

        if not(config.USE_TANIMOTO): #if not using tanimoto normalize between 0 and 1
            merged_array[:,config.COLUMN_MCES20] = LoadMCES.normalize_mces20(merged_array[:,config.COLUMN_MCES20],
                                                            max_value=config.MCES20_MAX_VALUE)

        # add the high similarity pairs
        if add_high_similarity_pairs:
            merged_array= LoadMCES.add_high_similarity_pairs_edit_distance(merged_array)
        # normalize
        # remove excess low pairs
        #merged_array = LoadMCES.remove_excess_low_pairs(merged_array)

        print(f'Number of pairs loaded: {merged_array.shape[0]}  ')
        return merged_array

    def merge_numpy_arrays(directory_path, prefix, use_edit_distance, use_multitask=False, add_high_similarity_pairs=False):
        '''
        load np arrays containing data as well as apply normalization
        '''
        if use_multitask:
            return LoadMCES.merge_numpy_arrays_multitask(directory_path, prefix,add_high_similarity_pairs=add_high_similarity_pairs)
        else:
            if use_edit_distance:
                return LoadMCES.merge_numpy_arrays_edit_distance(directory_path, prefix,)
            else:
                return LoadMCES.merge_numpy_arrays_mces(directory_path, prefix,)


    def remove_excess_low_pairs(indexes_tani, remove_percentage=0.95, max_value=5, target_column=2):
        '''
        remove the 90% of the low pairs to reduce the data loaded
        '''
        # get the sample size for the low range pairs
        sample_size = indexes_tani.shape[0] - int(remove_percentage*indexes_tani.shape[0])

        # filter by high or low similarity, assuming MCES distance
        indexes_tani_high = indexes_tani[indexes_tani[:,target_column]<max_value]
        indexes_tani_low = indexes_tani[indexes_tani[:,target_column]>=max_value]

        if remove_percentage>0:
            random_samples = np.random.randint(0,indexes_tani_low.shape[0], sample_size)
            
            # index
            indexes_tani_low = indexes_tani_low[random_samples]
    
        return np.concatenate((indexes_tani_low, indexes_tani_high), axis=0)


    @staticmethod
    def normalize_ed(ed, max_ed=5):
        # asuming series
        # normalize edit distance. the higher the mces the lower the similarity
        #mces_normalized = mces.apply(lambda x:x if x<=max_mces else max_mces)
        #return mces_normalized.apply(lambda x:(1-(x/max_mces)))

        ## asuming numpy
        print(f'Example of input ed: {ed}')
        ed_normalized = ed.copy()
        ed_normalized[ed_normalized >= max_ed] = max_ed 
        ed_normalized = 1 - (ed_normalized/max_ed)
        print(f'Example of normalized ed: {ed_normalized}')
        return ed_normalized

    @staticmethod
    def normalize_mces20(mcs20, max_value):
        # asuming series
        # normalize edit distance. the higher the mces the lower the similarity
        #mces_normalized = mces.apply(lambda x:x if x<=max_mces else max_mces)
        #return mces_normalized.apply(lambda x:(1-(x/max_mces)))

        ## asuming numpy
        print(f'Example of input mces: {mcs20}')
        mcs20_normalized = 1-mcs20/max_value
        print(f'Example of normalized mces: {mcs20_normalized}')
        return mcs20_normalized

    def load_mces_20_data(directory_path, prefix, number_folders):
        '''
        loads the mces with threshold 20 across different folders
        '''
        list_arrays= []
        for index in range(0, number_folders):
            array= LoadMCES.load_raw_data(directory_path=directory_path + str(index),
                                                    prefix=prefix) 
            list_arrays.append(array)

        # drop the lists that are empty
        list_arrays= [l for l in list_arrays if l.shape[0]>0]
        return np.concatenate(list_arrays, axis=0)