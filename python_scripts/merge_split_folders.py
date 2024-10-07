import os
import shutil


NUMBER_SPLITS=10
INPUT_PREFIX='indexes_tani_incremental_'
OUTPUT_PREFIX='indexes_tani_incremental_'
source_directory =      "/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925" # split folders
destination_directory = "/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925"



def copy_and_rename_files(source_dir, destination_dir, number_splits, folds=['train','val','test']):
    print('Calling function')

    save_index=0 #index of output numpy saved

    for k in folds:
        for split in range(0, number_splits):

            input_prefix=INPUT_PREFIX  + k 
            output_prefix=OUTPUT_PREFIX  + k 

            source_directory_split= source_dir + '_' + str(split)
            for file_name in os.listdir(source_directory_split):
                # Check if the file starts with 'indexes_tani_incremental_'
                if file_name.startswith(input_prefix):

                    # Define the new file name starting with 'STAR_'
                    new_file_name = output_prefix+ '_' + str(save_index)
                    
                    # Construct full source and destination paths
                    source_path = os.path.join(source_directory_split, file_name)
                    destination_path = os.path.join(destination_dir, new_file_name)
                    
                    # copy and rename the file
                    shutil.copy(source_path, destination_path)
                    print(f"Moved and renamed: {source_path} -> {destination_path}")
                    
                    save_index=save_index+1

# create folders
for i in range(0,NUMBER_SPLITS):
    target_path=destination_directory+'_'+ str(i)
    if not(os.path.exists(target_path)):
        os.mkdir(target_path)

copy_and_rename_files(source_directory, destination_directory, number_splits=NUMBER_SPLITS)
