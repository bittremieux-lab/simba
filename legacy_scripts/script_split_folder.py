import os
import shutil


NUMBER_SPLITS=10
INPUT_PREFIX='indexes_tani_incremental_'
OUTPUT_PREFIX='INPUT_SPECIFIC_PAIRS_indexes_tani_incremental_'
source_directory = "/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925"
destination_directory = "/scratch/antwerpen/209/vsc20939/data/preprocessing_mces_threshold20_newdata_20240925"



def move_and_rename_files(source_dir, destination_dir):
    print('Calling function')
    try:
        # List all files in the source directory
        for file_name in os.listdir(source_dir):
            # Check if the file starts with 'indexes_tani_incremental_'
            if file_name.startswith(INPUT_PREFIX):
                index= int(file_name.split('_')[-1].split('.')[0])

                target_split= str(index%NUMBER_SPLITS)
                
                # Define the new file name starting with 'STAR_'
                new_file_name = OUTPUT_PREFIX+ file_name.split(INPUT_PREFIX)[-1]
                
                # Construct full source and destination paths
                source_path = os.path.join(source_dir, file_name)
                destination_path = os.path.join(destination_dir + '_'+ target_split, new_file_name)
                
                # Move and rename the file
                shutil.move(source_path, destination_path)
                print(f"Moved and renamed: {file_name} -> {new_file_name}")
                
    except Exception as e:
        print(f"An error occurred: {e}")


def copy_spectra_objects(source_dir, destination_dir):
    ## copy spectra objects to each one of the subfolders
    try:
        # List all files in the source directory
        for file_name in os.listdir(source_dir):
            # Check if the file starts with 'indexes_tani_incremental_'
            for index in range(0,NUMBER_SPLITS):
                if file_name.endswith('.pkl'):
                    # Construct full source and destination paths
                    source_path = os.path.join(source_dir, file_name)
                    destination_path = os.path.join(destination_dir+'_'+str(index), file_name)
                    
                    # Move and rename the file
                    shutil.copy(source_path, destination_path)
                    print(f"Copied: {source_path} -> {destination_path}")
                
    except Exception as e:
        print(f"An error occurred: {e}")

# create folders
for i in range(0,NUMBER_SPLITS):
    target_path=destination_directory+'_'+ str(i)
    if not(os.path.exists(target_path)):
        os.mkdir(target_path)

move_and_rename_files(source_directory, destination_directory)
copy_spectra_objects(source_directory, destination_directory)