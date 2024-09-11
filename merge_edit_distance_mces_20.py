from src.load_mces.load_mces import LoadMCES

## CODE TO MERGE DATA FROM EDIT DISTANCE CALCULATIONS AND MCES 20

INPUT_FOLDER_ED= ""
INPUT_FOLDER_MCES_format=""    #IT IS ASSUMED THE MCES DATA IS SPLIT ACROSS SEVERAL FOLDERS
NUMBER_MCES_FOLDERS=1
OUTPUT_FOLDER= ""
SPLITS= ['_train', '_val', '_test']



    
for split in SPLITS:
    ## Load data EDIT DISTANCE
    edit_distance_data = LoadMCES.load_raw_data(directory_path=INPUT_FOLDER_ED,
                                                prefix="indexes_tani_incremental"+ split) 
                
    ## Load data MCES
    mces_data = LoadMCES.load_mces_20_data(directory_path=INPUT_FOLDER_MCES_format,
                                                prefix="indexes_tani_incremental"+ split, 
                                                number_folders=NUMBER_MCES_FOLDERS)
                                                
    ## Matching rows and create a new array

    ## Select only the column of MCES 20 and Edit distance

    ## Trasform MCES 20 to normalized MCES 20 (0-1) f(x)=1/(1 + log(x))

    ## Split the array and save
