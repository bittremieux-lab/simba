## Project

Development of a learned similarity model for MS/MS spectrums in order to predict structural similarity.


## Organization of code:

* src/ Source code
* src/transformers: Source code for declaration of transformer model
* slurm_scripts: Scripts for running jobs in HPC
* python_scripts: Python scripts for different tasks
* notebooks: Jupyter Notebooks

## NIST data

* Donwload the installer for NIST
* Install it in windows
* Install LIB2NIST
* Export the library files to MSP
* Parse MSP file (nist_loader.py in this repository)
 



## GPU problem solving

* Make sure that pytorch library is not the cpu one in order to use the gpu.

conda remove pytorch cudatoolkit
conda clean --all

conda install -c anaconda cudatoolkit (11.8)

* Instal PyTorch (GPU version compatible with CUDA verison):

conda install pytorch=2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

## Wetransfer

wget --user-agent Mozilla/4.0 '[your big address here]' -O dest_file_name

## GLOBUS:

Download the server, login and run it in the background

wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
 tar xzf globusconnectpersonal-latest.tgzwget 


./globusconnectpersonal

 ./globusconnectpersonal -start &

 globus transfer dff8c41a-9419-11ee-83dc-d5484943e99a:/user/antwerpen/209/vsc20939/best_model_20231207.cpkt ddb59aef-6d04-11e5-ba46-22000b92c6ec:~/best_model_gpu_20231207.cpkt



## VSC

Run an interactive session:

srun -p ampere_gpu --gpus=1 --pty bash


## Loading Edit distance data precomputed at UC Riverside
* use the notebook load_new_edit_distance_20240925 to load the  data from a csv with inchis, to csv divided with the smiles in them
* use the notebook matching_data_in_ming_data to match the pairs loaded with the spectra we have
* compress the output folder and send it to the vsc supercomputer in the supercomputer, split the folder into 30 nodes and run the computation of pairs
* use the script in python_scripts called script_split_folder.py to split the files into 10 subfolders
* run the script: script_all_matched_mces_bash.slurm.sh for computing the mces
* after the computation of mces is finished in the split folders, the results must be merged with the edit distance, using script merge_edit_distance_mces_20_v2.py


## Generation of Edit Distance/MCES 

* Run the script script_preprocessing_ed_mces_parallel.sh. You have to set the PREPROCESSING_DIR where the spectra must be previously saved. This script will generate the npy files for edit distance and mces.

* The results of edit distance and mces must be merged. 
## To train a multitask model using edit distance and mces

* run the script script_transformers_multitasking.sh
* run script_inference_multitasking.sh for inference
