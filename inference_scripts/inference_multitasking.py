

import os 

# In[268]:


import dill
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from pytorch_lightning.callbacks import ProgressBar
from simba.train_utils import TrainUtils
import matplotlib.pyplot as plt
from simba.config import Config
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from simba.parser import Parser
import random
from simba.weight_sampling import WeightSampling
from simba.losscallback import LossCallback
from simba.molecular_pairs_set import MolecularPairsSet
from simba.sanity_checks import SanityChecks
from simba.transformers.postprocessing import Postprocessing
from scipy.stats import spearmanr
import seaborn as sns
from simba.ordinal_classification.load_data_multitasking import LoadDataMultitasking
from simba.ordinal_classification.embedder_multitask import EmbedderMultitask
from sklearn.metrics import confusion_matrix, accuracy_score
from simba.load_mces.load_mces import LoadMCES   
from simba.performance_metrics.performance_metrics import PerformanceMetrics
import sys
import simba
sys.modules["src"] = simba

def remove_duplicates_array(array):
    seen = set()
    filtered_rows = []

    for row in array:
        # Create a tuple of the first two columns to check uniqueness
        key = tuple(sorted(row[:2]))  # Sort to account for unordered pairs
        if key not in seen:
            seen.add(key)
            filtered_rows.append(row)

    # Convert the filtered rows back to a NumPy array
    result = np.array(filtered_rows)
    return result

config = Config()
parser = Parser()
config = parser.update_config(config)

# In[274]:



# In[276]:


# In[277]:


config.bins_uniformise_INFERENCE=config.EDIT_DISTANCE_N_CLASSES-1


# In[278]:


config.use_uniform_data_INFERENCE = True


# ## Replicate standard regression training

# In[279]:


# In[280]:


# In[281]:
if not os.path.exists(config.CHECKPOINT_DIR):
    os.makedirs(config.CHECKPOINT_DIR)

# parameters
dataset_path = config.PREPROCESSING_DIR + config.PREPROCESSING_PICKLE_FILE
epochs = config.epochs
bins_uniformise_inference = config.bins_uniformise_INFERENCE
enable_progress_bar = config.enable_progress_bar
fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
model_code = config.MODEL_CODE


#
# In[282]:


print("loading file")
# Load the dataset from the pickle file
with open(dataset_path, "rb") as file:
    dataset = dill.load(file)

molecule_pairs_train = dataset["molecule_pairs_train"]
molecule_pairs_val = dataset["molecule_pairs_val"]
molecule_pairs_test = dataset["molecule_pairs_test"]

import copy
molecule_pairs_test_ed=   copy.deepcopy(molecule_pairs_test)
molecule_pairs_test_mces= copy.deepcopy(molecule_pairs_test)



# In[283]:
print('Loading pairs data ...')
#indexes_tani_multitasking_test = LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR, prefix='indexes_tani_incremental_test', 
#                                                             use_edit_distance=config.USE_EDIT_DISTANCE,
#                                                             use_multitask=config.USE_MULTITASK)
indexes_tani_multitasking_test = LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR_TRAIN, 
                                                             prefix='ed_mces_indexes_tani_incremental_test', 
                                                             use_edit_distance=config.USE_EDIT_DISTANCE,
                                                             use_multitask=config.USE_MULTITASK)

indexes_tani_multitasking_test = remove_duplicates_array(indexes_tani_multitasking_test)                          
molecule_pairs_test_ed.indexes_tani = indexes_tani_multitasking_test[:,0:3]


print(f'shape of similarity1: {molecule_pairs_test_ed.indexes_tani.shape}')

# add tanimotos

molecule_pairs_test_ed.tanimotos = indexes_tani_multitasking_test[:,3]

print(f'shape of similarity2: {molecule_pairs_test_ed.tanimotos.shape}')
print(f"Number of pairs for test: {len(molecule_pairs_test_ed)}")

# get the mces
molecule_pairs_test_mces.indexes_tani= indexes_tani_multitasking_test[:,[0,1,3]]
molecule_pairs_test_mces.tanimotos= indexes_tani_multitasking_test[:,3]


#best_model_path = model_path = data_folder + 'best_model_exhaustive_sampled_128n_20240618.ckpt'
#best_model_path = config.CHECKPOINT_DIR + f"best_model_n_steps-v9.ckpt"
#best_model_path = config.CHECKPOINT_DIR + f"last.ckpt"

if not(config.INFERENCE_USE_LAST_MODEL) and (os.path.exists(config.CHECKPOINT_DIR + f"best_model.ckpt")):     
    best_model_path = config.CHECKPOINT_DIR + config.BEST_MODEL_NAME
else:
    best_model_path = config.CHECKPOINT_DIR + f"last.ckpt"


#molecule_pairs_test = dataset["molecule_pairs_test"]
print(f"Number of molecule pairs: {len(molecule_pairs_test_ed)}")
print("Uniformize the data")
uniformed_molecule_pairs_test_ed, binned_molecule_pairs_ed = TrainUtils.uniformise(
    molecule_pairs_test_ed,
    number_bins=bins_uniformise_inference,
    return_binned_list=True,
    bin_sim_1=True,
    #bin_sim_1=False,
    ordinal_classification=True,
)  # do not treat sim==1 as another bin


uniformed_molecule_pairs_test_mces, binned_molecule_pairs_mces = TrainUtils.uniformise(
    molecule_pairs_test_mces,
    number_bins=bins_uniformise_inference,
    return_binned_list=True,
    bin_sim_1=False,
    #bin_sim_1=False,
    #ordinal_classification=True,
)  # do not treat sim==1 as another bin

# dataset_train = LoadData.from_molecule_pairs_to_dataset(m_train)
dataset_test_ed = LoadDataMultitasking.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_test_ed, max_num_peaks=int(config.TRANSFORMER_CONTEXT))
dataloader_test_ed = DataLoader(dataset_test_ed, batch_size=config.BATCH_SIZE, shuffle=False)

dataset_test_mces = LoadDataMultitasking.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_test_mces, max_num_peaks=int(config.TRANSFORMER_CONTEXT))
dataloader_test_mces = DataLoader(dataset_test_mces, batch_size=config.BATCH_SIZE, shuffle=False)

# In[ ]:


# Testinbest_model = Embedder.load_from_checkpoint(checkpoint_callback.best_model_path, d_model=64, n_layers=2)
trainer = pl.Trainer(max_epochs=2, enable_progress_bar=enable_progress_bar)
best_model = EmbedderMultitask.load_from_checkpoint(
    best_model_path,
    d_model=int(config.D_MODEL),
    n_layers=int(config.N_LAYERS),
    n_classes=config.EDIT_DISTANCE_N_CLASSES,
    use_gumbel=config.EDIT_DISTANCE_USE_GUMBEL,
    use_element_wise=True,
    use_cosine_distance=config.use_cosine_distance,
    use_edit_distance_regresion=config.USE_EDIT_DISTANCE_REGRESSION,
)

best_model.eval()

# ## Postprocessing

# In[ ]:

# prediction of ed
pred_test_ed = trainer.predict(
    best_model,
    dataloader_test_ed,
)

# prediction of mces
pred_test_mces = trainer.predict(
    best_model,
    dataloader_test_mces,
)
similarities_test1_ed, similarities_test2_ed = Postprocessing.get_similarities_multitasking(dataloader_test_ed)
similarities_test1_mces, similarities_test2_mces = Postprocessing.get_similarities_multitasking(dataloader_test_mces)

## Now assign the coorect similarity
similarities_test1 = similarities_test1_ed
similarities_test2 = similarities_test2_mces

# In[ ]:
def softmax(x):
        e_x = np.exp(x)  # Subtract max(x) for numerical stability
        return e_x / e_x.sum()


def which_index(p, threshold=0.5):
    return np.argmax(p)

def which_index_confident(p, threshold=0.50):
    # only predict confident predictions
    p_softmax= softmax(p)
    highest_pred = np.argmax(p_softmax)
    if p_softmax[highest_pred]>threshold:
        return np.argmax(p)
    else:
        return np.nan

def which_index_regression(p, max_index=5):
    ## the value of 0.2 must be the center of the second item

    index=np.round(p*max_index)
    # ad hoc solution
    #index=(-(np.round(p*max_index)))

    #index=np.clip(index, 0, 5)
    return index

#print(len(pred_test))
#print(pred_test[0])
#print(pred_test[0].shape)
#print(f'Shape of pred_test: {len(pred_test)}')
# flat the results
flat_pred_test1 = []
raw_flat_pred_test1 = []
confident_pred_test1=[]

flat_pred_test2 = []
confident_pred_test2=[]

flat_pred_test2 = [[p.item() for p in pred[1]] for pred in pred_test_mces]
flat_pred_test2= [item for sublist in flat_pred_test2 for item in sublist]
flat_pred_test2=np.array( flat_pred_test2)


flat_pred_test2_ed = []
flat_pred_test2_ed = [[p.item() for p in pred[1]] for pred in pred_test_ed]
flat_pred_test2_ed= [item for sublist in flat_pred_test2_ed for item in sublist]
flat_pred_test2_ed=np.array( flat_pred_test2_ed)
# In[250]:
#raw_flat_pred_test1=np.array(raw_flat_pred_test1)
#plt.figure()
#error_x= np.random.randint(0,100,raw_flat_pred_test1.shape[0])/200 - 0.5
#error_y= np.random.randint(0,100,raw_flat_pred_test1.shape[0])/200 - 0.5
#plt.scatter(5-(np.array(similarities_test1))+error_x, 5-5*(raw_flat_pred_test1)+error_y, alpha=0.1)
#plt.xlabel('edit distance')
#plt.ylabel('prediction')
#plt.grid()
#plt.savefig(config.CHECKPOINT_DIR + f"raw_edit_distance_scatter_plot_{config.MODEL_CODE}.png")


# lets extract the mces distance
flat_pred_test1 = [p[0] for p in pred_test_ed]
flat_pred_test1 = [[which_index(p) for p in p_list] for p_list in flat_pred_test1]
flat_pred_test1= [item for sublist in flat_pred_test1 for item in sublist]
flat_pred_test1=np.array(flat_pred_test1)

print(f'Example of edit distance prediction: {flat_pred_test1}')
# convert to numpy
confident_pred_test1=np.array(confident_pred_test1)



# get the results
similarities_test1=np.array(similarities_test1)
flat_pred_test1=np.array(flat_pred_test1)

similarities_test2=np.array(similarities_test2)
flat_pred_test2=np.array(flat_pred_test2)


print(f'Max value of similarities 1: {max(similarities_test1)}')
print(f'Min value of similarities 1: {min(similarities_test1)}')

# analyze errors and good predictions
#good_indexes = PerformanceMetrics.get_correct_predictions(similarities_test1_ed, flat_pred_test1, similarities_test2_ed, flat_pred_test2_ed,)
#bad_indexes =  PerformanceMetrics.get_bad_predictions(similarities_test1_ed, flat_pred_test1, similarities_test2_ed, flat_pred_test2_ed,)

#PerformanceMetrics.plot_molecules(uniformed_molecule_pairs_test_ed, similarities_test1_ed, similarities_test2_ed,
#                                            flat_pred_test1,flat_pred_test2_ed,  good_indexes, config, prefix='good')

#PerformanceMetrics.plot_molecules(uniformed_molecule_pairs_test_ed, similarities_test1_ed, similarities_test2_ed,
#                                           flat_pred_test1,flat_pred_test2_ed,  bad_indexes, config, prefix='bad')
# In[ ]:


len(similarities_test1)


# In[ ]:


similarities_test_cleaned1= similarities_test1[~np.isnan(flat_pred_test1)]
flat_pred_test_cleaned1= flat_pred_test1[~np.isnan(flat_pred_test1)]


# In[ ]:


len(similarities_test_cleaned1)


# In[ ]:


corr_model1, p_value_model1= spearmanr(similarities_test_cleaned1, flat_pred_test_cleaned1)


# In[ ]:


print(f'Correlation of edit distance model: {corr_model1}')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_cm(true, preds, config, file_name='cm.png'):
    # Compute the confusion matrix and accuracy
    cm = confusion_matrix(true, preds)
    accuracy = accuracy_score(true, preds)
    print("Accuracy:", accuracy)

    # Normalize the confusion matrix by the number of true instances per class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create the plot
    plt.figure(figsize=(10, 7))
    labels = ['>5', '4', '3', '2', '1', '0']
    
    # Plot the heatmap using the 'Blues' colormap
    im = plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    # Compute a threshold to decide the annotation text color
    threshold = cm_normalized.max() / 2.0

    # Annotate each cell with the percentage, using white text if the background is dark
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            text_color = "white" if cm_normalized[i, j] > threshold else "black"
            plt.text(j, i, f'{cm_normalized[i, j]:.2%}', ha='center', va='center', color=text_color)
    
    # Set tick labels and increase font size for clarity
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, fontsize=12)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=12)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.title(f'Confusion Matrix (Normalized), Acc: {accuracy:.2f}, Samples: {preds.shape[0]}', fontsize=16)
    
    # Save the plot
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, file_name))

plot_cm(similarities_test_cleaned1, flat_pred_test_cleaned1, config)
## analyze the impact of thresholding

similarities_test_cleaned_confident1= similarities_test1[~np.isnan(confident_pred_test1)]
flat_pred_test_cleaned_confident1= confident_pred_test1[~np.isnan(confident_pred_test1)]

confident_corr_model1, confident_p_value_model1= spearmanr(similarities_test_cleaned_confident1, flat_pred_test_cleaned_confident1)

print(f'Original size of predictions:{similarities_test1.shape}')
print(f'Confident size of predictions:{similarities_test_cleaned_confident1.shape}')
# In[ ]:


print(f'Correlation of model (confident): {confident_corr_model1}')

try:    
    plot_cm(similarities_test_cleaned_confident1, flat_pred_test_cleaned_confident1, config, file_name='confident_cm.png')
    # In[250]:
    plt.figure()
    np.random.seed(42)
    error_x= np.random.randint(0,100,flat_pred_test1.shape[0])/200 - 0.5
    error_y= np.random.randint(0,100,flat_pred_test1.shape[0])/200 - 0.5
    plt.scatter(5-(similarities_test1)+error_x, 5-flat_pred_test1+error_y, alpha=0.1)
    plt.xlabel('edit distance')
    plt.ylabel('prediction')
    plt.grid()
    plt.savefig(config.CHECKPOINT_DIR + f"edit_distance_scatter_plot_{config.MODEL_CODE}.png")
except:
    print('Problem generating cm matrix for confident predictions')


####### SECOND SIMILARITY ######

counts, bins= TrainUtils.count_ranges(similarities_test2, number_bins=5, bin_sim_1=False, max_value=1)


print('BEFORE BINING:')
print(f'Max value of similarities 2: {max(similarities_test2)}')
print(f'Min value of similarities 2: {min(similarities_test2)}')
print(f'Number of samples per bin for similarity 2: {counts}')
print(f'Bins for similarity 2: {bins}')

min_bin=min([c for c in counts if c>0])
print(f'Min bin for similarity 2: {min_bin}')

@staticmethod
def divide_predictions_in_bins(list_elements1, list_elements2, number_bins=5, bin_sim_1=False, min_bin=0, max_value=0):
    #count the instances in the  bins from 0 to 1
    # Group the values into the corresponding bins, adding one for sim=1

    list_elements1=list_elements1/max_value
    list_elements2=list_elements2/max_value
    output_elements1=np.array([])
    output_elements2=np.array([])


    if bin_sim_1:
        number_bins_effective = number_bins + 1
    else:
        number_bins_effective = number_bins

    for p in range(int(number_bins_effective)):
        if p==0: # cover all the possible values equal or lower than 0
            low = -np.inf

        if bin_sim_1:
            high = (p + 1) * (1 / number_bins)
        else:
            if p == (number_bins_effective - 1):
                high = np.inf
            else:
                high = (p + 1) * (1 / number_bins)

        list_elements1_temp = list_elements1[(list_elements1>=low) & (list_elements1<high)] 
        list_elements2_temp = list_elements2[(list_elements1>=low) & (list_elements1<high)] 

        #randomize the arrays
        if len(list_elements1_temp)>0:
            np.random.seed(42)
            random_indexes=np.random.randint(0,list_elements1_temp.shape[0], min_bin)
            output_elements1 = np.concatenate((output_elements1,list_elements1_temp[random_indexes]))
            output_elements2 = np.concatenate((output_elements2,list_elements2_temp[random_indexes]))

    return output_elements1, output_elements2

#similarities_test2, flat_pred_test2 = divide_predictions_in_bins(similarities_test2, flat_pred_test2, number_bins=5, bin_sim_1=False, min_bin=min_bin, 
#max_value=1)

print('')
print('AFTER BINING:')
print(f'Max value of similarities 2: {max(similarities_test2)}')
print(f'Min value of similarities 2: {min(similarities_test2)}')
print(f'Number of samples per bin for similarity 2: {counts}')
print(f'Bins for similarity 2: {bins}')

min_bin=min([c for c in counts if c>0])
print(f'Min bin for similarity 2: {min_bin}')
print('ground truth similarity 2')

print(similarities_test2)
print(similarities_test2.shape)

print('pred similarity 2')
print(flat_pred_test2)
print(flat_pred_test2.shape)
# In[ ]:


## Remove values correspoding to the threshold
similarities_test2_original=similarities_test2.copy()
similarities_test2= similarities_test2[similarities_test2_original != 0.5]
flat_pred_test2 =   flat_pred_test2[similarities_test2_original != 0.5]
corr_model2, p_value_model2= spearmanr(similarities_test2, flat_pred_test2)


# In[ ]:

if not(config.USE_TANIMOTO): #if using mces20, apply de-normalization to obtain scalar value sof MCES20
    similarities_test2= config.MCES20_MAX_VALUE*(1-similarities_test2)
    flat_pred_test2= config.MCES20_MAX_VALUE*(1-flat_pred_test2)
    
print(f'Correlation of tanimoto model: {corr_model2}')
sns.set_theme(style="ticks")
plot = sns.jointplot(x=similarities_test2, y=flat_pred_test2, kind="hex", color="#4CB391", joint_kws=dict(alpha=1, gridsize=15))
# Set x and y labels
plot.set_axis_labels("Ground truth Similarity", "Prediction", fontsize=12)
plot.fig.suptitle(f"Spearman Correlation:{corr_model2}", fontsize=16)
# Set x-axis limits
plot.ax_joint.set_xlim(0, 40)
# Set x-axis limits
plot.ax_joint.set_ylim(0, 40)
plt.savefig(config.CHECKPOINT_DIR + f"hexbin_plot_{config.MODEL_CODE}.png")



## save scatter plot
plt.scatter(similarities_test2, flat_pred_test2, alpha=0.5)
plt.xlabel('ground truth')
plt.ylabel('prediction')
plt.grid()
plt.savefig(config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png")

