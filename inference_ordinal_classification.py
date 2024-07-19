

import os 

# In[268]:


import dill
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from pytorch_lightning.callbacks import ProgressBar
from src.train_utils import TrainUtils
import matplotlib.pyplot as plt
from src.config import Config
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from src.parser import Parser
import random
from src.weight_sampling import WeightSampling
from src.losscallback import LossCallback
from src.molecular_pairs_set import MolecularPairsSet
from src.sanity_checks import SanityChecks
from src.transformers.postprocessing import Postprocessing
from scipy.stats import spearmanr
import seaborn as sns
from src.ordinal_classification.load_data_ordinal import LoadDataOrdinal
from src.ordinal_classification.embedder_ordinal import EmbedderOrdinal
from sklearn.metrics import confusion_matrix
from src.load_mces.load_mces import LoadMCES   
from src.performance_metrics.performance_metrics import PerformanceMetrics


config = Config()
parser = Parser()
config = parser.update_config(config)

# In[274]:


# In[274]:

config.USE_GUMBEL=False


# In[275]:


config.N_CLASSES=6


# In[276]:


# In[277]:


config.bins_uniformise_INFERENCE=config.N_CLASSES-1


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
uniformed_molecule_pairs_test = dataset["uniformed_molecule_pairs_test"]


# In[283]:
print('Loading pairs data ...')
molecule_pairs_test.indexes_tani = LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR, prefix='indexes_tani_incremental_test')
print(f"Number of pairs for test: {len(molecule_pairs_test)}")



#best_model_path = model_path = data_folder + 'best_model_exhaustive_sampled_128n_20240618.ckpt'
#best_model_path = config.CHECKPOINT_DIR + f"best_model_n_steps-v9.ckpt"
best_model_path = config.CHECKPOINT_DIR + f"last.ckpt"
#best_model_path = config.CHECKPOINT_DIR + f"best_model_n_steps.ckpt"

# In[ ]:


molecule_pairs_test.indexes_tani.shape


# In[ ]:


molecule_pairs_test = dataset["molecule_pairs_test"]
print(f"Number of molecule pairs: {len(molecule_pairs_test)}")
print("Uniformize the data")
uniformed_molecule_pairs_test, binned_molecule_pairs = TrainUtils.uniformise(
    molecule_pairs_test,
    number_bins=bins_uniformise_inference,
    return_binned_list=True,
    bin_sim_1=True,
    #bin_sim_1=False,
    ordinal_classification=True,
)  # do not treat sim==1 as another bin


# In[ ]:


binned_molecule_pairs[4].indexes_tani.shape


# In[ ]:


uniformed_molecule_pairs_test.indexes_tani


# In[ ]:


# dataset_train = LoadData.from_molecule_pairs_to_dataset(m_train)
dataset_test = LoadDataOrdinal.from_molecule_pairs_to_dataset(uniformed_molecule_pairs_test)
dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, shuffle=False)


# In[ ]:


# Testinbest_model = Embedder.load_from_checkpoint(checkpoint_callback.best_model_path, d_model=64, n_layers=2)
trainer = pl.Trainer(max_epochs=2, enable_progress_bar=enable_progress_bar)
best_model = EmbedderOrdinal.load_from_checkpoint(
    best_model_path,
    d_model=int(config.D_MODEL),
    n_layers=int(config.N_LAYERS),
    n_classes=config.N_CLASSES,
    use_gumbel=config.USE_GUMBEL,
    use_element_wise=True,
    use_cosine_distance=config.use_cosine_distance,
    
)


# ## Postprocessing

# In[ ]:


pred_test = trainer.predict(
    best_model,
    dataloader_test,
)
similarities_test = Postprocessing.get_similarities(dataloader_test)


# In[ ]:


plt.hist(similarities_test)


# In[ ]:


print(pred_test[0][4])
print(similarities_test[127])


# In[ ]:


pred_test[0][6]


# In[ ]:


np.argwhere(pred_test[0][0]>0.1)[0]


# In[ ]:


np.argwhere(pred_test[0][0]>0.9)[0].numel()


# In[ ]:


def which_index(p, threshold=0.5):
    #result= np.argwhere(p>threshold)[0]
     #
    #if result.numel()==0:
        #return np.argmax(p)
    #    return np.nan
    #else:
    #    return result[-1]
    return np.argmax(p)


# In[ ]:


# flat the results
flat_pred_test = []
for pred in pred_test:
    flat_pred_test = flat_pred_test + [which_index(p) for p in pred]
flat_pred_test=np.array( flat_pred_test)



# get the results
similarities_test=np.array(similarities_test)
flat_pred_test=np.array(flat_pred_test)

# analyze errors and good predictions
good_indexes = PerformanceMetrics.get_correct_predictions(similarities_test, flat_pred_test)
bad_indexes = PerformanceMetrics.get_bad_predictions(similarities_test, flat_pred_test)

PerformanceMetrics.plot_molecules(uniformed_molecule_pairs_test, similarities_test, flat_pred_test, good_indexes, config, prefix='good')
PerformanceMetrics.plot_molecules(uniformed_molecule_pairs_test, similarities_test, flat_pred_test, bad_indexes, config, prefix='bad')



# In[ ]:


len(similarities_test)


# In[ ]:


similarities_test_cleaned= similarities_test[~np.isnan(flat_pred_test)]
flat_pred_test_cleaned= flat_pred_test[~np.isnan(flat_pred_test)]


# In[ ]:


len(similarities_test_cleaned)


# In[ ]:


corr_model, p_value_model= spearmanr(similarities_test_cleaned, flat_pred_test_cleaned)


# In[ ]:


print(f'Correlation of model: {corr_model}')


# In[ ]:


# Compute the confusion matrix
cm = confusion_matrix(similarities_test_cleaned, flat_pred_test_cleaned)
# Normalize the confusion matrix by the number of true instances for each class
cm_normalized = cm.astype('float') / cm.sum()
# Plot the confusion matrix with percentages
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Normalized to Percentages)')
plt.savefig(config.CHECKPOINT_DIR + 'cm.png')
plt.show()


# In[250]:


plt.scatter(similarities_test, flat_pred_test, alpha=0.01)


# ##### 

# In[ ]:





# In[ ]:




