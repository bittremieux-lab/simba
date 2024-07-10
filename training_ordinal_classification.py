

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




config = Config()


# In[274]:


config.USE_GUMBEL=False


# In[275]:


config.N_CLASSES=6


# In[276]:


config.D_MODEL=128


# In[277]:


config.bins_uniformise_INFERENCE=config.N_CLASSES-1


# In[278]:


config.use_uniform_data_INFERENCE = True


# ## Replicate standard regression training

# In[279]:


# In[281]:


# parameters
dataset_path = config.dataset_path
epochs = config.epochs
bins_uniformise_inference = config.bins_uniformise_INFERENCE
enable_progress_bar = config.enable_progress_bar
fig_path = config.CHECKPOINT_DIR + f"scatter_plot_{config.MODEL_CODE}.png"
model_code = config.MODEL_CODE


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


print(f"Number of pairs for train: {len(molecule_pairs_train)}")
print(f"Number of pairs for val: {len(molecule_pairs_val)}")
print(f"Number of pairs for test: {len(molecule_pairs_test)}")
print(f"Number of pairs for uniform test: {len(uniformed_molecule_pairs_test)}")


# In[284]:


## CALCULATION OF WEIGHTS
train_binned_list, _ = TrainUtils.divide_data_into_bins_categories(
    molecule_pairs_train,
    config.N_CLASSES-1,
        bin_sim_1=True, 
)
#train_binned_list, _ = TrainUtils.divide_data_into_bins(
#    molecule_pairs_train,
#    config.N_CLASSES-1,
#    bin_sim_1=False, 
#)


# In[285]:


train_binned_list[1].indexes_tani.shape


# In[286]:


plt.hist(molecule_pairs_train.indexes_tani[molecule_pairs_train.indexes_tani[:,2]>0][:,2], bins=20)


# In[287]:


[t.indexes_tani.shape for t in train_binned_list]


# In[288]:


weights, range_weights = WeightSampling.compute_weights_categories(train_binned_list)
#weights, range_weights = WeightSampling.compute_weights(train_binned_list)


# In[289]:


weights


# In[290]:


range_weights


# In[291]:


weights_tr = WeightSampling.compute_sample_weights_categories(molecule_pairs_train, weights)
weights_val = WeightSampling.compute_sample_weights_categories(molecule_pairs_val, weights)
#weights_tr = WeightSampling.compute_sample_weights(molecule_pairs_train, weights)
#weights_val = WeightSampling.compute_sample_weights(molecule_pairs_val, weights)


# In[292]:


weights_val[(molecule_pairs_val.indexes_tani[:,2]<0.1)]


# In[293]:


weights_val[(molecule_pairs_val.indexes_tani[:,2]<0.21) & (molecule_pairs_val.indexes_tani[:,2]>0.19)]


# In[294]:


plt.hist(molecule_pairs_val.indexes_tani[:,2], bins=100)
plt.yscale('log')


# In[295]:


plt.hist(weights_val)
plt.yscale('log')


# In[296]:


dataset_train = LoadDataOrdinal.from_molecule_pairs_to_dataset(molecule_pairs_train, training=True)
# dataset_test = LoadData.from_molecule_pairs_to_dataset(m_test)
dataset_val = LoadDataOrdinal.from_molecule_pairs_to_dataset(molecule_pairs_val)


#best_model_path = model_path = data_folder + 'best_model_exhaustive_sampled_128n_20240618.ckpt'
#best_model_path = config.CHECKPOINT_DIR + f"best_model_n_steps-v9.ckpt"
best_model_path = config.CHECKPOINT_DIR + f"last.ckpt"


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


# In[ ]:


#list(pred_test)


# In[ ]:


flat_pred_test[0]


# ## Corr. Analysis

# In[ ]:


plt.hist(similarities_test)


# In[ ]:


similarities_test


# In[ ]:


flat_pred_test


# In[ ]:


similarities_test=np.array(similarities_test)
flat_pred_test=np.array(flat_pred_test)


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


corr_model


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
plt.show()


# In[250]:


plt.scatter(similarities_test, flat_pred_test, alpha=0.01)


# ##### 

# In[ ]:





# In[ ]:




