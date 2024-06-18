

import os


import numpy as np



data_folder= '/scratch/antwerpen/209/vsc20939/data/'
# datapath of tanimoto pairs computed
data_path =data_folder+ 'indexes_tani_train_exhaustive.npy'

# datapath of model similarities computed
matrix_similarities_path= data_folder+  'matrix_similarities.npy'
output_path=data_folder + 'indexes_tani_train_resampled.npy'


# load data
indexes_tani_train =np.load(data_path)


# In[8]:


indexes_tani_train_high_exhaustive= indexes_tani_train[indexes_tani_train[:,2]>0.5]


# In[9]:


indexes_tani_train_high_exhaustive.shape


# In[10]:


size_low_samples=indexes_tani_train_high_exhaustive.shape[0]


# ## Load the predictions to be used

# In[11]:


import pickle
#path = data_folder + 'merged_gnps_nist_20240319_unique_smiles_100_million_v2_no_identity.pkl'
#path = data_folder + 'merged_gnps_nist_20240516_150_millions.pkl'
#with open(path, 'rb') as file:
#    dataset = pickle.load(file)


# In[12]:


#indexes_tani_train = dataset['molecule_pairs_train'].indexes_tani


# In[13]:


matrix_similarities= np.load(matrix_similarities_path)


# In[14]:


# convert matrix similarities to the dimension of index train
from tqdm import tqdm


# In[15]:


new_column = np.zeros(indexes_tani_train.shape[0]).reshape(-1, 1)


# In[16]:


indexes_tani_train.shape


# In[17]:


new_column.shape


# In[18]:


indexes_tani_train= np.column_stack((indexes_tani_train, new_column))


# ## Match the predictions with the correspondent tanimoto

# In[ ]:


for i,row in tqdm(enumerate(indexes_tani_train)):
    x = int(row[0])
    y = int(row[1])
    indexes_tani_train[i,3] = matrix_similarities[x,y]


# ## Compute the difference between prediction and ground truth

# In[ ]:


# compute abs diff between predicition and ground truth
indexes_tani_train[:,3] = np.abs(indexes_tani_train[:,2]-indexes_tani_train[:,3])


# In[ ]:


indexes_tani_train[:,3] 


# In[ ]:


indexes_tani_train[:,2] 


# In[ ]:


indexes_tani_train.shape


# In[ ]:





# In[ ]:


#arg_sorted = np.argsort(indexes_tani_train[:,3])[::-1]
#indexes_tani_train = indexes_tani_train[arg_sorted]


# In[ ]:


arg_sorted = np.argsort(indexes_tani_train[:,3])[::-1]


# In[ ]:


arg_sorted[0]


# In[ ]:


indexes_tani_train = indexes_tani_train[arg_sorted]


# ## Order the pairs by difference between ground truth and prediction

# In[ ]:


indexes_tani_train[10361728]


# In[ ]:


indexes_tani_train_filtered = indexes_tani_train[0:size_low_samples]


# In[ ]:


indexes_tani_train_high_exhaustive.shape


# In[ ]:


indexes_tani_train_filtered.shape


# In[ ]:


indexes_tani_train_filtered = np.concatenate((indexes_tani_train_filtered[:,0:3],indexes_tani_train_high_exhaustive ))


# In[ ]:


indexes_tani_train_filtered.shape


# In[ ]:


indexes_tani_train_filtered = np.unique(indexes_tani_train_filtered, axis=0)


# In[ ]:


indexes_tani_train_filtered.shape


# In[ ]:


indexes_tani_train.shape


# In[ ]:


np.save(output_path, indexes_tani_train_filtered)


# In[ ]:


pwd


# ## Check the distribution of the sampled data

# In[ ]:


tani_sampled=indexes_tani_sorted[0:size_low_samples][:,3]


# In[ ]:


tani_sampled.shape


# In[ ]:


import matplotlib.pyplot as plt
results_hist=plt.hist(tani_sampled, bins=np.arange(0,10)/10)


# In[ ]:


results_hist[0]


# In[ ]:


results_hist[1]


# In[ ]:


for freq, bin in zip(results_hist[0],results_hist[1]):
    print(f'{bin},{freq}')


# In[ ]:




