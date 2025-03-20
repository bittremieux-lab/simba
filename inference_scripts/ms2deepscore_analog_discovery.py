
# %%
#import os
#os.chdir('/Users/sebas/projects/metabolomics')
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# %%
#from src.transformers.sklearn_model import SklearnModel
import gensim
from src.load_data import LoadData
from src.config import Config
from tqdm import tqdm
from src.loader_saver import LoaderSaver
import itertools
import numpy as np
from scipy.stats import spearmanr
import dill
from src.plotting import Plotting
from src.load_data import LoadData
from src.molecule_pairs_opt import MoleculePairsOpt
from sklearn.model_selection import train_test_split
from src.train_utils import TrainUtils
from src.preprocessor import Preprocessor
import pickle
import sys
from src.config import Config
from src.parser import Parser
from datetime import datetime
from ms2deepscore import MS2DeepScore
from src.loader_saver import LoaderSaver
from src.molecular_pairs_set import MolecularPairsSet
from scipy.stats import spearmanr
from src.transformers.embedder import Embedder
from src.transformers.encoder import Encoder
import matplotlib.pyplot as plt
from src.transformers.CustomDatasetEncoder import CustomDatasetEncoder
from src.transformers.load_data_encoder import LoadDataEncoder
from torch.utils.data import DataLoader
from src.analog_discovery.cosine_similarity import CosineSimilarity
from rdkit import Chem
from matchms.importing import load_from_mgf,load_from_msp
from matchms.similarity import ModifiedCosine
#from src.spec2vec_comparison import Spec2VecComparison
#from spec2vec import Spec2Vec
import tensorflow as tf
from ms2deepscore.models import load_model


# %% [markdown]
# ## params



# %%
data_folder= '/scratch/antwerpen/209/vsc20939/data/'
gnps_path =  data_folder + 'ALL_GNPS_NO_PROPOGATED_wb.mgf'
janssen_path = data_folder + 'drug_plus.mgf'
nist_path = data_folder + 'hr_msms_nist_all.MSP'
output_janssen_file= data_folder + 'all_spectrums_janssen.pkl'
dataset_path= data_folder +'merged_gnps_nist_20240311_unique_smiles_1_million.pkl'
model_path = data_folder + 'best_model_20240319_v2_512u_5_layers.ckpt'
model_spec2vec_file = data_folder + 'spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model'
model_ms2d_file = data_folder + 'ms2deepscore_positive_10k_1000_1000_1000_500.hdf5'

# %%
config=Config()

# %%
config.D_MODEL=512
config.N_LAYERS=5

# %%
similarity_name= 'ms2deepscore'

# %% [markdown]
# ## open janssen data

# %%
all_spectrums_janssen_matchms = [s for s in load_from_mgf(janssen_path)]

# %%
loader_saver = LoaderSaver(
        block_size=100,
        pickle_nist_path='',
        pickle_gnps_path='',
        pickle_janssen_path=output_janssen_file,
    )

# %%
all_spectrums_janssen_su = loader_saver.get_all_spectrums(
            janssen_path,
            100000000,
            use_tqdm=True,
            use_nist=False,
            config=config,
            use_janssen=True,
        )

# %% [markdown]
# ## open ref data

# %%
with open(dataset_path, 'rb') as file:
            dataset = dill.load(file)

# %%
all_spectrums_reference_su= dataset['molecule_pairs_train'].spectrums_original +\
                    dataset['molecule_pairs_val'].spectrums_original + \
                        dataset['molecule_pairs_test'].spectrums_original

# %%
all_spectrums_gnps_matchms = [s for s in load_from_mgf(gnps_path)]

# %%
all_spectrums_nist_matchms=  [s for s in load_from_msp(nist_path)]

# %%
all_spectrums_reference_matchms = all_spectrums_gnps_matchms + all_spectrums_nist_matchms

# %%
#compute all the hashes from the datasets
target_hashes = [s.spectrum_hash for s in all_spectrums_reference_su]

# %%
matchms_hashes=[]
for s in  tqdm(all_spectrums_reference_matchms):
    matchms_hashes.append(s.spectrum_hash())

# %%
len(target_hashes)

# %%
len(all_spectrums_reference_matchms)

# %%
matchms_hashes.index(target_hashes[0])

# %%
indexes_matched = [matchms_hashes.index(t) for t in tqdm(target_hashes)]

# %%
all_spectrums_reference = [all_spectrums_reference_matchms[index] for index in indexes_matched]

# %%
for i,(s_ref, s_su) in enumerate(zip(all_spectrums_reference, all_spectrums_reference_su)):
    new_metadata=s_ref.metadata.copy()
    new_metadata['smiles']=s_su.smiles
    all_spectrums_reference[i].metadata=new_metadata

# %% [markdown]
# ## Filter spectra from Janssen

# %%
su_hashes = [s.spectrum_hash for s in all_spectrums_janssen_su]

# %%
all_spectrums_janssen = [s for s in all_spectrums_janssen_matchms if s.spectrum_hash() in su_hashes]

# %% [markdown]
# ## Find those instances that are not in reference

# %%
canon_smiles_reference = [Chem.CanonSmiles(s.metadata['smiles']) for s in all_spectrums_reference]
canon_smiles_janssen =   [Chem.CanonSmiles(s.metadata['smiles']) for s in all_spectrums_janssen]
janssen_indexes_in_ref= [i for i,s in enumerate(canon_smiles_janssen) if s in canon_smiles_reference]
janssen_indexes_not_in_ref = [i for i,s in enumerate(canon_smiles_janssen) if s not in canon_smiles_reference]

# %%


# %%
len(janssen_indexes_in_ref),len(janssen_indexes_not_in_ref)

# %%
all_spectrums_janssen = [all_spectrums_janssen[index] for index in janssen_indexes_not_in_ref]

# %% [markdown]
# ## load model

# %%
#encoder= Encoder(model_path, D_MODEL=int(config.D_MODEL),N_LAYERS=int(config.N_LAYERS))

# %%
#similarity_model =ModifiedCosine(tolerance=0.1)

# %%
if similarity_name=='spec2vec':
    model = gensim.models.Word2Vec.load(model_spec2vec_file)
    similarity_model= Spec2Vec(
                model=model, intensity_weighting_power=0.5, allowed_missing_percentage=100.0
            )
    PREPROCESS_SPECTRUMS=True
elif similarity_name=='modified_cosine':
    similarity_model =ModifiedCosine(tolerance=0.1)
    PREPROCESS_SPECTRUMS=True
elif similarity_name == 'ms2deepscore':
    model_ms2d = load_model(model_ms2d_file)
    similarity_model= MS2DeepScore(model_ms2d)
    PREPROCESS_SPECTRUMS=False

# %% [markdown]
# ## Preprocessed spectrums

# %%
PREPROCESS_SPECTRUMS

# %%
preprocessed_all_spectrums_janssen =all_spectrums_janssen.copy()
if PREPROCESS_SPECTRUMS:
    for i,s in tqdm(enumerate(preprocessed_all_spectrums_janssen)):
        preprocessed_all_spectrums_janssen[i] = Spec2VecComparison.spectrum_processing(preprocessed_all_spectrums_janssen[i])
preprocessed_all_spectrums_janssen = [s for s in preprocessed_all_spectrums_janssen if ((s is not None)and (s.metadata['precursor_mz']>0))]
preprocessed_all_spectrums_janssen = [s for s in preprocessed_all_spectrums_janssen if s.mz[(s.mz >= 10) & (s.mz <= 1000)].shape[0]>0]

# %%
len(preprocessed_all_spectrums_janssen)

# %%
preprocessed_all_spectrums_janssen[0].mz.shape[0]

# %%
preprocessed_all_spectrums_reference =all_spectrums_reference.copy()
if PREPROCESS_SPECTRUMS:
    for i,s in tqdm(enumerate(preprocessed_all_spectrums_reference)):
        preprocessed_all_spectrums_reference[i] = Spec2VecComparison.spectrum_processing(preprocessed_all_spectrums_reference[i])
preprocessed_all_spectrums_reference = [s for s in preprocessed_all_spectrums_reference if ((s is not None)and (s.metadata['precursor_mz']>0))]
preprocessed_all_spectrums_reference = [s for s in preprocessed_all_spectrums_reference if s.mz[(s.mz >= 10) & (s.mz <= 1000)].shape[0]>0]

# %%
len(preprocessed_all_spectrums_reference)

# %%
#with open('preprocessed_all_spectrums_reference.pkl', 'wb') as file:
#        dictionary={'preprocessed_all_spectrums_reference':preprocessed_all_spectrums_reference}
#        dill.dump(dictionary,file)

#with open('preprocessed_all_spectrums_reference.pkl', 'rb') as file:
#        preprocessed_all_spectrums_reference=dill.load(file)['preprocessed_all_spectrums_reference']

# %%


# %% [markdown]
# ## compute similarities

# %%
from matchms import calculate_scores
if similarity_name != 'ms2deepscore':
    results_scores = calculate_scores(
                        preprocessed_all_spectrums_reference, preprocessed_all_spectrums_janssen, similarity_model
                    )
    results_tuple = [results_scores.scores_by_query(s, name='Spec2Vec', sort=True)[0] for s in preprocessed_all_spectrums_janssen]

else:
        #with tf.device('/device:CPU:0'): #execute on cpu
            results_scores = calculate_scores(
                        preprocessed_all_spectrums_reference, preprocessed_all_spectrums_janssen, similarity_model
                    )
            results_tuple = [results_scores.scores_by_query(s, name='MS2DeepScore', sort=True)[0] for s in preprocessed_all_spectrums_janssen]
        

# %%
spectrums_retrieved = [r[0] for r in results_tuple]
max_sim = [r[1] for r in results_tuple]

# %%
#spectrums_retrieved = [results_scores.scores_by_query(s, name='Spec2Vec', sort=True)[0][0] \
#                       for s in preprocessed_all_spectrums_janssen]

# %%
#max_sim = [results_scores.scores_by_query(s, name='Spec2Vec', sort=True)[0][1] \
#                       for s in preprocessed_all_spectrums_janssen]

# %% [markdown]
# ## Based on the similarities compute the similarity score of the match spectrum

# %%
smiles_retrieved = [s.metadata['smiles'] for s in spectrums_retrieved]

# %%
smiles_janssen = [s.metadata['smiles'] for s in preprocessed_all_spectrums_janssen]

# %%
from src.tanimoto import Tanimoto

# %%
tanimoto_retrieved = [Tanimoto.compute_tanimoto_from_smiles(s0,s1) for s0, s1 in zip(smiles_janssen, smiles_retrieved)]

# %%
_=plt.hist(max_sim, color='r',bins=10)
plt.xlabel('maximum predicted similarity found')
plt.ylabel('freq')
plt.grid()

# %%
plt.hist(tanimoto_retrieved)
plt.grid()
plt.xlabel('tanimoto similarity with reference spectra')
plt.ylabel('frequency')

# %%
tanimoto_retrieved= np.array(tanimoto_retrieved)

# %%
plt.scatter(tanimoto_retrieved, max_sim, alpha=0.5)
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()
plt.ylabel('prediction spec2vec')
plt.xlabel('tanimoto')

# %% [markdown]
# ## Check which is the spectra that has wrong predictions

# %%
from rdkit import Chem
from rdkit.Chem import rdFMCS
def calculate_mcs_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    # Perform MCS (Maximum Common Substructure) search
    mcs = rdFMCS.FindMCS([mol1, mol2])

    # Get SMARTS pattern from MCS result
    mcs_smarts = Chem.MolToSmarts(mcs.queryMol)
    
    # Calculate Tanimoto-like similarity
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
    #mcs_count = len(Chem.GetMolFrags(mcs_mol))
    mcs_count = mcs_mol.GetNumAtoms()
    similarity = mcs_count / (mol1.GetNumAtoms() + mol2.GetNumAtoms() - mcs_count)

    return similarity, mcs_mol
from rdkit.Chem import rdRascalMCES
def calculate_mces_sim(smiles1, smiles2, similarity_threshold=0.7):

    ad1 = Chem.MolFromSmiles(smiles1)
    ad2 = Chem.MolFromSmiles(smiles2)
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = similarity_threshold
    opts.returnEmptyMCES = True
    results = rdRascalMCES.FindMCES(ad1, ad2, opts)
    if len(results) != 0:
        similarity_tier1= results[0].tier1Sim
        similarity_tier2= results[0].tier2Sim
    
        if similarity_tier2 != -1:
            return similarity_tier2 #if the lower threshold is not surpassed
        else:
            return similarity_tier1
    else:
        return None

# %%


# %%
len(all_spectrums_janssen)

# %%
len(spectrums_retrieved)

# %%
bad_predictions = np.argsort(abs(tanimoto_retrieved-max_sim))[::-1]

# %%
target_spectra=bad_predictions[-2]

# %%
max_sim[target_spectra]

# %%
tanimoto_retrieved[target_spectra]

# %%
## mcs
sim_mcs, mol_mcs=calculate_mcs_similarity(smiles1=all_spectrums_janssen[target_spectra].metadata['smiles'], 
                         smiles2=spectrums_retrieved[target_spectra].metadata['smiles'])
sim_mcs

# %%
spectrums_retrieved[target_spectra].plot_against(all_spectrums_janssen[target_spectra])
#plt.xlim(0, 100)

# %%
sim_mces = calculate_mces_sim(smiles1=all_spectrums_janssen[target_spectra].metadata['smiles'], 
                         smiles2=spectrums_retrieved[target_spectra].metadata['smiles'])
sim_mces

# %%
mol_mcs

# %%
## plot the molecules

# %%
Chem.CanonSmiles(all_spectrums_janssen[target_spectra].metadata['smiles'])

# %%
from rdkit import Chem
mol_janssen = Chem.MolFromSmiles(all_spectrums_janssen[target_spectra].metadata['smiles'])
mol_janssen

# %%
Chem.CanonSmiles(spectrums_retrieved[target_spectra].metadata['smiles'])

# %%
mol_ref = Chem.MolFromSmiles(spectrums_retrieved[target_spectra].metadata['smiles'])
mol_ref

# %% [markdown]
# ## FIND SIMILARITY BASED ON MCS

# %%
len(smiles_retrieved)

# %%
len(smiles_janssen)

# %%
mces_sims=[]
for s0,s1 in tqdm(zip(smiles_janssen, smiles_retrieved)):
    similarity= calculate_mces_sim(s0, s1)
    mces_sims.append(similarity)

# %%
# Specify the bin width
bin_width = 0.1

# Calculate the number of bins based on the data range and bin width
bins = np.arange(0, 1 + bin_width, bin_width)

# %%
plt.hist(tanimoto_retrieved,alpha=0.5, label='tanimoto sim.', density=True, bins=bins)
plt.hist([m for m in mces_sims if m is not None],alpha=0.5, label='mces sim.', density=True, bins=bins)
plt.xlabel('similarity')
plt.ylabel('density')
plt.legend()
plt.grid()

# %% [markdown]
# ## saving of results

# %%
def get_identifier(s):
    if 'nistno' in s.metadata:
        return(s.metadata['nistno'])
    elif 'spectrum_id' in s.metadata:
        return (s.metadata['spectrum_id'])

# %%
all_spectrums_ref_identifiers =[]
for s in all_spectrums_reference:
    ident = get_identifier(s)
    all_spectrums_ref_identifiers.append(ident)

# %%
all_spectrums_retrieved_identifiers =[]
for s in spectrums_retrieved:
    ident = get_identifier(s)
    all_spectrums_retrieved_identifiers.append(ident)

# %%
original_spectrums_retrieved = [all_spectrums_reference[all_spectrums_ref_identifiers.index(ident)] for s,ident in zip(spectrums_retrieved, all_spectrums_retrieved_identifiers)]

# %%
len(original_spectrums_retrieved)

# %%
# Create a box plot
sim_list=[]
labels_list=[]
for s in ['ms2deepscore']:
    sim_retrieved= [m for m in mces_sims if m is not None]
    sim_list.append(sim_retrieved)
plt.figure(figsize=(20,5))
plt.boxplot(sim_list, labels=['ms2deepscore'])
#plt.boxplot(, labels=labels_list)

plt.ylim([0,1.1])
plt.ylabel('MECS Similarity')
plt.title('')
plt.grid()

# %%
results ={ 'preprocessed_all_spectrums_janssen':preprocessed_all_spectrums_janssen,
          'original_all_spectrums_janssen':all_spectrums_janssen,
            'smiles_janssen':smiles_janssen,
          'smiles_retrieved':smiles_retrieved,
          'spectrums_retrieved':spectrums_retrieved,
          'original_spectrums_retrieved':original_spectrums_retrieved,
            'tanimoto_retrieved':tanimoto_retrieved,
          'max_sim':max_sim,
          'mces_retrieved':mces_sims}

# %%
with open('./notebooks/discovery_search/results/'+similarity_name + '_results_analog_discovery_unknwon_compounds.pkl', 'wb') as f:
    dill.dump(results, f)

# %% [markdown]
# ## check what are the characteristics of the spectrums retrieved

# %%
spectrums_retrieved

# %%
spectrums_retrieved[5].metadata

# %%
bad_indexes= np.argsort(tanimoto_retrieved)[0:20]
good_indexes= [index for index in range(0, len(tanimoto_retrieved)) if index not in bad_indexes]

# %%
bad_spectrums=[spectrums_retrieved[index] for index in bad_indexes ]
good_spectrums=[spectrums_retrieved[index] for index in good_indexes ]

# %%
is_gnps_bad_spectrums = [s for s in bad_spectrums if 'spectrum_id' in s.metadata]
is_gnps_good_spectrums = [s for s in good_spectrums if 'spectrum_id' in s.metadata]

# %%
len(is_gnps_bad_spectrums)/len(bad_spectrums)

# %%
len(is_gnps_good_spectrums)/len(good_spectrums)

# %% [markdown]
# ## pass the same bad spectrum used in simba here

# %%
len(preprocessed_all_spectrums_janssen)

# %%
len(preprocessed_all_spectrums_reference)

# %%
target_spectrum_janssen = [s for s in preprocessed_all_spectrums_janssen if s.metadata['id']=='CUIHSIWYWATEQL'][0]

# %%
target_spectrum_janssen

# %%
target_spectrum_ref = [s for s in preprocessed_all_spectrums_reference if 'spectrum_id' in s.metadata]
target_spectrum_ref = [s for s in target_spectrum_ref if s.metadata['spectrum_id']=='CCMSLIB00003134614'][0]

# %%
target_spectrum_ref

# %%
target_results=calculate_scores( [target_spectrum_ref], [target_spectrum_janssen], similarity_model
                    )

# %%
target_results.scores_by_query(target_spectrum_janssen, name='Spec2Vec', sort=True)

# %% [markdown]
# ## what I got when I run the model?

# %%
results_scores.scores_by_query(target_spectrum_janssen, name='Spec2Vec', sort=True)

# %%
spectrums_retrieved

# %%


# %%



