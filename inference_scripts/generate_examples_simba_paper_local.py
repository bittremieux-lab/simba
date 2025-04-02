import sys
import os

# Make sure the root path is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Optional: change the working directory too
os.chdir(project_root)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import dill
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from matchms.similarity import ModifiedCosine
from pytorch_lightning.callbacks import ProgressBar
from simba.train_utils import TrainUtils
import matplotlib.pyplot as plt
from simba.config import Config
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

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
from simba.tanimoto import Tanimoto
from matchms import calculate_scores
import tensorflow as tf
from tqdm import tqdm
from matchms.Pipeline import Pipeline, create_workflow
from matchms.filtering.default_pipelines import DEFAULT_FILTERS
import matchms.filtering as msfilters


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
config.extra_info = "_generated_data_peak_dropout_more_data"
config.TRANSFORMER_CONTEXT = 100
config.use_cosine_distance = 1
config.BEST_MODEL_NAME = "best_model-final_performance.ckpt"
config.LR = 0.00001
config = parser.update_config(config)

config.CHECKPOINT_DIR = "/Users/sebas/projects/data/model_checkpoints_256_units_5_layers_1000_epochs_1e-05_lr_128_bs_generated_data_peak_dropout_more_data/"
config.PREPROCESSING_DIR = "/Users/sebas/projects/data/preprocessing_ed_mces_20250123/"
config.PREPROCESSING_DIR_TRAIN = (
    "/Users/sebas/projects/data/preprocessing_ed_mces_20250123/"
)
data_folder = "/Users/sebas/projects/data/"
model_ms2d_file = data_folder + "ms2deepscore_model.pt"

from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore

model_ms2d = load_model(model_ms2d_file)

# In[274]:


# In[276]:


# In[277]:


config.bins_uniformise_INFERENCE = config.EDIT_DISTANCE_N_CLASSES - 1


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

molecule_pairs_test_ed = copy.deepcopy(molecule_pairs_test)
molecule_pairs_test_mces = copy.deepcopy(molecule_pairs_test)


# In[283]:
print("Loading pairs data ...")
# indexes_tani_multitasking_test = LoadMCES.merge_numpy_arrays(config.PREPROCESSING_DIR, prefix='indexes_tani_incremental_test',
#                                                             use_edit_distance=config.USE_EDIT_DISTANCE,
#                                                             use_multitask=config.USE_MULTITASK)
indexes_tani_multitasking_test = LoadMCES.merge_numpy_arrays(
    config.PREPROCESSING_DIR_TRAIN,
    prefix="ed_mces_indexes_tani_incremental_test",
    use_edit_distance=config.USE_EDIT_DISTANCE,
    use_multitask=config.USE_MULTITASK,
)

indexes_tani_multitasking_test = remove_duplicates_array(indexes_tani_multitasking_test)


### Just for subsampling
# np.random.seed(42)
# random_indexes= np.random.randint(0,indexes_tani_multitasking_test.shape[0], 1000000)
# indexes_tani_multitasking_test = indexes_tani_multitasking_test[random_indexes]


molecule_pairs_test_ed.indexes_tani = indexes_tani_multitasking_test[:, 0:3]


print(f"shape of similarity1: {molecule_pairs_test_ed.indexes_tani.shape}")

# add tanimotos

molecule_pairs_test_ed.tanimotos = indexes_tani_multitasking_test[:, 3]

print(f"shape of similarity2: {molecule_pairs_test_ed.tanimotos.shape}")
print(f"Number of pairs for test: {len(molecule_pairs_test_ed)}")

# get the mces
molecule_pairs_test_mces.indexes_tani = indexes_tani_multitasking_test[:, [0, 1, 3]]
molecule_pairs_test_mces.tanimotos = indexes_tani_multitasking_test[:, 3]


# best_model_path = model_path = data_folder + 'best_model_exhaustive_sampled_128n_20240618.ckpt'
# best_model_path = config.CHECKPOINT_DIR + f"best_model_n_steps-v9.ckpt"
# best_model_path = config.CHECKPOINT_DIR + f"last.ckpt"

if not (config.INFERENCE_USE_LAST_MODEL) and (
    os.path.exists(config.CHECKPOINT_DIR + f"best_model.ckpt")
):
    best_model_path = config.CHECKPOINT_DIR + config.BEST_MODEL_NAME
else:
    best_model_path = config.CHECKPOINT_DIR + f"last.ckpt"


# molecule_pairs_test = dataset["molecule_pairs_test"]
print(f"Number of molecule pairs: {len(molecule_pairs_test_ed)}")
print("Uniformize the data")
uniformed_molecule_pairs_test_ed, binned_molecule_pairs_ed = TrainUtils.uniformise(
    molecule_pairs_test_ed,
    number_bins=5,
    return_binned_list=True,
    bin_sim_1=True,
    # bin_sim_1=False,
    ordinal_classification=True,
)  # do not treat sim==1 as another bin


uniformed_molecule_pairs_test_mces, binned_molecule_pairs_mces = TrainUtils.uniformise(
    molecule_pairs_test_mces,
    number_bins=10,
    return_binned_list=True,
    bin_sim_1=False,
    # bin_sim_1=False,
    # ordinal_classification=True,
)  # do not treat sim==1 as another bin


## Get the test spectra
n_spectra = len(uniformed_molecule_pairs_test_ed)
# n_spectra= len(uniformed_molecule_pairs_test_mces)
# indexes_0 = uniformed_molecule_pairs_test_mces.indexes_tani[:,0]
# indexes_1 = uniformed_molecule_pairs_test_mces.indexes_tani[:,1]
indexes_0 = uniformed_molecule_pairs_test_ed.indexes_tani[:, 0]
indexes_1 = uniformed_molecule_pairs_test_ed.indexes_tani[:, 1]
spectra0 = [
    uniformed_molecule_pairs_test_mces.get_original_spectrum_from_unique_index(index, 0)
    for index in indexes_0
]
spectra1 = [
    uniformed_molecule_pairs_test_mces.get_original_spectrum_from_unique_index(index, 1)
    for index in indexes_1
]

## Convert from spectrum utils to matchms
from simba.matchms_utils import MatchmsUtils

spectra0_mms = [MatchmsUtils.from_su_to_matchms(s) for s in spectra0]
spectra1_mms = [MatchmsUtils.from_su_to_matchms(s) for s in spectra1]

from matchms import Spectrum


def create_new_spectra(spectrums):
    new_s = []
    for s in spectrums:
        new_metadata = s.metadata
        new_metadata["ionmode"] = "positive"
        new_spectrum = Spectrum(
            mz=s.mz, intensities=s.intensities, metadata=new_metadata
        )
        new_s.append(new_spectrum)
    return new_s


spectra0_mms = create_new_spectra(spectra0_mms)
spectra1_mms = create_new_spectra(spectra1_mms)


# dataset_train = LoadData.from_molecule_pairs_to_dataset(m_train)
dataset_test_ed = LoadDataMultitasking.from_molecule_pairs_to_dataset(
    uniformed_molecule_pairs_test_ed, max_num_peaks=int(config.TRANSFORMER_CONTEXT)
)
dataloader_test_ed = DataLoader(
    dataset_test_ed, batch_size=config.BATCH_SIZE, shuffle=False
)

dataset_test_mces = LoadDataMultitasking.from_molecule_pairs_to_dataset(
    uniformed_molecule_pairs_test_mces, max_num_peaks=int(config.TRANSFORMER_CONTEXT)
)
dataloader_test_mces = DataLoader(
    dataset_test_mces, batch_size=config.BATCH_SIZE, shuffle=False
)

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
# prediction of mces
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

pred_test_mces = trainer.predict(
    best_model,
    dataloader_test_mces,
)

pred_test_ed = trainer.predict(
    best_model,
    dataloader_test_ed,
)
similarities_test1_mces, similarities_test2_mces = (
    Postprocessing.get_similarities_multitasking(dataloader_test_mces)
)
similarities_test1_ed, similarities_test2_ed = (
    Postprocessing.get_similarities_multitasking(dataloader_test_ed)
)

## Now assign the coorect similarity
# similarities_test1 = similarities_test1_mces
# similarities_test2 = similarities_test2_mces
similarities_test1 = similarities_test1_ed
similarities_test2 = similarities_test2_ed


# In[ ]:
def softmax(x):
    e_x = np.exp(x)  # Subtract max(x) for numerical stability
    return e_x / e_x.sum()


def which_index(p, threshold=0.5):
    return np.argmax(p)


def which_index_confident(p, threshold=0.50):
    # only predict confident predictions
    p_softmax = softmax(p)
    highest_pred = np.argmax(p_softmax)
    if p_softmax[highest_pred] > threshold:
        return np.argmax(p)
    else:
        return np.nan


def which_index_regression(p, max_index=5):
    ## the value of 0.2 must be the center of the second item

    index = np.round(p * max_index)
    # ad hoc solution
    # index=(-(np.round(p*max_index)))

    # index=np.clip(index, 0, 5)
    return index


# print(len(pred_test))
# print(pred_test[0])
# print(pred_test[0].shape)
# print(f'Shape of pred_test: {len(pred_test)}')
# flat the results
flat_pred_test1 = []
raw_flat_pred_test1 = []


flat_pred_test2 = []

# flat_pred_test2 = [[p.item() for p in pred[1]] for pred in pred_test_mces]
flat_pred_test2 = [[p.item() for p in pred[1]] for pred in pred_test_ed]
flat_pred_test2 = [item for sublist in flat_pred_test2 for item in sublist]
flat_pred_test2 = np.array(flat_pred_test2)


# flat_pred_test1 = [p[0] for p in pred_test_mces]
flat_pred_test1 = [p[0] for p in pred_test_ed]
flat_pred_test1 = [[which_index(p) for p in p_list] for p_list in flat_pred_test1]
flat_pred_test1 = [item for sublist in flat_pred_test1 for item in sublist]
flat_pred_test1 = np.array(flat_pred_test1)

print(f"Example of edit distance prediction: {flat_pred_test1}")


# get the results
similarities_test1 = np.array(similarities_test1)
flat_pred_test1 = np.array(flat_pred_test1)

similarities_test2 = np.array(similarities_test2)
flat_pred_test2 = np.array(flat_pred_test2)


print(f"Max value of similarities 1: {max(similarities_test1)}")
print(f"Min value of similarities 1: {min(similarities_test1)}")


print("Running modified cosine and MS2 deepscore")
similarity_model_mod = ModifiedCosine(tolerance=0.1)
similarity_model_ms2 = MS2DeepScore(model_ms2d)
pred_ms2 = []
pred_mod_cos = []
pipeline = Pipeline(
    create_workflow(
        query_filters=DEFAULT_FILTERS,
        score_computations=[[MS2DeepScore, {"model": model_ms2d}]],
    )
)


def apply_default_filters(spectrum: Spectrum) -> Spectrum:
    # Each filter is applied in sequence
    return msfilters.default_filters(spectrum)


for s0, s1 in tqdm(zip(spectra0_mms, spectra1_mms)):

    # print(f"s0 mz: {s0.mz}")
    # print(f"s1 mz: {s1.mz}")
    # print(f"s0 int: {s0.intensities}")
    # print(f"s1 int: {s1.intensities}")

    results_scores_mod = calculate_scores([s0], [s1], similarity_model_mod)

    s0_preprocessed = apply_default_filters(s0)
    s1_preprocessed = apply_default_filters(s1)
    results_scores_ms2 = calculate_scores(
        [s0_preprocessed], [s1_preprocessed], similarity_model_ms2
    )

    similarity_retrieved_ms2 = results_scores_ms2.to_array().transpose()[0][0]
    similarity_retrieved_mod = results_scores_mod.to_array().transpose()[0][0]

    retrieved_sim_ms2 = np.array(similarity_retrieved_ms2)
    # retrieved_sim=np.array(similarity_retrieved[0] if similarity_retrieved[1]>=6 else 0)
    retrieved_sim_mod = np.array(similarity_retrieved_mod[0])

    pred_ms2.append(retrieved_sim_ms2)
    pred_mod_cos.append(retrieved_sim_mod)

pred_ms2 = np.array(pred_ms2)
pred_mod_cos = np.array(pred_mod_cos)


print("Running tanimotos ...")
tanimotos = [
    Tanimoto.compute_tanimoto_from_smiles(s0.params["smiles"], s1.params["smiles"])
    for s0, s1 in zip(spectra0, spectra1)
]


## Get interesting examples
def filter_good_examples(
    similarities1,
    predictions1,
    similarities2,
    predictions2,
    pred_mod_cos,
    pred_ms2,
    tanimotos,
):
    sim_np_1 = np.array(similarities1)
    pred_np_1 = np.array(predictions1)

    sim_np_2 = np.array(similarities2)
    pred_np_2 = np.array(predictions2)

    # return np.argwhere((sim_np==pred_np)&(sim_np>0.1))
    # equal_edit_distance= (sim_np_1==pred_np_1) & (sim_np_1 < 5) & (sim_np_1 > 0) ## not equal molecules
    equal_edit_distance = sim_np_1 == pred_np_1

    # equal_edit_distance= (sim_np_1==pred_np_1)
    equal_mces = np.abs(sim_np_2 - pred_np_2) < 0.2

    low_mces = (sim_np_2 > 0.8) & (sim_np_2 != 1)
    # low_mces =  (sim_np_2>0.8)
    low_mod_cos = pred_mod_cos < 0.2
    high_mod_cos = pred_mod_cos > 0.7

    high_tanimoto = tanimotos > 0.8
    low_tanimoto = tanimotos < 0.3

    low_ms2 = (np.abs(pred_ms2 - tanimotos) > 0.5) & (
        pred_ms2 < 0.5
    )  ## the prediction is low and the differrence between tanimoto and ms2 is high

    condition_1 = (
        (equal_edit_distance & equal_mces & low_mces) & low_tanimoto & low_mod_cos
    )
    condition_2 = (
        (equal_edit_distance & equal_mces & low_mces) & low_tanimoto & high_mod_cos
    )
    condition_3 = (
        (equal_edit_distance & equal_mces & low_mces) & high_tanimoto & low_mod_cos
    )
    condition_4 = (
        (equal_edit_distance & equal_mces & low_mces) & high_tanimoto & high_mod_cos
    )
    condition_5 = (equal_edit_distance & equal_mces & low_mces) & low_ms2

    good_indexes_dict = {}

    good_indexes_dict["bad_ms2"] = np.argwhere(condition_5)
    good_indexes_dict["high_tani_high_mod"] = np.argwhere(condition_4)
    good_indexes_dict["high_tani_low_mod"] = np.argwhere(condition_3)
    good_indexes_dict["low_tani_high_mod"] = np.argwhere(condition_2)

    good_indexes_dict["bad_ms2_high_tani_low_mod"] = np.argwhere(
        condition_5 & condition_3
    )

    # return np.argwhere(condition_1|condition_2 |condition_3 )
    return good_indexes_dict


print(f"Tanimotos {tanimotos}")
tanimotos = [t if t is not None else np.nan for t in tanimotos]
tanimotos = np.array(tanimotos)


good_indexes_dict = filter_good_examples(
    similarities_test1,
    flat_pred_test1,
    similarities_test2,
    flat_pred_test2,
    pred_mod_cos,
    pred_ms2,
    tanimotos,
)


for k in good_indexes_dict:
    good_indexes = good_indexes_dict[k]
    prediction_results = {}
    prediction_results["similarities_ed"] = similarities_test1
    prediction_results["similarities_mces"] = similarities_test2
    prediction_results["predictions_ed"] = flat_pred_test1
    prediction_results["predictions_mces"] = flat_pred_test2
    prediction_results["pred_mod_cos"] = pred_mod_cos
    prediction_results["pred_ms2"] = pred_ms2
    # PerformanceMetrics.plot_molecules(uniformed_molecule_pairs_test_mces, prediction_results,  good_indexes, config, prefix=k)
    PerformanceMetrics.plot_molecules(
        uniformed_molecule_pairs_test_ed,
        prediction_results,
        good_indexes,
        config,
        prefix=k,
    )
