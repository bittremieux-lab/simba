#!/usr/bin/env python3
"""Analog discovery using only SIMBA and modified cosine.

Generated from run_analog_discovery_all_algorithms_incremental.ipynb.
"""


import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import spectrum_utils.plot as sup
from rdkit import Chem
from tqdm.auto import tqdm
from hydra import compose, initialize_config_dir

import simba
from simba.analog_discovery.simba_analog_discovery import AnalogDiscovery
from simba.core.chemistry.similarity_metrics import MolecularSimilarityMetrics as GroundTruth
from simba.utils.plotting_mces import Plotting
from simba.workflows.utils import load_spectra
from simba.core.models.simba_model import Simba
from simba.utils.config_utils import get_config_path
from simba.core.data.preprocessor import Preprocessor

try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.similarity import ModifiedCosine
except ImportError:
    MatchmsSpectrum = None
    ModifiedCosine = None



# ## Analog discovery using SIMBA and modified cosine sequentially


# This notebook runs the analog-discovery workflow sequentially with two retrieval backends:
# 
# 1. **SIMBA**, using MCES/edit-distance predictions and SIMBA ranking.
# 2. **Modified cosine**, using `matchms.similarity.ModifiedCosine`.
# 
# The notebook stores the outputs for all methods in a single pickle file: `analog_discovery_simba_modified_cosine_results.pkl`.
# 
# **Important:** results are now written incrementally after each algorithm finishes. For example, once SIMBA completes, the pickle is immediately updated before modified cosine starts. This avoids losing completed results if a later algorithm fails. The write is done atomically through a temporary `.tmp` file followed by `os.replace`.
# 
# For the original downstream inspection/plotting cells, set `PLOT_SCORING_METHOD` to choose which method is loaded as the active `ranking`.
# 
# The script also computes and plots normalized MCES distances for both algorithms using violin + boxplot overlays.


# ## Libraries


# ## Defining parameters


# Define the default configuration variables


# %% Cell 10
import os
written_spectra_file = '/home/spiedrahita/simba/all_spectrums_reference.pkl'
OPEN_REFERENCE_SPECTRA = False
WRITE_REFERENCE_SPECTRA = False

# Run both retrieval backends sequentially.
SCORING_METHODS = ["simba", "modified_cosine"]
VALID_SCORING_METHODS = {"simba", "modified_cosine"}
invalid_methods = set(SCORING_METHODS) - VALID_SCORING_METHODS
if invalid_methods:
    raise ValueError(f"Invalid scoring methods: {invalid_methods}. Valid methods: {VALID_SCORING_METHODS}")

# Method used by the downstream visualization/inspection cells after all algorithms finish.
# The pickle file always contains all methods, regardless of this value.
PLOT_SCORING_METHOD = "modified_cosine"
if PLOT_SCORING_METHOD not in SCORING_METHODS:
    raise ValueError(f"PLOT_SCORING_METHOD must be one of {SCORING_METHODS}, got {PLOT_SCORING_METHOD!r}")

USE_POSITIVE_DATA=True
USE_NEGATIVE_DATA=False
USE_PROTONIZED=True
USE_METADATA = True
USE_SIMBA_ORIGINAL = False
PUT_METADATA_TO_ZEROS_REFERENCE = False
PUT_METADATA_TO_ZEROS_QUERY = False
USE_TFS_SPECTRA=True
# Modified cosine parameters.
FRAGMENT_TOLERANCE = 0.1
PRECURSOR_TOLERANCE = 1.0  # kept explicit for reproducibility; matchms ModifiedCosine uses fragment tolerance here
MIN_MATCHED_PEAKS = 1


TOP_K = 10
RANDOM_SEED = 42
OUTPUT_DIR = "/data/simba_files_/"
if USE_TFS_SPECTRA:
    OUTPUT_PICKLE_FILE = os.path.join(OUTPUT_DIR, "analog_discovery_simba_modified_cosine_results_pos_tf.pkl")
else:
    OUTPUT_PICKLE_FILE = os.path.join(OUTPUT_DIR, "analog_discovery_simba_modified_cosine_results_pos_msnlib.pkl")
# Incremental persistence options.
# If True, already-finished methods found in OUTPUT_PICKLE_FILE are reused instead of recomputed.
RESUME_FROM_EXISTING_PICKLE = False

print(f"Will run scoring methods sequentially: {SCORING_METHODS}")
print(f"Combined results pickle: {OUTPUT_PICKLE_FILE}")
print(f"Downstream plots will use: {PLOT_SCORING_METHOD}")


# %% Cell 11
# The original SIMBA checkpoint does not use the extended metadata features.
if USE_SIMBA_ORIGINAL:
    USE_METADATA = False


# %% Cell 12
# Initialize Hydra config
config_path = get_config_path()
with initialize_config_dir(config_dir=str(config_path), version_base=None):
    cfg = compose(config_name="config")


# %% Cell 13
if USE_METADATA:
    cfg.model.features.use_ce=True
    cfg.model.features.use_ion_mode=True
    cfg.model.features.use_adduct=True
    cfg.model.features.use_ion_method=True
    cfg.model.features.use_ion_activation=True
    cfg.model.use_ce=True
    cfg.model.use_adduct=True
    cfg.model.use_ion_method=True
    cfg.model.use_ion_activation=True
    
else:
    cfg.model.features.use_ce=False
    cfg.model.features.use_ion_mode=False
    cfg.model.features.use_adduct=False
    cfg.model.features.use_ion_method=False
    cfg.model.features.use_ion_activation=False
    cfg.model.features.use_only_protonized_adducts=False
    cfg.model.use_ce=False
    cfg.model.use_adduct=False
    cfg.model.use_ion_method=False
    cfg.model.use_ion_activation=False
cfg.model.features.use_only_protonized_adducts=False
cfg.preprocessing="tfs_auto"
cfg.hardware.accelerator="cpu"


# 


# %% Cell 16
cfg.model.features.use_ce


# Location of model saved, reference spectra in mgf file (MASSSPECGYM), an query spectra (CASMI)


# %% Cell 19
if USE_SIMBA_ORIGINAL:
    model_location = (
        "/data/tutorial_files/best_model_20250422_only_massspecgym.ckpt"
    )
else:
    model_location = (
        "/data/simba_files_/training_files_new_encoding/ms2_merged_ref_auto_20260629/best_model.ckpt"
        #"/data/simba_files_/training_files_new_encoding/msn_reference_fixed_split_20260629/best_model.ckpt"
    )
   

#reference_file = "/data/tutorial_files/MassSpecGym.mgf"
#reference_file = "/data/tutorial_files/ALL_GNPS_NO_PROPOGATED_wb.mgf
if USE_TFS_SPECTRA:
    reference_file = "/data/simba_files/tfs_ms2_ref.mgf" 
else:
    reference_file = "/data/simba_files/msnlib_filtered.mgf" 
#reference_file = "/data/simba_files/nist_spectra_protonized.mgf" 

casmi_file = (
    "/data/tutorial_files/casmi_all_spectra.mgf"
)


# %% Cell 20
model_location


# ## Load spectra


# Let's load the reference spectra and query spectra. This code already carries out a preprocessing of the files obtaining only protonized adducts and spectra with at least more than 6 peaks.


# %% Cell 23
cfg.model.features.use_only_protonized_adducts



# %% Cell 25
import pickle
if OPEN_REFERENCE_SPECTRA:
    with open(written_spectra_file,'rb') as f:
        data= pickle.load(f)
    all_spectrums_reference = data['all_spectrums_reference']
else:
    all_spectrums_reference = load_spectra(
        reference_file, cfg, use_gnps_format=False, use_only_protonized_adducts=cfg.model.features.use_only_protonized_adducts,
        n_samples=10000000,
    )


# %% Cell 26
print(f'Number of spectra in reference: {len(all_spectrums_reference)}')


# %% Cell 27
print(
    f"Number of spectra loaded from reference: {len(all_spectrums_reference)}"
)


# %% Cell 28
all_spectrums_query = load_spectra(
    casmi_file, cfg, use_gnps_format=False, use_only_protonized_adducts=cfg.model.features.use_only_protonized_adducts
)


# %% Cell 29
print(f"Number of spectra loaded from query: {len(all_spectrums_query)}")


# %% Cell 30
from simba.core.data.preprocessor import Preprocessor
pp=Preprocessor()


# %% Cell 31
## Extra filtering
import copy
all_spectrums_reference_processed= [copy.deepcopy(s) for s in all_spectrums_reference]
all_spectrums_reference_processed = [pp.preprocess_spectrum(
            s,
            fragment_tol_mass=10,
            fragment_tol_mode="ppm",
            min_intensity=0.01,
            max_num_peaks=1000,
            scale_intensity='root',
        ) for s in all_spectrums_reference_processed]
    
all_spectrums_reference = [s_original for s_original, s_processed in zip(all_spectrums_reference,all_spectrums_reference_processed) if len(s_processed.mz)>=6]
#only ms2

all_spectrums_reference_new=[]
for i,s in enumerate(all_spectrums_reference):
    if 'mslevel' not in s.params:
        all_spectrums_reference[i].params['mslevel']=2
        all_spectrums_reference_new.append(all_spectrums_reference[i])
    else:
        if s.params['mslevel']=='2':
            all_spectrums_reference_new.append(all_spectrums_reference[i])
all_spectrums_reference =all_spectrums_reference_new


# %% Cell 32
len(all_spectrums_reference)


# %% Cell 33
import pickle
if WRITE_REFERENCE_SPECTRA:
    data={}
    with open(written_spectra_file, 'wb') as f:
        data['all_spectrums_reference']=all_spectrums_reference
        pickle.dump(data,f)


# %% Cell 34
len(all_spectrums_reference)


# %% Cell 35
all_spectrums_query[0].params['ce']


# %% Cell 37
## Refinement of 
metadata_fields= ['ce','ion_activation','ionization_method', 'adduct','ionmode',]

if USE_METADATA:
    if PUT_METADATA_TO_ZEROS_REFERENCE:
        for j,s in enumerate(all_spectrums_reference):
            all_spectrums_reference[j].params['ionization_method']='ESI'
            setattr(all_spectrums_reference[j], 'ionization_method', 'ESI')
            all_spectrums_reference[j].params['ce']='30'
            setattr(all_spectrums_reference[j], 'ce', '30' )
            all_spectrums_reference[j].params['ion_activation']='HCD'
            setattr(all_spectrums_reference[j], 'ion_activation', 'HCD' )
            all_spectrums_reference[j].params['adduct']='[M+H]+'
            setattr(all_spectrums_reference[j], 'adduct', '[M+H]+' )
            all_spectrums_reference[j].params['ionmode']='positive'
            setattr(all_spectrums_reference[j], 'ionmode', 'positive' )
    else:

        if not(USE_TFS_SPECTRA):
          for j,s in enumerate(all_spectrums_reference):
            if '[' in (s.params['collision_energy']):
                ce =s.params['collision_energy']
                ce = str(int(float(ce.strip('[]'))))
            else:
                ce= s.params['collision_energy']
            
            all_spectrums_reference[j].params['ce']=str(int(float(ce)))
            setattr(all_spectrums_reference[j], 'ce', str(int(float(ce))))

            all_spectrums_reference[j].params['ionization_method']=s.params['ion_source']
            setattr(all_spectrums_reference[j], 'ionization_method', s.params['ion_source'])

            all_spectrums_reference[j].params['ion_activation']=s.params['fragmentation_method']
            setattr(all_spectrums_reference[j], 'ion_activation', s.params['fragmentation_method'])

            all_spectrums_reference[j].params['adduct']=s.params['adduct']
            setattr(all_spectrums_reference[j], 'adduct', s.params['adduct'])

            all_spectrums_reference[j].params['ionmode']=s.params['ionmode'].lower()
            setattr(all_spectrums_reference[j], 'ionmode', s.params['ionmode'])
        
else:
    for j,s in enumerate(all_spectrums_reference):
        new_params = {}
        for k in s.params:
            if k not in metadata_fields:
                new_params[k]=s.params[k]
        all_spectrums_reference[j].params = new_params


# %% Cell 38
if USE_METADATA:
    if PUT_METADATA_TO_ZEROS_QUERY:
       for j,s in enumerate(all_spectrums_query):
            all_spectrums_query[j].params['ionization_method']='ESI'
            setattr(all_spectrums_query[j], 'ionization_method', 'ESI')
            all_spectrums_query[j].params['ce']='30'
            setattr(all_spectrums_query[j], 'ce', '30' )
            all_spectrums_query[j].params['ion_activation']='HCD'
            setattr(all_spectrums_query[j], 'ion_activation', 'HCD' )
            all_spectrums_query[j].params['adduct']='[M+H]+'
            setattr(all_spectrums_query[j], 'adduct', '[M+H]+' )
            all_spectrums_query[j].params['ionmode']='positive'
            setattr(all_spectrums_query[j], 'ionmode', 'positive' )


# %% Cell 40
all_spectrums_reference[0].ionization_method


# ## Remove smiles present in reference
print(f'Number of ref spectra: {len(all_spectrums_reference)}')

print('Print some examples of the reference spectra')
print(f'{all_spectrums_reference[0].params}')
print(f'{all_spectrums_reference[100].params}')
print(f'Features of spectrum 0:')
print(f'ce: {all_spectrums_reference[0].ce}')
print(f'ion_activation: {all_spectrums_reference[0].ion_activation}')
print(f'adduct: {all_spectrums_reference[0].adduct}')
print(f'ionmode: {all_spectrums_reference[0].ionmode}')

# %% Cell 42


# %% Cell 43
def canonicalize_smiles(smiles):
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

# Canonicalize query smiles
#reference_smiles = [
#    canonicalize_smiles(s.params["smiles"])
#    for s in all_spectrums_reference
#]

# Filter reference spectra whose canonical SMILES are NOT in the query set
#all_spectrums_query = [
#    spectrum
#    for spectrum in all_spectrums_query
#    if canonicalize_smiles(spectrum.params["smiles"]) not in reference_smiles
#]


# %% Cell 44
print(f'number of queries: {len(all_spectrums_query)}')


# ##  Let's check some spectra visually


# %% Cell 46
sup.spectrum(all_spectrums_query[2])


# ## Initialize model


# %% Cell 48
'''simba analog-discovery --model-path /data/simba_files/training_files_new_encoding/ms2_reference_fixed_split_fixed_encoding/best_model.ckpt \
  --query-spectra  /data/tutorial_files/casmi_all_spectra.mgf \
  --reference-spectra  /data/tutorial_files/casmi_all_spectra.mgf  \
  --output-dir ./results/all_queries/ \
  analog_discovery.device=gpu \
  analog_discovery.batch_size=64 \
  analog_discovery.save_individual_plots=false \
  model.features.use_only_protonized_adducts=0 \
  model.features.use_adduct=1 \
  model.features.use_ce=1 \
  model.features.use_ion_activation=1 \
  model.features.use_ion_method=1 \
  model.features.use_ion_mode=1 '''




### FITLER BY ION TYPE
if not(USE_POSITIVE_DATA):
     all_spectrums_reference = [s for s in all_spectrums_reference if s.params['ionmode']!= 'positive' ]
     all_spectrums_query = [s for s in all_spectrums_query if s.params['ionmode']!= 'positive' ]

if not(USE_NEGATIVE_DATA):
    all_spectrums_reference = [s for s in all_spectrums_reference if s.params['ionmode']!= 'negative' ]
    all_spectrums_query = [s for s in all_spectrums_query if s.params['ionmode']!= 'negative' ]


if USE_PROTONIZED:
     all_spectrums_reference = [s for s in all_spectrums_reference if 'M+H' in s.params['adduct']]
     all_spectrums_query = [s for s in all_spectrums_query if 'M+H' in s.params['adduct']]

# Here we load a simba model based on path specified in 'model_location'. The device to be used is set to 'cpu' unless you have access to a configures GPU. The argument cache_embeddings, allows to reuse embeddings already computed to accelerate future library searchs.


# %% Cell 50
cfg.model.use_ce


# %% Cell 51
if "simba" in SCORING_METHODS:
    simba_model = Simba(
        model_location, config=cfg, device="cpu", cache_embeddings=True,
    )
else:
    simba_model = None
    print("Skipping SIMBA model initialization because 'simba' is not in SCORING_METHODS.")


# %% Cell 52
cfg.model.use_ce


# ## Predictions


# %% Cell 54
if simba_model is not None:
    print("SIMBA model use_ce:", simba_model.model.use_ce)
else:
    print("SIMBA model not used.")


# Based on the simba model created let's predict the substructure edit distance (sim_ed) and MCES distance (sim_mces)


# %% Cell 56
def _get_first_existing(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _get_param_first(spectrum, names, default=None):
    params = getattr(spectrum, "params", {}) or {}
    metadata = getattr(spectrum, "metadata", {}) or {}
    for name in names:
        if name in params and params[name] is not None:
            return params[name]
        if name in metadata and metadata[name] is not None:
            return metadata[name]
    return default


def _extract_peaks(spectrum):
    mz = _get_first_existing(spectrum, ["mz", "m/z", "mzs"])
    intensities = _get_first_existing(spectrum, ["intensity", "intensities"])

    if mz is None and hasattr(spectrum, "peaks"):
        mz = _get_first_existing(spectrum.peaks, ["mz", "mzs"])
    if intensities is None and hasattr(spectrum, "peaks"):
        intensities = _get_first_existing(spectrum.peaks, ["intensities", "intensity"])

    mz = np.asarray(mz, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    keep = np.isfinite(mz) & np.isfinite(intensities) & (intensities > 0)
    mz = mz[keep]
    intensities = intensities[keep]

    order = np.argsort(mz)
    return mz[order], intensities[order]


def _extract_precursor_mz(spectrum):
    precursor = _get_first_existing(spectrum, ["precursor_mz", "precursor"])
    if precursor is None:
        precursor = _get_param_first(
            spectrum,
            ["precursor_mz", "precursor", "pepmass", "parent_mass", "precursor_mass"],
        )

    if isinstance(precursor, (list, tuple, np.ndarray)):
        precursor = precursor[0] if len(precursor) > 0 else None

    try:
        return float(precursor)
    except Exception:
        return None


def to_matchms_spectrum(spectrum):
    if MatchmsSpectrum is None:
        raise ImportError(
            "matchms is required for modified_cosine. "
            "Install with: pip install matchms"
        )

    mz, intensities = _extract_peaks(spectrum)
    metadata = dict(getattr(spectrum, "params", {}) or {})
    metadata.update(dict(getattr(spectrum, "metadata", {}) or {}))
    precursor_mz = _extract_precursor_mz(spectrum)

    if precursor_mz is not None:
        metadata["precursor_mz"] = precursor_mz

    return MatchmsSpectrum(mz=mz, intensities=intensities, metadata=metadata)


def score_modified_cosine(query_spectrum, reference_spectrum, modified_cosine):
    score = modified_cosine.pair(query_spectrum, reference_spectrum)

    # matchms versions may return either a tuple-like score or a structured object.
    if isinstance(score, tuple):
        cosine, n_matches = score
    elif hasattr(score, "item") and getattr(score, "dtype", None) is not None:
        item = score.item()
        cosine = item[0]
        n_matches = item[1]
    else:
        try:
            cosine = score["score"]
            n_matches = score["matches"]
        except Exception:
            cosine = float(score)
            n_matches = np.nan

    if np.isnan(cosine) or (not np.isnan(n_matches) and n_matches < MIN_MATCHED_PEAKS):
        return 0.0, int(0 if np.isnan(n_matches) else n_matches)

    return float(cosine), int(0 if np.isnan(n_matches) else n_matches)



def get_top_k_candidates(ranking, reference_spectra, k=10):
    top_indices = np.argsort(ranking, axis=1)[:, -k:][:, ::-1]
    top_scores = np.take_along_axis(ranking, top_indices, axis=1)
    top_spectra = [[reference_spectra[j] for j in row] for row in top_indices]
    return top_spectra, top_scores, top_indices


def get_spectrum_smiles(spectrum, default=None):
    """Safely extract a SMILES string from a spectrum object."""
    params = getattr(spectrum, "params", {}) or {}
    metadata = getattr(spectrum, "metadata", {}) or {}

    for key in ("smiles", "SMILES", "canonical_smiles", "canonicalsmiles"):
        value = params.get(key, metadata.get(key, None))
        if value is not None:
            return str(value)

    value = getattr(spectrum, "smiles", None)
    if value is not None:
        return str(value)

    return default


def build_top_k_smiles_matches(query_spectra, top_spectra, top_scores, top_indices):
    """Create a serializable list with query SMILES and its top-k match SMILES.

    Output format:
    [
        {
            "query_index": 0,
            "query_smiles": "...",
            "matches": [
                {"rank": 1, "reference_index": 123, "match_smiles": "...", "score": 0.99},
                ...
            ],
        },
        ...
    ]
    """
    rows = []

    for query_index, (query_spectrum, retrieved_group, score_group, index_group) in enumerate(
        zip(query_spectra, top_spectra, top_scores, top_indices)
    ):
        matches = []
        for rank, (retrieved_spectrum, score, reference_index) in enumerate(
            zip(retrieved_group, score_group, index_group),
            start=1,
        ):
            matches.append({
                "rank": int(rank),
                "reference_index": int(reference_index),
                "match_smiles": get_spectrum_smiles(retrieved_spectrum),
                "score": float(score) if np.isfinite(score) else np.nan,
            })

        rows.append({
            "query_index": int(query_index),
            "query_smiles": get_spectrum_smiles(query_spectrum),
            "matches": matches,
        })

    return rows


def save_simba_top10_smiles_tsv(simba_top10_smiles_matches, output_pickle_file=OUTPUT_PICKLE_FILE):
    """Also save SIMBA top-10 query/match SMILES as a TSV next to the pickle."""
    output_tsv_file = os.path.splitext(output_pickle_file)[0] + "_simba_top10_smiles_matches.tsv"
    os.makedirs(os.path.dirname(output_tsv_file) or ".", exist_ok=True)

    with open(output_tsv_file, "w") as f:
        f.write("query_index\tquery_smiles\trank\treference_index\tmatch_smiles\tsimba_score\n")
        for row in simba_top10_smiles_matches:
            query_index = row["query_index"]
            query_smiles = row["query_smiles"]
            for match in row["matches"]:
                f.write(
                    f"{query_index}\t{query_smiles}\t{match['rank']}\t"
                    f"{match['reference_index']}\t{match['match_smiles']}\t{match['score']}\n"
                )

    return output_tsv_file


def score_simba(query_spectra, reference_spectra):
    if simba_model is None:
        raise RuntimeError("SIMBA model was not initialized. Include 'simba' in SCORING_METHODS and rerun the notebook.")

    sim_ed, sim_mces = simba_model.predict(query_spectra, reference_spectra)
    ranking = AnalogDiscovery.compute_ranking(sim_mces, sim_ed)
    return {
        "ranking": ranking,
        "sim_ed": sim_ed,
        "sim_mces": sim_mces,
    }


def score_modified_cosine_matrix(query_spectra, reference_spectra):
    if ModifiedCosine is None:
        raise ImportError("matchms is required for modified_cosine. Install with: pip install matchms")

    modified_cosine = ModifiedCosine(
        tolerance=FRAGMENT_TOLERANCE,
        mz_power=0.0,
        intensity_power=0.5,
    )

    query_matchms = [to_matchms_spectrum(s) for s in tqdm(query_spectra, desc="Converting query spectra")]
    reference_matchms = [to_matchms_spectrum(s) for s in tqdm(reference_spectra, desc="Converting reference spectra")]

    n_query = len(query_matchms)
    n_reference = len(reference_matchms)
    ranking = np.zeros((n_query, n_reference), dtype=np.float32)
    n_matching_peaks = np.zeros((n_query, n_reference), dtype=np.int16)

    for i, q in enumerate(tqdm(query_matchms, desc="Modified cosine scoring query spectra")):
        for j, r in enumerate(reference_matchms):
            ranking[i, j], n_matching_peaks[i, j] = score_modified_cosine(q, r, modified_cosine)

    return {
        "ranking": ranking,
        "n_matching_peaks": n_matching_peaks,
        "fragment_tolerance": FRAGMENT_TOLERANCE,
        "precursor_tolerance": PRECURSOR_TOLERANCE,
        "min_matched_peaks": MIN_MATCHED_PEAKS,
    }


def run_scoring_method(method, query_spectra, reference_spectra):
    print("=" * 80)
    print(f"Running scoring method: {method}")
    print("=" * 80)

    if method == "simba":
        result = score_simba(query_spectra, reference_spectra)
    elif method == "modified_cosine":
        result = score_modified_cosine_matrix(query_spectra, reference_spectra)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

    ranking = result["ranking"]
    top_spectra, top_scores, top_indices = get_top_k_candidates(ranking, reference_spectra, k=TOP_K)

    result.update({
        "scoring_method": method,
        "ranking_shape": ranking.shape,
        "top_k": TOP_K,
        "spectrums_k_retrieved": top_spectra,
        "sim_k_retrieved": top_scores,
        "arg_max_k10": top_indices,
    })

    # Save, for SIMBA only, a compact serializable structure containing
    # each query SMILES and the SMILES of its top-10 retrieved matches.
    # This is stored inside the same output pickle under:
    # combined_results["results_by_method"]["simba"]["top10_smiles_matches"]
    if method == "simba":
        simba_top10_smiles_matches = build_top_k_smiles_matches(
            query_spectra=query_spectra,
            top_spectra=top_spectra,
            top_scores=top_scores,
            top_indices=top_indices,
        )
        simba_top10_smiles_tsv_file = save_simba_top10_smiles_tsv(
            simba_top10_smiles_matches,
            output_pickle_file=OUTPUT_PICKLE_FILE,
        )
        result["top10_smiles_matches"] = simba_top10_smiles_matches
        result["top10_smiles_matches_tsv_file"] = simba_top10_smiles_tsv_file
        print(f"Saved SIMBA top-{TOP_K} SMILES matches TSV: {simba_top10_smiles_tsv_file}")

    print(f"Finished {method}. ranking shape: {ranking.shape}")
    return result


def make_combined_results(results_by_method):
    """Build the combined results dictionary from the current method results."""
    return {
        "query_spectra": all_spectrums_query,
        "scoring_methods": SCORING_METHODS,
        "completed_methods": list(results_by_method.keys()),
        "top_k": TOP_K,
        "n_query_spectra": len(all_spectrums_query),
        "n_reference_spectra": len(all_spectrums_reference),
        "output_pickle_file": OUTPUT_PICKLE_FILE,
        "results_by_method": results_by_method,
    }


def save_combined_results_incremental(results_by_method, output_pickle_file=OUTPUT_PICKLE_FILE):
    """Atomically save all method results available so far.

    This is called immediately after each algorithm finishes, so completed
    algorithms are preserved even if a later algorithm fails.
    """
    os.makedirs(os.path.dirname(output_pickle_file) or ".", exist_ok=True)
    combined_results = make_combined_results(results_by_method)
    tmp_file = f"{output_pickle_file}.tmp"

    with open(tmp_file, "wb") as f:
        pickle.dump(combined_results, f)

    os.replace(tmp_file, output_pickle_file)
    print(
        f"Saved incremental pickle after {combined_results['completed_methods'][-1]}: "
        f"{output_pickle_file}"
    )
    print("Methods currently saved:", combined_results["completed_methods"])
    return combined_results


def load_existing_results_if_available(output_pickle_file=OUTPUT_PICKLE_FILE):
    """Optionally load already completed methods to avoid recomputation."""
    if not RESUME_FROM_EXISTING_PICKLE or not os.path.exists(output_pickle_file):
        return {}

    with open(output_pickle_file, "rb") as f:
        previous_results = pickle.load(f)

    loaded_results = previous_results.get("results_by_method", {})
    loaded_results = {
        method: result
        for method, result in loaded_results.items()
        if method in SCORING_METHODS
    }

    if loaded_results:
        print(
            "Loaded previously completed methods from pickle:",
            list(loaded_results.keys()),
        )
    return loaded_results


all_method_results = load_existing_results_if_available()

for method in SCORING_METHODS:
    if method in all_method_results:
        print(f"Skipping {method}; already present in {OUTPUT_PICKLE_FILE}.")
        continue

    all_method_results[method] = run_scoring_method(
        method,
        all_spectrums_query,
        all_spectrums_reference,
    )

    # Save immediately after this algorithm completes.
    combined_results = save_combined_results_incremental(all_method_results)

# Ensure combined_results exists even if every method was loaded from an existing pickle.
combined_results = make_combined_results(all_method_results)

if all_method_results:
    # Re-save once at the end to refresh metadata such as SCORING_METHODS/TOP_K if needed.
    tmp_file = f"{OUTPUT_PICKLE_FILE}.tmp"
    with open(tmp_file, "wb") as f:
        pickle.dump(combined_results, f)
    os.replace(tmp_file, OUTPUT_PICKLE_FILE)

print(f"Final combined pickle results: {OUTPUT_PICKLE_FILE}")
print("Available methods in pickle:", list(combined_results["results_by_method"].keys()))

# Load one selected method into the original variable names for the downstream exploration cells.
SCORING_METHOD = PLOT_SCORING_METHOD
_active_result = all_method_results[SCORING_METHOD]
ranking = _active_result["ranking"]
sim_ed = _active_result.get("sim_ed")
sim_mces = _active_result.get("sim_mces")
n_matching_peaks = _active_result.get("n_matching_peaks")
spectrums_k_retrieved = _active_result["spectrums_k_retrieved"]
sim_k_retrieved = _active_result["sim_k_retrieved"]
arg_max_k10 = _active_result["arg_max_k10"]
tanimoto_k_retrieved = sim_k_retrieved

print(f"Active method for downstream cells: {SCORING_METHOD}")
print("ranking shape:", ranking.shape)


# 


# %% Cell 60
if sim_mces is not None:
    sim_mces
else:
    print(f"sim_mces is not computed for {SCORING_METHOD}; ranking contains {SCORING_METHOD} scores.")


# The predictions of substructure edit distance are discretized between 0 and 5, being 5 having five or more modifications and 0 having zero modifications. Let's take 10,000 random predictions and check the distribution of the results. Higher substructure edit distances are more common since related molecules are scarse normally.


# %% Cell 62
if sim_ed is not None:
    flat = sim_ed.ravel()
    length = min(len(flat), 10000)
    idx = np.random.default_rng(RANDOM_SEED).choice(flat.size, size=length, replace=False)
    samples_ed = flat[idx]
    plt.hist(samples_ed, bins=20)
    plt.grid()
    plt.xlabel("Predicted substructure edit distance")
    plt.ylabel("Frequency")
else:
    print(f"Skipping sim_ed histogram for {SCORING_METHOD}.")


# The predictions of MCES distance are constrained to 0 to 40 edges. Let's take 10,000 random predictions and check the distribution of the results. Higher MCES distances are more common since related molecules are scarse normally.


# %% Cell 64
if sim_mces is not None:
    flat = sim_mces.ravel()
    length = min(len(flat), 10000)
    idx = np.random.default_rng(RANDOM_SEED).choice(flat.size, size=length, replace=False)
    samples_mces = flat[idx]
    plt.hist(samples_mces, bins=20)
    plt.grid()
    plt.xlabel("Predicted MCES distance")
    plt.ylabel("Frequency")
else:
    print(f"Skipping sim_mces histogram for {SCORING_METHOD}.")


# ## Reranking


# Based on the predictions of MCES and Edit distance we can rerank the results. Lower MCES distance and lower edit distances are higher in the rank. The MCES distance is used as primary metric to rank the predictions given its finer granularity. If 2 predictions have the same MCES distance, the one with the lower substructure edit distance is ranked higher.


# %% Cell 67
# ranking is computed in the prediction/scoring cell above.
# For SIMBA: ranking comes from AnalogDiscovery.compute_ranking(sim_mces, sim_ed).
# For modified_cosine: ranking is the modified cosine score matrix directly.
ranking.shape


# The rank is scaled to 0-1 (normalized to the number of comparisons with the reference library), where 1 means the highest ranking and 0 the lowest ranking.


# %% Cell 69
ranking.shape


# %% Cell 70
flat = ranking.ravel()
length = min(len(flat), 10000)
idx = np.random.default_rng(RANDOM_SEED).choice(flat.size, size=length, replace=False)
samples_ranking = flat[idx]
plt.hist(samples_ranking, bins=20)
plt.grid()
plt.xlabel("SIMBA ranking score" if SCORING_METHOD == "simba" else SCORING_METHOD)
plt.ylabel("Frequency")


# ## What is the matched spectra in the reference library for each query spectra?


# If we want to find this answer, we have to first select the query spectra we are interested. We can define a variable 'target_index' which indicates the position of the spectrum in the spectra loaded. From there, we can select the 10 highest SIMBA scores and filtering the match with the lowest MCES distance


# %% Cell 73
target_index = 4


# %% Cell 74
spectra_query = all_spectrums_query[target_index]


# %% Cell 75
Chem.MolFromSmiles(spectra_query.params["smiles"])


# %% Cell 109
import os
os.chdir('/home/spiedrahita/simba/')


# %% Cell 110
# Top-k candidates were already computed for every method in the sequential scoring cell.
# The active method for downstream cells is selected by PLOT_SCORING_METHOD.
print(f"Using precomputed top-{TOP_K} candidates for {SCORING_METHOD}.")
print("sim_k_retrieved shape:", sim_k_retrieved.shape)


# %% Cell 111
os.chdir('/home/spiedrahita/simba/')
import sys
sys.path.insert(0, "/home/spiedrahita/simba")
from legacy.old_scripts.simba.analog_discovery.mces import MCES    
from rdkit import Chem
import numpy as np


def safe_mces_sim(smiles1, smiles2, default=np.nan):
    if smiles1 is None or smiles2 is None:
        return default

    mol1 = Chem.MolFromSmiles(str(smiles1))
    mol2 = Chem.MolFromSmiles(str(smiles2))

    if mol1 is None or mol2 is None:
        print("Invalid SMILES:", smiles1, smiles2)
        return default

    try:
        return MCES.calculate_mces_sim(smiles1, smiles2)
    except Exception as e:
        print("MCES failed:", smiles1, smiles2, e)
        return default


mces_k_retrieved = [
    [
        safe_mces_sim(
            s.params.get("smiles"),
            spec_janssen.params.get("smiles")
        )
        for s in s_group
    ]
    for s_group, spec_janssen in zip(
        spectrums_k_retrieved,
        all_spectrums_query,
    )
]


# ### Compare normalized MCES distances across both algorithms
# 
# This section evaluates the top-k retrieved candidates from each algorithm. For every query spectrum and every method, it computes the MCES similarity against the retrieved candidates, keeps the best retrieved candidate, converts it to normalized MCES distance (`1 - MCES similarity`), and plots the two algorithms together with violin + boxplot overlays.


# %% Cell 113
def compute_norm_mces_distances_for_method(
    method_name,
    method_result,
    query_spectra,
):
    """Compute best normalized MCES distance per query for one retrieval method.

    For each query, we evaluate the method's top-k retrieved spectra using MCES
    similarity, select the retrieved candidate with the highest MCES similarity,
    and convert it to a normalized distance as 1 - similarity.
    """
    top_retrieved_spectra = method_result["spectrums_k_retrieved"]

    mces_k_retrieved = [
        [
            safe_mces_sim(
                retrieved_spectrum.params.get("smiles"),
                query_spectrum.params.get("smiles"),
            )
            for retrieved_spectrum in retrieved_group
        ]
        for retrieved_group, query_spectrum in tqdm(
            zip(top_retrieved_spectra, query_spectra),
            total=len(query_spectra),
            desc=f"Computing MCES for {method_name}",
        )
    ]

    best_indexes = []
    best_mces_sims = []
    norm_mces_distances = []

    for mces_group in mces_k_retrieved:
        values = np.asarray(mces_group, dtype=float)
        valid = np.isfinite(values)

        if not np.any(valid):
            best_indexes.append(None)
            best_mces_sims.append(np.nan)
            norm_mces_distances.append(np.nan)
            continue

        valid_positions = np.where(valid)[0]
        best_position = valid_positions[np.argmax(values[valid])]
        best_sim = float(values[best_position])

        best_indexes.append(int(best_position))
        best_mces_sims.append(best_sim)
        norm_mces_distances.append(1.0 - best_sim)

    norm_mces_distances = np.asarray(norm_mces_distances, dtype=float)

    return {
        "mces_k_retrieved": mces_k_retrieved,
        "best_indexes": best_indexes,
        "best_mces_sims": best_mces_sims,
        "norm_mces_distances": norm_mces_distances,
        "n_valid": int(np.isfinite(norm_mces_distances).sum()),
    }


mces_evaluation_by_method = {}
norm_mces_distances_by_method = {}

for method_name in SCORING_METHODS:
    method_result = combined_results["results_by_method"][method_name]
    evaluation = compute_norm_mces_distances_for_method(
        method_name,
        method_result,
        all_spectrums_query,
    )
    mces_evaluation_by_method[method_name] = evaluation
    norm_mces_distances_by_method[method_name] = evaluation["norm_mces_distances"]

    finite_distances = evaluation["norm_mces_distances"][np.isfinite(evaluation["norm_mces_distances"])]
    print(
        f"{method_name}: n={len(finite_distances)}, "
        f"median={np.nanmedian(finite_distances):.4f}, "
        f"mean={np.nanmean(finite_distances):.4f}"
    )


# %% Cell 114
methods_with_data = [
    method for method in SCORING_METHODS
    if np.isfinite(norm_mces_distances_by_method[method]).sum() > 0
]
plot_data = [
    norm_mces_distances_by_method[method][np.isfinite(norm_mces_distances_by_method[method])]
    for method in methods_with_data
]

if len(plot_data) == 0:
    raise ValueError("No valid normalized MCES distances were computed for any method.")

fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(methods_with_data)), 5))

violin = ax.violinplot(
    plot_data,
    positions=np.arange(1, len(methods_with_data) + 1),
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

# Overlay compact boxplots to show median and IQR on top of each violin.
ax.boxplot(
    plot_data,
    positions=np.arange(1, len(methods_with_data) + 1),
    widths=0.18,
    showfliers=False,
)

ax.set_xticks(np.arange(1, len(methods_with_data) + 1))
ax.set_xticklabels(methods_with_data, rotation=20, ha="right")
ax.set_ylabel("Normalized MCES distance")
ax.set_title(f"Top-{TOP_K} retrieval: normalized MCES distance by algorithm")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()

NORM_MCES_VIOLIN_PLOT_FILE = os.path.join(
    OUTPUT_DIR,
    "norm_mces_distances_simba_modified_cosine_violin_boxplot.png",
)
fig.savefig(NORM_MCES_VIOLIN_PLOT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {NORM_MCES_VIOLIN_PLOT_FILE}")


# %% Cell 115
# Store the all-method MCES evaluation and plot path in the same combined pickle.
combined_results["norm_mces_evaluation_by_method"] = mces_evaluation_by_method
combined_results["norm_mces_distances_by_method"] = norm_mces_distances_by_method
combined_results["norm_mces_violin_boxplot_file"] = NORM_MCES_VIOLIN_PLOT_FILE

tmp_file = f"{OUTPUT_PICKLE_FILE}.tmp"
with open(tmp_file, "wb") as f:
    pickle.dump(combined_results, f)
os.replace(tmp_file, OUTPUT_PICKLE_FILE)

print(f"Updated combined pickle results with all-method norm MCES distances: {OUTPUT_PICKLE_FILE}")


# %% Cell 116
best_indexes = [np.argmax([m for m in mces_group]) if len(mces_group)>=0 else 0 for mces_group in mces_k_retrieved ]


# %% Cell 117
# get the best tanimotos
spectrums_retrieved = [spectrums_k_retrieved[index_spectrum][best_index]  for index_spectrum, best_index in enumerate(best_indexes) if best_index is not None]
tanimoto_retrieved = [tanimoto_k_retrieved[index_spectrum][best_index]  for index_spectrum, best_index in enumerate(best_indexes) if best_index is not None]
mces_sims = [mces_k_retrieved[index_spectrum][best_index]  for index_spectrum, best_index in enumerate(best_indexes) if best_index is not None]

max_sim = [sim_k_retrieved[index_spectrum][best_index]  for index_spectrum, best_index in enumerate(best_indexes) if best_index is not None]


# %% Cell 118
len(spectrums_retrieved)


# %% Cell 119
if SCORING_METHOD == "simba":
    ed_pred_retrieved = [
        sim_ed[index_spectrum, arg_max_k10[index_spectrum]]
        for index_spectrum, best_index in enumerate(best_indexes)
        if best_index is not None
    ]
    mces_pred_retrieved = [
        sim_mces[index_spectrum, arg_max_k10[index_spectrum]]
        for index_spectrum, best_index in enumerate(best_indexes)
        if best_index is not None
    ]
else:
    ed_pred_retrieved = None
    mces_pred_retrieved = None
    print(f"SIMBA prediction arrays are not available for {SCORING_METHOD}.")


# %% Cell 120
smiles_retrieved= [s.smiles for s in spectrums_retrieved]


# %% Cell 121
max_sim = np.nanmax(sim_k_retrieved, axis=1)
_ = plt.hist(max_sim, bins=10)
plt.xlabel(f"maximum {SCORING_METHOD} score found")
plt.ylabel("freq")
plt.grid()


# %% Cell 122
plt.boxplot(tanimoto_retrieved)
plt.grid()


# %% Cell 123
all_smiles= [s.smiles for s in all_spectrums_query]


# %% Cell 124
from tqdm import tqdm


# %% Cell 125
sim_k_retrieved.shape


# %% Cell 126
sim_k_retrieved


# %% Cell 127
mces_sims=[]
for s0,s1 in tqdm(zip(all_smiles, smiles_retrieved)):
    try:
        similarity= MCES.calculate_mces_sim(s0, s1)
        mces_sims.append(similarity)
    except:
        print(f'Error processing smiles{s0, s1}')


# %% Cell 128
mces_sims


# %% Cell 129
plt.boxplot(np.array([m for m in mces_sims if not(np.isnan(m))]))
#plt.boxplot([m for m in mces_retrieved if m is not None])
plt.grid()
#plt.ylim([0,1.1])


# %% Cell 130
import matplotlib.pyplot as plt
import numpy as np

norm_mces_sims = [1 - m for m in mces_sims if m is not None and np.isfinite(m)]

plt.figure(figsize=(2, 5))
plt.boxplot(norm_mces_sims, showfliers=False)
plt.xticks([1], [SCORING_METHOD], fontsize=8)
plt.ylim([-0.05, 0.7])
plt.ylabel("Normalized MCES distance")
plt.grid(alpha=0.3)


# %% Cell 131
all_spectrums_query[0].params


# %% Cell 132
all_spectrums_reference[0].params['ce']


# %% Cell 133
# Update and save the combined pickle again, now including downstream MCES evaluation for the active method.
# The ranking/top-k outputs for both algorithms were already saved right after the sequential run.
combined_results["active_downstream_method"] = SCORING_METHOD
combined_results["active_downstream_metrics"] = {
    "norm_mces_sims": norm_mces_sims,
}

tmp_file = f"{OUTPUT_PICKLE_FILE}.tmp"
with open(tmp_file, "wb") as f:
    pickle.dump(combined_results, f)
os.replace(tmp_file, OUTPUT_PICKLE_FILE)

print(f"Updated combined pickle results: {OUTPUT_PICKLE_FILE}")
print("Saved methods:", list(combined_results["results_by_method"].keys()))
