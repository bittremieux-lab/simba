#!/usr/bin/env python3
"""
Analog discovery using modified cosine.

Converted from run_analog_discovery_modified_cosine.ipynb.
All figures are saved in the current workspace directory: .
"""


# # Analog discovery using modified cosine
#
# This notebook is a modified version of `run_analog_discovery_msn_tf_library.ipynb`.
#
# Main change: **SIMBA model inference is removed**. Candidate ranking is computed directly with **modified cosine similarity** between each query spectrum and each reference spectrum.
#
# Higher modified cosine = better spectral match.

# ## Libraries

# %% Cell 3
import copy
import pickle

import matplotlib


matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spectrum_utils.plot as sup
from hydra import compose, initialize_config_dir
from rdkit import Chem
from tqdm.auto import tqdm


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# We still use SIMBA utilities only for loading/preprocessing spectra and evaluation helpers.
# The SIMBA neural model is NOT used in this notebook.
from legacy.old_scripts.simba.analog_discovery.mces import MCES  # noqa: E402
from simba.core.chemistry.similarity_metrics import (  # noqa: E402
    MolecularSimilarityMetrics as GroundTruth,
)
from simba.core.data.preprocessor import Preprocessor  # noqa: E402
from simba.utils.config_utils import get_config_path  # noqa: E402
from simba.workflows.utils import load_spectra  # noqa: E402


try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.similarity import ModifiedCosine
except ImportError as e:
    raise ImportError(
        "This notebook requires matchms. Install it in your environment with: pip install matchms"
    ) from e


from pathlib import Path  # noqa: E402


FIGURE_DIR = Path(".")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(filename: str, dpi: int = 300):
    """Save the active matplotlib figure in the current workspace directory."""
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close()


# ## Parameters


# %% Cell 5
written_spectra_file = "/home/spiedrahita/simba/all_spectrums_reference.pkl"

OPEN_REFERENCE_SPECTRA = True
WRITE_REFERENCE_SPECTRA = False

# Keep this for compatibility with the original loading/preprocessing code.
# It does not affect the modified cosine score directly.
USE_METADATA = True
PUT_METADATA_TO_ZEROS = False

# Modified cosine parameters
FRAGMENT_TOLERANCE = 0.1
PRECURSOR_TOLERANCE = 1.0
MIN_MATCHED_PEAKS = 1

# Retrieval parameters
TOP_K = 10
RANDOM_SEED = 42


# %% Cell 6
# Initialize Hydra config
config_path = get_config_path()
with initialize_config_dir(config_dir=str(config_path), version_base=None):
    cfg = compose(config_name="config")

# These settings are only used by load_spectra/preprocessing.
cfg.model.features.use_ce = USE_METADATA
cfg.model.features.use_ion_mode = USE_METADATA
cfg.model.features.use_adduct = USE_METADATA
cfg.model.features.use_ion_method = USE_METADATA
cfg.model.features.use_ion_activation = USE_METADATA

cfg.model.use_ce = USE_METADATA
cfg.model.use_adduct = USE_METADATA
cfg.model.use_ion_method = USE_METADATA
cfg.model.use_ion_activation = USE_METADATA

cfg.model.features.use_only_protonized_adducts = False
cfg.preprocessing = "tfs_auto"
cfg.hardware.accelerator = "gpu"

# ## Input files


# %% Cell 8
reference_file = "/data/simba_files/msnlib_filtered.mgf"

casmi_file = "/data/tutorial_files/casmi_all_spectra.mgf"

print("Reference:", reference_file)
print("Query:", casmi_file)

# ## Load spectra


# %% Cell 10
if OPEN_REFERENCE_SPECTRA:
    with open(written_spectra_file, "rb") as f:
        data = pickle.load(f)
    all_spectrums_reference = data["all_spectrums_reference"]
else:
    all_spectrums_reference = load_spectra(
        reference_file,
        cfg,
        use_gnps_format=False,
        use_only_protonized_adducts=cfg.model.features.use_only_protonized_adducts,
        n_samples=10_000_000,
    )

print(f"Number of spectra loaded from reference: {len(all_spectrums_reference)}")


# %% Cell 11
all_spectrums_query = load_spectra(
    casmi_file,
    cfg,
    use_gnps_format=False,
    use_only_protonized_adducts=cfg.model.features.use_only_protonized_adducts,
)

print(f"Number of spectra loaded from query: {len(all_spectrums_query)}")

# ## Optional preprocessing/filtering


# %% Cell 13
pp = Preprocessor()

all_spectrums_reference_processed = [copy.deepcopy(s) for s in all_spectrums_reference]
all_spectrums_reference_processed = [
    pp.preprocess_spectrum(
        s,
        fragment_tol_mass=10,
        fragment_tol_mode="ppm",
        min_intensity=0.01,
        max_num_peaks=1000,
        scale_intensity="root",
    )
    for s in tqdm(
        all_spectrums_reference_processed, desc="Preprocessing reference spectra"
    )
]

# Keep original spectra whose processed version has enough peaks.
all_spectrums_reference = [
    s_original
    for s_original, s_processed in zip(
        all_spectrums_reference, all_spectrums_reference_processed
    )
    if len(s_processed.mz) >= 6
]

# Only MS2 spectra
all_spectrums_reference = [
    s for s in all_spectrums_reference if str(s.params.get("mslevel", "2")) == "2"
]

print(f"Reference spectra after filtering: {len(all_spectrums_reference)}")


# %% Cell 14
if WRITE_REFERENCE_SPECTRA:
    data = {"all_spectrums_reference": all_spectrums_reference}
    with open(written_spectra_file, "wb") as f:
        pickle.dump(data, f)

# ## Metadata normalization from the original notebook


# %% Cell 16
metadata_fields = ["ce", "ion_activation", "ionization_method", "adduct", "ionmode"]


def _set_param_and_attr(spectrum, key, value):
    spectrum.params[key] = value
    setattr(spectrum, key, value)


def _safe_collision_energy(s):
    ce = s.params.get("collision_energy", s.params.get("ce", "30"))
    ce = str(ce)
    if "[" in ce:
        ce = ce.strip("[]")
    try:
        return str(int(float(ce)))
    except Exception:
        return "30"


if USE_METADATA:
    if PUT_METADATA_TO_ZEROS:
        for s in all_spectrums_reference:
            _set_param_and_attr(s, "ionization_method", "ESI")
            _set_param_and_attr(s, "ce", "30")
            _set_param_and_attr(s, "ion_activation", "HCD")
            _set_param_and_attr(s, "adduct", "[M+H]+")
            _set_param_and_attr(s, "ionmode", "positive")
    else:
        for s in all_spectrums_reference:
            _set_param_and_attr(s, "ce", _safe_collision_energy(s))
            _set_param_and_attr(
                s,
                "ionization_method",
                s.params.get("ion_source", s.params.get("ionization_method", "")),
            )
            _set_param_and_attr(
                s,
                "ion_activation",
                s.params.get(
                    "fragmentation_method", s.params.get("ion_activation", "")
                ),
            )
            _set_param_and_attr(s, "adduct", s.params.get("adduct", ""))
            _set_param_and_attr(s, "ionmode", str(s.params.get("ionmode", "")).lower())
else:
    for s in all_spectrums_reference:
        s.params = {k: v for k, v in s.params.items() if k not in metadata_fields}


# %% Cell 17
if USE_METADATA and PUT_METADATA_TO_ZEROS:
    for s in all_spectrums_query:
        _set_param_and_attr(s, "ionization_method", "ESI")
        _set_param_and_attr(s, "ce", "30")
        _set_param_and_attr(s, "ion_activation", "HCD")
        _set_param_and_attr(s, "adduct", "[M+H]+")
        _set_param_and_attr(s, "ionmode", "positive")

# ## Visual check


# %% Cell 19
sup.spectrum(all_spectrums_query[0])

save_current_figure("query_spectrum_example.png")

# ## Convert spectra to matchms


# %% Cell 21
def _get_first_existing(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _get_param_first(spectrum, names, default=None):
    for name in names:
        if name in spectrum.params and spectrum.params[name] is not None:
            return spectrum.params[name]
    return default


def _extract_peaks(spectrum):
    mz = _get_first_existing(spectrum, ["mz", "m/z", "mzs"])
    intensities = _get_first_existing(spectrum, ["intensity", "intensities"])

    # spectrum_utils sometimes stores peaks as spectrum.peaks.mz / spectrum.peaks.intensities
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

    # pepmass can be a tuple/list such as (mz, intensity)
    if isinstance(precursor, (list, tuple, np.ndarray)):
        precursor = precursor[0] if len(precursor) > 0 else None

    try:
        return float(precursor)
    except Exception:
        return None


def to_matchms_spectrum(spectrum):
    mz, intensities = _extract_peaks(spectrum)
    metadata = dict(getattr(spectrum, "params", {}))
    precursor_mz = _extract_precursor_mz(spectrum)

    if precursor_mz is not None:
        metadata["precursor_mz"] = precursor_mz

    return MatchmsSpectrum(
        mz=mz,
        intensities=intensities,
        metadata=metadata,
    )


query_matchms = [
    to_matchms_spectrum(s)
    for s in tqdm(all_spectrums_query, desc="Converting query spectra")
]
reference_matchms = [
    to_matchms_spectrum(s)
    for s in tqdm(all_spectrums_reference, desc="Converting reference spectra")
]

print(
    f"Converted query spectra: {len(query_matchms)}; reference spectra: {len(reference_matchms)}"
)

# ## Compute modified cosine similarity matrix


# %% Cell 23
modified_cosine = ModifiedCosine(
    tolerance=FRAGMENT_TOLERANCE,
    mz_power=0.0,
    intensity_power=0.5,
)


def score_modified_cosine(query_spectrum, reference_spectrum):
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


n_query = len(query_matchms)
n_reference = len(reference_matchms)

ranking = np.zeros((n_query, n_reference), dtype=np.float32)
n_matching_peaks = np.zeros((n_query, n_reference), dtype=np.int16)

for i, q in enumerate(tqdm(query_matchms, desc="Scoring query spectra")):
    for j, r in enumerate(reference_matchms):
        ranking[i, j], n_matching_peaks[i, j] = score_modified_cosine(q, r)

# The `ranking` matrix replaces SIMBA's ranking. Here, **larger values mean better matches**.


# %% Cell 25
rng = np.random.default_rng(RANDOM_SEED)
flat = ranking.ravel()
length = min(len(flat), 10_000)
idx = rng.choice(flat.size, size=length, replace=False)

plt.hist(flat[idx], bins=30)
plt.grid()
plt.xlabel("Modified cosine")
plt.ylabel("Frequency")

save_current_figure("modified_cosine_score_distribution.png")

# ## Retrieve top-k matches for each query


# %% Cell 27
def get_top_k_candidates(ranking, reference_spectra, k=10):
    top_indices = np.argsort(ranking, axis=1)[:, -k:][:, ::-1]
    top_scores = np.take_along_axis(ranking, top_indices, axis=1)
    top_spectra = [[reference_spectra[j] for j in row] for row in top_indices]
    return top_spectra, top_scores, top_indices


spectrums_k_retrieved, modified_cosine_k_retrieved, arg_top_k = get_top_k_candidates(
    ranking,
    all_spectrums_reference,
    k=TOP_K,
)

modified_cosine_k_retrieved[:3]

# ## Inspect one query


# %% Cell 29
target_index = 4

spectra_query = all_spectrums_query[target_index]
top_indices = arg_top_k[target_index]
top_scores = modified_cosine_k_retrieved[target_index]
spectra_matches = spectrums_k_retrieved[target_index]

print("Top reference indices:", top_indices)
print("Top modified cosine scores:", top_scores)

print("Query SMILES:", spectra_query.params.get("smiles"))


# %% Cell 30
# Best spectral match according to modified cosine
best_match_index = top_indices[0]
spectra_match = spectra_matches[0]

print(f"Best reference index: {best_match_index}")
print(f"Modified cosine: {ranking[target_index, best_match_index]:.4f}")
print(f"Matched peaks: {n_matching_peaks[target_index, best_match_index]}")

print("Best match SMILES:", spectra_match.params.get("smiles"))


# %% Cell 31
sup.mirror(spectra_query, spectra_match)

save_current_figure("modified_cosine_best_match_mirror.png")

# ## Ground-truth structural comparison for the selected match


# %% Cell 33
ground_truth_mces = GroundTruth.compute_mces([spectra_query], [spectra_match])
ground_truth_ed = GroundTruth.compute_edit_distance([spectra_query], [spectra_match])

print(f"Modified cosine: {ranking[target_index, best_match_index]:.4f}")
print(f"Real MCES distance: {ground_truth_mces[0, 0]}")
print(f"Real edit distance: {ground_truth_ed[0, 0]}")

# ## Retrieval quality: choose the structurally best molecule within the top-k spectral hits


# %% Cell 35
# from legacy.old_scripts.simba.analog_discovery.mces import MCES


def safe_mces_sim(smiles1, smiles2, default=np.nan):
    if smiles1 is None or smiles2 is None:
        return default

    mol1 = Chem.MolFromSmiles(str(smiles1))
    mol2 = Chem.MolFromSmiles(str(smiles2))

    if mol1 is None or mol2 is None:
        return default

    try:
        return MCES.calculate_mces_sim(smiles1, smiles2)
    except Exception:
        return default


mces_k_retrieved = [
    [
        safe_mces_sim(
            s.params.get("smiles"),
            query_spectrum.params.get("smiles"),
        )
        for s in retrieved_group
    ]
    for retrieved_group, query_spectrum in tqdm(
        zip(spectrums_k_retrieved, all_spectrums_query),
        total=len(all_spectrums_query),
        desc="Computing MCES similarity for top-k hits",
    )
]


# %% Cell 36
best_indexes = [
    int(np.nanargmax(mces_group)) if np.any(np.isfinite(mces_group)) else None
    for mces_group in mces_k_retrieved
]

spectrums_retrieved = [
    spectrums_k_retrieved[i][best_idx]
    for i, best_idx in enumerate(best_indexes)
    if best_idx is not None
]

mces_sims = [
    mces_k_retrieved[i][best_idx]
    for i, best_idx in enumerate(best_indexes)
    if best_idx is not None
]

best_modified_cosine_scores = [
    modified_cosine_k_retrieved[i][best_idx]
    for i, best_idx in enumerate(best_indexes)
    if best_idx is not None
]

print(f"Retrieved spectra evaluated: {len(spectrums_retrieved)}")
print(f"Mean best top-{TOP_K} MCES similarity: {np.nanmean(mces_sims):.3f}")
print(
    f"Mean modified cosine of structurally best top-{TOP_K} hit: {np.nanmean(best_modified_cosine_scores):.3f}"
)


# %% Cell 37
plt.hist(best_modified_cosine_scores, bins=20)
plt.xlabel(f"Modified cosine of structurally best top-{TOP_K} hit")
plt.ylabel("Frequency")
plt.grid()

save_current_figure("structurally_best_hit_modified_cosine_distribution.png")


# %% Cell 38
plt.boxplot([m for m in mces_sims if np.isfinite(m)])
plt.ylabel(f"Best MCES similarity within top-{TOP_K}")
plt.grid()

save_current_figure("best_mces_similarity_topk_boxplot.png")


# %% Cell 39
norm_mces_distances = [1 - m for m in mces_sims if np.isfinite(m)]

plt.figure(figsize=(2, 5))
plt.boxplot(norm_mces_distances)
plt.xticks([1], ["Modified cosine"])
plt.ylabel("Normalized MCES distance")
plt.grid(alpha=0.3)

save_current_figure("normalized_mces_distance_boxplot.png")


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


norm_mces_sims = [1 - m for m in mces_sims if m is not None]

# Choose ONE palette and reuse it everywhere in the paper
colors = sns.color_palette("deep", n_colors=6)

plt.figure(figsize=(2, 5))

sns.violinplot(data=[norm_mces_sims], inner=None, scale="width", cut=0, palette=["r"])

sns.boxplot(
    data=[norm_mces_sims],
    width=0.2,
    showcaps=True,
    boxprops={"facecolor": "none"},
    showfliers=False,
    whiskerprops={"linewidth": 2},
    medianprops={"color": "black", "linewidth": 2},
)
stacked_labels = [
    "SIMBA with metadata",
]
plt.xticks(ticks=range(len([norm_mces_sims])), labels=stacked_labels, fontsize=8)

plt.ylim([-0.05, 0.7])
plt.ylabel("Normalized MCES distance")
plt.grid(alpha=0.3)
# ## Save results

save_current_figure("normalized_mces_distance_boxplot_violin.png")

# %% Cell 41
output_file = str(Path(".") / "modified_cosine_analog_discovery_results.npz")

np.savez_compressed(
    output_file,
    ranking=ranking,
    n_matching_peaks=n_matching_peaks,
    arg_top_k=arg_top_k,
    modified_cosine_k_retrieved=modified_cosine_k_retrieved,
)

print(f"Saved: {output_file}")
