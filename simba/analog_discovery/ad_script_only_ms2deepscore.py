#!/usr/bin/env python3
"""Analog discovery using only MS2DeepScore.

This script was adapted from the analog-discovery notebook workflow and the
provided ad_script.py, but removes SIMBA and modified-cosine scoring paths.

MS2DeepScore is forced to run on CPU before PyTorch/MS2DeepScore are imported,
which avoids CUDA kernel compatibility errors on unsupported GPUs.
"""

import os

# -----------------------------------------------------------------------------
# Force CPU before importing PyTorch/MS2DeepScore.
# This avoids: RuntimeError: CUDA error: no kernel image is available for execution
# on machines where CUDA is visible but the installed PyTorch wheel was not built
# for the GPU compute capability.
# -----------------------------------------------------------------------------
MS2DEEPSCORE_FORCE_CPU = True
if MS2DEEPSCORE_FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import copy
import pickle
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import spectrum_utils.plot as sup
from hydra import compose, initialize_config_dir
from rdkit import Chem
from tqdm.auto import tqdm

from matchms import Spectrum as MatchmsSpectrum
from matchms import calculate_scores
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model as load_ms2deepscore_model

if MS2DEEPSCORE_FORCE_CPU:
    import torch

    # Some MS2DeepScore versions choose the device internally with
    # torch.cuda.is_available(). Make it return False for this script so all
    # embeddings are computed on CPU.
    torch.cuda.is_available = lambda: False

from simba.core.data.preprocessor import Preprocessor
from simba.utils.config_utils import get_config_path
from simba.workflows.utils import load_spectra


# =============================================================================
# Parameters
# =============================================================================
SUFIX='unique_compounds'
REMOVE_REDUNDAT_SPECTRA=False
RETURN_UNIQUE_COMPOUNDS=True

written_spectra_file = "/home/spiedrahita/simba/all_spectrums_reference.pkl"
OPEN_REFERENCE_SPECTRA = False
WRITE_REFERENCE_SPECTRA = False

SCORING_METHOD = "ms2deepscore"
SCORING_METHODS = [SCORING_METHOD]

USE_METADATA = True
PUT_METADATA_TO_ZEROS_REFERENCE = False
PUT_METADATA_TO_ZEROS_QUERY = False

# MS2DeepScore parameters.
MS2DEEPSCORE_MODEL_FILE = "/data/ms2deepscore_model.pt"
MS2DEEPSCORE_MIN_PEAKS_IN_RANGE = 1
MS2DEEPSCORE_MIN_MZ = 10.0
MS2DEEPSCORE_MAX_MZ = 1000.0

TOP_K = 10
RANDOM_SEED = 42
OUTPUT_DIR = "/data/simba_files_/"
OUTPUT_PICKLE_FILE = os.path.join(OUTPUT_DIR, f"analog_discovery_ms2deepscore_results_tf_{SUFIX}.pkl")
RESUME_FROM_EXISTING_PICKLE = True

#reference_file = "/data/nist_spectra_protonized.mgf"

#reference_file= '/data/msnlib_filtered.mgf
reference_file = '/data/tfs_ms2_ref.mgf'
casmi_file = "/data/casmi_all_spectra.mgf"

print(f"Will run scoring method: {SCORING_METHOD}")
print(f"Results pickle: {OUTPUT_PICKLE_FILE}")


# =============================================================================
# Config and spectrum loading
# =============================================================================

config_path = get_config_path()
with initialize_config_dir(config_dir=str(config_path), version_base=None):
    cfg = compose(config_name="config")

if USE_METADATA:
    cfg.model.features.use_ce = True
    cfg.model.features.use_ion_mode = True
    cfg.model.features.use_adduct = True
    cfg.model.features.use_ion_method = True
    cfg.model.features.use_ion_activation = True
    cfg.model.use_ce = True
    cfg.model.use_adduct = True
    cfg.model.use_ion_method = True
    cfg.model.use_ion_activation = True
else:
    cfg.model.features.use_ce = False
    cfg.model.features.use_ion_mode = False
    cfg.model.features.use_adduct = False
    cfg.model.features.use_ion_method = False
    cfg.model.features.use_ion_activation = False
    cfg.model.features.use_only_protonized_adducts = False
    cfg.model.use_ce = False
    cfg.model.use_adduct = False
    cfg.model.use_ion_method = False
    cfg.model.use_ion_activation = False

cfg.model.features.use_only_protonized_adducts = False
cfg.preprocessing = "tfs_auto"
cfg.hardware.accelerator = "cpu"

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

all_spectrums_query = load_spectra(
    casmi_file,
    cfg,
    use_gnps_format=False,
    use_only_protonized_adducts=cfg.model.features.use_only_protonized_adducts,
)
print(f"Number of spectra loaded from query: {len(all_spectrums_query)}")


# =============================================================================
# Filtering and metadata normalization
# =============================================================================

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
    for s in all_spectrums_reference_processed
]

all_spectrums_reference = [
    s_original
    for s_original, s_processed in zip(
        all_spectrums_reference, all_spectrums_reference_processed
    )
    if len(s_processed.mz) >= 6
]

# Keep only MS2 spectra. If mslevel is missing, treat the spectrum as MS2.
all_spectrums_reference_new = []
for i, spectrum in enumerate(all_spectrums_reference):
    if "mslevel" not in spectrum.params:
        all_spectrums_reference[i].params["mslevel"] = 2
        all_spectrums_reference_new.append(all_spectrums_reference[i])
    elif str(spectrum.params["mslevel"]) == "2":
        all_spectrums_reference_new.append(all_spectrums_reference[i])
all_spectrums_reference = all_spectrums_reference_new

print(f"Number of reference spectra after filtering: {len(all_spectrums_reference)}")

if WRITE_REFERENCE_SPECTRA:
    with open(written_spectra_file, "wb") as f:
        pickle.dump({"all_spectrums_reference": all_spectrums_reference}, f)

metadata_fields = ["ce", "ion_activation", "ionization_method", "adduct", "ionmode"]

if USE_METADATA:
    if PUT_METADATA_TO_ZEROS_REFERENCE:
        for j, spectrum in enumerate(all_spectrums_reference):
            all_spectrums_reference[j].params["ionization_method"] = "ESI"
            setattr(all_spectrums_reference[j], "ionization_method", "ESI")
            all_spectrums_reference[j].params["ce"] = "30"
            setattr(all_spectrums_reference[j], "ce", "30")
            all_spectrums_reference[j].params["ion_activation"] = "HCD"
            setattr(all_spectrums_reference[j], "ion_activation", "HCD")
            all_spectrums_reference[j].params["adduct"] = "[M+H]+"
            setattr(all_spectrums_reference[j], "adduct", "[M+H]+")
            all_spectrums_reference[j].params["ionmode"] = "positive"
            setattr(all_spectrums_reference[j], "ionmode", "positive")
    else:
        for j, spectrum in enumerate(all_spectrums_reference):
            if "collision_energy" in spectrum.params:
                ce = spectrum.params["collision_energy"]
                if isinstance(ce, str) and "[" in ce:
                    ce = str(int(float(ce.strip("[]"))))
                all_spectrums_reference[j].params["ce"] = str(int(float(ce)))
                setattr(all_spectrums_reference[j], "ce", str(int(float(ce))))

            if "ion_source" in spectrum.params:
                all_spectrums_reference[j].params["ionization_method"] = spectrum.params["ion_source"]
                setattr(all_spectrums_reference[j], "ionization_method", spectrum.params["ion_source"])

            if "fragmentation_method" in spectrum.params:
                all_spectrums_reference[j].params["ion_activation"] = spectrum.params["fragmentation_method"]
                setattr(all_spectrums_reference[j], "ion_activation", spectrum.params["fragmentation_method"])

            if "adduct" in spectrum.params:
                all_spectrums_reference[j].params["adduct"] = spectrum.params["adduct"]
                setattr(all_spectrums_reference[j], "adduct", spectrum.params["adduct"])

            if "ionmode" in spectrum.params:
                all_spectrums_reference[j].params["ionmode"] = str(spectrum.params["ionmode"]).lower()
                setattr(all_spectrums_reference[j], "ionmode", str(spectrum.params["ionmode"]).lower())
else:
    for j, spectrum in enumerate(all_spectrums_reference):
        all_spectrums_reference[j].params = {
            k: v for k, v in spectrum.params.items() if k not in metadata_fields
        }

if USE_METADATA and PUT_METADATA_TO_ZEROS_QUERY:
    for j, spectrum in enumerate(all_spectrums_query):
        all_spectrums_query[j].params["ionization_method"] = "ESI"
        setattr(all_spectrums_query[j], "ionization_method", "ESI")
        all_spectrums_query[j].params["ce"] = "30"
        setattr(all_spectrums_query[j], "ce", "30")
        all_spectrums_query[j].params["ion_activation"] = "HCD"
        setattr(all_spectrums_query[j], "ion_activation", "HCD")
        all_spectrums_query[j].params["adduct"] = "[M+H]+"
        setattr(all_spectrums_query[j], "adduct", "[M+H]+")
        all_spectrums_query[j].params["ionmode"] = "positive"
        setattr(all_spectrums_query[j], "ionmode", "positive")


# =============================================================================
# Remove query molecules already present in the reference library
# =============================================================================

def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


reference_smiles = {
    canonicalize_smiles(s.params.get("smiles")) for s in all_spectrums_reference
}
reference_smiles.discard(None)

all_spectrums_query = [
    spectrum
    for spectrum in all_spectrums_query
    if canonicalize_smiles(spectrum.params.get("smiles")) not in reference_smiles
]
print(f"Number of query spectra after removing molecules in reference: {len(all_spectrums_query)}")

try:
    if len(all_spectrums_query) > 2:
        sup.spectrum(all_spectrums_query[2])
except Exception as exc:
    print(f"Skipping spectrum preview: {exc}")

if REMOVE_REDUNDAT_SPECTRA:
    print(f'Redundant spectra: Lets remove redundant spectra')
    added_spectra= []
    all_spectrums_no_redundant=[]
    for s, canon_s in zip(all_spectrums_reference,reference_smiles):
        if canon_s not in added_spectra:
            added_spectra.append(canon_s)
            all_spectrums_no_redundant.append(s)

    all_spectrums_reference = all_spectrums_no_redundant
    print(f'After removing redundant spectra we get {len(all_spectrums_reference)} spectra')


# =============================================================================
# MS2DeepScore scoring
# =============================================================================

def _get_first_existing(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _get_param_first(spectrum: Any, names: Iterable[str], default: Any = None) -> Any:
    params = getattr(spectrum, "params", {}) or {}
    metadata = getattr(spectrum, "metadata", {}) or {}
    for name in names:
        if name in params and params[name] is not None:
            return params[name]
        if name in metadata and metadata[name] is not None:
            return metadata[name]
    return default


def _extract_peaks(spectrum: Any) -> tuple[np.ndarray, np.ndarray]:
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


def _extract_precursor_mz(spectrum: Any) -> Optional[float]:
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


def to_matchms_spectrum(spectrum: Any) -> MatchmsSpectrum:
    mz, intensities = _extract_peaks(spectrum)
    metadata = dict(getattr(spectrum, "params", {}) or {})
    metadata.update(dict(getattr(spectrum, "metadata", {}) or {}))

    precursor_mz = _extract_precursor_mz(spectrum)
    if precursor_mz is not None:
        metadata["precursor_mz"] = precursor_mz

    return MatchmsSpectrum(mz=mz, intensities=intensities, metadata=metadata)


def has_ms2deepscore_compatible_peaks(spectrum: Any) -> bool:
    mz, intensities = _extract_peaks(spectrum)
    in_range = (
        (mz >= MS2DEEPSCORE_MIN_MZ)
        & (mz <= MS2DEEPSCORE_MAX_MZ)
        & (intensities > 0)
    )
    return int(np.sum(in_range)) >= MS2DEEPSCORE_MIN_PEAKS_IN_RANGE


def get_top_k_candidates(ranking: np.ndarray, reference_spectra: list, k: int = 10):
    top_indices = np.argsort(ranking, axis=1)[:, -k:][:, ::-1]
    top_scores = np.take_along_axis(ranking, top_indices, axis=1)
    top_spectra = [[reference_spectra[j] for j in row] for row in top_indices]
    return top_spectra, top_scores, top_indices

def get_top_k_candidates_unique(
    ranking,
    reference_spectra,
    k=10,
    unique_compounds=True,
):
    """
    Retrieve top-k candidates.

    If unique_compounds=True:
        1. Group all reference spectra by compound (canonical SMILES).
        2. For each query-compound pair, compute the MEDIAN of the original
           spectrum-level ranking scores.
        3. Rank compounds according to this median score (higher is better).
        4. Select the top-k compounds.
        5. For each selected compound, return the spectrum having the
           MAXIMUM original ranking score within that compound.

    Parameters
    ----------
    ranking : np.ndarray
        Shape (n_queries, n_reference_spectra).
        Original spectrum-level ranking, e.g.
        AnalogDiscovery.compute_ranking(...).

    reference_spectra : list
        Reference spectra corresponding to ranking columns.

    k : int
        Number of compounds/spectra to return.

    unique_compounds : bool
        If False, behaves as a standard spectrum-level top-k retrieval.

    Returns
    -------
    top_spectra : list[list]
        Representative spectrum for each selected compound.

    top_scores : np.ndarray
        Median compound ranking scores.
        Shape: (n_queries, k).

    top_indices : np.ndarray
        Reference-spectrum index of the highest-ranked spectrum
        belonging to each selected compound.
        Shape: (n_queries, k).
    """

    ranking = np.asarray(ranking)

    n_queries, n_references = ranking.shape

    if n_references != len(reference_spectra):
        raise ValueError(
            f"Ranking has {n_references} reference columns, "
            f"but reference_spectra has {len(reference_spectra)} spectra."
        )

    # ------------------------------------------------------------
    # Standard spectrum-level retrieval
    # ------------------------------------------------------------
    if not unique_compounds:

        k_eff = min(k, n_references)

        top_indices = np.argsort(
            ranking,
            axis=1,
        )[:, -k_eff:][:, ::-1]

        top_scores = np.take_along_axis(
            ranking,
            top_indices,
            axis=1,
        )

        top_spectra = [
            [reference_spectra[idx] for idx in row]
            for row in top_indices
        ]

        return top_spectra, top_scores, top_indices

    # ============================================================
    # Build compound -> spectrum indices mapping
    # ============================================================

    compound_to_indices = {}

    for ref_idx, spectrum in enumerate(reference_spectra):

        smiles = spectrum.params.get("smiles")

        smiles = canonicalize_smiles(smiles)

        if smiles is None:
            # Skip spectra without a valid compound identity.
            continue

        if smiles not in compound_to_indices:
            compound_to_indices[smiles] = []

        compound_to_indices[smiles].append(ref_idx)

    compound_groups = list(compound_to_indices.values())

    n_compounds = len(compound_groups)

    if n_compounds < k:
        raise ValueError(
            f"Only {n_compounds} unique compounds were found, "
            f"but k={k} was requested."
        )

    print(
        f"Ranking {n_compounds} unique compounds "
        f"from {n_references} reference spectra."
    )

    # ============================================================
    # Results
    # ============================================================

    all_top_indices = []
    all_top_scores = []
    all_top_spectra = []

    # ============================================================
    # Process every query
    # ============================================================

    for query_idx in range(n_queries):

        query_ranking = ranking[query_idx]

        compound_median_scores = np.empty(
            n_compounds,
            dtype=float,
        )

        compound_best_indices = np.empty(
            n_compounds,
            dtype=np.int64,
        )

        # --------------------------------------------------------
        # Aggregate spectra belonging to each compound
        # --------------------------------------------------------

        for compound_idx, spectrum_indices in enumerate(compound_groups):

            spectrum_indices = np.asarray(
                spectrum_indices,
                dtype=np.int64,
            )

            scores = query_ranking[spectrum_indices]

            # Compound score:
            # robust against compounds having many spectra.
            compound_median_scores[compound_idx] = np.nanmedian(
                scores
            )

            # Representative spectrum:
            # highest individually ranked spectrum for this compound.
            best_local_idx = np.nanargmax(scores)

            compound_best_indices[compound_idx] = (
                spectrum_indices[best_local_idx]
            )

        # --------------------------------------------------------
        # Rank compounds by MEDIAN score
        # Higher AnalogDiscovery ranking = better.
        # --------------------------------------------------------

        best_compound_indices = np.argsort(
            compound_median_scores
        )[-k:][::-1]

        # Compound-level scores used for retrieval
        selected_scores = compound_median_scores[
            best_compound_indices
        ]

        # Spectrum representing each selected compound
        selected_reference_indices = compound_best_indices[
            best_compound_indices
        ]

        selected_spectra = [
            reference_spectra[idx]
            for idx in selected_reference_indices
        ]

        all_top_scores.append(selected_scores)
        all_top_indices.append(selected_reference_indices)
        all_top_spectra.append(selected_spectra)

    # ============================================================
    # Convert to arrays
    # ============================================================

    top_scores = np.asarray(
        all_top_scores,
        dtype=float,
    )

    top_indices = np.asarray(
        all_top_indices,
        dtype=np.int64,
    )

    top_spectra = all_top_spectra

    return top_spectra, top_scores, top_indices


def _force_model_to_cpu(model: Any) -> Any:
    """Move a loaded MS2DeepScore/PyTorch model to CPU when possible."""
    if not MS2DEEPSCORE_FORCE_CPU:
        return model

    try:
        model.cpu()
    except Exception:
        pass

    try:
        model.to("cpu")
    except Exception:
        pass

    try:
        model.eval()
    except Exception:
        pass

    # Older/newer MS2DeepScore model wrappers can expose submodules as attrs.
    for attr_name in ("encoder", "base", "model"):
        submodule = getattr(model, attr_name, None)
        if submodule is None:
            continue
        try:
            submodule.cpu()
        except Exception:
            pass
        try:
            submodule.to("cpu")
        except Exception:
            pass

    return model


def _calculate_ms2deepscore_scores(reference_matchms, query_matchms, similarity_model):
    if MS2DEEPSCORE_FORCE_CPU:
        # Ensure no CUDA tensors are created even if another library imported
        # torch before CUDA_VISIBLE_DEVICES was set.
        import torch

        torch.cuda.is_available = lambda: False

    return calculate_scores(reference_matchms, query_matchms, similarity_model)

def score_ms2deepscore_matrix(query_spectra: list, reference_spectra: list) -> dict:
    if not os.path.exists(MS2DEEPSCORE_MODEL_FILE):
        raise FileNotFoundError(
            f"MS2DeepScore model file not found: {MS2DEEPSCORE_MODEL_FILE}. "
            "Set MS2DEEPSCORE_MODEL_FILE to your .hdf5 model path."
        )

    query_valid = np.array(
        [has_ms2deepscore_compatible_peaks(s) for s in query_spectra], dtype=bool
    )
    reference_valid = np.array(
        [has_ms2deepscore_compatible_peaks(s) for s in reference_spectra], dtype=bool
    )

    print(f"MS2DeepScore valid query spectra: {query_valid.sum()}/{len(query_valid)}")
    print(
        f"MS2DeepScore valid reference spectra: "
        f"{reference_valid.sum()}/{len(reference_valid)}"
    )

    ranking = np.zeros((len(query_spectra), len(reference_spectra)), dtype=np.float32)

    if query_valid.sum() == 0 or reference_valid.sum() == 0:
        print("No compatible spectra for MS2DeepScore. Returning an all-zero ranking matrix.")
        return {
            "ranking": ranking,
            "ms2deepscore_query_valid": query_valid,
            "ms2deepscore_reference_valid": reference_valid,
            "ms2deepscore_model_file": MS2DEEPSCORE_MODEL_FILE,
            "ms2deepscore_force_cpu": MS2DEEPSCORE_FORCE_CPU,
        }

    query_valid_indices = np.where(query_valid)[0]
    reference_valid_indices = np.where(reference_valid)[0]

    query_matchms = [
        to_matchms_spectrum(query_spectra[i])
        for i in tqdm(query_valid_indices, desc="Converting query spectra")
    ]
    reference_matchms = [
        to_matchms_spectrum(reference_spectra[j])
        for j in tqdm(reference_valid_indices, desc="Converting reference spectra")
    ]

    print(f"Loading MS2DeepScore model on CPU: {MS2DEEPSCORE_MODEL_FILE}")
    model_ms2d = load_ms2deepscore_model(MS2DEEPSCORE_MODEL_FILE)
    model_ms2d = _force_model_to_cpu(model_ms2d)
    similarity_model = MS2DeepScore(model_ms2d)

    # Keep the orientation from the comparison notebook:
    # calculate_scores(reference, query, similarity_model).to_array().transpose()
    results_scores = _calculate_ms2deepscore_scores(
        reference_matchms, query_matchms, similarity_model
    )
    valid_scores = results_scores.to_array().transpose().astype(np.float32)
    ranking[np.ix_(query_valid_indices, reference_valid_indices)] = valid_scores

    return {
        "ranking": ranking,
        "ms2deepscore_query_valid": query_valid,
        "ms2deepscore_reference_valid": reference_valid,
        "ms2deepscore_model_file": MS2DEEPSCORE_MODEL_FILE,
        "ms2deepscore_force_cpu": MS2DEEPSCORE_FORCE_CPU,
    }


def run_ms2deepscore(query_spectra: list, reference_spectra: list) -> dict:
    print("=" * 80)
    print("Running scoring method: ms2deepscore")
    print("=" * 80)

    result = score_ms2deepscore_matrix(query_spectra, reference_spectra)
    ranking = result["ranking"]
    if RETURN_UNIQUE_COMPOUNDS:
        function=get_top_k_candidates_unique
    else:
        function=get_top_k_candidates
    top_spectra, top_scores, top_indices = function(
        ranking, reference_spectra, k=TOP_K
    )

    result.update(
        {
            "scoring_method": SCORING_METHOD,
            "ranking_shape": ranking.shape,
            "top_k": TOP_K,
            "spectrums_k_retrieved": top_spectra,
            "sim_k_retrieved": top_scores,
            "arg_max_k10": top_indices,
        }
    )

    print(f"Finished ms2deepscore. ranking shape: {ranking.shape}")
    return result


def make_results(result: dict) -> dict:
    return {
        "scoring_methods": SCORING_METHODS,
        "completed_methods": SCORING_METHODS,
        "top_k": TOP_K,
        "n_query_spectra": len(all_spectrums_query),
        "n_reference_spectra": len(all_spectrums_reference),
        "output_pickle_file": OUTPUT_PICKLE_FILE,
        "results_by_method": {SCORING_METHOD: result},
        "ranking": result["ranking"],
        "spectrums_k_retrieved": result["spectrums_k_retrieved"],
        "sim_k_retrieved": result["sim_k_retrieved"],
        "arg_max_k10": result["arg_max_k10"],
    }


def save_results(results: dict, output_pickle_file: str = OUTPUT_PICKLE_FILE) -> None:
    os.makedirs(os.path.dirname(output_pickle_file) or ".", exist_ok=True)
    tmp_file = f"{output_pickle_file}.tmp"
    with open(tmp_file, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp_file, output_pickle_file)
    print(f"Saved results to: {output_pickle_file}")


def load_existing_results_if_available(output_pickle_file: str = OUTPUT_PICKLE_FILE) -> Optional[dict]:
    if not RESUME_FROM_EXISTING_PICKLE or not os.path.exists(output_pickle_file):
        return None

    with open(output_pickle_file, "rb") as f:
        previous_results = pickle.load(f)

    if SCORING_METHOD in previous_results.get("results_by_method", {}):
        print(f"Loaded existing MS2DeepScore results from: {output_pickle_file}")
        return previous_results

    return None


combined_results = load_existing_results_if_available()
if combined_results is None:
    ms2deepscore_result = run_ms2deepscore(all_spectrums_query, all_spectrums_reference)
    combined_results = make_results(ms2deepscore_result)
    save_results(combined_results)
else:
    ms2deepscore_result = combined_results["results_by_method"][SCORING_METHOD]

ranking = ms2deepscore_result["ranking"]
sim_ed = None
sim_mces = None
n_matching_peaks = None
spectrums_k_retrieved = ms2deepscore_result["spectrums_k_retrieved"]
sim_k_retrieved = ms2deepscore_result["sim_k_retrieved"]
arg_max_k10 = ms2deepscore_result["arg_max_k10"]
tanimoto_k_retrieved = sim_k_retrieved

print(f"Active method: {SCORING_METHOD}")
print("ranking shape:", ranking.shape)


# =============================================================================
# Basic score inspection plots
# =============================================================================

flat = ranking.ravel()
length = min(len(flat), 10_000)
if length > 0:
    idx = np.random.default_rng(RANDOM_SEED).choice(flat.size, size=length, replace=False)
    samples_ranking = flat[idx]
    plt.figure()
    plt.hist(samples_ranking, bins=20)
    plt.grid()
    plt.xlabel("MS2DeepScore score")
    plt.ylabel("Frequency")


# =============================================================================
# Optional MCES evaluation of top-k retrieved candidates
# =============================================================================

os.chdir("/home/spiedrahita/simba/")
import sys  # noqa: E402

sys.path.insert(0, "/home/spiedrahita/simba")
from legacy.old_scripts.simba.analog_discovery.mces import MCES  # noqa: E402


def safe_mces_sim(smiles1: Optional[str], smiles2: Optional[str], default=np.nan):
    if smiles1 is None or smiles2 is None:
        return default

    mol1 = Chem.MolFromSmiles(str(smiles1))
    mol2 = Chem.MolFromSmiles(str(smiles2))

    if mol1 is None or mol2 is None:
        print("Invalid SMILES:", smiles1, smiles2)
        return default

    try:
        return MCES.calculate_mces_sim(smiles1, smiles2)
    except Exception as exc:
        print("MCES failed:", smiles1, smiles2, exc)
        return default


mces_k_retrieved = [
    [
        safe_mces_sim(
            retrieved.params.get("smiles"),
            query.params.get("smiles"),
        )
        for retrieved in retrieved_group
    ]
    for retrieved_group, query in tqdm(
        zip(spectrums_k_retrieved, all_spectrums_query),
        total=len(all_spectrums_query),
        desc="Computing MCES for MS2DeepScore top-k",
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
finite_distances = norm_mces_distances[np.isfinite(norm_mces_distances)]
if len(finite_distances) > 0:
    print(
        f"ms2deepscore: n={len(finite_distances)}, "
        f"median={np.nanmedian(finite_distances):.4f}, "
        f"mean={np.nanmean(finite_distances):.4f}"
    )

    fig, ax = plt.subplots(figsize=(3.5, 5))
    ax.violinplot([finite_distances], positions=[1], showmeans=False, showmedians=False, showextrema=False)
    ax.boxplot([finite_distances], positions=[1], widths=0.18, showfliers=False)
    ax.set_xticks([1])
    ax.set_xticklabels([SCORING_METHOD], rotation=20, ha="right")
    ax.set_ylabel("Normalized MCES distance")
    ax.set_title(f"Top-{TOP_K} retrieval: normalized MCES distance")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    norm_mces_plot_file = os.path.join(
        OUTPUT_DIR,
        "norm_mces_distances_ms2deepscore_violin_boxplot.png",
    )
    fig.savefig(norm_mces_plot_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {norm_mces_plot_file}")
else:
    norm_mces_plot_file = None
    print("No valid normalized MCES distances were computed.")

combined_results["active_downstream_method"] = SCORING_METHOD
combined_results["norm_mces_evaluation_by_method"] = {
    SCORING_METHOD: {
        "mces_k_retrieved": mces_k_retrieved,
        "best_indexes": best_indexes,
        "best_mces_sims": best_mces_sims,
        "norm_mces_distances": norm_mces_distances,
        "n_valid": int(np.isfinite(norm_mces_distances).sum()),
    }
}
combined_results["norm_mces_distances_by_method"] = {
    SCORING_METHOD: norm_mces_distances,
}
combined_results["norm_mces_violin_boxplot_file"] = norm_mces_plot_file

save_results(combined_results)
print("Saved methods:", list(combined_results["results_by_method"].keys()))


