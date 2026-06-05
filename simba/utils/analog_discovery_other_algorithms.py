#!/usr/bin/env python3
"""
Analog discovery using either modified cosine or Spec2Vec.

Set:
    SCORING_METHOD = "modified_cosine"
or:
    SCORING_METHOD = "spec2vec"
"""

import copy
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spectrum_utils.plot as sup
from rdkit import Chem
from tqdm.auto import tqdm

from hydra import compose, initialize_config_dir

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from legacy.old_scripts.simba.analog_discovery.mces import MCES
from simba.workflows.utils import load_spectra
from simba.utils.config_utils import get_config_path
from simba.core.data.preprocessor import Preprocessor
from simba.core.chemistry.similarity_metrics import MolecularSimilarityMetrics as GroundTruth

try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.similarity import ModifiedCosine
except ImportError as e:
    raise ImportError("Install matchms with: pip install matchms") from e


# =============================================================================
# Parameters
# =============================================================================

#SCORING_METHOD = "modified_cosine"
SCORING_METHOD = "spec2vec"

written_spectra_file = "/home/spiedrahita/simba/all_spectrums_reference.pkl"

OPEN_REFERENCE_SPECTRA = True
WRITE_REFERENCE_SPECTRA = False

USE_METADATA = True
PUT_METADATA_TO_ZEROS = False

reference_file = "/data/simba_files/msnlib_filtered.mgf"
casmi_file = "/data/tutorial_files/casmi_all_spectra.mgf"

TOP_K = 10
RANDOM_SEED = 42

FIGURE_DIR = Path(".")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Modified cosine parameters
FRAGMENT_TOLERANCE = 0.1
MIN_MATCHED_PEAKS = 1

# Spec2Vec parameters
SPEC2VEC_MODEL_FILE = "/data/simba_files/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
SPEC2VEC_INTENSITY_WEIGHTING_POWER = 0.5
SPEC2VEC_ALLOWED_MISSING_PERCENTAGE = 100.0
SPEC2VEC_N_DECIMALS = 2


# =============================================================================
# Helpers
# =============================================================================

def save_current_figure(filename: str, dpi: int = 300):
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close()


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


def normalize_metadata(spectra):
    metadata_fields = ["ce", "ion_activation", "ionization_method", "adduct", "ionmode"]

    if USE_METADATA:
        if PUT_METADATA_TO_ZEROS:
            for s in spectra:
                _set_param_and_attr(s, "ionization_method", "ESI")
                _set_param_and_attr(s, "ce", "30")
                _set_param_and_attr(s, "ion_activation", "HCD")
                _set_param_and_attr(s, "adduct", "[M+H]+")
                _set_param_and_attr(s, "ionmode", "positive")
        else:
            for s in spectra:
                _set_param_and_attr(s, "ce", _safe_collision_energy(s))
                _set_param_and_attr(
                    s,
                    "ionization_method",
                    s.params.get("ion_source", s.params.get("ionization_method", "")),
                )
                _set_param_and_attr(
                    s,
                    "ion_activation",
                    s.params.get("fragmentation_method", s.params.get("ion_activation", "")),
                )
                _set_param_and_attr(s, "adduct", s.params.get("adduct", ""))
                _set_param_and_attr(s, "ionmode", str(s.params.get("ionmode", "")).lower())
    else:
        for s in spectra:
            s.params = {k: v for k, v in s.params.items() if k not in metadata_fields}

    return spectra


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


def get_top_k_candidates(ranking, reference_spectra, k=10):
    top_indices = np.argsort(ranking, axis=1)[:, -k:][:, ::-1]
    top_scores = np.take_along_axis(ranking, top_indices, axis=1)
    top_spectra = [[reference_spectra[j] for j in row] for row in top_indices]
    return top_spectra, top_scores, top_indices


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


# =============================================================================
# Scoring methods
# =============================================================================

def compute_modified_cosine_ranking(query_matchms, reference_matchms):
    modified_cosine = ModifiedCosine(
        tolerance=FRAGMENT_TOLERANCE,
        mz_power=0.0,
        intensity_power=0.5,
    )

    def score_modified_cosine(query_spectrum, reference_spectrum):
        score = modified_cosine.pair(query_spectrum, reference_spectrum)

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

        if np.isnan(cosine) or (
            not np.isnan(n_matches) and n_matches < MIN_MATCHED_PEAKS
        ):
            return 0.0, 0

        return float(cosine), int(0 if np.isnan(n_matches) else n_matches)

    n_query = len(query_matchms)
    n_reference = len(reference_matchms)

    ranking = np.zeros((n_query, n_reference), dtype=np.float32)
    n_matching_peaks = np.zeros((n_query, n_reference), dtype=np.int16)

    for i, q in enumerate(tqdm(query_matchms, desc="Scoring with modified cosine")):
        for j, r in enumerate(reference_matchms):
            ranking[i, j], n_matching_peaks[i, j] = score_modified_cosine(q, r)

    return ranking, {"n_matching_peaks": n_matching_peaks}


def compute_spec2vec_ranking(query_matchms, reference_matchms):
    try:
        from gensim.models import Word2Vec
        from spec2vec import SpectrumDocument
        from spec2vec import Spec2Vec
    except ImportError as e:
        raise ImportError(
            "Install Spec2Vec dependencies with: pip install spec2vec gensim"
        ) from e

    spec2vec_model = Word2Vec.load(SPEC2VEC_MODEL_FILE)

    spec2vec_similarity = Spec2Vec(
        model=spec2vec_model,
        intensity_weighting_power=SPEC2VEC_INTENSITY_WEIGHTING_POWER,
        allowed_missing_percentage=SPEC2VEC_ALLOWED_MISSING_PERCENTAGE,
    )

    query_documents = [
        SpectrumDocument(s, n_decimals=SPEC2VEC_N_DECIMALS)
        for s in tqdm(query_matchms, desc="Creating query Spec2Vec documents")
    ]

    reference_documents = [
        SpectrumDocument(s, n_decimals=SPEC2VEC_N_DECIMALS)
        for s in tqdm(reference_matchms, desc="Creating reference Spec2Vec documents")
    ]

    def score_spec2vec(query_document, reference_document):
        score = spec2vec_similarity.pair(query_document, reference_document)

        try:
            score = float(score)
        except Exception:
            score = float(score["score"])

        if np.isnan(score):
            return 0.0

        return score

    n_query = len(query_documents)
    n_reference = len(reference_documents)

    ranking = np.zeros((n_query, n_reference), dtype=np.float32)

    for i, q in enumerate(tqdm(query_documents, desc="Scoring with Spec2Vec")):
        for j, r in enumerate(reference_documents):
            ranking[i, j] = score_spec2vec(q, r)

    return ranking, {}


# =============================================================================
# Main
# =============================================================================

def main():
    if SCORING_METHOD not in {"modified_cosine", "spec2vec"}:
        raise ValueError(
            "SCORING_METHOD must be either 'modified_cosine' or 'spec2vec'"
        )

    print(f"Using scoring method: {SCORING_METHOD}")
    print("Reference:", reference_file)
    print("Query:", casmi_file)

    config_path = get_config_path()

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="config")

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

    print(f"Reference spectra loaded: {len(all_spectrums_reference)}")

    all_spectrums_query = load_spectra(
        casmi_file,
        cfg,
        use_gnps_format=False,
        use_only_protonized_adducts=cfg.model.features.use_only_protonized_adducts,
    )

    print(f"Query spectra loaded: {len(all_spectrums_query)}")

    pp = Preprocessor()

    all_spectrums_reference_processed = [
        copy.deepcopy(s) for s in all_spectrums_reference
    ]

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
            all_spectrums_reference_processed,
            desc="Preprocessing reference spectra",
        )
    ]

    all_spectrums_reference = [
        s_original
        for s_original, s_processed in zip(
            all_spectrums_reference,
            all_spectrums_reference_processed,
        )
        if len(s_processed.mz) >= 6
    ]

    all_spectrums_reference = [
        s for s in all_spectrums_reference
        if str(s.params.get("mslevel", "2")) == "2"
    ]

    print(f"Reference spectra after filtering: {len(all_spectrums_reference)}")

    if WRITE_REFERENCE_SPECTRA:
        with open(written_spectra_file, "wb") as f:
            pickle.dump(
                {"all_spectrums_reference": all_spectrums_reference},
                f,
            )

    all_spectrums_reference = normalize_metadata(all_spectrums_reference)
    all_spectrums_query = normalize_metadata(all_spectrums_query)

    sup.spectrum(all_spectrums_query[0])
    save_current_figure("query_spectrum_example.png")

    query_matchms = [
        to_matchms_spectrum(s)
        for s in tqdm(all_spectrums_query, desc="Converting query spectra")
    ]

    reference_matchms = [
        to_matchms_spectrum(s)
        for s in tqdm(all_spectrums_reference, desc="Converting reference spectra")
    ]

    print(
        f"Converted query spectra: {len(query_matchms)}; "
        f"reference spectra: {len(reference_matchms)}"
    )

    if SCORING_METHOD == "modified_cosine":
        ranking, extra_results = compute_modified_cosine_ranking(
            query_matchms,
            reference_matchms,
        )
        score_label = "Modified cosine"
    else:
        ranking, extra_results = compute_spec2vec_ranking(
            query_matchms,
            reference_matchms,
        )
        score_label = "Spec2Vec similarity"

    print("Ranking shape:", ranking.shape)

    rng = np.random.default_rng(RANDOM_SEED)
    flat = ranking.ravel()
    length = min(len(flat), 10_000)
    idx = rng.choice(flat.size, size=length, replace=False)

    plt.hist(flat[idx], bins=30)
    plt.grid()
    plt.xlabel(score_label)
    plt.ylabel("Frequency")
    save_current_figure(f"{SCORING_METHOD}_score_distribution.png")

    spectrums_k_retrieved, scores_k_retrieved, arg_top_k = get_top_k_candidates(
        ranking,
        all_spectrums_reference,
        k=TOP_K,
    )

    target_index = 4

    spectra_query = all_spectrums_query[target_index]
    top_indices = arg_top_k[target_index]
    top_scores = scores_k_retrieved[target_index]
    spectra_matches = spectrums_k_retrieved[target_index]

    print("Top reference indices:", top_indices)
    print(f"Top {score_label} scores:", top_scores)
    print("Query SMILES:", spectra_query.params.get("smiles"))

    best_match_index = top_indices[0]
    spectra_match = spectra_matches[0]

    print(f"Best reference index: {best_match_index}")
    print(f"{score_label}: {ranking[target_index, best_match_index]:.4f}")

    if SCORING_METHOD == "modified_cosine":
        n_matching_peaks = extra_results["n_matching_peaks"]
        print(
            "Matched peaks:",
            n_matching_peaks[target_index, best_match_index],
        )

    print("Best match SMILES:", spectra_match.params.get("smiles"))

    sup.mirror(spectra_query, spectra_match)
    save_current_figure(f"{SCORING_METHOD}_best_match_mirror.png")

    ground_truth_mces = GroundTruth.compute_mces([spectra_query], [spectra_match])
    ground_truth_ed = GroundTruth.compute_edit_distance([spectra_query], [spectra_match])

    print(f"{score_label}: {ranking[target_index, best_match_index]:.4f}")
    print(f"Real MCES distance: {ground_truth_mces[0, 0]}")
    print(f"Real edit distance: {ground_truth_ed[0, 0]}")

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

    best_indexes = [
        int(np.nanargmax(mces_group))
        if np.any(np.isfinite(mces_group))
        else None
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

    best_scores = [
        scores_k_retrieved[i][best_idx]
        for i, best_idx in enumerate(best_indexes)
        if best_idx is not None
    ]

    print(f"Retrieved spectra evaluated: {len(spectrums_retrieved)}")
    print(f"Mean best top-{TOP_K} MCES similarity: {np.nanmean(mces_sims):.3f}")
    print(
        f"Mean {score_label} of structurally best top-{TOP_K} hit: "
        f"{np.nanmean(best_scores):.3f}"
    )

    plt.hist(best_scores, bins=20)
    plt.xlabel(f"{score_label} of structurally best top-{TOP_K} hit")
    plt.ylabel("Frequency")
    plt.grid()
    save_current_figure(
        f"structurally_best_hit_{SCORING_METHOD}_score_distribution.png"
    )

    plt.boxplot([m for m in mces_sims if np.isfinite(m)])
    plt.ylabel(f"Best MCES similarity within top-{TOP_K}")
    plt.grid()
    save_current_figure(f"{SCORING_METHOD}_best_mces_similarity_topk_boxplot.png")

    norm_mces_distances = [1 - m for m in mces_sims if np.isfinite(m)]

    plt.figure(figsize=(2, 5))
    plt.boxplot(norm_mces_distances)
    plt.xticks([1], [SCORING_METHOD])
    plt.ylabel("Normalized MCES distance")
    plt.grid(alpha=0.3)
    save_current_figure(f"{SCORING_METHOD}_normalized_mces_distance_boxplot.png")

    try:
        import seaborn as sns

        plt.figure(figsize=(2, 5))

        sns.violinplot(
            data=[norm_mces_distances],
            inner=None,
            scale="width",
            cut=0,
            palette=["r"],
        )

        sns.boxplot(
            data=[norm_mces_distances],
            width=0.2,
            showcaps=True,
            boxprops={"facecolor": "none"},
            showfliers=False,
            whiskerprops={"linewidth": 2},
            medianprops={"color": "black", "linewidth": 2},
        )

        plt.xticks(
            ticks=[0],
            labels=[SCORING_METHOD],
            fontsize=8,
        )

        plt.ylim([-0.05, 0.7])
        plt.ylabel("Normalized MCES distance")
        plt.grid(alpha=0.3)

        save_current_figure(
            f"{SCORING_METHOD}_normalized_mces_distance_boxplot_violin.png"
        )

    except ImportError:
        print("Seaborn not installed. Skipping violin plot.")

    output_file = str(Path(".") / f"{SCORING_METHOD}_analog_discovery_results.npz")

    save_dict = {
        "ranking": ranking,
        "arg_top_k": arg_top_k,
        "scores_k_retrieved": scores_k_retrieved,
        "mces_sims": np.asarray(mces_sims, dtype=float),
        "best_scores": np.asarray(best_scores, dtype=float),
    }

    save_dict.update(extra_results)

    np.savez_compressed(output_file, **save_dict)

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()