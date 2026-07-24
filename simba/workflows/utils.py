"""Workflow utility functions."""

import copy
import hashlib
from pathlib import Path

import dill
from rdkit import Chem

from simba.core.data.loaders import LoadData, LoaderSaver
from simba.core.data.preprocessor import Preprocessor
from simba.core.data.spectrum import SpectrumExt
from simba.utils.logger_setup import logger


def _spectra_cache_path(
    file_name: str,
    min_peaks: int,
    n_samples: int,
    use_gnps_format: bool,
    use_only_protonized_adducts: bool,
) -> Path:
    """Build a cache path next to `file_name`, keyed by loader params + source
    file identity (mtime + size), so a changed .mgf or different params can't
    silently hit a stale cache.
    """
    src = Path(file_name)
    stat = src.stat()
    key = (
        f"{src.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|"
        f"{min_peaks}|{n_samples}|{use_gnps_format}|{use_only_protonized_adducts}"
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return src.with_suffix(f".spectra_cache.{digest}.pkl")


def filter_invalid_smiles(spectra: list[SpectrumExt]) -> list[SpectrumExt]:
    """Remove spectra whose SMILES cannot be parsed by RDKit.

    Spectra with empty/missing SMILES are left untouched.
    Only non-empty SMILES that RDKit cannot parse are removed.
    """
    unique_smiles = {s.smiles or s.params.get("smiles", "") for s in spectra}
    unique_smiles.discard("")
    valid_smiles = {smi for smi in unique_smiles if Chem.MolFromSmiles(smi) is not None}
    invalid_smiles = unique_smiles - valid_smiles

    if not invalid_smiles:
        return spectra

    valid = [
        s
        for s in spectra
        if (s.smiles or s.params.get("smiles", "")) not in invalid_smiles
    ]
    logger.warning(
        f"Removed {len(spectra) - len(valid)} spectra with unparseable SMILES "
        f"({len(invalid_smiles)} distinct SMILES affected). "
        f"Examples: {list(invalid_smiles)[:3]}"
    )
    return valid


def load_spectra(
    file_name: str,
    cfg,
    min_peaks: int = 6,
    n_samples: int = 500000,
    use_gnps_format: bool = False,
    use_only_protonized_adducts: bool = True,
) -> list[SpectrumExt]:
    """Load and preprocess spectra from a file.

    Parameters
    ----------
    file_name : str
        The path to the file containing the spectra.
    cfg : DictConfig
        Hydra configuration object containing parameters.
    min_peaks : int, optional
        The minimum number of peaks a spectrum must have to be included, by default 6.
    n_samples : int, optional
        The number of samples to load, by default 500000.
    use_gnps_format : bool, optional
        Whether to use GNPS format for loading, by default False.
    use_only_protonized_adducts : bool, optional
        Whether to use only protonized adducts, by default True.

    Returns
    -------
    List[SpectrumExt]
        A list of preprocessed SpectrumExt objects.
    """
    cache_path = None
    if file_name.endswith(".mgf"):
        cache_path = _spectra_cache_path(
            file_name,
            min_peaks,
            n_samples,
            use_gnps_format,
            use_only_protonized_adducts,
        )
        if cache_path.exists():
            logger.info(f"Loading cached parsed spectra from {cache_path}...")
            try:
                with open(cache_path, "rb") as f:
                    return dill.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to load spectra cache ({type(e).__name__}: {e}); "
                    "re-parsing from source."
                )

    # load
    logger.info(f"Starting to load spectra from {file_name}...")
    if file_name.endswith(".mgf"):
        loader_saver = LoaderSaver(
            block_size=100,
            pickle_nist_path=None,
            pickle_gnps_path=None,
            pickle_janssen_path=None,
        )
        all_spectra = loader_saver.get_all_spectra(
            file_name,
            n_samples,
            use_tqdm=True,
            use_nist=False,
            cfg=cfg,
            use_janssen=not (use_gnps_format),
            use_only_protonized_adducts=use_only_protonized_adducts,
        )
    elif file_name.endswith(".pkl"):
        all_spectra = LoadData.get_all_spectra_casmi(
            file_name,
            cfg=cfg,
        )
    else:
        logger.error("Error: unrecognized file extension")
        return []

    logger.info(
        f"Loaded {len(all_spectra)} spectra from file. "
        f"Unique molecules: {len({s.params.get('smiles', 'N/A') for s in all_spectra})}"
    )

    # preprocess
    logger.info("Starting spectrum preprocessing (filtering, normalization)...")
    all_spectra_processed = [copy.deepcopy(s) for s in all_spectra]

    pp = Preprocessor()
    # remove extra peaks
    all_spectra_processed = [
        pp.preprocess_spectrum(
            s,
            fragment_tol_mass=10,
            fragment_tol_mode="ppm",
            min_intensity=0.01,
            max_num_peaks=1000,
            scale_intensity=None,
        )
        for s in all_spectra_processed
    ]

    # remove spectra that does not have at least min peaks
    filtered_spectra = [
        s_original
        for s_original, s_processed in zip(
            all_spectra, all_spectra_processed, strict=False
        )
        if len(s_processed.mz) >= min_peaks
    ]
    logger.info(f"{len(filtered_spectra)} spectra remaining after filtering.")

    # Additional logging for filtering stage
    failed_filtering = len(all_spectra) - len(filtered_spectra)
    unique_molecules_original = len(
        {s.params.get("smiles", "N/A") for s in all_spectra}
    )
    unique_molecules_filtered = len(
        {s.params.get("smiles", "N/A") for s in filtered_spectra}
    )

    logger.info(
        f"Filtering summary: {failed_filtering} spectra removed due to insufficient peaks ({min_peaks} required). "
        f"Unique molecules before: {unique_molecules_original}, after: {unique_molecules_filtered}"
    )

    # Filter out spectra with unparseable SMILES to prevent C++ aborts downstream
    filtered_spectra = filter_invalid_smiles(filtered_spectra)

    if cache_path is not None:
        try:
            with open(cache_path, "wb") as f:
                dill.dump(filtered_spectra, f)
            logger.info(f"Cached parsed spectra to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write spectra cache ({type(e).__name__}: {e})")

    return filtered_spectra
