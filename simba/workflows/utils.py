"""Workflow utility functions."""

import copy

from simba.core.data.loaders import LoadData, LoaderSaver
from simba.core.data.preprocessor import Preprocessor
from simba.core.data.spectrum import SpectrumExt
from simba.utils.logger_setup import logger


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
        f"Unique molecules: {len(set(s.params.get('smiles', 'N/A') for s in all_spectra))}"
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
        set(s.params.get("smiles", "N/A") for s in all_spectra)
    )
    unique_molecules_filtered = len(
        set(s.params.get("smiles", "N/A") for s in filtered_spectra)
    )

    logger.info(
        f"Filtering summary: {failed_filtering} spectra removed due to insufficient peaks ({min_peaks} required). "
        f"Unique molecules before: {unique_molecules_original}, after: {unique_molecules_filtered}"
    )

    return filtered_spectra
