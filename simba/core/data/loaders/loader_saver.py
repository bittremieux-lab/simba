import math
from typing import IO

import dill

from simba.core.data.loaders.loaders import LoadData
from simba.core.data.spectrum import SpectrumExt


class LoaderSaver:
    """
    class that implements the incremental loading of data and saving
    """

    def __init__(
        self,
        nist_line_number=0,
        block_size=100,
        pickle_nist_path="../data/all_spectrums_nist.pkl",
        pickle_gnps_path="../data/all_spectrums_gnps.pkl",
        pickle_janssen_path="../data/all_spectrums_janssen.pkl",
    ):
        self.pickle_nist_path = pickle_nist_path
        self.pickle_gnps_path = pickle_gnps_path
        self.pickle_janssen_path = pickle_janssen_path
        self.nist_line_number = (
            nist_line_number  # the current line number of the msp file loaded
        )
        self.block_size = block_size  # number of spectra to be saved per block

    def get_all_spectra(
        self,
        file: str | IO,
        num_samples: int = 10,
        compute_classes: bool = False,
        use_tqdm: bool = True,
        use_nist: bool = False,
        cfg=None,
        use_janssen: bool = False,
        use_only_protonized_adducts: bool = True,
    ) -> list[SpectrumExt]:
        """
        Get all spectra from a file.
        If a pickle path is provided, it will save the loaded spectra to that path.

        Parameters
        ----------
        file : Union[str, IO]
            The file path or file object to load spectra from.
        num_samples : int, optional
            The number of samples to load, by default 10.
        compute_classes : bool, optional
            Whether to compute classes for the spectra, by default False.
        use_tqdm : bool, optional
            Whether to use tqdm for progress indication, by default True.
        use_nist : bool, optional
            Whether the file is in NIST format, by default False.
        cfg : DictConfig, optional
            Hydra configuration object, by default None.
        use_janssen : bool, optional
            Whether the file is in Janssen format, by default False.
        use_only_protonized_adducts : bool, optional
            Whether to use only protonized adducts, by default True.

        Returns
        -------
        List[SpectrumExt]
            A list of loaded spectrums.
        """

        if use_janssen:
            spectra = LoadData.get_all_spectra_mgf(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                cfg=cfg,
                use_gnps_format=not (use_janssen),
                use_only_protonized_adducts=use_only_protonized_adducts,
            )  # Janssen data does not use the GNPS format
            if self.pickle_janssen_path is not None:
                self.save_pickle(self.pickle_janssen_path, spectra)
        elif use_nist:
            spectra = self.load_and_save_nist(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                cfg=cfg,
            )
        else:
            spectra = LoadData.get_all_spectra_mgf(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                cfg=cfg,
                use_only_protonized_adducts=use_only_protonized_adducts,
            )
            if self.pickle_gnps_path is not None:
                self.save_pickle(self.pickle_gnps_path, spectra)

        return spectra

    def load_and_save_nist(
        self,
        file,
        num_samples=100,
        compute_classes=False,
        use_tqdm=True,
        cfg=None,
    ):
        # get the number of lines of the file
        with open(file) as f:
            line_count = sum(1 for line in f)

        print(f"The NIST file contains {line_count} lines")

        # number of blocks
        number_of_blocks = math.ceil(num_samples / self.block_size)
        current_line_number = self.nist_line_number
        all_spectra = []

        for _ in range(0, number_of_blocks):
            print(
                f"Starting loading spectra with block size {self.block_size} in the following spectrum index {len(all_spectra)} and line number {current_line_number}"
            )

            spectra, current_line_number = LoadData.get_all_spectra_nist(
                file=file,
                num_samples=self.block_size,  # get N spectra per time
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                cfg=cfg,
                initial_line_number=current_line_number,
            )

            print(f"length of spectra retrieved:{len(spectra)}")
            all_spectra = all_spectra + spectra

            print(
                f"Saving spectra with block size {self.block_size} in the following spectrum index {len(all_spectra)} and updated line number {current_line_number}"
            )
            self.save_pickle(self.pickle_nist_path, all_spectra)

            # break out of the loop if the line number is equal or exceeds the max number of lines
            if line_count - 1 <= current_line_number:
                print("We got to the end of the NIST file")
                break

        return all_spectra

    def save_pickle(
        self,
        file_path,
        spectra=None,
    ):
        dataset = {
            "spectrums": spectra,
        }
        with open(file_path, "wb") as file:
            dill.dump(dataset, file)
