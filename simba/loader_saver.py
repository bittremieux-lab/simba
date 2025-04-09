from simba.load_data import LoadData
import dill

import math


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

    def get_all_spectrums(
        self,
        file,
        num_samples=10,
        compute_classes=False,
        use_tqdm=True,
        use_nist=False,
        config=None,
        use_janssen=False,
    ):

        if use_janssen:
            spectrums = LoadData.get_all_spectrums_mgf(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                config=config,
                use_gnps_format=not (use_janssen),
            )  # Janssen data does not use the GNPS format
            if self.pickle_janssen_path is not None:
                self.save_pickle(self.pickle_janssen_path, spectrums)
        elif use_nist:

            spectrums = self.load_and_save_nist(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                config=config,
            )
        else:
            spectrums = LoadData.get_all_spectrums_mgf(
                file=file,
                num_samples=num_samples,
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                config=config,
            )
            if self.pickle_gnps_path is not None:
                self.save_pickle(self.pickle_gnps_path, spectrums)

        return spectrums

    def load_and_save_nist(
        self, file, num_samples=100, compute_classes=False, use_tqdm=True, config=None
    ):

        # get the number of lines of the file
        with open(file, "r") as f:
            line_count = sum(1 for line in f)

        print(f"The NIST file contains {line_count} lines")

        # number of blocks
        number_of_blocks = math.ceil(num_samples / self.block_size)
        current_line_number = self.nist_line_number
        all_spectrums = []

        for m in range(0, number_of_blocks):
            print(
                f"Starting loading spectrums with block size {self.block_size} in the following spectrum index {len(all_spectrums)} and line number {current_line_number}"
            )

            spectrums, current_line_number = LoadData.get_all_spectrums_nist(
                file=file,
                num_samples=self.block_size,  # get N spectra per time
                compute_classes=compute_classes,
                use_tqdm=use_tqdm,
                config=config,
                initial_line_number=current_line_number,
            )

            print(f"lenght of spectrums retrievied:{len(spectrums)}")
            all_spectrums = all_spectrums + spectrums

            print(
                f"Saving spectrums with block size {self.block_size} in the following spectrum index {len(all_spectrums)} and updated line number {current_line_number}"
            )
            self.save_pickle(self.pickle_nist_path, all_spectrums)

            # break out of the loop if the line number is equal or exceeds the max number of lines
            if line_count - 1 <= current_line_number:
                print("We got to the end of the NIST file")
                break

        return all_spectrums

    # def load_pickle():
    def save_pickle(
        self,
        file_path,
        spectrums=None,
    ):
        dataset = {
            "spectrums": spectrums,
        }
        with open(file_path, "wb") as file:
            dill.dump(dataset, file)
