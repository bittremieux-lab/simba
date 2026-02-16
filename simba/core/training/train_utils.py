import random
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from simba.core.chemistry.tanimoto import Tanimoto
from simba.core.data.molecular_pairs import MolecularPairsSet
from simba.core.data.molecule_pairs_opt import MoleculePairsOpt
from simba.core.data.preprocessing import PreprocessingUtils
from simba.core.data.spectrum import SpectrumExt
from simba.core.models.ordinal.ordinal_classification import (
    OrdinalClassification,
)
from simba.utils.logger_setup import logger


class TrainUtils:
    @staticmethod
    def compute_unique_combinations(molecule_pairs, high_sim=1):
        lenght_total = len(molecule_pairs.spectra)
        indexes_np = np.zeros((lenght_total, 3))
        print(f"number of pairs: {lenght_total}")
        for index, _ in enumerate(molecule_pairs.spectra):
            indexes_np[index, 0] = index
            indexes_np[index, 1] = index
            indexes_np[index, 2] = high_sim

        new_indexes_np = np.concatenate(
            (molecule_pairs.pair_distances, indexes_np), axis=0
        )

        new_indexes_np = np.unique(new_indexes_np, axis=0)
        # add info to
        new_molecule_pairs = MoleculePairsOpt(
            unique_spectra=molecule_pairs.spectra,
            pair_distances=new_indexes_np,
            original_spectra=molecule_pairs.original_spectra,
            df_smiles=molecule_pairs.df_smiles,
        )

        return new_molecule_pairs

    @staticmethod
    def train_val_test_split_bms(
        spectra: list[SpectrumExt],
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[SpectrumExt], list[SpectrumExt], list[SpectrumExt]]:
        """
        Split data into train, validation, and test sets based on Murcko scaffolds
        ensuring that scaffolds do not overlap between sets.

        Parameters
        ----------
        spectra: List[SpectrumExt]
            List of SpectrumExt objects to be split.
        val_split: float
            Proportion of data to be used for validation.
        test_split: float
            Proportion of data to be used for testing.
        seed: int
            Random seed for reproducibility.

        Returns
        -------
        Tuple[List[SpectrumExt], List[SpectrumExt], List[SpectrumExt]]
            Three lists containing the training, validation, and test spectra.
        """
        # Use a consistent RNG for reproducibility across parallel processes
        rng = np.random.RandomState(seed)

        # get the percentage of training data
        train_split = 1 - val_split - test_split
        # get the murcko scaffold
        bms = [s.murcko_scaffold for s in spectra]

        # count the unique elements
        unique_values, counts = np.unique(bms, return_counts=True)
        idx_no_bms = np.where(unique_values == "")
        if len(idx_no_bms[0]) > 0:
            logger.info(
                f"{counts[np.where(unique_values == '')][0]}/{len(bms)} spectra without bms"
            )

        # remove the appearence of not identified bms
        unique_values = unique_values[unique_values != ""]

        # randomize using numpy's RNG for deterministic shuffling
        rng.shuffle(unique_values)

        # get indexes
        train_index = int((train_split) * (len(unique_values)))
        val_index = train_index + int(val_split * (len(unique_values)))

        # get elements
        train_bms = unique_values[0:train_index]
        val_bms = unique_values[train_index:val_index]
        test_bms = unique_values[val_index:]

        # get data
        spectrums_train = [s for s in spectra if s.murcko_scaffold in train_bms]
        spectrums_val = [s for s in spectra if s.murcko_scaffold in val_bms]
        spectrums_test = [s for s in spectra if s.murcko_scaffold in test_bms]
        return spectrums_train, spectrums_val, spectrums_test

    @staticmethod
    def get_combination_indexes(num_samples, combination_length=2):
        # Define the number of elements in each combination (e.g., 2 for pairs of indexes)
        return list(combinations(range(num_samples), combination_length))

    @staticmethod
    def generate_random_combinations(num_samples, num_combinations):
        all_indices = list(range(num_samples))

        for _ in range(num_combinations):
            random_indices = random.sample(
                all_indices, 2
            )  # Generate random combination of 2 indices
            yield random_indices

    @staticmethod
    def compute_all_fingerprints(all_spectrums):
        fingerprints = []

        # mols = [Chem.MolFromSmiles(s.params['smiles']) if (s.params['smiles'] != '' and s.params['smiles'] != 'N/A') else None
        #        for s in all_spectrums ]
        # fingerprints = [Chem.RDKFingerprint(m) if m is not None else None for m in mols ]

        for i in range(0, len(all_spectrums)):
            fp = Tanimoto.compute_fingerprint(all_spectrums[i].params["smiles"])
            fingerprints.append(fp)
        return fingerprints

    @staticmethod
    def precompute_min_max_indexes(
        all_spectrums, min_mass_diff, max_mass_diff, use_tqdm
    ):
        """
        precompute the min and max indexes for molecule pair computation
        """

        print("Precomputing min and max index")
        df = pd.DataFrame()

        # get mz
        total_mz = np.array([s.precursor_mz for s in all_spectrums])
        df["index"] = [i for i, s in enumerate(all_spectrums)]
        for i, _ in tqdm(enumerate(all_spectrums)):
            # compute max and min
            diff_total_max = total_mz - (all_spectrums[i].precursor_mz + max_mass_diff)
            diff_total_min = total_mz - (all_spectrums[i].precursor_mz + min_mass_diff)
            min_mz_index = np.where(diff_total_min > 0)[0]
            max_mz_index = np.where(diff_total_max > 0)[0]  # get list

            min_mz_index = min_mz_index[0] if len(min_mz_index) > 0 else 0
            max_mz_index = (
                max_mz_index[0] if len(max_mz_index) > 0 else len(all_spectrums) - 1
            )
            df.loc[i, "min_index"] = min_mz_index
            df.loc[i, "max_index"] = max_mz_index
            # print(f'min_index: {min_mz_index},max_index:{max_mz_index}')
        return df

    @staticmethod
    def get_unique_spectra(all_spectra):
        """
        table witht he information of indexes per unique smiles

        Parameters
        ----------
        all_spectra : List[SpectrumExt]
            List of SpectrumExt objects.

        Returns
        -------
        Tuple[List[SpectrumExt], pd.DataFrame]
            A tuple containing a list of unique SpectrumExt objects and a DataFrame with smiles metadata.
        """
        # convert to canonical smiles
        canon_smiles = [Chem.CanonSmiles(s.smiles) for s in all_spectra]

        # get all metadata associated with the spectra
        all_mz = [s.precursor_mz for s in all_spectra]
        all_charge = [s.precursor_charge for s in all_spectra]
        all_library = [s.library for s in all_spectra]
        all_inchi = [s.inchi for s in all_spectra]
        all_bms = [s.murcko_scaffold for s in all_spectra]
        all_superclass = [s.superclass for s in all_spectra]
        all_classe = [s.classe for s in all_spectra]
        all_subclass = [s.subclass for s in all_spectra]

        unique_smiles = np.unique(canon_smiles)
        # map unique smiles to spectrum indexes
        smiles_to_spectra_map = {
            s: [i for i, c in enumerate(canon_smiles) if c == s] for s in unique_smiles
        }

        df_smiles = pd.DataFrame()
        df_smiles["canon_smiles"] = list(unique_smiles)
        df_smiles["indexes"] = [smiles_to_spectra_map[k] for k in unique_smiles]
        df_smiles["number_indexes"] = [  # TODO: rename to num_spectra
            len(smiles_to_spectra_map[k]) for k in unique_smiles
        ]

        indexes_original = [
            canon_smiles.index(u_s) for u_s in unique_smiles
        ]  # first index of each unique smiles

        df_smiles["mz"] = [all_mz[u_s] for u_s in indexes_original]
        df_smiles["charge"] = [all_charge[u_s] for u_s in indexes_original]
        df_smiles["library"] = [all_library[u_s] for u_s in indexes_original]
        df_smiles["inchi"] = [all_inchi[u_s] for u_s in indexes_original]
        df_smiles["bms"] = [all_bms[u_s] for u_s in indexes_original]
        df_smiles["superclass"] = [all_superclass[u_s] for u_s in indexes_original]
        df_smiles["classe"] = [all_classe[u_s] for u_s in indexes_original]
        df_smiles["subclass"] = [all_subclass[u_s] for u_s in indexes_original]

        # create dummy spectra for the unique smiles
        spectra_unique = TrainUtils.create_dummy_spectra(df_smiles)
        # order spectra by charge and precursor mz
        spectra_unique_ordered = PreprocessingUtils.order_spectra_by_mz(spectra_unique)
        # reindex df_smiles
        canon_smiles_not_ordered = [s.smiles for s in spectra_unique]
        canon_smiles_ordered = [s.smiles for s in spectra_unique_ordered]

        new_indexes = [canon_smiles_ordered.index(s) for s in canon_smiles_not_ordered]
        df_smiles.set_index(pd.Index(new_indexes), inplace=True)
        df_smiles = df_smiles.sort_index()
        return spectra_unique_ordered, df_smiles

    @staticmethod
    def create_dummy_spectra(df_smiles: pd.DataFrame) -> list[SpectrumExt]:
        """
        Create dummy spectra based on the smiles information and associated metadata.
        The spectra will have empty mz and intensity arrays.

        Parameters
        ----------
        df_smiles : pd.DataFrame
            DataFrame containing smiles and associated metadata.

        Returns
        -------
        List[SpectrumExt]
            A list of dummy SpectrumExt objects.
        """
        # Use DataFrame.itertuples for faster iteration and preallocate arrays
        zeros_array = np.zeros(1)
        nan_value = np.nan
        dummy_spectra = [
            SpectrumExt(
                identifier=str(row.Index),
                precursor_mz=row.mz,
                precursor_charge=row.charge,
                mz=zeros_array,
                intensity=zeros_array,
                retention_time=nan_value,
                params={"smiles": row.canon_smiles},
                library=row.library,
                inchi=row.inchi,
                smiles=row.canon_smiles,
                ionmode=None,
                adduct=None,
                ce=None,
                ion_activation="",
                ionization_method="",
                bms=row.bms,
                superclass=row.superclass,
                classe=row.classe,
                subclass=row.subclass,
            )
            for row in df_smiles.itertuples()
        ]
        return dummy_spectra

    @staticmethod
    def compute_all_tanimoto_results_unique(
        spectra_original,
        max_combinations=1000000,
        limit_low_tanimoto=True,
        max_low_pairs=0.5,
        use_tqdm=True,
        max_mass_diff=None,  # maximum number of elements in which we stop adding new items
        min_mass_diff=0,
        num_workers=15,
        MIN_SIM=0.8,
        MAX_SIM=1,
        high_tanimoto_range=0.5,
        use_exhaustive=True,
    ):
        """
        compute tanimoto results using unique spectrums
        """

        print("Computing tanimoto results based on unique smiles")

        function_tanimoto = (
            TrainUtils.compute_all_tanimoto_results_exhaustive
            if use_exhaustive
            else TrainUtils.compute_all_tanimoto_results
        )

        spectrums_unique, df_smiles = TrainUtils.get_unique_spectra(spectra_original)

        molecule_pairs_unique = function_tanimoto(
            spectrums_unique,
            max_combinations=max_combinations,
            limit_low_tanimoto=limit_low_tanimoto,
            max_low_pairs=max_low_pairs,
            use_tqdm=use_tqdm,
            max_mass_diff=max_mass_diff,  # maximum number of elements in which we stop adding new items
            min_mass_diff=min_mass_diff,
            num_workers=num_workers,
            MIN_SIM=MIN_SIM,
            MAX_SIM=MAX_SIM,
            high_tanimoto_range=high_tanimoto_range,
        )
        return MoleculePairsOpt(
            original_spectra=spectra_original,
            unique_spectra=spectrums_unique,
            df_smiles=df_smiles,
            pair_distances=molecule_pairs_unique.pair_distances,
        )

    @staticmethod
    def compute_all_tanimoto_results(
        all_spectrums,
        max_combinations=1000000,
        limit_low_tanimoto=True,
        max_low_pairs=0.5,
        use_tqdm=True,
        max_mass_diff=None,  # maximum number of elements in which we stop adding new items
        min_mass_diff=0,
        num_workers=15,
        MIN_SIM=0.8,
        MAX_SIM=1,
        high_tanimoto_range=0.5,
    ):
        print("Starting computation of molecule pairs")
        print(datetime.now())
        # asume the spectra is already ordered previously
        # all_spectrums = PreprocessingUtils.order_spectrums_by_mz(all_spectrums)

        # indexes=[]
        indexes_np = np.zeros((max_combinations, 3))
        indexes_np = MoleculePairsOpt.adjust_data_format(np.array(indexes_np))

        counter_indexes = 0
        # Iterate through the list to form pairsi

        print("Computing all the tanimoto results")
        if use_tqdm:
            # Initialize tqdm with the total number of iterations
            progress_bar = tqdm(total=max_combinations, desc="Processing")
            # progress_bar = tqdm(total=len(all_spectrums), desc="Processing")
        # Compute all the fingerprints:
        print("Compute all the fingerprints")
        fingerprints = TrainUtils.compute_all_fingerprints(all_spectrums)

        # get random indexes for the first part of the pair
        # random_i_np = np.random.randint(0, len(all_spectrums)-2, max_combinations)

        print(f"Number of workers: {num_workers}")
        counter_indexes = 0

        # precompute min and max index
        df_precomputed_indexes = TrainUtils.precompute_min_max_indexes(
            all_spectrums,
            min_mass_diff=min_mass_diff,
            max_mass_diff=max_mass_diff,
            use_tqdm=use_tqdm,
        )

        while counter_indexes < (max_combinations):
            # to use the whole range or only a small range?
            use_all_range = np.random.randint(0, 2)

            i = np.random.randint(0, len(all_spectrums))
            if use_all_range == 1:
                min_mz_index = i
                max_mz_index = len(all_spectrums) - 1
            else:
                min_mz_index = df_precomputed_indexes.loc[i, "min_index"]
                max_mz_index = df_precomputed_indexes.loc[i, "max_index"]

            # get the other index
            j = random.randint(min_mz_index, max_mz_index)

            # order indexes to avoid duplicates
            i, j = min(i, j), max(i, j)

            # Submit the task to the executor
            tani = Tanimoto.compute_tanimoto(
                fingerprints[i],
                fingerprints[j],
            )

            if (tani is not None) and (
                (counter_indexes < (max_low_pairs * max_combinations))
                or (tani > high_tanimoto_range)
            ):
                indexes_np[counter_indexes, 0] = i
                indexes_np[counter_indexes, 1] = j
                indexes_np[counter_indexes, 2] = tani
                counter_indexes = counter_indexes + 1
                if use_tqdm:
                    progress_bar.update(1)

        # avoid duplicates:
        print(f"Number of effective pairs originally computed: {indexes_np.shape[0]} ")
        indexes_np = np.unique(indexes_np, axis=0)

        # remove reordered

        print(f"Number of effective pairs retrieved: {indexes_np.shape[0]} ")
        # molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes)
        molecular_pair_set = MolecularPairsSet(
            spectra=all_spectrums, pair_distances=indexes_np
        )

        print(datetime.now())
        return molecular_pair_set

    @staticmethod
    def compute_all_tanimoto_results_exhaustive(
        all_spectrums,
        max_combinations=1000000,
        limit_low_tanimoto=True,
        max_low_pairs=0.5,
        use_tqdm=True,
        max_mass_diff=None,  # maximum number of elements in which we stop adding new items
        min_mass_diff=0,
        num_workers=15,
        MIN_SIM=0.8,
        MAX_SIM=1,
        high_tanimoto_range=0.5,
    ):
        print("Starting computation of molecule pairs")
        print(datetime.now())
        # asume the spectra is already ordered previously
        # all_spectrums = PreprocessingUtils.order_spectrums_by_mz(all_spectrums)

        # indexes=[]
        first_row = 0
        max_row = int(len(all_spectrums))
        M = max_row - first_row
        N = len(all_spectrums)

        exhaustive_combinations = M * N
        indexes_np = np.zeros(((exhaustive_combinations), 3))
        # indexes_np = MoleculePairsOpt.adjust_data_format(np.array(indexes_np))

        counter_indexes = 0
        # Iterate through the list to form pairsi

        print("Computing all the tanimoto results")
        if use_tqdm:
            # Initialize tqdm with the total number of iterations
            progress_bar = tqdm(total=exhaustive_combinations, desc="Processing")
            # progress_bar = tqdm(total=len(all_spectrums), desc="Processing")
        # Compute all the fingerprints:
        print("Compute all the fingerprints")
        fingerprints = TrainUtils.compute_all_fingerprints(all_spectrums)

        # get random indexes for the first part of the pair
        # random_i_np = np.random.randint(0, len(all_spectrums)-2, max_combinations)

        print(f"Number of workers: {num_workers}")
        counter_indexes = 0

        # precompute min and max index
        _ = TrainUtils.precompute_min_max_indexes(
            all_spectrums,
            min_mass_diff=min_mass_diff,
            max_mass_diff=max_mass_diff,
            use_tqdm=use_tqdm,
        )

        for i in range(first_row, max_row):
            # to use the whole range or only a small range?
            for j in range(0, len(all_spectrums)):
                # Submit the task to the executor
                tani = Tanimoto.compute_tanimoto(
                    fingerprints[i],
                    fingerprints[j],
                )

                indexes_np[counter_indexes, 0] = i
                indexes_np[counter_indexes, 1] = j
                if tani is None:
                    indexes_np[counter_indexes, 2] = 2
                else:
                    indexes_np[counter_indexes, 2] = tani

                if use_tqdm:
                    progress_bar.update(1)

                counter_indexes = counter_indexes + 1

        print(f"Number of combinations computed: {counter_indexes}")
        # print('Remove similarities not computed')
        # indexes_np = indexes_np[indexes_np[:,2]<=1]
        #
        # avoid duplicates:
        # print(f"Number of effective pairs originally computed: {indexes_np.shape[0]} ")
        # indexes_np = np.unique(indexes_np, axis=0)

        # remove reordered

        print(f"Number of effective pairs retrieved: {indexes_np.shape[0]} ")
        # molecular_pair_set= MolecularPairsSet(spectrums=all_spectrums,indexes_tani= indexes)
        molecular_pair_set = MolecularPairsSet(
            spectra=all_spectrums, pair_distances=indexes_np
        )

        print(datetime.now())
        return molecular_pair_set

    @staticmethod
    def count_ranges(
        list_elements: np.ndarray,
        number_bins: int = 5,
        bin_sim_1: bool = False,
        max_value: float = 1,
    ) -> tuple[list[int], list[float]]:
        """
        count the number of elements in the different bins

        Parameters
        ----------
        list_elements : list or np.array
            List of numerical values to be binned.
        number_bins : int
            Number of bins to divide the data into.
        bin_sim_1 : bool
            If True, treat the maximum value (e.g., 1) as a separate bin.
        max_value : float
            Maximum value for normalization (default is 1).

        Returns
        -------
        Tuple[List[int], List[float]]
            A tuple containing two lists:
            - counts: Number of elements in each bin.
            - bins: The lower bound of each bin.
        """
        # count the instances in the  bins from 0 to 1
        # Group the values into the corresponding bins, adding one for sim=1
        counts = []
        bins = []

        # normalize the elements of list_elements based on max_value
        list_elements_norm = list_elements / max_value

        number_bins_effective = number_bins + 1 if bin_sim_1 else number_bins

        for p in range(int(number_bins_effective)):
            low = -np.inf if p == 0 else p * (1 / number_bins)

            if bin_sim_1:
                high = (p + 1) * (1 / number_bins)
            else:
                if p == (number_bins_effective - 1):
                    high = np.inf
                else:
                    high = (p + 1) * (1 / number_bins)

            list_elements_temp = list_elements_norm[
                (list_elements_norm >= low) & (list_elements_norm < high)
            ]
            counts.append(len(list_elements_temp))
            bins.append(low)
        return counts, bins

    @staticmethod
    def divide_data_into_bins(
        molecule_pairs,
        number_bins,
        bin_sim_1=False,  # if you want to try sim=1 as a different bin
    ):
        # Initialize lists to store values for each bin
        binned_molecule_pairs = []

        # Group the values into the corresponding bins, adding one for sim=1
        number_bins_effective = number_bins + 1 if bin_sim_1 else number_bins

        for p in range(int(number_bins_effective)):
            low = p * (1 / number_bins)

            if bin_sim_1:
                high = (p + 1) * (1 / number_bins)
            else:
                if p == (number_bins_effective - 1):
                    high = 1 + 0.1
                else:
                    high = (p + 1) * (1 / number_bins)

            # temp_molecule_pairs = [m for m in molecule_pairs if ((m.similarity>=low) and (m.similarity<high))]
            # check the similarity
            # temp_indexes_tani = np.array([ row for row in molecule_pairs.pair_distances if ((row[2]>=low) and (row[2]<high)) ])
            pair_distances_temp = molecule_pairs.pair_distances[
                (molecule_pairs.pair_distances[:, 2] >= low)
                & (molecule_pairs.pair_distances[:, 2] < high)
            ]

            if molecule_pairs.extra_distances is not None:
                extra_distances_temp = molecule_pairs.extra_distances[
                    (molecule_pairs.pair_distances[:, 2] >= low)
                    & (molecule_pairs.pair_distances[:, 2] < high)
                ]
            else:
                extra_distances_temp = None

            temp_molecule_pairs = MoleculePairsOpt(
                unique_spectra=molecule_pairs.spectra,
                pair_distances=pair_distances_temp,
                df_smiles=molecule_pairs.df_smiles,
                original_spectra=molecule_pairs.original_spectra,
                extra_distances=extra_distances_temp,
            )
            binned_molecule_pairs.append(temp_molecule_pairs)

        # get minimum bin size
        min_bin = min([len(b) for b in binned_molecule_pairs])
        return binned_molecule_pairs, min_bin

    @staticmethod
    def divide_data_into_bins_categories(
        molecule_pairs: MoleculePairsOpt,
        number_bins,
        bin_sim_1=False,  # if you want to try sim=1 as a different bin
    ):
        """
        divide data into bins using ordinal classification approach
        """
        # Initialize lists to store values for each bin
        binned_molecule_pairs = []

        # Group the values into the corresponding bins, adding one for sim=1
        number_bins_effective = number_bins + 1 if bin_sim_1 else number_bins

        # convert it to an integer
        bin_size = 1 / number_bins
        # target = np.ceil(molecule_pairs.pair_distances[:, 2]/bin_size)
        target = OrdinalClassification.custom_random(
            molecule_pairs.pair_distances[:, 2] / bin_size
        )
        for p in range(int(number_bins_effective)):
            # low = p * (1 / number_bins)

            # if bin_sim_1:
            #    high = (p + 1) * (1 / number_bins)
            # else:
            #    if p == (number_bins_effective - 1):
            #        high = 1 + 0.1
            #    else:
            #        high = (p + 1) * (1 / number_bins)

            # temp_molecule_pairs = [m for m in molecule_pairs if ((m.similarity>=low) and (m.similarity<high))]
            # check the similarity
            # temp_indexes_tani = np.array([ row for row in molecule_pairs.pair_distances if ((row[2]>=low) and (row[2]<high)) ])
            pair_dists_temp = molecule_pairs.pair_distances[(target == p)]

            if molecule_pairs.extra_distances is not None:
                extra_dists_temp = molecule_pairs.extra_distances[(target == p)]
            else:
                extra_dists_temp = None
            temp_molecule_pairs = MoleculePairsOpt(
                unique_spectra=molecule_pairs.spectra,
                pair_distances=pair_dists_temp,
                df_smiles=molecule_pairs.df_smiles,
                original_spectra=molecule_pairs.original_spectra,
                extra_distances=extra_dists_temp,
            )
            binned_molecule_pairs.append(temp_molecule_pairs)

        # get minimum bin size
        min_bin = min([len(b) for b in binned_molecule_pairs])
        return binned_molecule_pairs, min_bin

    @staticmethod
    def uniformise(
        molecule_pairs,
        number_bins=3,
        return_binned_list=False,
        bin_sim_1=True,  # if you want to treat sim=1 as another bin
        seed=42,
        ordinal_classification=False,
    ):
        """
        get a uniform distribution of labels between 0 and 1
        """

        # initialize random seed
        random.seed(seed)
        np.random.seed(seed)

        # choose function
        function = (
            TrainUtils.divide_data_into_bins_categories
            if ordinal_classification
            else TrainUtils.divide_data_into_bins
        )

        # min_bin = TrainUtils.get_min_bin(molecule_pairs, number_bins)
        binned_molecule_pairs, min_bin = function(
            molecule_pairs, number_bins, bin_sim_1=bin_sim_1
        )

        uniform_molecule_pairs = None

        for target_molecule_pairs in binned_molecule_pairs:
            sampled_rows = np.random.choice(
                target_molecule_pairs.pair_distances.shape[0],
                size=min_bin,
                replace=False,
            )
            sampled_indexes_tani = target_molecule_pairs.pair_distances[sampled_rows]

            ## check if there are tanimotos as second similarity metric appended
            if target_molecule_pairs.extra_distances is not None:
                tanimotos = target_molecule_pairs.extra_distances[sampled_rows]
            else:
                tanimotos = None

            sampled_molecule_pairs = MoleculePairsOpt(
                unique_spectra=target_molecule_pairs.spectra,
                original_spectra=target_molecule_pairs.original_spectra,
                pair_distances=sampled_indexes_tani,
                df_smiles=target_molecule_pairs.df_smiles,
                extra_distances=tanimotos,
            )
            # add to the final list

            if uniform_molecule_pairs is None:
                uniform_molecule_pairs = sampled_molecule_pairs
            else:
                uniform_molecule_pairs = uniform_molecule_pairs + sampled_molecule_pairs

        # insert spectrum vectors
        # uniform_molecule_pairs = TrainUtils.insert_spectrum_vector_into_molecule_pairs(uniform_molecule_pairs)

        if return_binned_list:
            return uniform_molecule_pairs, binned_molecule_pairs
        else:
            return uniform_molecule_pairs

    @staticmethod
    def get_data_from_indexes(spectrums, indexes):
        return [
            (
                spectrums[p[0]].spectrum_vector,
                TrainUtils.get_global_variables(spectrums[p[0]]),
                spectrums[p[1]].spectrum_vector,
                TrainUtils.get_global_variables(spectrums[p[1]]),
            )
            for p in indexes
        ]

    @staticmethod
    def get_global_variables(spectrum):
        """
        get global variables from a spectrum such as precursor mass
        """
        list_global_variables = [
            spectrum.precursor_mz,
            spectrum.precursor_charge,
        ]
        return np.array(list_global_variables)
