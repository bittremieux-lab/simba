import random
from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from simba.core.data.molecule_pairs import MoleculePairsOpt
from simba.core.data.preprocessor import PreprocessingUtils
from simba.core.data.spectrum import SpectrumExt
from simba.utils.binning import round_to_ordinal
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
    def _scaffold_bucket(scaffold: str, n_buckets: int = 10) -> int:
        """Return a deterministic bucket index in [0, n_buckets) for a scaffold string.

        Uses SHA-256 so the assignment depends only on the scaffold identity,
        not on what other scaffolds happen to be in the dataset.
        """
        import hashlib

        digest = hashlib.sha256(scaffold.encode()).hexdigest()
        return int(digest, 16) % n_buckets

    @staticmethod
    def train_val_test_split_bms(
        spectra: list[SpectrumExt],
        n_buckets: int = 10,
        train_buckets: list[int] = None,
        val_buckets: list[int] = None,
        test_buckets: list[int] = None,
        force_scaffold_split: bool = False,
    ) -> tuple[list[SpectrumExt], list[SpectrumExt], list[SpectrumExt]]:
        """
        Split spectra into train / val / test.

        If any spectrum has a non-null ``fold`` attribute (e.g. "train"/"val"/"test"
        as shipped in MassSpecGym MGF files), those labels are used directly for all
        spectra and the scaffold-hashing logic is skipped entirely — unless
        ``force_scaffold_split=True``, which ignores predefined labels and always
        applies Murcko scaffold hashing.

        Otherwise each scaffold is assigned to a bucket by SHA-256(scaffold) % n_buckets.

        Default: n_buckets=10, train=[0..7], val=[8], test=[9]
        Spectra whose scaffold is empty are put in train.

        Parameters
        ----------
        spectra:
            List of SpectrumExt objects to be split.
        n_buckets:
            Total number of hash buckets (modulus for SHA-256 hash).
        train_buckets:
            Bucket numbers that go to train (default 0-7).
        val_buckets:
            Bucket numbers that go to val (default [8]).
        test_buckets:
            Bucket numbers that go to test (default [9]).
        force_scaffold_split:
            If True, ignore predefined fold labels and always use Murcko scaffold
            hashing. Useful when the dataset ships with splits (e.g. MassSpecGym)
            but you want a scaffold-based split instead.
        """
        # Use predefined fold labels when present (e.g. MassSpecGym)
        if not force_scaffold_split and any(
            getattr(s, "fold", None) is not None for s in spectra
        ):
            spectrums_train, spectrums_val, spectrums_test = [], [], []
            unlabeled = 0
            for s in spectra:
                fold = getattr(s, "fold", None)
                if fold == "val":
                    spectrums_val.append(s)
                elif fold == "test":
                    spectrums_test.append(s)
                else:
                    spectrums_train.append(s)
                    if fold != "train":
                        unlabeled += 1
            if unlabeled:
                logger.info(
                    f"{unlabeled} spectra with unknown fold label assigned to train"
                )
            total = len(spectra)
            logger.info(
                f"Using predefined fold labels – Train: {len(spectrums_train)} "
                f"({100 * len(spectrums_train) / total:.1f}%), "
                f"Val: {len(spectrums_val)} ({100 * len(spectrums_val) / total:.1f}%), "
                f"Test: {len(spectrums_test)} ({100 * len(spectrums_test) / total:.1f}%)"
            )
            return spectrums_train, spectrums_val, spectrums_test

        if train_buckets is None:
            train_buckets = list(range(n_buckets - 2))
        if val_buckets is None:
            val_buckets = [n_buckets - 2]
        if test_buckets is None:
            test_buckets = [n_buckets - 1]

        val_set = set(val_buckets)
        test_set = set(test_buckets)

        bms = [s.murcko_scaffold for s in spectra]
        no_bms = sum(1 for b in bms if not b)
        if no_bms:
            logger.info(f"{no_bms}/{len(bms)} spectra without bms")

        spectrums_train, spectrums_val, spectrums_test = [], [], []
        for s, scaffold in zip(spectra, bms):
            if not scaffold:
                spectrums_train.append(s)
                continue
            bucket = TrainUtils._scaffold_bucket(scaffold, n_buckets)
            if bucket in test_set:
                spectrums_test.append(s)
            elif bucket in val_set:
                spectrums_val.append(s)
            else:
                spectrums_train.append(s)

        total = len(spectra)
        logger.info(
            f"Split sizes – Train: {len(spectrums_train)} ({100 * len(spectrums_train) / total:.1f}%), "
            f"Val: {len(spectrums_val)} ({100 * len(spectrums_val) / total:.1f}%), "
            f"Test: {len(spectrums_test)} ({100 * len(spectrums_test) / total:.1f}%) "
            f"[train buckets: {sorted(train_buckets)}, val: {sorted(val_buckets)}, test: {sorted(test_buckets)}]"
        )

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
        logger.info(f"Finding unique spectra from {len(all_spectra)} total spectra...")

        # convert to canonical smiles; guard against invalid SMILES that would
        # cause a C++ abort inside CanonSmiles -> MolToSmiles(None)
        canon_smiles = []
        for s in all_spectra:
            mol = Chem.MolFromSmiles(s.smiles)
            if mol is None:
                logger.warning(f"Could not parse SMILES, using raw: {s.smiles}")
                canon_smiles.append(s.smiles)
            else:
                canon_smiles.append(Chem.MolToSmiles(mol))

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
        logger.info(
            f"Found {len(unique_smiles)} unique SMILES from {len(all_spectra)} spectra (compression: {len(all_spectra) / len(unique_smiles):.2f}x)"
        )

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

        logger.info(
            f"Created {len(spectra_unique_ordered)} unique dummy spectra from {len(all_spectra)} input spectra"
        )

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
        target = round_to_ordinal(molecule_pairs.pair_distances[:, 2] / bin_size)
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
