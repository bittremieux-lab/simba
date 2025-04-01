from simba.train_utils import TrainUtils


class SanityChecks:

    @staticmethod
    def sanity_checks_ids(
        molecules_pairs_train,
        molecules_pairs_val,
        molecules_pairs_test,
        uniformed_molecule_pairs_test,
    ):
        """
        no train ids in test/val
        """
        ids_train = [s.spectrum_hash for s in molecules_pairs_train.spectrums]
        ids_val = [s.spectrum_hash for s in molecules_pairs_val.spectrums]
        ids_test = [s.spectrum_hash for s in molecules_pairs_test.spectrums]
        

        is_any_train_in_val = any([(id in ids_train) for id in [ids_val]])
        is_any_train_in_test = any([(id in ids_train) for id in [ids_test]])

        return not (
            is_any_train_in_val + is_any_train_in_test 
        )

    @staticmethod
    def sanity_checks_bms(
        molecules_pairs_train,
        molecules_pairs_val,
        molecules_pairs_test,
        uniformed_molecule_pairs_test,
    ):
        """
        different mruck scaffold between train and test
        """
        bms_train = [s.murcko_scaffold for s in molecules_pairs_train.spectrums]
        bms_val = [s.murcko_scaffold for s in molecules_pairs_val.spectrums]
        bms_test = [s.murcko_scaffold for s in molecules_pairs_test.spectrums]

        is_any_bms_train_in_val = any([(id in bms_train) for id in [bms_val]])
        is_any_bms_train_in_test = any([(id in bms_train) for id in [bms_test]])

        return not (
            is_any_bms_train_in_val
            + is_any_bms_train_in_test
       
        )

    # check distribution of similarities
    def check_distribution_similarities(molecule_pairs, bins=10):
        train_binned_list, _ = TrainUtils.divide_data_into_bins(molecule_pairs, bins)
        samples_per_range = [len(t) for t in train_binned_list]
        bins = [(n / bins) for n in range(len(samples_per_range))]

        return samples_per_range, bins
