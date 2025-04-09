from simba.tanimoto import Tanimoto
import numpy as np


class AnalogDiscovery:

    def get_analog_spectrums_matchms(
        results_scores,
        spectrums_reference,
        spectrums_query,
    ):
        """
        based on finding k=10 best ranked spectrums and then finding the best tanimoto
        """
        spectrums_k_retrieved, max_k_sim = AnalogDiscovery.get_k_ranking_matchms(
            results_scores, spectrums_query
        )
        spectrums_retrieved, tanimoto_retrieved, max_sim = (
            AnalogDiscovery.get_best_tanimoto_from_k_ranking(
                spectrums_reference, spectrums_query, spectrums_k_retrieved, max_k_sim
            )
        )
        return (
            np.array(spectrums_retrieved),
            np.array(tanimoto_retrieved),
            np.array(max_sim),
        )

    def get_k_candidates(
        similarities,
        spectrums_reference,
        spectrums_query,
        k=100,
        return_arg_max_k10=False,
    ):
        arg_max_k10 = np.argsort(-similarities, axis=1)[:, 0:k]
        sim_k_retrieved = np.take_along_axis(similarities, arg_max_k10, axis=1)

        spectrums_k_retrieved = [
            [spectrums_reference[ind] for ind in ind_group] for ind_group in arg_max_k10
        ]
        smiles_k_retrieved = [
            [s.smiles for s in s_group] for s_group in spectrums_k_retrieved
        ]
        smiles_janssen = [s.smiles for s in spectrums_query]

        # get all k tanimotos
        tanimoto_k_retrieved = [
            [Tanimoto.compute_tanimoto_from_smiles(s0, s1) for s1 in s1_group]
            for s0, s1_group in zip(smiles_janssen, smiles_k_retrieved)
        ]
        if not (return_arg_max_k10):
            return spectrums_k_retrieved, tanimoto_k_retrieved, sim_k_retrieved
        else:
            return (
                spectrums_k_retrieved,
                tanimoto_k_retrieved,
                sim_k_retrieved,
                arg_max_k10,
            )

    def get_analog_spectrums_su(
        similarities, spectrums_reference, spectrums_query, k=10
    ):
        """
        finding analogs using spectrum utils
        """

        arg_max_k10 = np.argsort(-similarities, axis=1)[:, 0:k]
        sorted_k_similarities = np.take_along_axis(similarities, arg_max_k10, axis=1)
        spectrums_k_retrieved = [
            [spectrums_reference[ind] for ind in ind_group] for ind_group in arg_max_k10
        ]
        smiles_k_retrieved = [
            [s.smiles for s in s_group] for s_group in spectrums_k_retrieved
        ]
        smiles_janssen = [s.smiles for s in spectrums_query]

        # get all k tanimotos
        tanimoto_k_retrieved = [
            [Tanimoto.compute_tanimoto_from_smiles(s0, s1) for s1 in s1_group]
            for s0, s1_group in zip(smiles_janssen, smiles_k_retrieved)
        ]

        # get the best tanimoto
        best_index_retrieved = [np.argmax(t) for t in tanimoto_k_retrieved]
        spectrums_retrieved = [
            s_group[ind]
            for ind, s_group in zip(best_index_retrieved, spectrums_k_retrieved)
        ]
        tanimoto_retrieved = [max(t) for t in tanimoto_k_retrieved]
        max_sim = [
            s_group[ind]
            for ind, s_group in zip(best_index_retrieved, sorted_k_similarities)
        ]

        return (
            np.array(spectrums_retrieved),
            np.array(tanimoto_retrieved),
            np.array(max_sim),
        )

    def get_k_ranking_matchms(results_scores, spectrums_query, k=10):
        """
        from the scores obtained, get the 10 best matches
        """
        # results_tuple = [results_scores.scores_by_query(s, name='Spec2Vec', sort=True)[0:k] for s in preprocessed_all_spectrums_janssen]
        results_tuple = [
            results_scores.scores_by_query(s, name="ModifiedCosine_score", sort=True)[
                0:k
            ]
            for s in spectrums_query
        ]

        spectrums_k_retrieved = [[r[0] for r in r_group] for r_group in results_tuple]
        max_sim = [[r[1] for r in r_group] for r_group in results_tuple]

        return spectrums_k_retrieved, max_sim

    def get_best_tanimoto_from_k_ranking(
        spectrums_reference, spectrums_query, spectrums_k_retrieved, max_k_sim
    ):
        """
        get the best tanimoto score from a set of spectrums
        """
        # compute all tanimotos
        smiles_k_retrieved = [
            [s.metadata["smiles"] for s in s_group] for s_group in spectrums_k_retrieved
        ]
        smiles_k_janssen = [s.metadata["smiles"] for s in spectrums_query]
        tanimoto_k_retrieved = [
            [Tanimoto.compute_tanimoto_from_smiles(s0, s1) for s1 in s1_group]
            for s0, s1_group in zip(smiles_k_janssen, smiles_k_retrieved)
        ]

        ## get the best index
        best_index_retrieved = [np.argmax(t) for t in tanimoto_k_retrieved]
        tanimoto_retrieved = [np.max(t) for t in tanimoto_k_retrieved]
        spectrums_retrieved = [
            s_group[ind]
            for ind, s_group in zip(best_index_retrieved, spectrums_k_retrieved)
        ]

        max_sim = [max_k_sim[index] for index in best_index_retrieved]
        return spectrums_retrieved, tanimoto_retrieved, max_sim

    def get_rank_of_best_candidate(
        similarities,
        smiles_janssen,
        smiles_janssen_loaded,
        smiles_reference,
        smiles_best_candidate_loaded,
    ):

        argsort_similarities = np.argsort(-similarities, axis=1)
        list_rankings = []
        for s_janssen, arg_sim_row in zip(smiles_janssen, argsort_similarities):
            # find the smiles that is mapped in the loaded data
            if s_janssen in smiles_janssen_loaded:
                smiles_mapped_loaded_index = smiles_janssen_loaded.index(s_janssen)

                #    get the best candidate smile
                best_candidate_smile = smiles_best_candidate_loaded[
                    smiles_mapped_loaded_index
                ]

                sort_reference_smiles = [
                    smiles_reference[index] for index in arg_sim_row
                ]

                try:
                    rank = sort_reference_smiles.index(best_candidate_smile)

                    list_rankings.append(rank)
                except:
                    print(
                        "The best candidate is not found in the current reference data. Possibly this is spectra was filtered out"
                    )
        return list_rankings
