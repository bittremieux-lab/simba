from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
from tqdm import tqdm
from matchms.similarity import FingerprintSimilarity
from matchms.filtering import add_fingerprint
from matchms import calculate_scores
import tensorflow as tf


class MS2DeepScoreComparison:

    def get_ms2deepscore_similarity(model_ms2d_file):

        # Load pretrained model
        model_ms2d = load_model(model_ms2d_file)
        return MS2DeepScore(model_ms2d)

    def compute_ms2deepscore(
        molecule_pairs,
        molecule_pairs_indexes,
        original_spectrum_match_hash,
        target_hashes_subset,
        similarity_ms2,
        compute_tanimoto=False,  # if to retrieve the similarity from the molecular pairs
    ):
        tanimotos = []
        scores_ms2d = []

        spectrum_found_0_ms_all = []
        spectrum_found_1_ms_all = []
        molecules_filtered = []
        for i in tqdm(molecule_pairs_indexes):
            m = molecule_pairs[i]
            hash_0 = m.spectrum_object_0.spectrum_hash
            hash_1 = m.spectrum_object_1.spectrum_hash

            spectrum_found_0_ms = next(
                s
                for s, t in zip(original_spectrum_match_hash, target_hashes_subset)
                if t == hash_0
            )
            spectrum_found_1_ms = next(
                s
                for s, t in zip(original_spectrum_match_hash, target_hashes_subset)
                if t == hash_1
            )

            if (spectrum_found_0_ms is not None) and (spectrum_found_1_ms is not None):
                spectrum_found_0_ms_all.append(spectrum_found_0_ms)
                spectrum_found_1_ms_all.append(spectrum_found_1_ms)
                molecules_filtered.append(m)
            # else:
            #    tanimotos.append(None)

        results = calculate_scores(
            spectrum_found_0_ms_all, spectrum_found_1_ms_all, similarity_ms2
        )
        sim_array = results.to_array()

        spectrum_found_0_ms_all = [add_fingerprint(s) for s in spectrum_found_0_ms_all]
        spectrum_found_1_ms_all = [add_fingerprint(s) for s in spectrum_found_1_ms_all]

        print(sim_array)
        for i, (s0, s1, m) in enumerate(
            zip(spectrum_found_0_ms_all, spectrum_found_1_ms_all, molecules_filtered)
        ):

            # try:
            #    results_tuple = results.scores_by_query([s1], name='MS2DeepScore')
            #    score = results_tuple[i][1]
            if compute_tanimoto:
                tanimoto_measure = FingerprintSimilarity(similarity_measure="jaccard")

                tani = tanimoto_measure.pair(s0, s1)
            else:
                tani = m.similarity
            tanimotos.append(tani)
            score = sim_array[i, i]
            scores_ms2d.append(score)
        # except:
        #    print(i)
        #    print('Problem while computing MS2')
        # for spectrum_found_0_ms, spectrum_found_1_ms in zip(spectrum_found_0_ms_all, spectrum_found_1_ms_all):
        #    # calculate scores
        #    if (spectrum_found_0_ms is not None) and (spectrum_found_1_ms is not None):

        #        score = similarity_ms2.pair(spectrum_found_0_ms, spectrum_found_1_ms)
        #        scores_ms2d.append(score)
        #    else:
        #        tanimotos.append(None)
        #        scores_ms2d.append(None)

        return tanimotos, scores_ms2d
