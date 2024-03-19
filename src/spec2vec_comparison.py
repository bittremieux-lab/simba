import gensim
import matchms.filtering as msfilters

from spec2vec import Spec2Vec
from tqdm import tqdm
from matchms import calculate_scores
from matchms.similarity import FingerprintSimilarity
from matchms.filtering import add_fingerprint


class Spec2VecComparison:
    """
    comparison with other dl approaches
    """

    def get_spec2vec_similarity(model_spec2vec_file):
        model = gensim.models.Word2Vec.load(model_spec2vec_file)
        return Spec2Vec(
            model=model, intensity_weighting_power=0.5, allowed_missing_percentage=100.0
        )

    def spectrum_processing(s):
        """This is how one would typically design a desired pre- and post-
        processing pipeline."""
        s = msfilters.default_filters(s)
        s = msfilters.add_parent_mass(s)
        s = msfilters.normalize_intensities(s)
        s = msfilters.reduce_to_number_of_peaks(
            s, n_required=10, ratio_desired=None, n_max=500
        )
        # s = msfilters.reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5, n_max=500)
        s = msfilters.select_by_mz(s, mz_from=0, mz_to=1000)
        s = msfilters.add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
        s = msfilters.require_minimum_number_of_peaks(s, n_required=10)
        return s

    def compute_scores_tanimoto(
        molecule_pairs,
        preprocessed_spectrums,
        target_hashes_subset,
        spec2vec_similarity,
        compute_tanimoto=False,  # if to retrieve the similarity from the molecular pairs
    ):
        tanimotos = []
        scores = []
        for m in tqdm(molecule_pairs):

            hash_0 = m.spectrum_object_0.spectrum_hash
            hash_1 = m.spectrum_object_1.spectrum_hash

            # get right spectra
            spectrum_found_0_ms = next(
                s
                for s, t in zip(preprocessed_spectrums, target_hashes_subset)
                if t == hash_0
            )
            spectrum_found_1_ms = next(
                s
                for s, t in zip(preprocessed_spectrums, target_hashes_subset)
                if t == hash_1
            )

            # calculate scores
            if (spectrum_found_0_ms is not None) and (spectrum_found_1_ms is not None):
                score = calculate_scores(
                    [spectrum_found_0_ms], [spectrum_found_1_ms], spec2vec_similarity
                )

                score = score.scores_by_query(spectrum_found_1_ms, sort=True)[0][1]

                if compute_tanimoto:
                    tanimoto_measure = FingerprintSimilarity(
                        similarity_measure="jaccard"
                    )
                    tani = tanimoto_measure.pair(
                        spectrum_found_0_ms, spectrum_found_1_ms
                    )
                else:
                    tani = m.similarity
                tanimotos.append(tani)
                # tanimotos.append(m.similarity)
                scores.append(score)
            else:
                tanimotos.append(None)
                scores.append(None)
        return tanimotos, scores
