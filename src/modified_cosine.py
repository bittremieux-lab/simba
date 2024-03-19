from matchms.similarity import ModifiedCosine
from tqdm import tqdm
from matchms import calculate_scores


class ModCosine:
    @staticmethod
    def get_mod_cosine():
        return ModifiedCosine(tolerance=0.1)

    def compute_scores_tanimoto(
        molecule_pairs,
        preprocessed_spectrums,
        target_hashes_subset,
        modified_cosine,
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
            not_none = (spectrum_found_0_ms is not None) and (
                spectrum_found_1_ms is not None
            )

            if not_none:
                precursor_mz_positive = (
                    spectrum_found_0_ms.metadata["precursor_mz"] > 0
                ) and (spectrum_found_1_ms.metadata["precursor_mz"] > 0)

                if precursor_mz_positive:

                    if compute_tanimoto:
                        tanimoto_measure = FingerprintSimilarity(
                            similarity_measure="jaccard"
                        )
                        tani = tanimoto_measure.pair(
                            spectrum_found_0_ms, spectrum_found_1_ms
                        )
                    else:
                        tani = m.similarity
                    score = modified_cosine.pair(
                        spectrum_found_0_ms, spectrum_found_1_ms
                    )
                    tanimotos.append(tani)
                    # tanimotos.append(m.similarity)
                    score = float(score["score"])
                    scores.append(score)
                else:
                    tanimotos.append(None)
                    scores.append(None)
            else:
                print("Some Nones in spectrums")
            #    tanimotos.append(None)
            #    scores.append(None)
        return tanimotos, scores
