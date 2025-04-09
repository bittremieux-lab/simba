class SimilarityMetric:

    def compute_scores_tanimoto(
        molecule_pairs,
        preprocessed_spectrums,
        target_hashes_subset,
        similarity,
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
                    [spectrum_found_0_ms], [spectrum_found_1_ms], similarity
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
