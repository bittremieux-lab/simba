import numpy as np


class AnalogDiscovery:
    @staticmethod
    def compute_ranking(similarities_mces, similarities_ed, max_value_2_int=5):
        """
        based on mces and edit distance rerank.
        If 2 matches have the same mces, choose the one with lowest edit distance
        """
        similarities_mces_integer = np.round(similarities_mces)
        # Preallocate the ranking array with the same shape as similarities1.
        ranking_total = np.zeros(similarities_mces.shape, dtype=int)

        # Process each row (or each set of values) individually.
        for row_index, (row_sim, row_int, row_int2) in enumerate(
            zip(
                similarities_mces,
                similarities_mces_integer,
                similarities_ed,
                strict=False,
            )
        ):
            # Use lexsort with a composite key:
            #   - Primary: similarities1_integer (ascending)
            #   - Secondary: similarities2_integer (ascending)
            #   - Tertiary: similarities1 (descending, so use -row_sim)
            #
            # Note: np.lexsort uses the last key as the primary key.
            sorted_indices = np.lexsort((row_sim, row_int2, row_int))

            # Now assign ranking values based on sorted order.
            # Here the best (first in sorted_indices) gets rank 0,
            # the next gets rank 1, etc.
            ranking = np.empty_like(sorted_indices)
            ranking[sorted_indices] = np.arange(len(row_sim))

            # Store the ranking for this row.
            ranking_total[row_index] = ranking

        # normalizing
        ranking_total = 1 - ranking_total / ranking_total.shape[1]
        return ranking_total

    @staticmethod
    def compute_ranking_with_precursor_mz(
        similarities_mces,
        similarities_ed,
        precursor_mz_differences,
    ):
        """
        Rank candidates using:

        1. Lowest MCES
        2. Lowest edit distance
        3. Lowest absolute precursor m/z difference

        All three arrays must have shape:
            (n_queries, n_references)

        Returns
        -------
        np.ndarray
            Normalized rankings where 1.0 is the best candidate and
            0.0 is the worst candidate.
        """
        similarities_mces = np.asarray(similarities_mces, dtype=int)
        similarities_ed = np.asarray(similarities_ed, dtype=int)
        precursor_mz_differences = np.asarray(
            precursor_mz_differences,
            dtype=float,
        )

        if not (
            similarities_mces.shape
            == similarities_ed.shape
            == precursor_mz_differences.shape
        ):
            raise ValueError(
                "MCES, edit distance, and precursor m/z differences "
                "must have the same shape."
            )

        ranking_total = np.zeros(similarities_mces.shape, dtype=float)

        for row_index, (row_mces, row_ed, row_mz_diff) in enumerate(
            zip(
                similarities_mces,
                similarities_ed,
                precursor_mz_differences,
                strict=False,
            )
        ):
            # np.lexsort uses the last key as the primary key.
            #
            # Primary:   lowest MCES
            # Secondary: lowest edit distance
            # Tertiary:  lowest precursor m/z difference
            sorted_indices = np.lexsort(
                (
                    row_mz_diff,
                    row_ed,
                    row_mces,
                )
            )

            ranking = np.empty(len(sorted_indices), dtype=int)
            ranking[sorted_indices] = np.arange(len(sorted_indices))

            # Best candidate -> 1.0
            # Worst candidate -> 0.0
            if len(ranking) > 1:
                ranking_total[row_index] = (
                    1.0 - ranking / (len(ranking) - 1)
                )
            else:
                ranking_total[row_index] = 1.0

        return ranking_total
