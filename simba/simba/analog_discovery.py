

class AnalogDiscovery:
    def compute_ranking(similarities_mces, similarities_ed, max_value_2_int=5):
    '''

    '''
    similarities_mces_integer= np.round(similarities_mces)
    # Preallocate the ranking array with the same shape as similarities1.
    ranking_total = np.zeros(similarities_mces.shape, dtype=int)
    
    # Process each row (or each set of values) individually.
    for row_index, (row_sim, row_int, row_int2) in enumerate(zip(similarities_mces, similarities_mces_integer, similarities_ed)):
        # Use lexsort with a composite key:
        #   - Primary: similarities1_integer (ascending)
        #   - Secondary: similarities2_integer (ascending)
        #   - Tertiary: similarities1 (descending, so use -row_sim)
        #
        # Note: np.lexsort uses the last key as the primary key.
        sorted_indices = np.lexsort( ( row_sim, row_int2, row_int ) )
        
        # Now assign ranking values based on sorted order.
        # Here the best (first in sorted_indices) gets rank 0,
        # the next gets rank 1, etc.
        ranking = np.empty_like(sorted_indices)
        ranking[sorted_indices] = np.arange(len(row_sim))
        
        # Store the ranking for this row.
        ranking_total[row_index] = ranking

    #normalizing
    ranking_total =1 - ranking_total/ranking_total.shape[1]
    return ranking_total